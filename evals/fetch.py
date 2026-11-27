"""Source acquisition: resolve, download, verify and ledger the eval recordings.

Nothing here runs implicitly. ``run_eval.py fetch`` is the only entry point, it refuses to download a source whose
license has not been accepted with ``--accept-terms``, and every download is appended to ``LEDGER.json`` with its
URL, byte count, SHA-256 and license so the audio's provenance is auditable. Audio stays under ``recordings/`` (gitignored).
"""

from __future__ import annotations

import hashlib
import html as htmllib
import json
import re
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from evals import REPO_ROOT
from evals.manifest import Source, SourcesFile
from evals.stems import AUDIO_EXTS, _audio_files

USER_AGENT = "AetherMix-Eval/0.1 (+local evaluation; contact via repository)"
CAMBRIDGE_INDEX = "https://www.cambridge-mt.com/ms/mtk/"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_META = "https://archive.org/metadata/{id}"
ARCHIVE_DL = "https://archive.org/download/{id}/{name}"
LEDGER = "LEDGER.json"


class FetchError(RuntimeError):
    pass


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download(url: str, dest: Path, max_bytes: int) -> tuple[int, str]:
    """Stream to dest.part, enforce a size cap, return (bytes, sha256)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha256()
    total = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        length = resp.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise FetchError(f"{url}: {int(length) / 1e6:.0f} MB exceeds the {max_bytes / 1e6:.0f} MB cap (raise --max-gb)")
        while chunk := resp.read(1 << 20):
            total += len(chunk)
            if total > max_bytes:
                raise FetchError(f"{url}: exceeded the {max_bytes / 1e6:.0f} MB cap")
            h.update(chunk)
            out.write(chunk)
    tmp.replace(dest)
    return total, h.hexdigest()


# ---------------------------------------------------------------------------
# Pure resolvers (unit tested against fixtures; the live hosts have not been exercised from this environment)
# ---------------------------------------------------------------------------


def resolve_cambridge_zip(page_html: str, artist: str, tier: str | None) -> list[str]:
    """Candidate ZIP URLs for an artist, best first. The library index lists no instruments, so match by name."""
    text = htmllib.unescape(page_html)
    out: list[str] = []
    for m in re.finditer(re.escape(htmllib.unescape(artist)), text, flags=re.IGNORECASE):
        window = text[m.end() : m.end() + 6000]
        for link in re.finditer(r"""href=["']([^"']+?\.zip)["']""", window, flags=re.IGNORECASE):
            url = urljoin(CAMBRIDGE_INDEX, link.group(1))
            if url not in out:
                out.append(url)
        if out:
            break
    want_excerpt = tier == "excerpt"
    out.sort(key=lambda u: (("excerpt" in u.lower() or "edited" in u.lower()) != want_excerpt))
    return out


def find_archive_ids(page_html: str) -> list[str]:
    ids: list[str] = []
    for m in re.finditer(r"archive\.org/(?:details|download|embed)/([A-Za-z0-9_.\-]+)", page_html):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def pick_archive_audio(meta: dict[str, Any], target_minutes: float) -> list[str]:
    """Pick sequential MP3/OGG files from an archive.org item until ~target_minutes of audio."""
    files = [f for f in meta.get("files", []) if str(f.get("name", "")).lower().endswith((".mp3", ".ogg"))]
    mp3 = [f for f in files if "64kb" in str(f.get("name", "")).lower() or "64kb" in str(f.get("format", "")).lower()] or files
    mp3.sort(key=lambda f: f["name"])
    chosen: list[str] = []
    secs = 0.0
    for f in mp3:
        try:
            secs += float(f.get("length", 0) or 0)
        except ValueError:
            pass
        chosen.append(f["name"])
        if secs >= target_minutes * 60.0 or len(chosen) >= 12:
            break
    return chosen


def select_librispeech_speakers(speakers_txt: str, subset: str, gender: str, n: int) -> list[str]:
    """Speaker ids from SPEAKERS.TXT (`ID | SEX | SUBSET | MINUTES | NAME`) for one subset and gender."""
    want = "F" if gender == "female" else "M"
    ids: list[tuple[float, str]] = []
    for line in speakers_txt.splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[1] == want and parts[2] == subset:
            ids.append((-float(parts[3] or 0), parts[0]))
    return [i for _, i in sorted(ids)[:n]]


def _safe_member(dest: Path, name: str) -> Path | None:
    target = (dest / name).resolve()
    return target if str(target).startswith(str(dest.resolve())) else None


def safe_extract_zip(zpath: Path, dest: Path) -> list[Path]:
    """Extract audio members only, refusing paths that escape dest (zip-slip)."""
    out: list[Path] = []
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in AUDIO_EXTS or Path(info.filename).name.startswith("._"):
                continue
            target = _safe_member(dest, info.filename)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


# ---------------------------------------------------------------------------
# Status and ledger
# ---------------------------------------------------------------------------


@dataclass
class SourceStatus:
    id: str
    slots: list[str]
    method: str
    present: bool
    files: int
    license: str
    commercial_use: str
    needs_terms: bool


def status(sources: SourcesFile, root: Path) -> list[SourceStatus]:
    rows: list[SourceStatus] = []
    for s in sources.sources:
        lic = sources.licenses[s.license]
        if s.fetch.method == "repo_asset":
            present = bool(s.fetch.asset and (REPO_ROOT / s.fetch.asset).exists())
            n = int(present)
        else:
            n = len(_audio_files(root / s.id))
            present = n > 0
        rows.append(SourceStatus(s.id, s.slots, s.fetch.method, present, n, lic.id, lic.commercial_use,
                                 s.fetch.method in ("cambridge_mt", "openslr_librispeech")))
    return rows


def _ledger_add(root: Path, entry: dict[str, Any]) -> None:
    path = root / LEDGER
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    data = [e for e in data if e.get("id") != entry["id"]] + [entry]
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _entry(source: Source, sources: SourcesFile, urls: list[str], nbytes: int, digest: str) -> dict[str, Any]:
    lic = sources.licenses[source.license]
    return {"id": source.id, "license": lic.id, "commercial_use": lic.commercial_use, "terms_url": lic.terms_url,
            "attribution": lic.attribution, "urls": urls, "bytes": nbytes, "sha256": digest,
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds")}


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def fetch_source(source: Source, sources: SourcesFile, root: Path, accept_terms: bool, max_gb: float, dry_run: bool) -> str:
    """Fetch one source. Returns a one-line outcome. Raises FetchError for unrecoverable problems."""
    m = source.fetch.method
    dest = root / source.id
    cap = int(max_gb * 1e9)
    lic = sources.licenses[source.license]
    if m in ("local", "synth"):
        return f"manual: {source.fetch.hint or 'no automatic download'}"
    if m == "repo_asset":
        return "repo asset (nothing to download)"
    if m in ("cambridge_mt", "openslr_librispeech") and not accept_terms:
        return (f"SKIPPED: accept the terms first ({lic.terms_url}); commercial_use={lic.commercial_use}. "
                f"Re-run with --accept-terms once you have read them.")
    if _audio_files(dest):
        return f"already present ({len(_audio_files(dest))} files)"
    if m == "cambridge_mt":
        page = _http_get(CAMBRIDGE_INDEX).decode("utf-8", "ignore")
        cands = resolve_cambridge_zip(page, source.fetch.anchor or source.title, source.fetch.tier)
        if not cands:
            raise FetchError(f"{source.id}: no ZIP found near '{source.fetch.anchor}' on {CAMBRIDGE_INDEX}; download it by hand into {dest}")
        if dry_run:
            return f"would download {cands[0]}"
        zpath = root / "_downloads" / f"{source.id}.zip"
        nbytes, digest = _download(cands[0], zpath, cap)
        files = safe_extract_zip(zpath, dest)
        zpath.unlink(missing_ok=True)
        _ledger_add(root, _entry(source, sources, [cands[0]], nbytes, digest))
        return f"downloaded {nbytes / 1e6:.0f} MB, {len(files)} audio files"
    if m in ("archive_org", "librivox_page"):
        aid = source.fetch.archive_id
        if m == "librivox_page":
            page = _http_get(source.page_url or "").decode("utf-8", "ignore")
            ids = find_archive_ids(page)
            if not ids:
                raise FetchError(f"{source.id}: no archive.org item found on {source.page_url}")
            aid = ids[0]
        assert aid
        meta = json.loads(_http_get(ARCHIVE_META.format(id=aid)))
        names = pick_archive_audio(meta, source.fetch.target_minutes or 10.0)
        if not names:
            raise FetchError(f"{source.id}: archive item {aid} has no mp3/ogg files")
        urls = [ARCHIVE_DL.format(id=aid, name=n) for n in names]
        if dry_run:
            return f"would download {len(urls)} files from archive.org/{aid}"
        total, h = 0, hashlib.sha256()
        for n, u in zip(names, urls, strict=True):
            nb, dg = _download(u, dest / Path(n).name, cap)
            total += nb
            h.update(dg.encode())
        _ledger_add(root, _entry(source, sources, urls, total, h.hexdigest()))
        return f"downloaded {len(urls)} files, {total / 1e6:.0f} MB"
    if m == "openslr_librispeech":
        if dry_run:
            return f"would download {LIBRISPEECH_URL} (~337 MB) and stitch 3 {source.fetch.gender} speakers"
        tpath = root / "_downloads" / "dev-clean.tar.gz"
        if not tpath.exists():
            nbytes, digest = _download(LIBRISPEECH_URL, tpath, cap)
        else:
            nbytes, digest = tpath.stat().st_size, "cached"
        _extract_librispeech(tpath, dest, source.fetch.gender or "female", 3)
        _ledger_add(root, _entry(source, sources, [LIBRISPEECH_URL], nbytes, digest))
        return f"extracted speakers into {dest}"
    raise FetchError(f"{source.id}: unsupported fetch method {m}")


def _extract_librispeech(tar_path: Path, dest: Path, gender: str, n: int) -> None:
    with tarfile.open(tar_path, "r:gz") as tf:
        spk = tf.extractfile("LibriSpeech/SPEAKERS.TXT")
        speakers = select_librispeech_speakers(spk.read().decode("utf-8", "ignore") if spk else "", "dev-clean", gender, n)
        if not speakers:
            raise FetchError("no matching speakers in SPEAKERS.TXT")
        dest.mkdir(parents=True, exist_ok=True)
        for member in tf:
            parts = member.name.split("/")
            if member.isfile() and member.name.endswith(".flac") and len(parts) > 3 and parts[2] in speakers:
                target = _safe_member(dest, "/".join(parts[2:]))
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(member)
                if f:
                    target.write_bytes(f.read())
