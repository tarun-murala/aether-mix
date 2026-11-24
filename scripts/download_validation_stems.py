#!/usr/bin/env python3
"""Download real studio multitrack stems, real human speech, and Silero VAD ONNX.

Downloads into:
  - recordings/stems/             (gitignored real studio multitrack WAV stems)
  - recordings/stems/speech/      (real human spoken word WAV recordings)
  - ~/.aethermix/models/silero_vad.onnx
"""

from __future__ import annotations

import argparse
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STEMS_DIR = REPO_ROOT / "recordings" / "stems"
SPEECH_DIR = STEMS_DIR / "speech"
SILERO_PATH = Path.home() / ".aethermix" / "models" / "silero_vad.onnx"

SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SPEECH_URLS = {
    "pastor_male.wav": "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0030_8k.wav",
    "speaker_female.wav": "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
}
CAMBRIDGE_INDEX_URL = "https://www.cambridge-mt.com/ms/mtk/"
MEDLEYDB_SAMPLE_URL = "https://zenodo.org/record/1438309/files/MedleyDB_Sample.tar.gz"


def download_file(url: str, dest: Path, *, max_mb: float | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[cached] {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "AetherMix-Validator/0.5.1"})
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        if total and max_mb and int(total) > max_mb * 1024 * 1024:
            raise RuntimeError(f"File too large ({int(total)/1e6:.1f} MB > {max_mb} MB)")
        tmp = dest.with_suffix(dest.suffix + ".part")
        downloaded = 0
        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
        tmp.replace(dest)
    print(f"[ok] {dest.name}: {downloaded / 1e6:.2f} MB")
    return dest


def pick_cambridge_excerpt_zip() -> str | None:
    """Fetch Cambridge-MT catalog and pick a compact multitrack excerpt ZIP (pop/rock/acoustic)."""
    req = urllib.request.Request(
        CAMBRIDGE_INDEX_URL, headers={"User-Agent": "AetherMix-Validator/0.5.1"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    # Find direct multitrack zip links (preferring compact excerpts < 150 MB)
    zips = re.findall(r'https?://[^"\'\s<>]+\.zip', html, flags=re.IGNORECASE)
    # Deduplicate preserving order
    seen: set[str] = set()
    candidates: list[str] = []
    for z in zips:
        if z not in seen and ("cambridge" in z.lower() or "mtk" in z.lower() or "multitrack" in z.lower()):
            seen.add(z)
            candidates.append(z)

    # Prefer excerpt zips (smaller, faster, full band chorus/verse) over 1GB full zips
    excerpts = [z for z in candidates if "full" not in z.lower()]
    ordered = excerpts + [z for z in candidates if z not in excerpts]
    for url in ordered[:12]:
        try:
            head_req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "AetherMix-Validator/0.5.1"}
            )
            with urllib.request.urlopen(head_req, timeout=15) as h:
                cl = int(h.headers.get("Content-Length") or 0)
                if 15 * 1024 * 1024 <= cl <= 180 * 1024 * 1024:
                    print(f"[selected Cambridge-MT] {url} ({cl / 1e6:.1f} MB)")
                    return url
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[skip] {url}: {exc}")
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download real validation stems & Silero VAD ONNX")
    parser.add_argument("--source", choices=["auto", "cambridge", "medleydb"], default="auto")
    args = parser.parse_args()

    # 1. Silero VAD ONNX
    download_file(SILERO_URL, SILERO_PATH)

    # 2. Real human speech WAVs
    for fname, url in SPEECH_URLS.items():
        download_file(url, SPEECH_DIR / fname)

    # 3. Real studio multitracks
    multitrack_dir = STEMS_DIR / "multitrack"
    existing_wavs = list(multitrack_dir.rglob("*.wav")) if multitrack_dir.exists() else []
    if len(existing_wavs) >= 4:
        print(f"[cached] Found {len(existing_wavs)} WAV stems under {multitrack_dir}")
        return 0

    multitrack_dir.mkdir(parents=True, exist_ok=True)
    downloaded = False

    if args.source in ("auto", "cambridge"):
        try:
            url = pick_cambridge_excerpt_zip()
            if url:
                zpath = STEMS_DIR / "cambridge_multitrack.zip"
                download_file(url, zpath, max_mb=220.0)
                print(f"[extract] {zpath} -> {multitrack_dir}")
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(multitrack_dir)
                downloaded = True
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            print(f"[warn] Cambridge-MT download failed: {exc}")

    if not downloaded and args.source in ("auto", "medleydb"):
        tpath = STEMS_DIR / "MedleyDB_Sample.tar.gz"
        download_file(MEDLEYDB_SAMPLE_URL, tpath, max_mb=500.0)
        print(f"[extract] {tpath} -> {multitrack_dir}")
        with tarfile.open(tpath, "r:gz") as tf:
            tf.extractall(multitrack_dir)
        downloaded = True

    wavs = sorted(multitrack_dir.rglob("*.wav"))
    print(f"[done] Extracted {len(wavs)} WAV stems:")
    for w in wavs[:30]:
        print(f"  - {w.relative_to(multitrack_dir)} ({w.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
