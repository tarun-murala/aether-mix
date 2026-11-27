"""Stem discovery, classification, loading, and deterministic synthetic stand-ins.

A *source* (one multitrack, one sermon, one applause clip) is resolved to *parts* (speech, lead, bv1, bv2, choir,
keys, guitar, drums, bass, room). Real audio is read from ``<stems_root>/<source id>/``; when a source is not on
disk a deterministic synthetic part is generated instead and flagged ``synthetic`` so reports never present a
stand-in as real audio.
"""

from __future__ import annotations

import fnmatch
import math
import re
import shutil
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from evals import REPO_ROOT, SAMPLE_RATE
from evals.manifest import Source

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".m4a"}
MAX_PART_SECONDS = 900.0
# Target active-window RMS (dBFS) per part, taken from the nominal_rms_db values in engine/schemas/service_plan.json.
PART_TARGET_DB = {
    "speech": -18.0, "lead": -18.0, "bv1": -20.0, "bv2": -20.0, "choir": -20.0, "keys": -16.0,
    "guitar": -16.0, "drums": -14.0, "bass": -18.0, "room": -24.0,
}
# slot -> parts that slot's source may provide (used to choose what to extract and what to audition)
SLOT_PARTS = {
    "keys": ["keys"], "acoustic": ["guitar"], "electric": ["guitar"], "bass": ["bass"], "drums": ["drums"],
    "vocal_song": ["lead", "bv1", "bv2", "keys", "guitar", "drums", "bass"], "choir": ["choir"],
    "sermon": ["speech"], "room": ["room"],
}

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("drums", re.compile(r"bass[\s_-]*drum|\bbd\b|kick|snare|\btom|hi[\s_-]*hat|\bhh\b|overhead|\boh\b|cymbal|drum|\bkit\b|perc|room[\s_-]*(l|r|mic)?\b", re.IGNORECASE)),
    ("bv", re.compile(r"\bbv|backing|backup|harmon|\bbgv|choir|gang|\bbvx|bvox", re.IGNORECASE)),
    ("lead", re.compile(r"lead[\s_-]*(vox|vocal)?|\bvox\b|\bvoc|vocal|singer|\blv\b|\bmain[\s_-]*v", re.IGNORECASE)),
    ("bass", re.compile(r"\bbass\b|\bbs\b|bassgtr|bass[\s_-]*(di|amp|mic)|sub[\s_-]*bass|\bdi\b[\s_-]*bass", re.IGNORECASE)),
    ("keys", re.compile(r"piano|keys|keyboard|rhodes|organ|synth|\bpad\b|wurli|\bpno\b|\bkbd\b|\bepiano|\bep\b|strings", re.IGNORECASE)),
    ("guitar", re.compile(r"gtr|guitar|acoustic|\bac[\s_-]|\belec|\beg\b|\bag\b|strum|riff|\bdi\b", re.IGNORECASE)),
]


def classify_filename(name: str) -> str:
    """Best-effort part from a stem filename. The Cambridge-MT index carries no instrument lists, so this is a
    heuristic; override with ``picks`` in sources.yaml and confirm with ``run_eval.py audition``."""
    stem = re.sub(r"[_.\-]+", " ", Path(name).stem)
    for part, pat in _PATTERNS:
        if pat.search(stem):
            return part
    return "other"


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read any common audio file as float32 (n, ch). Falls back to ffmpeg for formats libsndfile lacks (mp3, m4a)."""
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise
        cmd = [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SAMPLE_RATE), "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy(), SAMPLE_RATE


def to_48k_stereo(data: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        g = math.gcd(SAMPLE_RATE, sr)
        data = np.stack([resample_poly(data[:, c], SAMPLE_RATE // g, sr // g) for c in range(data.shape[1])], axis=1)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return np.ascontiguousarray(data[:, :2], dtype=np.float32)


def trim_leading_silence(x: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    thr = 10.0 ** (threshold_db / 20.0)
    mono = np.abs(x).mean(axis=1)
    win = 2400
    for i in range(0, max(1, len(mono) - win), win // 2):
        if float(np.sqrt(np.mean(mono[i : i + win] ** 2))) >= thr:
            return x[i:]
    return x


def normalize_active_rms(x: np.ndarray, target_db: float, peak_ceiling: float = 0.84) -> np.ndarray:
    """Scale so the median RMS of active 100 ms windows hits target_db, without exceeding the peak ceiling."""
    flat = x.mean(axis=1)
    win = 4800
    n = (len(flat) // win) * win
    if n == 0:
        return x
    e = (flat[:n].reshape(-1, win) ** 2).mean(axis=1)
    active = e[e > 10.0 ** (-45.0 / 10.0)]
    ref = math.sqrt(float(np.median(active))) if active.size else math.sqrt(float(e.mean()) + 1e-12)
    if ref < 1e-6:
        return x
    gain = 10.0 ** (target_db / 20.0) / ref
    peak = float(np.max(np.abs(x))) or 1.0
    gain = min(gain, peak_ceiling / peak) if peak * gain > peak_ceiling else gain
    return (x * gain).astype(np.float32)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


@dataclass
class PartSource:
    """A readable stream of one part. ``read`` returns (frames, 2) float32 and loops at the end of real audio."""

    source_id: str
    part: str
    synthetic: bool
    data: np.ndarray | None = None
    seed: int = 0
    files: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float | None:
        return None if self.data is None else len(self.data) / SAMPLE_RATE

    def read(self, start_sample: int, frames: int) -> np.ndarray:
        if self.data is None:
            return synth_block(self.part, start_sample, frames, self.seed)
        n = len(self.data)
        if start_sample >= 0 and start_sample + frames <= n:
            return self.data[start_sample : start_sample + frames]
        idx = (start_sample + np.arange(frames)) % n
        return self.data[idx]


def _audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._"))


def _load_norm(path: Path) -> np.ndarray:
    data, sr = read_audio(path)
    return trim_leading_silence(to_48k_stereo(data, sr))


def _sum_files(files: list[Path], pan_alternate: bool = False) -> np.ndarray | None:
    if not files:
        return None
    cap = int(MAX_PART_SECONDS * SAMPLE_RATE)
    arrays = [_load_norm(f)[:cap] for f in files]
    n = max(len(a) for a in arrays)
    out = np.zeros((n, 2), dtype=np.float32)
    for i, a in enumerate(arrays):
        if pan_alternate and np.allclose(a[:, 0], a[:, 1]):
            ch = i % 2
            out[: len(a), ch] += a[:, 0]
        else:
            out[: len(a)] += a
    return out


def classify_folder(source: Source, root: Path) -> dict[str, list[Path]]:
    """part -> files, honouring `picks` overrides, for audition and loading."""
    files = _audio_files(root / source.id)
    parts: dict[str, list[Path]] = {}
    if source.picks:
        for part, globs in source.picks.items():
            parts[part] = [f for f in files if any(fnmatch.fnmatch(f.name, g) for g in globs)]
        return parts
    slot = source.slots[0]
    if slot == "sermon":
        return {"speech": files}
    if slot == "room":
        return {"room": files}
    if slot == "choir":
        return {"choir": [f for f in files if classify_filename(f.name) in ("bv", "lead", "other")] or files}
    for f in files:
        part = classify_filename(f.name)
        key = {"bv": "bv"}.get(part, part)
        parts.setdefault(key, []).append(f)
    return parts


def load_parts(source: Source, root: Path, seed: int) -> dict[str, PartSource]:
    """Load every part a source can supply; absent parts are omitted (callers fall back to synthetic)."""
    grouped = classify_folder(source, root)
    out: dict[str, PartSource] = {}

    def add(part: str, arr: np.ndarray | None, files: list[Path]) -> None:
        if arr is None or len(arr) < SAMPLE_RATE:
            return
        arr = normalize_active_rms(arr, PART_TARGET_DB[part])
        out[part] = PartSource(source.id, part, False, data=arr, seed=seed, files=tuple(f.name for f in files))

    for part, files in grouped.items():
        if part == "bv":
            bvs = files
            if bvs:
                add("bv1", _sum_files(bvs[:1]), bvs[:1])
                add("bv2", _sum_files(bvs[1:2] or bvs[:1]), bvs[1:2] or bvs[:1])
        elif part == "lead":
            add("lead", _sum_files(files), files)
        elif part in ("speech", "room", "keys", "guitar", "drums", "bass"):
            add(part, _sum_files(files), files)
        elif part == "choir":
            add("choir", _sum_files(files, pan_alternate=True), files)
    if source.slots[0] in ("acoustic", "electric") and "guitar" not in out:  # solo-guitar projects with odd names
        other = grouped.get("other", []) + grouped.get("keys", [])
        add("guitar", _sum_files(other), other)
    return out


def part_for(parts: dict[str, PartSource], source_id: str, part: str, seed: int) -> PartSource:
    return parts.get(part) or PartSource(source_id, part, True, data=None, seed=seed)


# ---------------------------------------------------------------------------
# Deterministic synthetic stand-ins (stateless: any block can be generated from its absolute sample index)
# ---------------------------------------------------------------------------

_PART_IDS = {p: i for i, p in enumerate(sorted(PART_TARGET_DB))}


def _rng(seed: int, part: str, start: int) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)) & 0xFFFFFFFF, _PART_IDS.get(part, 99), max(0, int(start)) // 960])


def _harmonics(ph: np.ndarray, n: int, rolloff: float) -> np.ndarray:
    return sum((1.0 / (k**rolloff)) * np.sin(k * ph) for k in range(1, n + 1))


def synth_block(part: str, start: int, frames: int, seed: int) -> np.ndarray:
    """(frames, 2) float32 at 48 kHz for the absolute sample range [start, start+frames)."""
    t = (start + np.arange(frames, dtype=np.float64)) / SAMPLE_RATE
    rng = _rng(seed, part, start)
    sid = zlib.crc32(f"{seed}".encode()) & 0xFF
    L = R = np.zeros(frames)
    if part == "speech":
        f_base = 120.0 + (sid % 7) * 14.0  # per-source voice pitch
        talk = 0.5 * (1.0 + np.tanh(6.0 * np.sin(2.0 * np.pi * t / 5.6 + 0.4)))  # ~5.6 s phrases with pauses
        syll = 0.65 + 0.35 * np.sin(2.0 * np.pi * 3.8 * t)
        ph = 2.0 * np.pi * (f_base * t - (30.0 / (2.0 * np.pi * 2.1)) * np.cos(2.0 * np.pi * 2.1 * t)
                            - (10.0 / (2.0 * np.pi * 5.3)) * np.cos(2.0 * np.pi * 5.3 * t))
        glottal = sum((1.0 / (k**0.75)) * np.sin(k * ph + 0.3 * k) for k in range(2, 15))
        breath = np.convolve(rng.standard_normal(frames), np.array([0.2, 0.3, 0.3, 0.2]), mode="same")
        L = R = 0.13 * talk * syll * (glottal + 1.4 * breath)
    elif part in ("lead", "bv1", "bv2", "choir"):
        base = {"lead": 240.0, "bv1": 295.0, "bv2": 365.0, "choir": 330.0}[part] + (sid % 5) * 6.0
        ph = 2.0 * np.pi * (base * t - (22.0 / (2.0 * np.pi * 5.5)) * np.cos(2.0 * np.pi * 5.5 * t))
        note = 0.55 + 0.45 * np.abs(np.sin(2.0 * np.pi * t / 3.2))  # sustained notes with breaths
        amp = {"lead": 0.075, "bv1": 0.055, "bv2": 0.055, "choir": 0.06}[part]
        bed = np.convolve(rng.standard_normal(frames), np.array([0.25, 0.5, 0.25]), mode="same")
        L = amp * note * (_harmonics(ph, 12, 0.7) + 0.45 * bed)
        R = L if part != "choir" else amp * note * (_harmonics(ph * 1.26, 9, 0.7) + 0.45 * bed)
    elif part == "keys":
        chord = (0.45 * np.sin(2.0 * np.pi * 261.63 * t) + 0.35 * np.sin(2.0 * np.pi * 329.63 * t)
                 + 0.25 * np.sin(2.0 * np.pi * 392.0 * t) + 0.15 * np.sin(2.0 * np.pi * 523.25 * t))
        chord_r = (0.45 * np.sin(2.0 * np.pi * 261.63 * t + 0.4) + 0.35 * np.sin(2.0 * np.pi * 329.63 * t + 0.7)
                   + 0.25 * np.sin(2.0 * np.pi * 392.0 * t + 1.1) + 0.15 * np.sin(2.0 * np.pi * 523.25 * t + 1.4))
        L, R = 0.14 * chord, 0.14 * chord_r
    elif part == "guitar":
        env = 0.65 + 0.35 * np.abs(np.sin(2.0 * np.pi * 2.5 * t))
        g = 0.15 * env * (0.45 * np.sin(2.0 * np.pi * 330.0 * t) + 0.35 * np.sin(2.0 * np.pi * 660.0 * t)
                          + 0.25 * np.sin(2.0 * np.pi * 990.0 * t) + 0.15 * np.sin(2.0 * np.pi * 1320.0 * t))
        L, R = g, 0.95 * g
    elif part == "drums":
        beat = np.mod(t, 0.5)
        kick = np.exp(-beat * 18.0) * np.sin(2.0 * np.pi * 65.0 * t)
        snare = np.exp(-np.mod(t + 0.25, 0.5) * 22.0) * (0.6 * np.sin(2.0 * np.pi * 420.0 * t) + 0.6 * rng.standard_normal(frames))
        hat = 0.25 * rng.standard_normal(frames) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * t))
        d = 0.36 * (kick + snare + hat)
        L, R = d, 0.92 * d
    elif part == "bass":
        env = 0.6 + 0.4 * np.abs(np.sin(2.0 * np.pi * 1.0 * t))
        b = 0.30 * env * (0.25 * np.sin(2.0 * np.pi * 110.0 * t) + 0.35 * np.sin(2.0 * np.pi * 550.0 * t))
        L, R = b, 0.93 * b
    elif part == "room":
        burst = 0.6 + 0.4 * np.abs(np.sin(2.0 * np.pi * 7.0 * t))
        L = 0.12 * burst * rng.standard_normal(frames)
        R = 0.12 * burst * rng.standard_normal(frames)
    return np.stack([L, R], axis=1).astype(np.float32)


def repo_asset_path(rel: str) -> Path:
    return REPO_ROOT / rel


def load_source(source: Source, root: Path, seed: int) -> dict[str, PartSource]:
    """Resolve a source to parts: repo assets are read in place, everything else comes from ``root/<id>/``."""
    if source.fetch.method == "repo_asset" and source.fetch.asset:
        path = repo_asset_path(source.fetch.asset)
        if path.exists():
            arr = normalize_active_rms(to_48k_stereo(*read_audio(path)), PART_TARGET_DB["room"])
            return {"room": PartSource(source.id, "room", False, data=arr, seed=seed, files=(path.name,))}
        return {}
    return load_parts(source, root, seed)
