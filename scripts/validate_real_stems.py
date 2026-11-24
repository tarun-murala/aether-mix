#!/usr/bin/env python3
"""Holistic Production Multi-Track Stem Mixer & Validation Harness for AetherMix v5.1.

Loads REAL downloaded studio multitrack WAV stems (MedleyDB / Cambridge-MT) +
REAL recorded human speech + REAL Silero VAD ONNX neural model, runs the full
4-step soundcheck calibration, executes a closed-loop 8-act live church service
through AetherMix's SafetyFirewall, SceneClassifier, AudioTap, VADWorker,
OffsetLearner, and RulesManager, renders listenable before/after 48 kHz stereo
WAV mixes, and evaluates 12 strict production-readiness gates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core.audio_tap import AudioTap
from engine.core.offset_learner import OffsetLearner
from engine.core.protocol import (
    AutonomyMode,
    BridgeStatePayload,
    MixingRules,
    ServicePhase,
    ServicePlan,
    cc_to_db,
    cross_validate_plan_and_rules,
    db_to_cc,
)
from engine.core.rules_manager import RulesManager
from engine.core.safety_firewall import SafetyFirewall
from engine.core.scene_classifier import SceneClassifier
from engine.core.vad_worker import VADWorker

SAMPLE_RATE = 48000
BLOCK_FRAMES = 960  # 20 ms = 50 Hz
STEMS_ROOT = REPO_ROOT / "recordings" / "stems"
OUTPUT_DIR = REPO_ROOT / "recordings" / "validation"
DEFAULT_PLAN = REPO_ROOT / "engine" / "schemas" / "service_plan.json"
DEFAULT_RULES = REPO_ROOT / "engine" / "schemas" / "mixing_rules.json"
SILERO_MODEL_PATH = Path.home() / ".aethermix" / "models" / "silero_vad.onnx"


# ---------------------------------------------------------------------------
# Real Audio Loading, Resampling & MedleyDB / Generic Stem Discovery
# ---------------------------------------------------------------------------


def trim_leading_silence(audio: np.ndarray, threshold_db: float = -42.0) -> np.ndarray:
    """Trim leading studio count-in silence so real stems begin on active audio."""
    thresh = 10.0 ** (threshold_db / 20.0)
    mono = np.mean(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    # Use 50ms moving RMS-like window to skip tiny clicks
    win = 2400
    for i in range(0, max(1, len(mono) - win), win // 2):
        if float(np.sqrt(np.mean(mono[i : i + win] ** 2))) >= thresh:
            return audio[i:]
    return audio


def load_wav_48k(
    path: Path, *, stereo: bool = False, target_rms_db: float | None = None, trim: bool = False
) -> np.ndarray:
    """Load any WAV/FLAC file, resample cleanly to 48 kHz float32, and normalize active RMS."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        g = math.gcd(SAMPLE_RATE, sr)
        up = SAMPLE_RATE // g
        down = sr // g
        cols = [resample_poly(data[:, c], up, down).astype(np.float32) for c in range(data.shape[1])]
        data = np.stack(cols, axis=1)

    if stereo:
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        else:
            data = data[:, :2]
    else:
        data = np.mean(data, axis=1, keepdims=False)

    if trim:
        data = trim_leading_silence(data)

    if target_rms_db is not None:
        # Compute RMS over active windows (> -45 dBFS) so silence pauses don't over-boost peaks
        flat = data.reshape(-1)
        win = 4800
        active_energies: list[float] = []
        for i in range(0, len(flat) - win + 1, win):
            e = float(np.mean(flat[i : i + win] ** 2))
            if e > 10.0 ** (-45.0 / 10.0):
                active_energies.append(e)
        ref_rms = math.sqrt(float(np.median(active_energies))) if active_energies else float(
            np.sqrt(np.mean(flat**2) + 1e-12)
        )
        if ref_rms > 1e-6:
            target_lin = 10.0 ** (target_rms_db / 20.0)
            gain = target_lin / ref_rms
            # Prevent clipping individual stem above -1.5 dBFS
            peak = float(np.max(np.abs(data)))
            if peak * gain > 0.84:
                gain = 0.84 / peak
            data = (data * gain).astype(np.float32)
    return data.astype(np.float32)


def loop_slice(audio: np.ndarray, start_sample: int, frames: int) -> np.ndarray:
    """Extract `frames` samples seamlessly looping around the end of `audio`."""
    n = audio.shape[0]
    if n == 0:
        return np.zeros((frames,) + audio.shape[1:], dtype=np.float32)
    idx = (np.arange(frames) + start_sample) % n
    return audio[idx]


@dataclass
class RealStemPack:
    source_name: str
    pastor_speech: np.ndarray  # mono 48k
    lead_vocal: np.ndarray  # mono 48k
    bv1: np.ndarray  # mono 48k
    bv2: np.ndarray  # mono 48k
    keys_stereo: np.ndarray  # (N, 2) 48k
    guitars_stereo: np.ndarray  # (N, 2) 48k
    drums_bass_quad: np.ndarray  # (N, 4) 48k
    applause_stereo: np.ndarray  # (N, 2) 48k
    stem_files: dict[str, str]


def _classify_instrument_label(label: str) -> str | None:
    s = label.lower()
    if any(k in s for k in ("vocal", "singer", "voice", "vox", "choir", "soprano", "tenor", "alto")):
        if any(b in s for b in ("back", "bv", "harm", "chorus", "choir")):
            return "bv"
        return "lead_vox"
    if any(k in s for k in ("drum", "kick", "snare", "hat", "perc", "cymbal", "bass", "double bass")):
        return "drums_bass"
    if any(k in s for k in ("piano", "key", "synth", "organ", "pad", "rhodes", "clav", "wurlitzer", "string")):
        return "keys"
    if any(k in s for k in ("guitar", "gtr", "mandolin", "banjo", "ukulele")):
        return "guitars"
    return None


def discover_real_stems(stems_root: Path = STEMS_ROOT) -> RealStemPack:
    """Discover and map real downloaded MedleyDB / Cambridge-MT WAV stems + real speech."""
    speech_male = stems_root / "speech" / "pastor_male.wav"
    speech_female = stems_root / "speech" / "speaker_female.wav"
    if not speech_male.exists():
        raise FileNotFoundError(
            f"Missing real speech file {speech_male}. Run scripts/download_validation_stems.py first."
        )
    pastor_a = load_wav_48k(speech_male, stereo=False, target_rms_db=-16.0, trim=False)
    pastor_a = trim_leading_silence(pastor_a, threshold_db=-24.0)
    if speech_female.exists():
        pastor_b = load_wav_48k(speech_female, stereo=False, target_rms_db=-16.0, trim=False)
        pastor_b = trim_leading_silence(pastor_b, threshold_db=-24.0)
        pastor_speech = np.concatenate([pastor_a, np.zeros(8000, dtype=np.float32), pastor_b])
    else:
        pastor_speech = pastor_a

    # Inspect MedleyDB songs by METADATA.yaml (ignoring macOS ._* AppleDouble files)
    multitrack_dir = stems_root / "multitrack"
    meta_files = sorted(
        p for p in multitrack_dir.rglob("*_METADATA.yaml") if not p.name.startswith("._")
    )

    buckets: dict[str, list[tuple[str, Path]]] = {
        "lead_vox": [],
        "bv": [],
        "keys": [],
        "guitars": [],
        "drums_bass": [],
        "other": [],
    }
    best_song_name = "Real Multi-Track Corpus"
    best_score = -1
    best_song_buckets: dict[str, list[tuple[str, Path]]] | None = None

    for mf in meta_files:
        meta = yaml.safe_load(mf.read_text(encoding="utf-8")) or {}
        song_dir = mf.parent
        stem_dir_name = meta.get("stem_dir", "")
        stem_dir = song_dir / stem_dir_name if stem_dir_name else song_dir
        song_buckets: dict[str, list[tuple[str, Path]]] = {k: [] for k in buckets}

        for _sid, sinfo in sorted((meta.get("stems") or {}).items()):
            inst = str(sinfo.get("instrument", ""))
            fname = str(sinfo.get("filename", ""))
            comp = str(sinfo.get("component", ""))
            wav_path = stem_dir / fname
            if not wav_path.exists() or fname.startswith("._"):
                continue
            role = _classify_instrument_label(inst) or _classify_instrument_label(fname) or "other"
            # In MedleyDB, additional vocal stems without component='melody' are harmony / BVs
            if role == "lead_vox" and song_buckets["lead_vox"] and comp != "melody":
                role = "bv"
            song_buckets[role].append((f"{song_dir.name}/{fname} [{inst}]", wav_path))
            buckets[role].append((f"{song_dir.name}/{fname} [{inst}]", wav_path))

        coverage = sum(1 for k in ("lead_vox", "bv", "keys", "guitars", "drums_bass") if song_buckets[k])
        if coverage > best_score:
            best_score = coverage
            best_song_name = song_dir.name
            best_song_buckets = song_buckets

    # Prefer stems from the best single song, borrowing any missing role from companion MedleyDB songs
    chosen = {k: list(v) for k, v in (best_song_buckets or buckets).items()}
    for role in ("lead_vox", "bv", "keys", "guitars", "drums_bass", "other"):
        if not chosen[role] and buckets[role]:
            chosen[role] = list(buckets[role])

    # Fallback: scan generic WAVs if no METADATA.yaml (e.g., Cambridge-MT)
    if not any(chosen.values()):
        for wav in sorted(multitrack_dir.rglob("*.wav")):
            if wav.name.startswith("._") or "mix" in wav.stem.lower():
                continue
            role = _classify_instrument_label(wav.stem) or "other"
            chosen[role].append((wav.name, wav))

    # Find common song active offset for time-aligned MedleyDB stems so vocals & guitars stay in sync!
    def _sum_mono(items: list[tuple[str, Path]], target_db: float, start_s: float = 15.0) -> tuple[np.ndarray, str]:
        if not items:
            return np.zeros(SAMPLE_RATE * 10, dtype=np.float32), "none"
        tracks = [load_wav_48k(p, stereo=False, target_rms_db=target_db) for _, p in items[:2]]
        min_len = min(len(t) for t in tracks)
        off = min(int(start_s * SAMPLE_RATE), max(0, min_len - SAMPLE_RATE * 20))
        mix = sum(t[off:min_len] for t in tracks) / math.sqrt(len(tracks))
        return mix.astype(np.float32), ", ".join(lbl for lbl, _ in items[:2])

    def _sum_stereo(items: list[tuple[str, Path]], target_db: float, start_s: float = 15.0) -> tuple[np.ndarray, str]:
        if not items:
            return np.zeros((SAMPLE_RATE * 10, 2), dtype=np.float32), "none"
        tracks = [load_wav_48k(p, stereo=True, target_rms_db=target_db) for _, p in items[:3]]
        min_len = min(t.shape[0] for t in tracks)
        off = min(int(start_s * SAMPLE_RATE), max(0, min_len - SAMPLE_RATE * 20))
        mix = sum(t[off:min_len] for t in tracks) / math.sqrt(len(tracks))
        return mix.astype(np.float32), ", ".join(lbl for lbl, _ in items[:3])

    stem_files: dict[str, str] = {"ch1_pastor": f"{speech_male.name} + {speech_female.name}"}

    # Offset into active chorus/verse (~22s in LizNelson_Rainfall where all vocals + guitars are active)
    lead_vox, stem_files["ch2_lead_vox"] = _sum_mono(
        chosen["lead_vox"] or chosen["other"], target_db=-18.0, start_s=22.0
    )
    if chosen["bv"]:
        bv1, stem_files["ch3_bv1"] = _sum_mono(chosen["bv"][:1], target_db=-20.0, start_s=22.0)
        bv2, stem_files["ch4_bv2"] = _sum_mono(chosen["bv"][1:2] or chosen["bv"][:1], target_db=-20.0, start_s=22.0)
    else:
        shift = min(len(lead_vox) // 3, SAMPLE_RATE * 4)
        bv1 = np.roll(lead_vox, shift) * 0.75
        bv2 = np.roll(lead_vox, shift * 2) * 0.72
        stem_files["ch3_bv1"] = f"offset-real-vox ({stem_files['ch2_lead_vox']})"
        stem_files["ch4_bv2"] = f"offset-real-vox-2 ({stem_files['ch2_lead_vox']})"

    # Keys/Pad: use dedicated keys stem if present, else real sustained melodic stems (violin/flute/clean elec)
    keys_candidates = chosen["keys"] or chosen["other"][:2]
    keys_stereo, stem_files["ch5_keys"] = _sum_stereo(keys_candidates, target_db=-16.0, start_s=10.0)
    guitars_stereo, stem_files["ch7_guitars"] = _sum_stereo(
        chosen["guitars"] or chosen["other"], target_db=-16.0, start_s=22.0
    )
    drums_candidates = chosen["drums_bass"] or chosen["other"][-1:]
    drums_st, stem_files["ch9_drums"] = _sum_stereo(drums_candidates, target_db=-14.0, start_s=10.0)
    # Build 4-channel bus (L/R drums + L/R bass/low-end weight)
    drums_quad = np.concatenate([drums_st, drums_st * 0.85], axis=1).astype(np.float32)

    # Realistic audience applause stem (broadband high-flatness envelope-modulated crowd noise)
    rng = np.random.default_rng(2026)
    appl_len = SAMPLE_RATE * 12
    appl_l = rng.standard_normal(appl_len).astype(np.float32) * 0.10
    appl_r = rng.standard_normal(appl_len).astype(np.float32) * 0.10
    t_ax = np.arange(appl_len, dtype=np.float32) / SAMPLE_RATE
    env = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.35 * t_ax)
    applause_stereo = np.stack([appl_l * env, appl_r * env], axis=1).astype(np.float32)
    stem_files["ch13_room"] = "synthetic_broadband_applause_room"

    return RealStemPack(
        source_name=best_song_name,
        pastor_speech=pastor_speech,
        lead_vocal=lead_vox,
        bv1=bv1,
        bv2=bv2,
        keys_stereo=keys_stereo,
        guitars_stereo=guitars_stereo,
        drums_bass_quad=drums_quad,
        applause_stereo=applause_stereo,
        stem_files=stem_files,
    )


# ---------------------------------------------------------------------------
# Lightweight Real-Time Hall Reverb & Click-Free Closed-Loop Console Mixer
# ---------------------------------------------------------------------------


class SimpleHallReverb:
    """Deterministic stereo Schroeder comb-filter hall reverb for audible reverb-send validation."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        delays = [int(sample_rate * d) for d in (0.0297, 0.0371, 0.0411, 0.0437)]
        self._bufs_l = [np.zeros(d, dtype=np.float32) for d in delays]
        self._bufs_r = [np.zeros(d + 23, dtype=np.float32) for d in delays]
        self._idx_l = [0] * len(delays)
        self._idx_r = [0] * len(delays)
        self._decay = 0.76

    def process_block(self, mono_send: np.ndarray) -> np.ndarray:
        n = len(mono_send)
        out_l = np.zeros(n, dtype=np.float32)
        out_r = np.zeros(n, dtype=np.float32)
        for c in range(len(self._bufs_l)):
            bl = self._bufs_l[c]
            br = self._bufs_r[c]
            dl, dr = len(bl), len(br)
            il, ir = self._idx_l[c], self._idx_r[c]
            # Vectorized chunked comb tap (n=960 < dl>=1425 so no self-overlap within one block!)
            idx_l = (np.arange(n) + il) % dl
            idx_r = (np.arange(n) + ir) % dr
            yl = bl[idx_l].copy()
            yr = br[idx_r].copy()
            bl[idx_l] = mono_send + self._decay * yl
            br[idx_r] = mono_send + self._decay * yr
            self._idx_l[c] = (il + n) % dl
            self._idx_r[c] = (ir + n) % dr
            out_l += yl
            out_r += yr
        return np.stack([out_l * 0.25, out_r * 0.25], axis=1).astype(np.float32)


class ClosedLoopConsoleMixer:
    """Simulates Apple MainStage Concert-level faders + reverb sends with click-free ramping."""

    def __init__(self, plan: ServicePlan) -> None:
        self.plan = plan
        self.reverb = SimpleHallReverb(SAMPLE_RATE)
        self._prev_vol_lin: dict[str, float] = {ch.id: 0.0 for ch in plan.channels}
        self._prev_rev_lin: dict[str, float] = {ch.id: 0.0 for ch in plan.channels}
        # Slight stereo panning for natural concert soundstage
        self._pan: dict[str, tuple[float, float]] = {
            "ch1_pastor": (0.707, 0.707),
            "ch2_lead_vox": (0.707, 0.707),
            "ch3_bv1": (0.82, 0.57),
            "ch4_bv2": (0.57, 0.82),
        }

    @staticmethod
    def _db_to_lin(db: float) -> float:
        if math.isinf(db) or db <= -59.5:
            return 0.0
        return float(10.0 ** (db / 20.0))

    def mix_block(
        self,
        raw_16ch: np.ndarray,
        vol_cc_map: dict[str, int],
        reverb_cc_map: dict[str, int | None],
        *,
        unity_raw: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Render a 20 ms stereo master block + per-channel post-fader RMS energies."""
        n = raw_16ch.shape[0]
        master = np.zeros((n, 2), dtype=np.float32)
        rev_send_mono = np.zeros(n, dtype=np.float32)
        post_energies: dict[str, float] = {}

        for ch in self.plan.channels:
            if unity_raw:
                target_vol = 1.0
                target_rev = 0.0
            else:
                v_db = cc_to_db(vol_cc_map.get(ch.id, 0), ch.cc_taper)
                target_vol = self._db_to_lin(v_db)
                r_cc = reverb_cc_map.get(ch.id)
                target_rev = (
                    self._db_to_lin(cc_to_db(r_cc, ch.reverb_cc_taper))
                    if (r_cc is not None and ch.reverb_cc is not None)
                    else 0.0
                )

            prev_v = target_vol if unity_raw else self._prev_vol_lin[ch.id]
            prev_r = target_rev if unity_raw else self._prev_rev_lin[ch.id]
            ramp_v = np.linspace(prev_v, target_vol, n, dtype=np.float32)
            ramp_r = np.linspace(prev_r, target_rev, n, dtype=np.float32)
            if not unity_raw:
                self._prev_vol_lin[ch.id] = target_vol
                self._prev_rev_lin[ch.id] = target_rev

            taps = [t - 1 for t in ch.tap_channels]
            if len(taps) == 1:
                mono = raw_16ch[:, taps[0]]
                gained = mono * ramp_v
                pl, pr = self._pan.get(ch.id, (0.707, 0.707))
                master[:, 0] += gained * pl
                master[:, 1] += gained * pr
                if target_rev > 0.0 or prev_r > 0.0:
                    rev_send_mono += mono * ramp_r
                post_energies[ch.id] = float(np.mean(gained**2))
            elif len(taps) == 2:
                sl = raw_16ch[:, taps[0]] * ramp_v
                sr = raw_16ch[:, taps[1]] * ramp_v
                master[:, 0] += sl
                master[:, 1] += sr
                post_energies[ch.id] = float(0.5 * (np.mean(sl**2) + np.mean(sr**2)))
            else:
                sl = 0.5 * (raw_16ch[:, taps[0]] + raw_16ch[:, taps[2]]) * ramp_v
                sr = 0.5 * (raw_16ch[:, taps[1]] + raw_16ch[:, taps[3]]) * ramp_v
                master[:, 0] += sl
                master[:, 1] += sr
                post_energies[ch.id] = float(0.5 * (np.mean(sl**2) + np.mean(sr**2)))

        if not unity_raw:
            master += self.reverb.process_block(rev_send_mono)

        # Transparent master headroom trim (-3 dB bus headroom) + soft peak limiter at -1.0 dBFS (~0.891)
        master *= 0.707
        peak = float(np.max(np.abs(master)))
        ceil = 10.0 ** (-1.0 / 20.0)
        if peak > ceil:
            master = np.tanh(master / ceil) * ceil
        return master.astype(np.float32), post_energies


# ---------------------------------------------------------------------------
# Realistic Live Stage Audio Composer (With Acoustic Stage Bleed + Howl)
# ---------------------------------------------------------------------------


def compose_stage_block(
    pack: RealStemPack,
    cursor: int,
    *,
    pastor_on: bool = False,
    lead_on: bool = False,
    bv_on: bool = False,
    keys_on: bool = False,
    guitars_on: bool = False,
    drums_on: bool = False,
    applause_on: bool = False,
    stage_bleed: bool = True,
    feedback_ch_idx: int | None = None,
    feedback_amp: float = 0.0,
    feedback_freq_hz: float = 2480.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assemble one realistic 20 ms 16-channel pre-fader live stage block from real WAV stems."""
    block = np.zeros((BLOCK_FRAMES, 16), dtype=np.float32)
    # Realistic quiet stage mic noise floor (~ -58 dBFS)
    block[:, :12] = rng.standard_normal((BLOCK_FRAMES, 12)).astype(np.float32) * (10.0 ** (-58.0 / 20.0))

    keys = loop_slice(pack.keys_stereo, cursor, BLOCK_FRAMES) if keys_on else np.zeros((BLOCK_FRAMES, 2), np.float32)
    gtr = loop_slice(pack.guitars_stereo, cursor, BLOCK_FRAMES) if guitars_on else np.zeros((BLOCK_FRAMES, 2), np.float32)
    drm = loop_slice(pack.drums_bass_quad, cursor, BLOCK_FRAMES) if drums_on else np.zeros((BLOCK_FRAMES, 4), np.float32)

    block[:, 4:6] += keys
    block[:, 6:8] += gtr
    block[:, 8:12] += drm

    # Acoustic stage spill into live vocal & pulpit microphones (-16 dB pulpit, -14 dB stage vocals)
    band_mono = (
        0.5 * (keys[:, 0] + keys[:, 1])
        + 0.5 * (gtr[:, 0] + gtr[:, 1])
        + 0.25 * np.sum(drm, axis=1)
    )
    if stage_bleed:
        block[:, 0] += band_mono * (10.0 ** (-16.0 / 20.0))
        bleed_vox = band_mono * (10.0 ** (-14.0 / 20.0))
        block[:, 1] += bleed_vox
        block[:, 2] += bleed_vox
        block[:, 3] += bleed_vox

    if pastor_on:
        block[:, 0] += loop_slice(pack.pastor_speech, cursor, BLOCK_FRAMES)
    if lead_on:
        block[:, 1] += loop_slice(pack.lead_vocal, cursor, BLOCK_FRAMES)
    if bv_on:
        block[:, 2] += loop_slice(pack.bv1, cursor, BLOCK_FRAMES)
        block[:, 3] += loop_slice(pack.bv2, cursor, BLOCK_FRAMES)
    if applause_on:
        block[:, 12:14] += loop_slice(pack.applause_stereo, cursor, BLOCK_FRAMES)
    else:
        # Quiet room HVAC floor
        block[:, 12:14] = rng.standard_normal((BLOCK_FRAMES, 2)).astype(np.float32) * (10.0 ** (-54.0 / 20.0))

    if feedback_ch_idx is not None and feedback_amp > 0.0:
        t = (np.arange(BLOCK_FRAMES, dtype=np.float64) + cursor) / float(SAMPLE_RATE)
        block[:, feedback_ch_idx] += (feedback_amp * np.sin(2.0 * np.pi * feedback_freq_hz * t)).astype(
            np.float32
        )

    return block


# ---------------------------------------------------------------------------
# Holistic Production Validation Runner
# ---------------------------------------------------------------------------


def run_production_validation(stems_root: Path = STEMS_ROOT, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pack = discover_real_stems(stems_root)

    plan = ServicePlan.model_validate_json(DEFAULT_PLAN.read_text(encoding="utf-8"))
    rules = MixingRules.model_validate_json(DEFAULT_RULES.read_text(encoding="utf-8"))
    cross_validate_plan_and_rules(plan, rules)

    audio_tap = AudioTap(plan, headless=True)
    vad_worker = VADWorker(plan, model_path=SILERO_MODEL_PATH)
    vad_worker.start()
    scene_classifier = SceneClassifier(plan)
    offset_learner = OffsetLearner(plan)
    rules_manager = RulesManager(plan, rules)
    firewall = SafetyFirewall(plan, rules, rules_manager, offset_learner)
    console = ClosedLoopConsoleMixer(plan)
    raw_console = ClosedLoopConsoleMixer(plan)
    rng = np.random.default_rng(42)

    silero_onnx_active = getattr(vad_worker, "_ort_session", None) is not None

    # ------------------------------------------------------------------
    # Step 0: Run REAL 4-Step Sequential Soundcheck Calibration!
    # ------------------------------------------------------------------
    # Soundcheck band at chorus energy (~20s into stems, matching live FOH soundcheck)
    cal_cursor = 20 * SAMPLE_RATE
    silence_blocks = [
        compose_stage_block(pack, cal_cursor + i * BLOCK_FRAMES, stage_bleed=False, rng=rng)
        for i in range(100)  # 2.0s room silence
    ]
    noise_floors = audio_tap.capture_room_silence(silence_blocks)

    spoken_blocks = [
        compose_stage_block(
            pack, i * BLOCK_FRAMES, pastor_on=True, lead_on=True, bv_on=True, stage_bleed=False, rng=rng
        )
        for i in range(250)  # 5.0s isolated vocals/pastor
    ]
    spoken_noms = audio_tap.capture_input_nominals("spoken", spoken_blocks)

    band_blocks = [
        compose_stage_block(
            pack, cal_cursor + i * BLOCK_FRAMES, keys_on=True, guitars_on=True, drums_on=True, stage_bleed=False, rng=rng
        )
        for i in range(250)  # 5.0s isolated band
    ]
    band_noms = audio_tap.capture_input_nominals("band", band_blocks)

    bleed_blocks = [
        compose_stage_block(
            pack, cal_cursor + i * BLOCK_FRAMES, keys_on=True, guitars_on=True, drums_on=True, stage_bleed=True, rng=rng
        )
        for i in range(250)  # 5.0s band plays, nobody speaks
    ]
    bleed_offsets = audio_tap.capture_bleed_floor(bleed_blocks)

    # Seed bridge state synced at SPOKEN_WORD baselines, arm all channels, enter ACTIVE
    init_cc: dict[str, int] = {}
    for ch in plan.channels:
        v_db = rules.phase_baselines_db[ServicePhase.SPOKEN_WORD][ch.id]
        init_cc[f"{ch.id}:vol"] = db_to_cc(v_db, ch.cc_taper)
        if ch.reverb_cc is not None:
            r_db = rules.phase_reverb_baselines_db[ServicePhase.SPOKEN_WORD][ch.id]
            init_cc[f"{ch.id}:reverb"] = db_to_cc(r_db, ch.reverb_cc_taper)
    firewall.hydrate_from_bridge(BridgeStatePayload(last_sent_cc=init_cc, controller_connected=True, engine_connected=True))
    for ch in plan.channels:
        firewall.confirm_sync(ch.id)
        firewall.set_channel_armed(ch.id, True)
    firewall.set_mode(AutonomyMode.ACTIVE)

    # ------------------------------------------------------------------
    # 8-Act Real-Stem Live Service Timeline (72 seconds = 3,600 ticks @ 50 Hz)
    # ------------------------------------------------------------------
    acts: list[dict[str, Any]] = [
        # (name, duration_s, expected_phase, service_step_idx, stage_kwargs)
        {
            "name": "Act 1: Spoken Welcome (Real Male Speech)",
            "dur_s": 10.0,
            "expected_phase": ServicePhase.SPOKEN_WORD,
            "step_idx": 3,
            "stage": {"pastor_on": True},
        },
        {
            "name": "Act 2: Acoustic Worship Verse (Real Piano/Guitars/Lead Vox)",
            "dur_s": 10.0,
            "expected_phase": ServicePhase.WORSHIP_LOW,
            "step_idx": 1,
            "stage": {"lead_on": True, "keys_on": True, "guitars_on": True},
        },
        {
            "name": "Act 3: Full Band Chorus + Stage Bleed (WORSHIP_HIGH)",
            "dur_s": 12.0,
            "expected_phase": ServicePhase.WORSHIP_HIGH,
            "step_idx": 0,
            "stage": {"lead_on": True, "bv_on": True, "keys_on": True, "guitars_on": True, "drums_on": True},
        },
        {
            "name": "Act 4: Pastor Interrupt Over Live Band",
            "dur_s": 8.0,
            "expected_phase": None,  # reflex test during WORSHIP_HIGH
            "step_idx": 0,
            "stage": {"pastor_on": True, "keys_on": True, "guitars_on": True, "drums_on": True},
        },
        {
            "name": "Act 5: Pastoral Prayer Over Real Piano Pad (PRAYER_UNDERSCORE)",
            "dur_s": 10.0,
            "expected_phase": ServicePhase.PRAYER_UNDERSCORE,
            "step_idx": 2,
            "stage": {"pastor_on": True, "keys_on": True},
        },
        {
            "name": "Act 6: Sermon + Injected Open-Mic Feedback Howl on Ch 2",
            "dur_s": 8.0,
            "expected_phase": ServicePhase.SPOKEN_WORD,
            "step_idx": 3,
            "stage": {"pastor_on": True},
            "inject_feedback": True,
        },
        {
            "name": "Act 7: Human Override Zero-Drift & Smart Rollback Check",
            "dur_s": 8.0,
            "expected_phase": ServicePhase.WORSHIP_HIGH,
            "step_idx": 0,
            "stage": {"lead_on": True, "bv_on": True, "keys_on": True, "guitars_on": True, "drums_on": True},
            "human_override_test": True,
        },
        {
            "name": "Act 8: Applause Transition + Speech-Over-Intro Song Start",
            "dur_s": 10.0,
            "expected_phase": ServicePhase.TRANSITION,
            "step_idx": 4,
            "stage": {"applause_on": True},
            "speech_over_intro": True,
        },
    ]

    # Authoritative live CC map (simulating MainStage state)
    vol_cc_map: dict[str, int] = {
        ch.id: init_cc[f"{ch.id}:vol"] for ch in plan.channels
    }
    rev_cc_map: dict[str, int | None] = {
        ch.id: init_cc.get(f"{ch.id}:reverb") for ch in plan.channels
    }

    raw_mix_blocks: list[np.ndarray] = []
    aether_mix_blocks: list[np.ndarray] = []

    tick_latencies_ms: list[float] = []
    phase_correct = 0
    phase_total = 0

    # Detailed telemetry accumulators for production gates
    act4_speech_open_latency_s: float | None = None
    act2_instrument_open_latency_s: float | None = None
    act3_false_speech_open_ticks = 0
    act5_false_instrument_open_ticks = 0
    non_feedback_false_cuts = 0
    act6_feedback_cut_latency_s: float | None = None
    act7_max_human_drift_db = 0.0
    act7_rollback_respected_lock = True
    act8_speech_over_intro_ducked = False

    # Intelligibility tracking: Pastor vs Band post-fader energy ratios (raw vs AetherMix)
    intelligibility_raw_db: list[float] = []
    intelligibility_mix_db: list[float] = []
    per_act_lufs: dict[str, list[float]] = {}

    now_s = 1000.0
    cursor = 0
    last_phase_eval_s = now_s - 2.0
    last_macro = ServicePhase.SPOKEN_WORD
    active_phase = ServicePhase.SPOKEN_WORD
    last_mixed_master = np.zeros((BLOCK_FRAMES, 2), dtype=np.float32)

    # Act 4 starts right at the beginning of a spoken sentence (t=0s of pastor_speech)
    # so speech_open latency measures true onset from first spoken word rather than a silence pause
    act4_pastor_offset = 0
    for act in acts:
        act_name = str(act["name"])
        per_act_lufs[act_name] = []
        n_ticks = round(act["dur_s"] * 50)
        act_start_s = now_s
        human_keys_db_target: float | None = None
        human_moved_at_tick: int | None = None
        rollback_tested = False

        for tick_idx in range(n_ticks):
            t_in_act = tick_idx * 0.02
            stage_kw = dict(act["stage"])

            # Act 6: inject realistic growing 2480 Hz feedback ring on Ch 2 from t=2.5s..5.0s
            fb_ch = None
            fb_amp = 0.0
            if act.get("inject_feedback") and 2.5 <= t_in_act <= 5.0:
                fb_ch = 1  # ch2_lead_vox
                fb_progress = (t_in_act - 2.5) / 2.5
                fb_amp = 0.04 * (10.0 ** (fb_progress * 1.1))  # realistic exponential ring growth

            # Act 8: applause 0..6s (verifies TRANSITION lock), then pastor says "let's stand and sing" + drums & keys at 6..10s
            if act.get("speech_over_intro") and t_in_act >= 6.0:
                stage_kw = {"pastor_on": True, "keys_on": True, "drums_on": True}

            stem_cursor = (act4_pastor_offset + tick_idx * BLOCK_FRAMES) if act_name.startswith("Act 4") else cursor
            block = compose_stage_block(
                pack,
                stem_cursor,
                feedback_ch_idx=fb_ch,
                feedback_amp=fb_amp,
                rng=rng,
                **stage_kw,
            )
            # In Act 4 (Pastor Interrupt over loud band), pastor projects +3 dB into close pulpit mic
            if act_name.startswith("Act 4"):
                block[:, 0] *= 10.0 ** (3.0 / 20.0)
            # Closed-loop post-fader stereo master from previous 20ms block!
            block[:, 14:16] = last_mixed_master

            # Real-time timed tick
            t0 = time.perf_counter()
            audio_tap.push_block(block)
            metrics = audio_tap.poll_metrics()
            vad_probs, sung_vocals = vad_worker.process_frame(metrics)
            scene_classifier.observe_tick(metrics, vad_probs, sung_vocals)

            # Tier 2 at 0.5 Hz (every 2.0 s)
            if now_s - last_phase_eval_s >= 2.0 - 1e-9:
                decision = scene_classifier.update(
                    metrics=metrics,
                    vad_probs=vad_probs,
                    sung_vocals=sung_vocals,
                    clap_probs=None,
                    service_step_idx=int(act["step_idx"]),
                    use_service_plan=True,
                    now_s=now_s,
                )
                last_phase_eval_s = now_s
                active_phase = decision.active_phase
                last_macro = decision.last_macro_phase

            # Act 7: simulate human grabbing ch5_keys at t=1.5s, then verify 0.00 dB drift & 👎 rollback
            if act.get("human_override_test"):
                if tick_idx == 75:  # t = 1.5s
                    ch_keys = plan.channel_by_id("ch5_keys")
                    human_cc = db_to_cc(-6.5, ch_keys.cc_taper)
                    vol_cc_map["ch5_keys"] = human_cc
                    firewall.on_human_cc("ch5_keys", "vol", human_cc, human_seq=1, now_s=now_s)
                    human_keys_db_target = cc_to_db(human_cc, ch_keys.cc_taper)
                    human_moved_at_tick = tick_idx
                elif tick_idx == 200 and not rollback_tested:  # t = 4.0s (during touch lock)
                    firewall.handle_thumbs_down(now_s=now_s, freeze_learning=False)
                    rollback_tested = True

            out = firewall.tick(
                metrics=metrics,
                vad_probs=vad_probs,
                sung_vocals=sung_vocals,
                active_phase=active_phase,
                last_macro_phase=last_macro,
                dt_s=0.02,
                now_s=now_s,
                tick_elapsed_ms=1.0,
            )
            tick_ms = (time.perf_counter() - t0) * 1000.0
            tick_latencies_ms.append(tick_ms)

            # Apply agent writes to simulated MainStage console
            for w in out.agent_writes:
                if rollback_tested and tick_idx == 200 and w.channel_id == "ch5_keys" and w.control == "vol":
                    act7_rollback_respected_lock = False
                if w.control == "vol":
                    vol_cc_map[w.channel_id] = w.cc
                elif w.control == "reverb":
                    rev_cc_map[w.channel_id] = w.cc

            # Render BOTH raw unmixed reference AND AetherMix closed-loop live mix!
            raw_stereo, raw_e = raw_console.mix_block(block, vol_cc_map, rev_cc_map, unity_raw=True)
            mix_stereo, mix_e = console.mix_block(block, vol_cc_map, rev_cc_map, unity_raw=False)
            last_mixed_master = mix_stereo
            raw_mix_blocks.append(raw_stereo)
            aether_mix_blocks.append(mix_stereo)

            # --- Collect Production Gate Metrics ---
            per_act_lufs[act_name].append(metrics.master_lufs_short_term)

            # Settled phase accuracy (after 4.5s hysteresis settle window in each labeled act;
            # exclude Act 8's post-6s speech-over-intro song start from TRANSITION ground truth)
            in_labeled_window = (
                act["expected_phase"] is not None
                and t_in_act >= 4.5
                and not (act.get("speech_over_intro") and t_in_act >= 6.0)
            )
            if in_labeled_window:
                phase_total += 1
                if active_phase == act["expected_phase"]:
                    phase_correct += 1

            # Act 2: instrument_open latency on real stems (note: Act 1 pastor speech_open tail
            # holds -6 dB duck cap for 3s, so keys initially open from -16 dB -> -10 dB!)
            if (
                act_name.startswith("Act 2")
                and act2_instrument_open_latency_s is None
                and out.channels_state["ch5_keys"].instrument_open
                and out.channels_state["ch5_keys"].vol_current_db >= -10.5
            ):
                act2_instrument_open_latency_s = now_s - act_start_s

            # Act 3: zero false pastor speech_open under real band stage bleed
            if act_name.startswith("Act 3") and out.channels_state["ch1_pastor"].speech_open:
                act3_false_speech_open_ticks += 1

            # Act 4: pastor speech_open latency on REAL male human voice over full real band!
            if act_name.startswith("Act 4") and (
                act4_speech_open_latency_s is None
                and out.channels_state["ch1_pastor"].vol_current_db >= -1.0
            ):
                act4_speech_open_latency_s = now_s - act_start_s

            # Act 5: zero false instrument_open during PRAYER_UNDERSCORE
            if act_name.startswith("Act 5") and out.channels_state["ch5_keys"].instrument_open:
                act5_false_instrument_open_ticks += 1

            # Intelligibility comparison in Acts 4 & 5 (Pastor vs Band)
            if (act_name.startswith(("Act 4", "Act 5"))) and t_in_act >= 1.5:
                raw_band = raw_e["ch5_keys"] + raw_e["ch7_guitars"] + raw_e["ch9_drums"] + 1e-12
                mix_band = mix_e["ch5_keys"] + mix_e["ch7_guitars"] + mix_e["ch9_drums"] + 1e-12
                if raw_e["ch1_pastor"] > 1e-5:
                    intelligibility_raw_db.append(10.0 * math.log10((raw_e["ch1_pastor"] + 1e-12) / raw_band))
                    intelligibility_mix_db.append(10.0 * math.log10((mix_e["ch1_pastor"] + 1e-12) / mix_band))

            # Anti-feedback precision & recall
            for ev in out.events:
                if ev.category.value == "SAFETY" and "feedback" in ev.summary.lower():
                    if act.get("inject_feedback") and 2.5 <= t_in_act <= 5.5:
                        if act6_feedback_cut_latency_s is None:
                            act6_feedback_cut_latency_s = t_in_act - 2.5
                    else:
                        non_feedback_false_cuts += 1

            # Act 7: zero-drift check on human-placed ch5_keys
            if human_keys_db_target is not None and human_moved_at_tick is not None and tick_idx > human_moved_at_tick:
                drift = abs(out.channels_state["ch5_keys"].vol_current_db - human_keys_db_target)
                act7_max_human_drift_db = max(act7_max_human_drift_db, drift)

            # Act 8: speech-over-intro duck-capped instrument open (starts at t=6.0s)
            if act.get("speech_over_intro") and t_in_act >= 6.5 and (
                out.channels_state["ch9_drums"].instrument_open
                and out.channels_state["ch9_drums"].vol_target_db <= -6.0
            ):
                act8_speech_over_intro_ducked = True

            cursor += BLOCK_FRAMES
            now_s += 0.02

    vad_worker.stop()

    # Concatenate and write 48 kHz 24-bit WAV deliverables!
    raw_audio = np.concatenate(raw_mix_blocks, axis=0)
    mix_audio = np.concatenate(aether_mix_blocks, axis=0)
    raw_wav_path = output_dir / "01_raw_unmixed_sum.wav"
    mix_wav_path = output_dir / "02_aethermix_production_mix.wav"
    sf.write(str(raw_wav_path), raw_audio, SAMPLE_RATE, subtype="PCM_24")
    sf.write(str(mix_wav_path), mix_audio, SAMPLE_RATE, subtype="PCM_24")

    # Audio quality metrics
    peak_mix_dbfs = 20.0 * math.log10(float(np.max(np.abs(mix_audio))) + 1e-12)
    clipped_samples = int(np.sum(np.abs(mix_audio) >= 10.0 ** (-0.5 / 20.0)))
    max_step_jump = float(np.max(np.abs(np.diff(mix_audio, axis=0))))

    p95_ms = float(np.percentile(tick_latencies_ms, 95))
    p99_ms = float(np.percentile(tick_latencies_ms, 99))
    phase_acc = (phase_correct / phase_total) if phase_total > 0 else 0.0
    intel_gain_db = (
        float(np.mean(intelligibility_mix_db) - np.mean(intelligibility_raw_db))
        if intelligibility_mix_db
        else 0.0
    )

    gates: dict[str, dict[str, Any]] = {
        "1_silero_vad_onnx_real_model": {
            "pass": silero_onnx_active,
            "detail": f"Silero ONNX active={silero_onnx_active} ({SILERO_MODEL_PATH.name})",
        },
        "2_soundcheck_bleed_calibration": {
            "pass": -22.0 <= bleed_offsets.get("ch1_pastor", 0.0) <= -6.0,
            "detail": f"ch1_pastor bleed_offset={bleed_offsets.get('ch1_pastor'):.2f} dB (3-source -16 dB spill)",
        },
        "3_real_audio_phase_accuracy": {
            "pass": phase_acc >= 0.85,
            "detail": f"{phase_acc * 100:.1f}% ({phase_correct}/{phase_total} settled ticks, gate >= 85%)",
        },
        "4_real_speech_open_latency": {
            "pass": act4_speech_open_latency_s is not None and act4_speech_open_latency_s < 1.0,
            "detail": f"{act4_speech_open_latency_s:.3f}s (< 1.0s gate)" if act4_speech_open_latency_s is not None else "NOT TRIGGERED",
        },
        "5_real_instrument_open_latency": {
            "pass": act2_instrument_open_latency_s is not None and act2_instrument_open_latency_s < 1.5,
            "detail": f"{act2_instrument_open_latency_s:.3f}s" if act2_instrument_open_latency_s is not None else "NOT TRIGGERED",
        },
        "6_zero_false_bleed_and_prayer_opens": {
            "pass": act3_false_speech_open_ticks == 0 and act5_false_instrument_open_ticks == 0,
            "detail": f"worship_bleed_false_speech={act3_false_speech_open_ticks}, prayer_false_inst={act5_false_instrument_open_ticks}",
        },
        "7_pastor_intelligibility_gain": {
            "pass": intel_gain_db >= 5.0,
            "detail": f"+{intel_gain_db:.2f} dB Pastor-to-Band SNR improvement vs raw sum (gate >= +5.0 dB)",
        },
        "8_real_anti_feedback_precision_recall": {
            "pass": non_feedback_false_cuts == 0 and act6_feedback_cut_latency_s is not None and act6_feedback_cut_latency_s < 1.0,
            "detail": f"false_cuts_on_real_music={non_feedback_false_cuts}, howl_cut_latency={act6_feedback_cut_latency_s}",
        },
        "9_zero_clipping_and_headroom": {
            "pass": clipped_samples == 0 and peak_mix_dbfs < -0.5,
            "detail": f"peak={peak_mix_dbfs:.2f} dBFS, clipped_samples={clipped_samples}",
        },
        "10_loop_compute_budget_with_onnx": {
            "pass": p95_ms < 5.0 and p99_ms <= 25.0,
            "detail": f"p95={p95_ms:.2f}ms (<5ms), p99={p99_ms:.2f}ms (<=25ms)",
        },
        "11_human_override_zero_drift_and_rollback": {
            "pass": act7_max_human_drift_db < 0.01 and act7_rollback_respected_lock,
            "detail": f"max_drift={act7_max_human_drift_db:.4f} dB, rollback_respected_lock={act7_rollback_respected_lock}",
        },
        "12_speech_over_intro_duck_cap": {
            "pass": act8_speech_over_intro_ducked,
            "detail": f"speech_over_intro_duck_capped={act8_speech_over_intro_ducked}",
        },
    }

    report = {
        "corpus": pack.source_name,
        "stem_files": pack.stem_files,
        "duration_s": round(len(mix_audio) / SAMPLE_RATE, 2),
        "outputs": {
            "raw_unmixed_wav": str(raw_wav_path),
            "aethermix_mix_wav": str(mix_wav_path),
        },
        "calibration": {
            "noise_floors_vband_db": {k: round(v, 2) for k, v in noise_floors.items()},
            "spoken_nominals": {k: [round(a, 2), round(b, 2)] for k, (a, b) in spoken_noms.items()},
            "band_nominals": {k: [round(a, 2), round(b, 2)] for k, (a, b) in band_noms.items()},
            "bleed_offsets_db": {k: round(v, 2) for k, v in bleed_offsets.items()},
        },
        "per_act_median_lufs": {
            k: round(float(np.median(v)), 2) for k, v in per_act_lufs.items() if v
        },
        "max_sample_step": round(max_step_jump, 4),
        "gates": gates,
        "all_passed": all(g["pass"] for g in gates.values()),
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== AetherMix v5.1 Real-Stem Holistic Production Validation ===")
    print(f"Corpus: {pack.source_name}")
    for k, v in pack.stem_files.items():
        print(f"  {k:12s} <- {v}")
    print("\nPer-Act Short-Term LUFS:")
    for act_name, lufs in report["per_act_median_lufs"].items():
        print(f"  {lufs:6.2f} LUFS | {act_name}")
    print("\nProduction Gates:")
    passed_cnt = 0
    for gname, ginfo in gates.items():
        status = "PASS" if ginfo["pass"] else "FAIL"
        if ginfo["pass"]:
            passed_cnt += 1
        print(f"  [{status}] {gname}: {ginfo['detail']}")
    print(f"\nOverall: {passed_cnt}/{len(gates)} gates passed")
    print(f"Rendered WAVs:\n  Raw:       {raw_wav_path}\n  AetherMix: {mix_wav_path}")
    print(f"Report:      {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AetherMix on real studio multitrack stems")
    parser.add_argument("--stems-root", type=Path, default=STEMS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    report = run_production_validation(args.stems_root, args.output_dir)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
