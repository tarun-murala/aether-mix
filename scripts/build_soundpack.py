#!/usr/bin/env python3
"""Compose, render, master, and package the AetherMix Multi-Instrument Sound Pack.

Uses Apple CoreAudio's built-in General MIDI DLS SoundFont (`gs_instruments.dls`)
via `scripts/render_soundpack.swift` (AVAudioEngine offline rendering with
AVAudioUnitSampler, AVAudioUnitReverb, and AVAudioUnitDelay) to build 5 distinct,
seamless-looping multi-instrument tunes mapped 1-to-1 to AetherMix / MainStage's
16-channel bus layout, plus real public-domain spoken word (Ch 1) and stereo
auditorium applause (Ch 13-14).

Output directory (packaged in repo, ~6-9 MB total as 48 kHz OGG Vorbis):
  assets/soundpack/
    manifest.json
    common/
      pastor_speech.ogg       # Ch 1 (real OSR human speech, 48 kHz mono)
      room_applause.ogg       # Ch 13-14 (natural stereo applause, 48 kHz)
    01_morning_grace/         # Warm Worship Ballad (72 BPM, G major)
    02_anthem_rise/           # Upbeat Modern Pop-Rock (124 BPM, D major)
    03_sunday_soul/           # Gospel / Rhodes & B3 Organ (92 BPM, Bb major)
    04_acoustic_campfire/     # Unplugged Folk Acoustic (108 BPM, C major)
    05_still_waters/          # Ambient Prayer & Post-Rock Swell (66 BPM, E minor)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, filtfilt, resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
SOUNDPACK_DIR = REPO_ROOT / "assets" / "soundpack"
SWIFT_RENDERER_SRC = REPO_ROOT / "scripts" / "render_soundpack.swift"
SPEECH_DIR = REPO_ROOT / "recordings" / "stems" / "speech"
SAMPLE_RATE = 48000

# General MIDI Drum Map (Channel 10 / 0-indexed 9)
KICK = 36
RIM = 37
SNARE = 38
CLAP = 39
LOW_TOM = 41
CHH = 42
FLOOR_TOM = 43
PEDAL_HH = 44
MID_TOM = 45
OHH = 46
HI_TOM = 48
CRASH = 49
RIDE = 51
RIDE_BELL = 53
TAMB = 54
CRASH2 = 57
SHAKER = 70


@dataclass
class TuneSpec:
    id: str
    title: str
    genre: str
    bpm: float
    key: str
    bars: int
    recommended_phase: str
    description: str


def beat_to_sec(beat: float, bpm: float) -> float:
    return beat * (60.0 / bpm)


def humanize(
    rng: np.random.Generator,
    start_s: float,
    vel: int,
    *,
    jitter_s: float = 0.008,
    vel_jitter: int = 6,
) -> tuple[float, int]:
    t = max(0.0, start_s + float(rng.uniform(-jitter_s, jitter_s)))
    v = int(np.clip(vel + int(rng.integers(-vel_jitter, vel_jitter + 1)), 18, 125))
    return t, v


def add_strum(
    notes_out: list[dict[str, Any]],
    rng: np.random.Generator,
    chord: list[int],
    start_s: float,
    dur_s: float,
    vel: int,
    *,
    down: bool = True,
    spread_s: float = 0.018,
) -> None:
    ordered = chord if down else list(reversed(chord))
    step = spread_s / max(1, len(ordered) - 1) if len(ordered) > 1 else 0.0
    for idx, n in enumerate(ordered):
        t, v = humanize(rng, start_s + idx * step, vel - (0 if down else 6), jitter_s=0.004, vel_jitter=5)
        notes_out.append({"start": t, "duration": max(0.08, dur_s - idx * step), "note": int(n), "velocity": v})


def make_vibrato_ccs(duration_s: float, *, depth: int = 44, rate_hz: float = 0.25) -> list[dict[str, Any]]:
    """Gentle musical modulation (CC 1) + breathing expression (CC 11) so vocals sound alive."""
    ccs: list[dict[str, Any]] = []
    t = 0.0
    while t < duration_s:
        mod = int(np.clip(depth + 12 * math.sin(2 * math.pi * rate_hz * t), 16, 72))
        expr = int(np.clip(108 + 14 * math.sin(2 * math.pi * (rate_hz * 0.7) * t + 0.5), 82, 124))
        ccs.append({"time": round(t, 3), "cc": 1, "value": mod})
        ccs.append({"time": round(t, 3), "cc": 11, "value": expr})
        t += 0.25
    return ccs


def make_pedal_ccs(bars: int, bpm: float, changes_per_bar: int = 1) -> list[dict[str, Any]]:
    ccs: list[dict[str, Any]] = []
    beats_per_change = 4.0 / changes_per_bar
    total_changes = bars * changes_per_bar
    for i in range(total_changes):
        t = beat_to_sec(i * beats_per_change, bpm)
        if i > 0:
            ccs.append({"time": round(max(0.0, t - 0.04), 3), "cc": 64, "value": 0})
        ccs.append({"time": round(t + 0.03, 3), "cc": 64, "value": 127})
    return ccs


# ---------------------------------------------------------------------------
# 5 Distinct Musical Arrangements (Keys, Guitars, Drums, Bass, Lead & BVs)
# ---------------------------------------------------------------------------


def arrange_morning_grace() -> tuple[TuneSpec, dict[str, list[dict[str, Any]]]]:
    """Tune 1: Warm Worship Ballad — 72 BPM, G Major, 8 bars (26.67 s)."""
    bpm = 72.0
    bars = 8
    dur = beat_to_sec(bars * 4, bpm)
    rng = np.random.default_rng(101)
    spec = TuneSpec(
        id="01_morning_grace",
        title="Morning Grace",
        genre="Warm Worship Ballad",
        bpm=bpm,
        key="G Major",
        bars=bars,
        recommended_phase="WORSHIP_LOW",
        description="Grand piano arpeggios, warm strings/pad, fingerpicked acoustic & clean electric swells, soft ballad drums, fingered bass, and expressive 3-part harmonies.",
    )

    # Progression (8 bars): G | D/F# | Em7 | Cadd9 | G | D | Em7 | Cadd9
    chords_piano = [
        [55, 62, 67, 71, 74],  # G
        [54, 62, 66, 69, 74],  # D/F#
        [52, 59, 64, 67, 74],  # Em7
        [48, 55, 60, 62, 67],  # Cadd9
        [55, 62, 67, 71, 74],
        [50, 57, 62, 66, 69],  # D
        [52, 59, 64, 67, 71],  # Em7
        [48, 55, 60, 62, 67],  # Cadd9
    ]
    chords_guitar = [
        [43, 50, 55, 59, 62, 67],
        [42, 50, 57, 62, 66],
        [40, 47, 52, 55, 59, 64],
        [48, 52, 55, 60, 62, 67],
    ] * 2
    bass_roots = [43, 42, 40, 36, 43, 38, 40, 36]

    piano_notes: list[dict[str, Any]] = []
    pad_notes: list[dict[str, Any]] = []
    ac_gtr_notes: list[dict[str, Any]] = []
    el_gtr_notes: list[dict[str, Any]] = []
    drum_notes: list[dict[str, Any]] = []
    bass_notes: list[dict[str, Any]] = []
    lead_notes: list[dict[str, Any]] = []
    bv1_notes: list[dict[str, Any]] = []
    bv2_notes: list[dict[str, Any]] = []

    for bar in range(bars):
        b0 = bar * 4.0
        chord = chords_piano[bar]
        # Grand piano: rolling ballad eighth-note arpeggio
        arp_idx = [0, 1, 2, 3, 4, 3, 2, 1]
        for step, ci in enumerate(arp_idx):
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), 82 if step % 4 == 0 else 68)
            piano_notes.append({"start": t, "duration": beat_to_sec(0.85, bpm), "note": chord[ci], "velocity": v})
        # Warm Pad: lush whole-bar chord
        for n in chord[1:]:
            pad_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(3.95, bpm), "note": n, "velocity": 66})

        # Steel Acoustic: fingerpicked arpeggio pattern
        gch = chords_guitar[bar]
        pick_order = [0, 2, 3, 4, 3, 2, 4, 3]
        for step, gi in enumerate(pick_order):
            n = gch[min(gi, len(gch) - 1)]
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), 76 if step == 0 else 64)
            ac_gtr_notes.append({"start": t, "duration": beat_to_sec(0.9, bpm), "note": n, "velocity": v})

        # Clean Electric: atmospheric upper triad swell on beats 1 & 3
        for half in (0.0, 2.0):
            add_strum(
                el_gtr_notes, rng, [gch[-3] + 12, gch[-2] + 12, gch[-1] + 12],
                beat_to_sec(b0 + half, bpm), beat_to_sec(1.85, bpm), 62, spread_s=0.045,
            )

        # Ballad Drums: kick on 1 & 3.5, warm rim/snare on 3, gentle 8th hats, tom fill on bar 7
        for kb, kv in [(0.0, 96), (2.5, 82), (3.5, 74)]:
            t, v = humanize(rng, beat_to_sec(b0 + kb, bpm), kv, jitter_s=0.006)
            drum_notes.append({"start": t, "duration": 0.25, "note": KICK, "velocity": v})
        snare_note = RIM if bar < 4 else SNARE
        for sb in (1.0, 3.0):
            t, v = humanize(rng, beat_to_sec(b0 + sb, bpm), 72 if bar < 4 else 86, jitter_s=0.006)
            drum_notes.append({"start": t, "duration": 0.25, "note": snare_note, "velocity": v})
        for step in range(8):
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), 62 if step % 2 == 0 else 44, jitter_s=0.005)
            drum_notes.append({"start": t, "duration": 0.15, "note": CHH, "velocity": v})
        if bar in (0, 4):
            drum_notes.append({"start": beat_to_sec(b0, bpm), "duration": 1.6, "note": CRASH, "velocity": 82})
        if bar == 7:
            for ti, tn in enumerate([HI_TOM, MID_TOM, FLOOR_TOM, FLOOR_TOM]):
                drum_notes.append({"start": beat_to_sec(b0 + 3.0 + ti * 0.25, bpm), "duration": 0.35, "note": tn, "velocity": 78 + ti * 5})

        # Fingered Bass: warm whole/half notes with passing fifth
        root = bass_roots[bar]
        bass_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(2.4, bpm), "note": root, "velocity": 94})
        bass_notes.append({"start": beat_to_sec(b0 + 2.5, bpm), "duration": beat_to_sec(0.9, bpm), "note": root + 7, "velocity": 84})
        bass_notes.append({"start": beat_to_sec(b0 + 3.5, bpm), "duration": beat_to_sec(0.45, bpm), "note": root + 5, "velocity": 80})

    # Lyrical Worship Melody (G Major: G4=67, A4=69, B4=71, C5=72, D5=74, E5=76)
    melody = [
        (0.0, 1.4, 71), (1.5, 0.9, 74), (2.5, 1.2, 71),
        (4.0, 1.4, 69), (5.5, 0.9, 71), (6.5, 1.1, 69),
        (8.0, 1.4, 67), (9.5, 0.9, 71), (10.5, 1.3, 74),
        (12.0, 1.6, 72), (14.0, 1.4, 71),
        (16.0, 1.3, 74), (17.5, 0.9, 76), (18.5, 1.2, 74),
        (20.0, 1.4, 71), (21.5, 0.9, 74), (22.5, 1.1, 71),
        (24.0, 1.4, 69), (25.5, 0.9, 67), (26.5, 1.2, 71),
        (28.0, 1.8, 67), (30.2, 1.1, 69),
    ]
    for b, d, n in melody:
        t, v = humanize(rng, beat_to_sec(b, bpm), 96, jitter_s=0.01)
        lead_notes.append({"start": t, "duration": beat_to_sec(d, bpm), "note": n, "velocity": v})
        # Harmonies join gently in bars 1-4 and fuller in bars 5-8
        if b >= 8.0:
            bv1_notes.append({"start": t + 0.01, "duration": beat_to_sec(d, bpm), "note": n + 4 if n in (67, 71, 72) else n + 3, "velocity": 82})
            bv2_notes.append({"start": t + 0.015, "duration": beat_to_sec(d, bpm), "note": n - 5, "velocity": 80})

    vib = make_vibrato_ccs(dur, depth=46, rate_hz=0.35)
    layers = {
        "keys": [
            {"name": "grand_piano", "program": 0, "gain_db": 0.0, "pan": -0.15, "reverb_wet": 24.0, "notes": piano_notes, "ccs": make_pedal_ccs(bars, bpm, 1)},
            {"name": "warm_pad", "program": 89, "gain_db": -6.0, "pan": 0.15, "reverb_wet": 36.0, "notes": pad_notes},
        ],
        "guitars": [
            {"name": "steel_acoustic", "program": 25, "gain_db": 0.0, "pan": -0.35, "reverb_wet": 18.0, "notes": ac_gtr_notes},
            {"name": "clean_swells", "program": 27, "gain_db": -3.5, "pan": 0.35, "reverb_wet": 38.0, "delay_wet": 22.0, "delay_time": 0.416, "delay_feedback": 30.0, "notes": el_gtr_notes},
        ],
        "drums": [
            {"name": "ballad_kit", "program": 0, "is_drum": True, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 14.0, "notes": drum_notes},
        ],
        "bass": [
            {"name": "fingered_bass", "program": 33, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 4.0, "notes": bass_notes},
        ],
        "lead_vox": [
            {"name": "lead_choir", "program": 52, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 20.0, "notes": lead_notes, "ccs": vib},
            {"name": "lead_ooh_layer", "program": 53, "gain_db": -6.0, "pan": 0.0, "reverb_wet": 16.0, "notes": lead_notes, "ccs": vib},
        ],
        "bv1": [
            {"name": "bv1_harmony", "program": 52, "gain_db": 0.0, "pan": -0.1, "reverb_wet": 22.0, "notes": bv1_notes, "ccs": vib},
        ],
        "bv2": [
            {"name": "bv2_harmony", "program": 53, "gain_db": 0.0, "pan": 0.1, "reverb_wet": 22.0, "notes": bv2_notes, "ccs": vib},
        ],
    }
    return spec, layers


def arrange_anthem_rise() -> tuple[TuneSpec, dict[str, list[dict[str, Any]]]]:
    """Tune 2: Upbeat Modern Worship / Pop-Rock — 124 BPM, D Major, 16 bars (30.97 s)."""
    bpm = 124.0
    bars = 16
    dur = beat_to_sec(bars * 4, bpm)
    rng = np.random.default_rng(202)
    spec = TuneSpec(
        id="02_anthem_rise",
        title="Anthem Rise",
        genre="Modern Pop-Rock Anthem",
        bpm=bpm,
        key="D Major",
        bars=bars,
        recommended_phase="WORSHIP_HIGH",
        description="Driving bright piano octaves, dotted-eighth delay electric riff, overdriven rhythm guitars, energetic rock drum kit, picked bass, and anthemic chorus harmonies.",
    )

    # 4-bar cycle repeated 4x: Bm7 | G | D | A
    chords_piano = [[59, 62, 66, 69], [55, 59, 62, 67], [50, 57, 62, 66], [57, 61, 64, 69]] * 4
    power_chords = [[47, 54, 59], [43, 50, 55], [50, 57, 62], [45, 52, 57]] * 4
    bass_roots = [35, 43, 38, 45] * 4
    dotted_8th_s = beat_to_sec(0.75, bpm)  # ~0.363 s authentic U2/worship dotted-8th delay!

    piano_notes: list[dict[str, Any]] = []
    synth_notes: list[dict[str, Any]] = []
    delay_gtr_notes: list[dict[str, Any]] = []
    od_gtr_notes: list[dict[str, Any]] = []
    drum_notes: list[dict[str, Any]] = []
    bass_notes: list[dict[str, Any]] = []
    lead_notes: list[dict[str, Any]] = []
    bv1_notes: list[dict[str, Any]] = []
    bv2_notes: list[dict[str, Any]] = []

    riff_pattern = [74, 73, 74, 78, 74, 73, 71, 69]  # Iconic D major shimmer hook

    for bar in range(bars):
        b0 = bar * 4.0
        ch = chords_piano[bar]
        # Driving Bright Piano quarter/eighth pulse
        for step in range(8):
            vel = 96 if step % 2 == 0 else 78
            for n in ch:
                t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), vel, jitter_s=0.005)
                piano_notes.append({"start": t, "duration": beat_to_sec(0.42, bpm), "note": n, "velocity": v})
        # Soaring Synth Pad/Brass swell on chorus bars
        for n in ch:
            synth_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(3.9, bpm), "note": n + 12, "velocity": 72})

        # Dotted-8th Delay Clean Electric Riff
        for step in range(4):
            note = riff_pattern[(bar * 2 + step) % len(riff_pattern)]
            t, v = humanize(rng, beat_to_sec(b0 + step * 1.0, bpm), 92, jitter_s=0.005)
            delay_gtr_notes.append({"start": t, "duration": beat_to_sec(0.45, bpm), "note": note, "velocity": v})

        # Overdriven Power Chords (driving 8ths)
        pc = power_chords[bar]
        for step in range(8):
            accent = 104 if step in (0, 3, 6) else 84
            add_strum(od_gtr_notes, rng, pc, beat_to_sec(b0 + step * 0.5, bpm), beat_to_sec(0.42, bpm), accent, down=(step % 2 == 0), spread_s=0.01)

        # Driving Rock Drums
        for kb in (0.0, 1.5, 2.0, 2.5):
            t, v = humanize(rng, beat_to_sec(b0 + kb, bpm), 114 if kb in (0.0, 2.0) else 98, jitter_s=0.004)
            drum_notes.append({"start": t, "duration": 0.2, "note": KICK, "velocity": v})
        for sb in (1.0, 3.0):
            t, v = humanize(rng, beat_to_sec(b0 + sb, bpm), 112, jitter_s=0.004)
            drum_notes.append({"start": t, "duration": 0.2, "note": SNARE, "velocity": v})
        for step in range(8):
            hh = OHH if step == 7 else CHH
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), 82 if step % 2 == 0 else 64, jitter_s=0.004)
            drum_notes.append({"start": t, "duration": 0.14, "note": hh, "velocity": v})
        if bar % 4 == 0:
            drum_notes.append({"start": beat_to_sec(b0, bpm), "duration": 1.2, "note": CRASH, "velocity": 102})
        if bar in (7, 15):
            for ti, tn in enumerate([SNARE, HI_TOM, MID_TOM, FLOOR_TOM]):
                drum_notes.append({"start": beat_to_sec(b0 + 3.0 + ti * 0.25, bpm), "duration": 0.22, "note": tn, "velocity": 98 + ti * 4})

        # Picked Rock Bass (pumping 8th notes with octave jump on beat 4)
        root = bass_roots[bar]
        for step in range(8):
            n = root + 12 if step == 7 else root
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), 104 if step % 2 == 0 else 88, jitter_s=0.004)
            bass_notes.append({"start": t, "duration": beat_to_sec(0.42, bpm), "note": n, "velocity": v})

    # Anthemic Melody in natural human vocal range (D4=62, E4=64, F#4=66, G4=67, A4=69, B4=71)
    phrase = [
        (0.0, 1.3, 66), (1.5, 1.3, 64), (3.0, 0.9, 62),
        (4.0, 1.8, 62), (6.0, 1.4, 59),
        (8.0, 1.3, 66), (9.5, 1.3, 67), (11.0, 0.9, 66),
        (12.0, 1.8, 64), (14.0, 1.4, 62),
    ]
    for rep in range(4):
        offset = rep * 16.0
        for b, d, n in phrase:
            t, v = humanize(rng, beat_to_sec(offset + b, bpm), 106, jitter_s=0.008)
            lead_notes.append({"start": t, "duration": beat_to_sec(d, bpm), "note": n, "velocity": v})
            bv1_notes.append({"start": t + 0.01, "duration": beat_to_sec(d, bpm), "note": n + 3 if n in (59, 64, 67) else n + 4, "velocity": 90})
            bv2_notes.append({"start": t + 0.012, "duration": beat_to_sec(d, bpm), "note": n - 5, "velocity": 88})

    vib = make_vibrato_ccs(dur, depth=46, rate_hz=0.45)
    layers = {
        "keys": [
            {"name": "bright_piano", "program": 1, "gain_db": 0.0, "pan": -0.2, "reverb_wet": 18.0, "notes": piano_notes},
            {"name": "polysynth_pad", "program": 90, "gain_db": -6.5, "pan": 0.2, "reverb_wet": 28.0, "notes": synth_notes},
        ],
        "guitars": [
            {"name": "delay_clean_lead", "program": 27, "gain_db": 0.0, "pan": -0.35, "reverb_wet": 24.0, "delay_wet": 36.0, "delay_time": round(dotted_8th_s, 3), "delay_feedback": 38.0, "notes": delay_gtr_notes},
            {"name": "overdrive_rhythm", "program": 29, "gain_db": -2.0, "pan": 0.35, "reverb_wet": 14.0, "notes": od_gtr_notes},
        ],
        "drums": [
            {"name": "rock_kit", "program": 0, "is_drum": True, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 16.0, "notes": drum_notes},
        ],
        "bass": [
            {"name": "picked_bass", "program": 34, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 3.0, "notes": bass_notes},
        ],
        "lead_vox": [
            {"name": "anthem_lead", "program": 52, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 22.0, "notes": lead_notes, "ccs": vib},
            {"name": "anthem_lead_formant", "program": 54, "gain_db": -5.5, "pan": 0.0, "reverb_wet": 18.0, "notes": lead_notes, "ccs": vib},
        ],
        "bv1": [
            {"name": "anthem_bv1", "program": 52, "gain_db": 0.0, "pan": -0.15, "reverb_wet": 24.0, "notes": bv1_notes, "ccs": vib},
            {"name": "anthem_bv1_formant", "program": 54, "gain_db": -6.0, "pan": -0.15, "reverb_wet": 20.0, "notes": bv1_notes, "ccs": vib},
        ],
        "bv2": [
            {"name": "anthem_bv2", "program": 52, "gain_db": 0.0, "pan": 0.15, "reverb_wet": 24.0, "notes": bv2_notes, "ccs": vib},
            {"name": "anthem_bv2_formant", "program": 54, "gain_db": -6.0, "pan": 0.15, "reverb_wet": 20.0, "notes": bv2_notes, "ccs": vib},
        ],
    }
    return spec, layers


def arrange_sunday_soul() -> tuple[TuneSpec, dict[str, list[dict[str, Any]]]]:
    """Tune 3: Gospel / Rhodes & B3 Organ Groove — 92 BPM, Bb Major, 8 bars (20.87 s)."""
    bpm = 92.0
    bars = 8
    dur = beat_to_sec(bars * 4, bpm)
    rng = np.random.default_rng(303)
    spec = TuneSpec(
        id="03_sunday_soul",
        title="Sunday Soul",
        genre="Gospel Rhodes & Organ Groove",
        bpm=bpm,
        key="Bb Major",
        bars=bars,
        recommended_phase="WORSHIP_HIGH",
        description="Soulful Fender Rhodes comping, Hammond B3 organ swells, funky jazz-clean guitar chords, pocket gospel drums with ghost notes, walking bass, and rich gospel choir.",
    )

    # Gospel ii9 - V13 - Imaj9 - VI7alt progression in Bb Major
    chords = [
        [60, 63, 67, 70, 74],  # Cm9
        [57, 63, 67, 70, 74],  # F13
        [58, 62, 65, 69, 72],  # Bbmaj9
        [55, 59, 63, 65, 70],  # G7#5
    ] * 2
    bass_lines = [
        [(0.0, 36), (1.5, 36), (2.0, 43), (3.0, 39), (3.5, 40)],  # C walk to F
        [(0.0, 41), (1.5, 41), (2.0, 48), (3.0, 45), (3.5, 44)],  # F walk to Bb
        [(0.0, 34), (1.5, 34), (2.0, 41), (3.0, 38), (3.5, 39)],  # Bb walk to G
        [(0.0, 43), (1.5, 43), (2.0, 41), (3.0, 39), (3.5, 38)],  # G walk to C
    ] * 2

    rhodes_notes: list[dict[str, Any]] = []
    organ_notes: list[dict[str, Any]] = []
    gtr_notes: list[dict[str, Any]] = []
    drum_notes: list[dict[str, Any]] = []
    bass_notes: list[dict[str, Any]] = []
    lead_notes: list[dict[str, Any]] = []
    bv1_notes: list[dict[str, Any]] = []
    bv2_notes: list[dict[str, Any]] = []

    for bar in range(bars):
        b0 = bar * 4.0
        ch = chords[bar]
        # Soulful syncopated Rhodes comping
        for cb, cd, cv in [(0.0, 1.2, 92), (1.5, 0.9, 82), (2.75, 0.9, 88)]:
            add_strum(rhodes_notes, rng, ch, beat_to_sec(b0 + cb, bpm), beat_to_sec(cd, bpm), cv, spread_s=0.022)
        # Hammond B3 Organ swells
        for n in ch[:4]:
            organ_notes.append({"start": beat_to_sec(b0 + 0.05, bpm), "duration": beat_to_sec(3.75, bpm), "note": n, "velocity": 76})

        # Funky Jazz Clean Guitar syncopated offbeat stabs
        for gb in (0.5, 1.5, 2.5, 3.25):
            add_strum(gtr_notes, rng, ch[1:4], beat_to_sec(b0 + gb, bpm), beat_to_sec(0.28, bpm), 82, down=(gb in (0.5, 2.5)), spread_s=0.012)

        # Pocket Gospel Drums (slight 16th swing feel)
        for kb, kv in [(0.0, 108), (1.75, 92), (2.5, 102)]:
            t, v = humanize(rng, beat_to_sec(b0 + kb, bpm), kv, jitter_s=0.005)
            drum_notes.append({"start": t, "duration": 0.22, "note": KICK, "velocity": v})
        for sb, sv in [(1.0, 106), (2.25, 52), (3.0, 108), (3.75, 58)]:  # Snare + ghost notes!
            t, v = humanize(rng, beat_to_sec(b0 + sb, bpm), sv, jitter_s=0.006)
            drum_notes.append({"start": t, "duration": 0.2, "note": SNARE, "velocity": v})
        for step in range(8):
            swing = 0.03 if step % 2 == 1 else 0.0
            cym = RIDE_BELL if (bar >= 4 and step % 2 == 0) else (RIDE if bar >= 4 else CHH)
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5 + swing, bpm), 78 if step % 2 == 0 else 58, jitter_s=0.004)
            drum_notes.append({"start": t, "duration": 0.2, "note": cym, "velocity": v})
        if bar in (0, 4):
            drum_notes.append({"start": beat_to_sec(b0, bpm), "duration": 1.3, "note": CRASH2, "velocity": 90})

        # Walking Gospel Bass
        for bb, bn in bass_lines[bar]:
            t, v = humanize(rng, beat_to_sec(b0 + bb, bpm), 102, jitter_s=0.005)
            bass_notes.append({"start": t, "duration": beat_to_sec(0.65, bpm), "note": bn, "velocity": v})

    # Soulful Gospel Melody (Bb major pentatonic / gospel scale: Bb4=70, C5=72, D5=74, F5=77, G5=79)
    melody = [
        (0.0, 0.9, 72), (1.0, 0.8, 74), (2.0, 1.4, 77),
        (4.0, 0.9, 74), (5.0, 0.9, 72), (6.0, 1.4, 70),
        (8.0, 1.2, 74), (9.5, 0.9, 72), (10.5, 1.3, 70),
        (12.0, 1.4, 67), (13.75, 1.3, 70),
        (16.0, 1.0, 74), (17.25, 0.8, 77), (18.25, 1.4, 79),
        (20.0, 1.1, 77), (21.5, 0.9, 74), (22.5, 1.3, 72),
        (24.0, 1.2, 74), (25.5, 0.9, 72), (26.5, 1.3, 70),
        (28.0, 1.6, 70), (30.0, 1.2, 72),
    ]
    for b, d, n in melody:
        t, v = humanize(rng, beat_to_sec(b, bpm), 102, jitter_s=0.009)
        lead_notes.append({"start": t, "duration": beat_to_sec(d, bpm), "note": n, "velocity": v})
        bv1_notes.append({"start": t + 0.01, "duration": beat_to_sec(d, bpm), "note": n + 3, "velocity": 88})
        bv2_notes.append({"start": t + 0.015, "duration": beat_to_sec(d, bpm), "note": n - 5, "velocity": 86})

    vib = make_vibrato_ccs(dur, depth=50, rate_hz=0.45)
    layers = {
        "keys": [
            {"name": "fender_rhodes", "program": 4, "gain_db": 0.0, "pan": -0.25, "reverb_wet": 18.0, "notes": rhodes_notes, "ccs": make_pedal_ccs(bars, bpm, 1)},
            {"name": "hammond_b3", "program": 18, "gain_db": -5.5, "pan": 0.25, "reverb_wet": 24.0, "notes": organ_notes},
        ],
        "guitars": [
            {"name": "jazz_clean_gtr", "program": 26, "gain_db": 0.0, "pan": 0.15, "reverb_wet": 16.0, "notes": gtr_notes},
        ],
        "drums": [
            {"name": "gospel_kit", "program": 0, "is_drum": True, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 12.0, "notes": drum_notes},
        ],
        "bass": [
            {"name": "soul_bass", "program": 33, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 3.0, "notes": bass_notes},
        ],
        "lead_vox": [
            {"name": "gospel_lead", "program": 52, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 20.0, "notes": lead_notes, "ccs": vib},
        ],
        "bv1": [
            {"name": "gospel_bv1", "program": 52, "gain_db": 0.0, "pan": -0.2, "reverb_wet": 24.0, "notes": bv1_notes, "ccs": vib},
        ],
        "bv2": [
            {"name": "gospel_bv2", "program": 53, "gain_db": 0.0, "pan": 0.2, "reverb_wet": 24.0, "notes": bv2_notes, "ccs": vib},
        ],
    }
    return spec, layers


def arrange_acoustic_campfire() -> tuple[TuneSpec, dict[str, list[dict[str, Any]]]]:
    """Tune 4: Unplugged Folk / Indie Acoustic — 108 BPM, C Major, 12 bars (26.67 s)."""
    bpm = 108.0
    bars = 12
    dur = beat_to_sec(bars * 4, bpm)
    rng = np.random.default_rng(404)
    spec = TuneSpec(
        id="04_acoustic_campfire",
        title="Acoustic Campfire",
        genre="Unplugged Folk Acoustic",
        bpm=bpm,
        key="C Major",
        bars=bars,
        recommended_phase="WORSHIP_LOW",
        description="Strummed steel acoustic guitar, nylon counter-melody, upright double bass, organic shaker/tambourine/rimshot percussion, soft acoustic piano & warm strings.",
    )

    # 4-bar folk progression repeated 3x: C | Fmaj7 | Am7 | G
    chords = [[48, 55, 60, 64, 67], [41, 48, 53, 57, 64], [45, 52, 57, 60, 67], [43, 50, 55, 59, 62]] * 3
    bass_roots = [36, 41, 45, 43] * 3

    piano_notes: list[dict[str, Any]] = []
    strings_notes: list[dict[str, Any]] = []
    steel_notes: list[dict[str, Any]] = []
    nylon_notes: list[dict[str, Any]] = []
    drum_notes: list[dict[str, Any]] = []
    bass_notes: list[dict[str, Any]] = []
    lead_notes: list[dict[str, Any]] = []
    bv1_notes: list[dict[str, Any]] = []
    bv2_notes: list[dict[str, Any]] = []

    # Classic folk strum: D - D U - U D U
    strum_pattern = [(0.0, True, 92), (1.0, True, 84), (1.5, False, 74), (2.5, False, 76), (3.0, True, 86), (3.5, False, 74)]

    for bar in range(bars):
        b0 = bar * 4.0
        ch = chords[bar]
        # Warm Acoustic Piano block chords on 1 & 3
        for pb in (0.0, 2.0):
            add_strum(piano_notes, rng, [n + 12 for n in ch[:4]], beat_to_sec(b0 + pb, bpm), beat_to_sec(1.8, bpm), 76, spread_s=0.015)
        # Subtle Warm Strings legato pad
        for n in ch[2:]:
            strings_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(3.9, bpm), "note": n + 12, "velocity": 58})

        # Steel Acoustic Strumming
        for sb, down, sv in strum_pattern:
            add_strum(steel_notes, rng, ch, beat_to_sec(b0 + sb, bpm), beat_to_sec(0.55, bpm), sv, down=down, spread_s=0.018)

        # Nylon Acoustic fingerpicked counter-melody
        for step in range(4):
            n = ch[(step + 2) % len(ch)] + 12
            t, v = humanize(rng, beat_to_sec(b0 + step * 1.0 + 0.5, bpm), 72, jitter_s=0.008)
            nylon_notes.append({"start": t, "duration": beat_to_sec(0.75, bpm), "note": n, "velocity": v})

        # Organic Acoustic Percussion (kick + cross-stick + shaker + tambourine)
        for kb in (0.0, 2.0, 2.5):
            t, v = humanize(rng, beat_to_sec(b0 + kb, bpm), 92 if kb == 0 else 78, jitter_s=0.006)
            drum_notes.append({"start": t, "duration": 0.2, "note": KICK, "velocity": v})
        for rb in (1.0, 3.0):
            t, v = humanize(rng, beat_to_sec(b0 + rb, bpm), 82, jitter_s=0.006)
            drum_notes.append({"start": t, "duration": 0.16, "note": RIM, "velocity": v})
            drum_notes.append({"start": t + 0.005, "duration": 0.18, "note": TAMB, "velocity": 68})
        for step in range(8):
            t, v = humanize(rng, beat_to_sec(b0 + step * 0.5, bpm), 62 if step % 2 == 0 else 44, jitter_s=0.006)
            drum_notes.append({"start": t, "duration": 0.12, "note": SHAKER, "velocity": v})

        # Upright Double Bass root-fifth bounce
        root = bass_roots[bar]
        bass_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(1.7, bpm), "note": root, "velocity": 96})
        bass_notes.append({"start": beat_to_sec(b0 + 2.0, bpm), "duration": beat_to_sec(1.4, bpm), "note": root + 7, "velocity": 86})
        bass_notes.append({"start": beat_to_sec(b0 + 3.5, bpm), "duration": beat_to_sec(0.45, bpm), "note": root + 5, "velocity": 80})

    # Intimate Folk Melody in warm human vocal range (C4=60, D4=62, E4=64, F4=65, G4=67, A4=69)
    phrase = [
        (0.0, 1.3, 64), (1.5, 0.9, 65), (2.5, 1.2, 67),
        (4.0, 1.4, 64), (5.5, 0.9, 60), (6.5, 1.1, 64),
        (8.0, 1.3, 67), (9.5, 0.9, 65), (10.5, 1.2, 64),
        (12.0, 1.6, 62), (14.0, 1.3, 60),
    ]
    for rep in range(3):
        off = rep * 16.0
        for b, d, n in phrase:
            t, v = humanize(rng, beat_to_sec(off + b, bpm), 94, jitter_s=0.009)
            lead_notes.append({"start": t, "duration": beat_to_sec(d, bpm), "note": n, "velocity": v})
            if rep >= 1:
                bv1_notes.append({"start": t + 0.01, "duration": beat_to_sec(d, bpm), "note": n + 4 if n in (60, 65) else n + 3, "velocity": 80})
                bv2_notes.append({"start": t + 0.015, "duration": beat_to_sec(d, bpm), "note": n - 5, "velocity": 78})

    vib = make_vibrato_ccs(dur, depth=42, rate_hz=0.38)
    layers = {
        "keys": [
            {"name": "folk_piano", "program": 0, "gain_db": -1.0, "pan": -0.2, "reverb_wet": 18.0, "notes": piano_notes, "ccs": make_pedal_ccs(bars, bpm, 1)},
            {"name": "warm_strings", "program": 48, "gain_db": -7.0, "pan": 0.25, "reverb_wet": 28.0, "notes": strings_notes},
        ],
        "guitars": [
            {"name": "steel_strum", "program": 25, "gain_db": 0.0, "pan": -0.3, "reverb_wet": 16.0, "notes": steel_notes},
            {"name": "nylon_counter", "program": 24, "gain_db": -2.5, "pan": 0.35, "reverb_wet": 20.0, "notes": nylon_notes},
        ],
        "drums": [
            {"name": "acoustic_perc", "program": 0, "is_drum": True, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 12.0, "notes": drum_notes},
        ],
        "bass": [
            {"name": "upright_bass", "program": 32, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 6.0, "notes": bass_notes},
        ],
        "lead_vox": [
            {"name": "folk_lead", "program": 52, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 18.0, "notes": lead_notes, "ccs": vib},
            {"name": "folk_lead_formant", "program": 54, "gain_db": -5.5, "pan": 0.0, "reverb_wet": 16.0, "notes": lead_notes, "ccs": vib},
        ],
        "bv1": [
            {"name": "folk_bv1", "program": 52, "gain_db": 0.0, "pan": -0.15, "reverb_wet": 20.0, "notes": bv1_notes, "ccs": vib},
            {"name": "folk_bv1_formant", "program": 54, "gain_db": -6.0, "pan": -0.15, "reverb_wet": 18.0, "notes": bv1_notes, "ccs": vib},
        ],
        "bv2": [
            {"name": "folk_bv2", "program": 52, "gain_db": 0.0, "pan": 0.15, "reverb_wet": 20.0, "notes": bv2_notes, "ccs": vib},
            {"name": "folk_bv2_formant", "program": 54, "gain_db": -6.0, "pan": 0.15, "reverb_wet": 18.0, "notes": bv2_notes, "ccs": vib},
        ],
    }
    return spec, layers


def arrange_still_waters() -> tuple[TuneSpec, dict[str, list[dict[str, Any]]]]:
    """Tune 5: Ambient Prayer Underscore & Cinematic Swell — 66 BPM, E Minor, 8 bars (29.09 s)."""
    bpm = 66.0
    bars = 8
    dur = beat_to_sec(bars * 4, bpm)
    rng = np.random.default_rng(505)
    spec = TuneSpec(
        id="05_still_waters",
        title="Still Waters",
        genre="Ambient Prayer & Cinematic Pad",
        bpm=bpm,
        key="E Minor",
        bars=bars,
        recommended_phase="PRAYER_UNDERSCORE",
        description="Lush atmospheric pads, soft felt grand piano chords, ambient echo clean electric arpeggios, cinematic floor-tom heartbeat, fretless/sub bass, and ethereal choir oohs.",
    )

    # Em9 | Cmaj7 | G | Dsus4 repeated 2x
    chords = [
        [52, 59, 64, 66, 67, 71],  # Em9
        [48, 55, 60, 64, 67, 71],  # Cmaj7
        [43, 50, 55, 59, 62, 67],  # G
        [50, 57, 62, 67, 69, 74],  # Dsus4
    ] * 2
    bass_roots = [40, 36, 43, 38] * 2

    pad_notes: list[dict[str, Any]] = []
    halo_notes: list[dict[str, Any]] = []
    piano_notes: list[dict[str, Any]] = []
    gtr_notes: list[dict[str, Any]] = []
    drum_notes: list[dict[str, Any]] = []
    bass_notes: list[dict[str, Any]] = []
    lead_notes: list[dict[str, Any]] = []
    bv1_notes: list[dict[str, Any]] = []
    bv2_notes: list[dict[str, Any]] = []

    for bar in range(bars):
        b0 = bar * 4.0
        ch = chords[bar]
        for n in ch[1:5]:
            pad_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(3.98, bpm), "note": n, "velocity": 72})
        for n in ch[3:]:
            halo_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(3.98, bpm), "note": n + 12, "velocity": 54})
        # Delicate Felt Piano chord + top shimmer notes
        add_strum(piano_notes, rng, ch[1:5], beat_to_sec(b0, bpm), beat_to_sec(3.6, bpm), 62, spread_s=0.05)
        piano_notes.append({"start": beat_to_sec(b0 + 2.0, bpm), "duration": beat_to_sec(1.8, bpm), "note": ch[-1] + 12, "velocity": 56})

        # Ambient Clean Electric echoing arpeggio
        for step in range(4):
            n = ch[(step + 2) % len(ch)] + 12
            t, v = humanize(rng, beat_to_sec(b0 + step * 1.0, bpm), 64, jitter_s=0.01)
            gtr_notes.append({"start": t, "duration": beat_to_sec(1.1, bpm), "note": n, "velocity": v})

        # Cinematic Floor Tom Heartbeat Pulse
        for tb, tv in [(0.0, 82), (1.0, 68), (2.0, 78), (3.0, 64), (3.5, 58)]:
            t, v = humanize(rng, beat_to_sec(b0 + tb, bpm), tv, jitter_s=0.008)
            drum_notes.append({"start": t, "duration": 0.45, "note": FLOOR_TOM if tb in (0.0, 2.0) else LOW_TOM, "velocity": v})
        if bar in (0, 4):
            drum_notes.append({"start": beat_to_sec(b0, bpm), "duration": 2.4, "note": CRASH, "velocity": 62})

        # Deep Fretless Bass long notes
        bass_notes.append({"start": beat_to_sec(b0, bpm), "duration": beat_to_sec(3.85, bpm), "note": bass_roots[bar], "velocity": 86})

    # Gentle Ethereal Vocal Swell in warm human range (B3=59, C4=60, D4=62, E4=64, F#4=66, G4=67)
    melody = [
        (0.5, 2.8, 64), (4.5, 2.8, 64), (8.5, 2.8, 67), (12.5, 2.8, 62),
        (16.5, 2.8, 67), (20.5, 2.8, 66), (24.5, 2.8, 64), (28.5, 2.8, 62),
    ]
    for b, d, n in melody:
        t, v = humanize(rng, beat_to_sec(b, bpm), 78, jitter_s=0.012)
        lead_notes.append({"start": t, "duration": beat_to_sec(d, bpm), "note": n, "velocity": v})
        bv1_notes.append({"start": t + 0.02, "duration": beat_to_sec(d, bpm), "note": n + 3, "velocity": 70})
        bv2_notes.append({"start": t + 0.02, "duration": beat_to_sec(d, bpm), "note": n - 5, "velocity": 68})

    vib = make_vibrato_ccs(dur, depth=52, rate_hz=0.28)
    layers = {
        "keys": [
            {"name": "warm_pad", "program": 89, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 46.0, "notes": pad_notes},
            {"name": "halo_pad", "program": 94, "gain_db": -7.5, "pan": 0.25, "reverb_wet": 52.0, "notes": halo_notes},
            {"name": "felt_piano", "program": 0, "gain_db": -2.0, "pan": -0.2, "reverb_wet": 38.0, "notes": piano_notes, "ccs": make_pedal_ccs(bars, bpm, 1)},
        ],
        "guitars": [
            {"name": "ambient_echo_gtr", "program": 27, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 44.0, "delay_wet": 42.0, "delay_time": 0.454, "delay_feedback": 46.0, "notes": gtr_notes},
        ],
        "drums": [
            {"name": "cinematic_toms", "program": 0, "is_drum": True, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 34.0, "notes": drum_notes},
        ],
        "bass": [
            {"name": "fretless_bass", "program": 35, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 12.0, "notes": bass_notes},
        ],
        "lead_vox": [
            {"name": "ethereal_lead", "program": 52, "gain_db": 0.0, "pan": 0.0, "reverb_wet": 36.0, "notes": lead_notes, "ccs": vib},
            {"name": "ethereal_lead_formant", "program": 54, "gain_db": -5.5, "pan": 0.0, "reverb_wet": 32.0, "notes": lead_notes, "ccs": vib},
        ],
        "bv1": [
            {"name": "ethereal_bv1", "program": 52, "gain_db": 0.0, "pan": -0.2, "reverb_wet": 38.0, "notes": bv1_notes, "ccs": vib},
            {"name": "ethereal_bv1_formant", "program": 54, "gain_db": -6.0, "pan": -0.2, "reverb_wet": 34.0, "notes": bv1_notes, "ccs": vib},
        ],
        "bv2": [
            {"name": "ethereal_bv2", "program": 52, "gain_db": 0.0, "pan": 0.2, "reverb_wet": 38.0, "notes": bv2_notes, "ccs": vib},
            {"name": "ethereal_bv2_formant", "program": 54, "gain_db": -6.0, "pan": 0.2, "reverb_wet": 34.0, "notes": bv2_notes, "ccs": vib},
        ],
    }
    return spec, layers


# ---------------------------------------------------------------------------
# Mastering, Seamless Loop Crossfade & OGG Vorbis Encoding
# ---------------------------------------------------------------------------


def apply_studio_fx(
    audio: np.ndarray,
    *,
    reverb_wet: float = 0.0,
    delay_wet: float = 0.0,
    delay_time_s: float = 0.363,
    delay_feedback: float = 0.36,
) -> np.ndarray:
    """Apply clean stereo ping-pong delay and warm studio hall reverb."""
    out = audio.astype(np.float32, copy=True)
    if out.ndim == 1:
        out = np.repeat(out[:, None], 2, axis=1)

    if delay_wet > 0.01:
        tap_frames = max(1, round(delay_time_s * SAMPLE_RATE))
        wet_buf = np.zeros_like(out)
        gain = delay_wet
        for tap in range(1, 5):
            shift = tap * tap_frames
            if shift >= len(out):
                break
            # Ping-pong alternate L/R taps with warm high-cut decay
            if tap % 2 == 1:
                wet_buf[shift:, 0] += out[:-shift, 1] * gain
                wet_buf[shift:, 1] += out[:-shift, 0] * (gain * 0.65)
            else:
                wet_buf[shift:, 0] += out[:-shift, 0] * (gain * 0.65)
                wet_buf[shift:, 1] += out[:-shift, 1] * gain
            gain *= delay_feedback
        b_lp, a_lp = butter(2, 4200.0 / (0.5 * SAMPLE_RATE), btype="lowpass")
        for c in range(2):
            wet_buf[:, c] = filtfilt(b_lp, a_lp, wet_buf[:, c]).astype(np.float32)
        out = out + wet_buf

    if reverb_wet > 0.01:
        rng = np.random.default_rng(777)
        ir_len = int(1.45 * SAMPLE_RATE)
        t = np.arange(ir_len, dtype=np.float32) / SAMPLE_RATE
        # Smooth diffused stereo hall impulse response with 18 ms pre-delay
        pre = int(0.018 * SAMPLE_RATE)
        env = np.exp(-t / 0.42).astype(np.float32)
        ir_l = rng.standard_normal(ir_len).astype(np.float32) * env
        ir_r = rng.standard_normal(ir_len).astype(np.float32) * env
        ir_l[:pre] = 0.0
        ir_r[:pre] = 0.0
        b_lp, a_lp = butter(2, 5200.0 / (0.5 * SAMPLE_RATE), btype="lowpass")
        ir_l = filtfilt(b_lp, a_lp, ir_l).astype(np.float32)
        ir_r = filtfilt(b_lp, a_lp, ir_r).astype(np.float32)
        norm = max(1e-6, float(np.sqrt(np.sum(ir_l**2))))
        ir_l /= norm
        ir_r /= norm
        wet_l = fftconvolve(out[:, 0], ir_l, mode="full")[: len(out)].astype(np.float32)
        wet_r = fftconvolve(out[:, 1], ir_r, mode="full")[: len(out)].astype(np.float32)
        out = np.stack([out[:, 0] + wet_l * reverb_wet, out[:, 1] + wet_r * reverb_wet], axis=1)

    return out


def master_and_loop(
    audio: np.ndarray,
    *,
    loop_frames: int,
    mono: bool,
    target_rms_db: float,
    hp_hz: float | None = None,
    reverb_wet: float = 0.0,
    delay_wet: float = 0.0,
    delay_time_s: float = 0.363,
    vocal_warmth: bool = False,
) -> np.ndarray:
    """Apply studio FX, fold tail into loop start for click-free seamless looping, HPF, & normalize."""
    if audio.ndim == 1:
        audio = audio[:, None]

    if vocal_warmth:
        # Gentle asymmetric tube-preamp harmonic warmth (adds realistic 2nd & 3rd vocal overtones
        # like real human vocal cords + slight studio breath presence)
        pk = max(1e-6, float(np.max(np.abs(audio))))
        normed = audio / pk
        warm = np.tanh(1.65 * normed + 0.22 * (normed**2)) - 0.14 * float(np.mean(normed**2))
        audio = (0.72 * normed + 0.28 * warm).astype(np.float32) * pk

    if reverb_wet > 0.01 or delay_wet > 0.01:
        audio = apply_studio_fx(
            audio, reverb_wet=reverb_wet, delay_wet=delay_wet, delay_time_s=delay_time_s
        )

    # Fold reverb/release tail past loop_frames back into start so loop wraps organically!
    core = audio[:loop_frames].copy()
    tail = audio[loop_frames:]
    if len(tail) > 0:
        n_fold = min(len(tail), loop_frames)
        fade_out = np.linspace(1.0, 0.0, n_fold, dtype=np.float32)[:, None]
        core[:n_fold] += tail[:n_fold] * fade_out

    # Tiny 12 ms edge micro-fade to guarantee zero DC click at wrap
    edge = min(576, loop_frames // 20)
    ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
    core[:edge] *= ramp
    core[-edge:] *= ramp[::-1]

    if mono:
        core = np.mean(core, axis=1, keepdims=True)
    elif core.shape[1] == 1:
        core = np.repeat(core, 2, axis=1)

    if hp_hz is not None:
        b, a = butter(2, hp_hz / (0.5 * SAMPLE_RATE), btype="highpass")
        for c in range(core.shape[1]):
            core[:, c] = filtfilt(b, a, core[:, c]).astype(np.float32)

    # Normalize over active 100 ms windows (> -48 dBFS) so rests don't over-boost peaks
    flat = core.reshape(-1)
    win = 4800
    energies = [
        float(np.mean(flat[i : i + win] ** 2))
        for i in range(0, max(1, len(flat) - win + 1), win)
        if float(np.mean(flat[i : i + win] ** 2)) > 10.0 ** (-48.0 / 10.0)
    ]
    ref_rms = math.sqrt(float(np.median(energies))) if energies else float(np.sqrt(np.mean(flat**2) + 1e-12))
    if ref_rms > 1e-6:
        gain = (10.0 ** (target_rms_db / 20.0)) / ref_rms
        peak = float(np.max(np.abs(core)))
        if peak * gain > 0.88:
            gain = 0.88 / peak
        core = (core * gain).astype(np.float32)

    return core[:, 0] if mono else core


def write_ogg(audio: np.ndarray, out_ogg: Path, bitrate: str = "128k") -> None:
    """Encode float32 48 kHz audio to crisp compact OGG Vorbis via ffmpeg (or soundfile fallback)."""
    out_ogg.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        try:
            sf.write(str(tmp_wav), audio, SAMPLE_RATE, subtype="FLOAT")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(tmp_wav),
                    "-c:a",
                    "libvorbis",
                    "-b:a",
                    bitrate,
                    "-ar",
                    str(SAMPLE_RATE),
                    str(out_ogg),
                ],
                check=True,
            )
        finally:
            tmp_wav.unlink(missing_ok=True)
    else:
        sf.write(str(out_ogg), audio, SAMPLE_RATE, format="OGG", subtype="VORBIS")


def build_common_stems(out_dir: Path) -> dict[str, str]:
    """Package real OSR recorded human speech (Ch 1) and natural stereo applause (Ch 13-14)."""
    common_dir = out_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real recorded speech (Open Speech Repository public-domain male & female Harvard sentences)
    speech_clips: list[np.ndarray] = []
    for fname in ("pastor_male.wav", "speaker_female.wav"):
        p = SPEECH_DIR / fname
        if p.exists():
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            if sr != SAMPLE_RATE:
                g = math.gcd(SAMPLE_RATE, sr)
                mono = resample_poly(mono, SAMPLE_RATE // g, sr // g).astype(np.float32)
            # Trim leading silence > -26 dBFS
            win = 2400
            thresh = 10.0 ** (-26.0 / 20.0)
            for i in range(0, max(1, len(mono) - win), win // 2):
                if float(np.sqrt(np.mean(mono[i : i + win] ** 2))) >= thresh:
                    mono = mono[i:]
                    break
            speech_clips.append(mono)
            speech_clips.append(np.zeros(int(0.45 * SAMPLE_RATE), dtype=np.float32))

    if not speech_clips:
        raise FileNotFoundError(
            f"Missing real speech WAVs under {SPEECH_DIR}. Run `python scripts/download_validation_stems.py` first."
        )

    speech_all = np.concatenate(speech_clips)
    # Keep first ~42 seconds so both male and female voices are packaged compactly
    speech_all = speech_all[: int(42.0 * SAMPLE_RATE)]
    speech_master = master_and_loop(
        speech_all, loop_frames=len(speech_all), mono=True, target_rms_db=-16.5, hp_hz=85.0
    )
    write_ogg(speech_master, common_dir / "pastor_speech.ogg", bitrate="96k")

    # 2. Natural stereo auditorium applause (Ch 13-14 room mics, 12 s seamless loop)
    rng = np.random.default_rng(909)
    n_app = int(12.0 * SAMPLE_RATE)
    noise = rng.standard_normal((n_app, 2)).astype(np.float32)
    b_bp, a_bp = butter(2, [450.0 / (0.5 * SAMPLE_RATE), 5800.0 / (0.5 * SAMPLE_RATE)], btype="bandpass")
    applause = np.stack([filtfilt(b_bp, a_bp, noise[:, c]) for c in range(2)], axis=1).astype(np.float32)
    # Add overlapping realistic palm clap transients across the room
    for _ in range(1100):
        pos = int(rng.integers(0, n_app - 1600))
        pan_l = float(rng.uniform(0.25, 0.95))
        pan_r = math.sqrt(max(0.05, 1.0 - pan_l * pan_l))
        amp = float(rng.uniform(0.35, 1.15))
        t_c = np.arange(1400, dtype=np.float32) / SAMPLE_RATE
        clap_env = np.exp(-t_c / float(rng.uniform(0.008, 0.022)))
        clap = rng.standard_normal(1400).astype(np.float32) * clap_env * amp
        applause[pos : pos + 1400, 0] += clap * pan_l
        applause[pos : pos + 1400, 1] += clap * pan_r
    applause_master = master_and_loop(applause, loop_frames=n_app, mono=False, target_rms_db=-18.0, hp_hz=200.0)
    write_ogg(applause_master, common_dir / "room_applause.ogg", bitrate="112k")

    return {
        "pastor_speech": "common/pastor_speech.ogg",
        "room_applause": "common/room_applause.ogg",
    }


def compile_swift_renderer(bin_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("CLANG_MODULE_CACHE_PATH", "/tmp/clang-module-cache")
    print(f"[compile] {SWIFT_RENDERER_SRC.name} -> {bin_path}")
    subprocess.run(["swiftc", "-O", str(SWIFT_RENDERER_SRC), "-o", str(bin_path)], check=True, env=env)


def main() -> int:
    SOUNDPACK_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aethermix_render_") as tmpdir:
        tmp = Path(tmpdir)
        renderer_bin = tmp / "render_soundpack"
        compile_swift_renderer(renderer_bin)

        arrangers = [
            arrange_morning_grace,
            arrange_anthem_rise,
            arrange_sunday_soul,
            arrange_acoustic_campfire,
            arrange_still_waters,
        ]

        # Target nominal RMS levels matching engine/schemas/service_plan.json
        stem_targets: dict[str, tuple[bool, float, float | None, str]] = {
            "lead_vox": (True, -18.0, 100.0, "96k"),
            "bv1": (True, -20.0, 120.0, "96k"),
            "bv2": (True, -20.0, 120.0, "96k"),
            "keys": (False, -16.0, 45.0, "144k"),
            "guitars": (False, -16.0, 75.0, "144k"),
            "drums": (False, -14.5, 32.0, "144k"),
            "bass": (False, -15.5, 28.0, "112k"),
        }

        manifest_tunes: list[dict[str, Any]] = []
        for arrange_fn in arrangers:
            spec, stem_layers = arrange_fn()
            loop_dur_s = beat_to_sec(spec.bars * 4, spec.bpm)
            loop_frames = round(loop_dur_s * SAMPLE_RATE)
            render_dur_s = loop_dur_s + 2.5  # capture full reverb/release tail for seamless foldback
            tune_dir = SOUNDPACK_DIR / spec.id
            tune_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[tune] {spec.id}: {spec.title} ({spec.genre} · {spec.bpm:.0f} BPM · {spec.key} · {loop_dur_s:.2f}s)")
            job_paths: list[str] = []
            wav_map: dict[str, Path] = {}
            for stem_name, layers in stem_layers.items():
                wav_path = tmp / f"{spec.id}_{stem_name}.wav"
                job_path = tmp / f"{spec.id}_{stem_name}.json"
                job_path.write_text(
                    json.dumps(
                        {
                            "sample_rate": SAMPLE_RATE,
                            "duration": render_dur_s,
                            "output_wav": str(wav_path),
                            "layers": layers,
                        }
                    ),
                    encoding="utf-8",
                )
                job_paths.append(str(job_path))
                wav_map[stem_name] = wav_path

            # Render all 7 stems for this tune in one fast CoreAudio batch invocation
            subprocess.run([str(renderer_bin), *job_paths], check=True)

            stem_relpaths: dict[str, str] = {}
            for stem_name, wav_path in wav_map.items():
                mono, target_rms, hp_hz, bitrate = stem_targets[stem_name]
                layers_cfg = stem_layers[stem_name]
                rev_wet = max((float(l.get("reverb_wet", 0.0)) for l in layers_cfg), default=0.0) / 100.0
                del_wet = max((float(l.get("delay_wet", 0.0)) for l in layers_cfg), default=0.0) / 100.0
                del_time = next((float(l["delay_time"]) for l in layers_cfg if "delay_time" in l), 0.363)
                raw, _ = sf.read(str(wav_path), dtype="float32", always_2d=True)
                mastered = master_and_loop(
                    raw,
                    loop_frames=loop_frames,
                    mono=mono,
                    target_rms_db=target_rms,
                    hp_hz=hp_hz,
                    reverb_wet=rev_wet * 0.85,
                    delay_wet=del_wet * 0.85,
                    delay_time_s=del_time,
                    vocal_warmth=(stem_name in ("lead_vox", "bv1", "bv2")),
                )
                out_ogg = tune_dir / f"{stem_name}.ogg"
                write_ogg(mastered, out_ogg, bitrate=bitrate)
                stem_relpaths[stem_name] = f"{spec.id}/{stem_name}.ogg"
                size_kb = out_ogg.stat().st_size / 1024.0
                print(f"  - {stem_name:10s} -> {out_ogg.name:14s} ({size_kb:5.1f} KB)")

            manifest_tunes.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "genre": spec.genre,
                    "bpm": spec.bpm,
                    "key": spec.key,
                    "bars": spec.bars,
                    "duration_s": round(loop_dur_s, 3),
                    "recommended_phase": spec.recommended_phase,
                    "description": spec.description,
                    "stems": stem_relpaths,
                }
            )

        common_map = build_common_stems(SOUNDPACK_DIR)
        manifest = {
            "schema_version": 1,
            "sample_rate": SAMPLE_RATE,
            "channel_map": {
                "ch1": "Pastor Mic (common/pastor_speech.ogg)",
                "ch2": "Lead Vocal (<tune>/lead_vox.ogg)",
                "ch3": "Backup Vocal 1 (<tune>/bv1.ogg)",
                "ch4": "Backup Vocal 2 (<tune>/bv2.ogg)",
                "ch5_6": "Keys / Pad / Organ / Piano Stereo (<tune>/keys.ogg)",
                "ch7_8": "Guitars Stereo (<tune>/guitars.ogg)",
                "ch9_10": "Drums Stereo (<tune>/drums.ogg)",
                "ch11_12": "Bass Stereo (<tune>/bass.ogg)",
                "ch13_14": "Room Mics / Applause Stereo (common/room_applause.ogg)",
                "ch15_16": "Master Return Tap (routed back from MainStage or engine monitor)",
            },
            "common": common_map,
            "tunes": manifest_tunes,
        }
        (SOUNDPACK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(p.stat().st_size for p in SOUNDPACK_DIR.rglob("*") if p.is_file())
    print(f"\n[done] Packaged 5 multi-instrument tunes + speech + applause in {SOUNDPACK_DIR}")
    print(f"       Total size: {total_bytes / (1024 * 1024):.2f} MB ({len(list(SOUNDPACK_DIR.rglob('*.ogg')))} OGG stems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
