"""Compile a service script + source stems into a deterministic 16-channel 48 kHz timeline.

The timeline yields (960, 16) blocks (one 50 Hz engine tick) with taps 1-12 pre-fader, room on 13-14 and the
master slots 15-16 left empty (the runner fills them from the emulated console, or MainStage returns them live).
It also produces the ground-truth markers and the scoring events derived from each layer's ``expect`` block.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evals import BLOCK_FRAMES, DEFAULT_STEMS_ROOT, SAMPLE_RATE
from evals.manifest import META_PARTS, PART_COLUMNS, Layer, Service, ServicesFile, SourcesFile
from evals.stems import PartSource, load_source, part_for

TAP_GROUPS = ((0,), (1,), (2,), (3,), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
BAND_COLS = (4, 5, 6, 7, 8, 9, 10, 11)


@dataclass
class SegmentSpan:
    id: str
    title: str
    phase: str | None
    t0: float
    t1: float
    step_idx: int | None
    expect: dict[str, Any]


@dataclass
class Event:
    kind: str  # speech_onset | instrument_open | no_instrument_open | probe_instrument_open | speech_over_intro_cap
    #            | ducking | feedback_cut | no_speech_open | human_move
    t0: float
    t1: float
    segment: str
    layer: str | None = None
    channels: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerRT:
    id: str
    part: str
    cols: tuple[int, ...]
    source: PartSource | None
    t0: float
    t1: float
    offset_s: float
    gain: float
    fade_out_s: float | None
    howl: dict[str, float] | None = None


@dataclass
class Timeline:
    service: Service
    profile: str
    segments: list[SegmentSpan]
    layers: list[LayerRT]
    events: list[Event]
    trims_db: dict[str, float]
    bleed_gain: float
    seed: int
    total_s: float
    synthetic_parts: list[str]

    @property
    def n_ticks(self) -> int:
        return round(self.total_s * SAMPLE_RATE / BLOCK_FRAMES)

    def segment_at(self, t: float) -> SegmentSpan:
        for s in self.segments:
            if s.t0 <= t < s.t1:
                return s
        return self.segments[-1]

    def block(self, tick: int) -> np.ndarray:
        start = tick * BLOCK_FRAMES
        t_abs = (start + np.arange(BLOCK_FRAMES)) / SAMPLE_RATE
        t_lo, t_hi = t_abs[0], t_abs[-1]
        blk = np.zeros((BLOCK_FRAMES, 16), dtype=np.float64)
        rng = np.random.default_rng([self.seed, 7, tick])
        for lay in self.layers:
            if lay.t1 <= t_lo or lay.t0 > t_hi:
                continue
            gain = np.full(BLOCK_FRAMES, lay.gain, dtype=np.float64)
            active = (t_abs >= lay.t0) & (t_abs < lay.t1)
            if lay.fade_out_s:
                db = -60.0 * np.clip((t_abs - (lay.t1 - lay.fade_out_s)) / lay.fade_out_s, 0.0, 1.0)
                gain *= 10.0 ** (db / 20.0)
            gain *= active
            if lay.howl is not None:
                h = lay.howl
                amp = h["start_amp"] * 10.0 ** (h["growth_db_per_s"] * np.clip(t_abs - lay.t0, 0, None) / 20.0)
                blk[:, int(h["col"])] += gain * np.minimum(amp, 0.5) * np.sin(2.0 * np.pi * h["freq_hz"] * t_abs)
                continue
            assert lay.source is not None
            j0 = max(0, int(np.ceil((lay.t0 - t_lo) * SAMPLE_RATE - 1e-9)))  # first sample inside the layer
            if j0 >= BLOCK_FRAMES:
                continue
            src_start = round((t_lo + j0 / SAMPLE_RATE - lay.t0 + lay.offset_s) * SAMPLE_RATE)
            seg = lay.source.read(max(0, src_start), BLOCK_FRAMES - j0).astype(np.float64) * gain[j0:, None]
            if len(lay.cols) == 1:
                blk[j0:, lay.cols[0]] += seg.mean(axis=1)
            else:
                blk[j0:, lay.cols[0]] += seg[:, 0]
                blk[j0:, lay.cols[1]] += seg[:, 1]
        for grp in TAP_GROUPS:
            g = 10.0 ** (self.trims_db[str(grp[0])] / 20.0)
            for c in grp:
                blk[:, c] *= g
        band = 0.33 * (0.5 * (blk[:, 4] + blk[:, 5]) + 0.5 * (blk[:, 6] + blk[:, 7]) + 0.5 * (blk[:, 8] + blk[:, 9])
                       + 0.5 * (blk[:, 10] + blk[:, 11]))
        for c in range(4):
            blk[:, c] += band * self.bleed_gain
        blk[:, :12] += rng.standard_normal((BLOCK_FRAMES, 12)) * 10.0 ** (-58.0 / 20.0)
        hum = 10.0 ** (-52.0 / 20.0) * (0.7 * np.sin(2 * np.pi * 120 * t_abs) + 0.3 * np.sin(2 * np.pi * 240 * t_abs))
        blk[:, 12] += hum
        blk[:, 13] += hum
        return np.clip(blk, -1.0, 1.0).astype(np.float32)

    def markers(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for s in self.segments:
            rows.append({"t0_s": s.t0, "t1_s": s.t1, "kind": "segment", "id": s.id, "phase": s.phase or "UNSCORED", "detail": s.title})
        for e in self.events:
            rows.append({"t0_s": e.t0, "t1_s": e.t1, "kind": e.kind, "id": e.layer or e.segment, "phase": "", "detail": ",".join(e.channels)})
        rows.sort(key=lambda r: (r["t0_s"], r["kind"] != "segment"))
        return rows

    def write_markers_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["t0_s", "t1_s", "kind", "id", "phase", "detail"])
            w.writeheader()
            for r in self.markers():
                w.writerow({**r, "t0_s": f"{r['t0_s']:.3f}", "t1_s": f"{r['t1_s']:.3f}"})


def audio_onset_s(ps: PartSource, offset_s: float, dur_s: float, thr_db: float = -40.0, search_s: float = 30.0) -> float | None:
    """Seconds after the layer start until the part first rises above thr_db (real speech and songs rarely start on beat)."""
    thr = 10.0 ** (thr_db / 20.0)
    for k in range(int(min(dur_s, search_s) * SAMPLE_RATE / BLOCK_FRAMES)):
        blk = ps.read(round(offset_s * SAMPLE_RATE) + k * BLOCK_FRAMES, BLOCK_FRAMES)
        if float(np.sqrt(np.mean(blk.astype(np.float64) ** 2))) >= thr:
            return k * BLOCK_FRAMES / SAMPLE_RATE
    return None


def _expand_parts(part: str) -> tuple[str, ...]:
    return META_PARTS.get(part, (part,))


def build_timeline(
    services: ServicesFile,
    sources: SourcesFile,
    service_id: str,
    profile: str = "quick",
    stems_root: Path = DEFAULT_STEMS_ROOT,
    bleed_db: float | None = None,
    trim_range_db: float | None = None,
) -> Timeline:
    service = next(s for s in services.services if s.id == service_id)
    by_id = sources.by_id()
    loaded: dict[str, dict[str, PartSource]] = {}
    for slot, sid in service.slots.items():
        loaded[slot] = load_source(by_id[sid], stems_root, service.seed)

    rng = np.random.default_rng(service.seed)
    span = services.trim_range_db if trim_range_db is None else trim_range_db
    trims = {str(g[0]): float(rng.uniform(-span, span)) for g in TAP_GROUPS}

    segs: list[SegmentSpan] = []
    layers: list[LayerRT] = []
    events: list[Event] = []
    cursor: dict[tuple[str, str], float] = {}
    synthetic: set[str] = set()
    t = 0.0
    for seg in services.segments_for(profile):
        dur = float(seg.dur[profile])
        t0, t1 = t, t + dur
        segs.append(SegmentSpan(seg.id, seg.title, seg.phase, t0, t1, seg.step_idx, dict(seg.expect)))
        for lay in seg.layers:
            _compile_layer(lay, seg.id, t0, t1, service, by_id, loaded, cursor, layers, events, synthetic)
        for act in seg.actions:
            ta = t0 + act.at_frac * dur
            events.append(Event("human_move", ta, min(t1, ta + act.hold_s), seg.id, None, [act.channel],
                                {"delta_db": act.delta_db, "hold_s": act.hold_s}))
        if seg.expect.get("no_speech_open"):
            events.append(Event("no_speech_open", t0, t1, seg.id, None, ["ch1_pastor"]))
        if seg.expect.get("no_instrument_open"):
            events.append(Event("no_instrument_open", t0, t1, seg.id, None, list(seg.expect["no_instrument_open"])))
        t = t1
    bleed = services.bleed_db if bleed_db is None else bleed_db
    return Timeline(service, profile, segs, layers, events, trims, 10.0 ** (bleed / 20.0), service.seed, t, sorted(synthetic))


def _compile_layer(
    lay: Layer, seg_id: str, t0: float, t1: float, service: Service, by_id: dict[str, Any],
    loaded: dict[str, dict[str, PartSource]], cursor: dict[tuple[str, str], float],
    layers: list[LayerRT], events: list[Event], synthetic: set[str],
) -> None:
    start = t0 + (lay.at_s if lay.at_s is not None else lay.at_frac * (t1 - t0))
    if lay.dur_s is not None:
        end = start + lay.dur_s
    elif lay.dur_frac is not None:
        end = start + lay.dur_frac * (t1 - t0)
    else:
        end = t1
    end = min(end, t1)
    if lay.part == "howl":
        layers.append(LayerRT(lay.id, "howl", (), None, start, end, 0.0, 10.0 ** (lay.gain_db / 20.0), None,
                              howl={"col": float(lay.col or 0), "freq_hz": lay.freq_hz, "start_amp": lay.start_amp,
                                    "growth_db_per_s": lay.growth_db_per_s}))
    else:
        assert lay.src is not None
        offset = cursor.get((lay.src, lay.part), 0.0) if lay.continue_ else lay.src_offset_s
        sid = service.slots[lay.src]
        for part in _expand_parts(lay.part):
            ps = part_for(loaded[lay.src], sid, part, service.seed)
            if ps.synthetic:
                synthetic.add(f"{lay.src}:{sid}:{part}")
            layers.append(LayerRT(f"{lay.id}.{part}", part, PART_COLUMNS[part], ps, start, end, offset,
                                  10.0 ** (lay.gain_db / 20.0), lay.fade_out_s))
        cursor[(lay.src, lay.part)] = offset + (end - start)
    onsets = [audio_onset_s(l.source, l.offset_s, l.t1 - l.t0) for l in layers if l.id.startswith(f"{lay.id}.") and l.source]
    onsets = [o for o in onsets if o is not None]
    t_onset = start + (min(onsets) if onsets else 0.0) if lay.part != "howl" else start
    for kind, val in lay.expect.items():
        chans = [val] if isinstance(val, str) else list(val) if isinstance(val, (list, tuple)) else []
        if kind == "speech_onset":
            events.append(Event("speech_onset", start, end, seg_id, lay.id, ["ch1_pastor"], {"t_onset": t_onset}))
        elif kind == "speech_over_intro_cap":
            events.append(Event("speech_over_intro_cap", start, end, seg_id, lay.id,
                                ["ch5_keys", "ch7_guitars", "ch9_drums"]))
        else:
            events.append(Event(kind, start, end, seg_id, lay.id, chans, {"t_onset": t_onset}))
