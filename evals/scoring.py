"""Score one service run (TickLog + Timeline) against the gates in manifest/gates.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from engine.core.protocol import MixingRules, ServicePhase

from evals.manifest import Gate, GatesFile
from evals.runner import CHANNELS, DT, PHASES, TickLog
from evals.timeline import Timeline

OPEN_WINDOW_S = 3.0  # how long to wait for a reflex before calling it missed
SPEECH_OPEN_FADER_DB = -1.0  # same open criterion as scripts/simulate_service.py
CLOSED_FADER_DB = -6.0  # an onset only counts for the reflex when the pastor fader was below this
FB_RELEASE_TAIL_S = 8.0


@dataclass
class GateResult:
    gate: str
    service: str
    value: float | None
    passed: bool | None  # None = not measured / informational
    detail: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


def judge(gate: Gate, value: float | None) -> bool | None:
    if value is None or gate.informational:
        return None
    return not (
        (gate.min is not None and not value >= gate.min)
        or (gate.max is not None and not value <= gate.max)
        or (gate.equals is not None and value != gate.equals)
    )


def _i(t: float) -> int:
    return round(t / DT)


def _clip(log: TickLog, a: float, b: float) -> tuple[int, int]:
    return max(0, _i(a)), min(len(log.t), _i(b))


def macro_f1(truth: list[str], pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    per: dict[str, dict[str, float]] = {}
    labels = sorted(set(truth))
    t_arr, p_arr = np.array(truth), np.array(pred)
    for lab in labels:
        tp = int(np.sum((t_arr == lab) & (p_arr == lab)))
        fp = int(np.sum((t_arr != lab) & (p_arr == lab)))
        fn = int(np.sum((t_arr == lab) & (p_arr != lab)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = {"precision": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0, "support": int(np.sum(t_arr == lab))}
    return (float(np.mean([v["f1"] for v in per.values()])) if per else 0.0), per


def score_run(log: TickLog, tl: Timeline, gates: GatesFile, rules: MixingRules, settle_s: float) -> list[GateResult]:
    sid = log.service
    out: list[GateResult] = []

    def add(name: str, value: float | None, detail: str, evidence: list[dict[str, Any]] | None = None) -> None:
        out.append(GateResult(name, sid, None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value),
                              judge(gates.gates[name], value), detail, evidence or []))

    # --- phase classification -------------------------------------------------
    truth: list[str] = []
    pred: list[str] = []
    seg_acc: list[dict[str, Any]] = []
    for s in tl.segments:
        if s.phase is None:
            continue
        a, b = _clip(log, s.t0 + settle_s, s.t1)
        if b <= a:
            continue
        p = [PHASES[x] for x in log.phase[a:b]]
        truth += [s.phase] * len(p)
        pred += p
        seg_acc.append({"segment": s.id, "truth": s.phase, "accuracy": round(float(np.mean([x == s.phase for x in p])), 3),
                        "modal_pred": max(set(p), key=p.count)})
    if truth:
        f1, per = macro_f1(truth, pred)
        add("phase_macro_f1", f1, f"macro F1 {f1:.3f} over {len(truth)} settled ticks; per label {per}", seg_acc)
    else:
        add("phase_macro_f1", None, "no labelled segments in this profile")

    ev = lambda kind: [e for e in tl.events if e.kind == kind]

    # --- speech-open ----------------------------------------------------------
    lat: list[float] = []
    misses = 0
    ev_rows: list[dict[str, Any]] = []
    for e in ev("speech_onset"):
        i0 = _i(e.params.get("t_onset", e.t0))
        pre = log.vol_db["ch1_pastor"][i0 - 1] if i0 > 0 else 0.0
        if pre > CLOSED_FADER_DB:
            ev_rows.append({"layer": e.layer, "t": round(e.t0, 2), "result": "skipped (fader already open)", "pre_db": round(float(pre), 1)})
            continue
        hi = min(len(log.t), i0 + _i(OPEN_WINDOW_S))
        hit = next((k for k in range(i0, hi) if log.speech_open[k] and log.vol_db["ch1_pastor"][k] >= SPEECH_OPEN_FADER_DB), None)
        if hit is None:
            misses += 1
            ev_rows.append({"layer": e.layer, "t": round(e.t0, 2), "result": "MISSED", "pre_db": round(float(pre), 1)})
        else:
            lat.append((hit - i0) * DT)
            ev_rows.append({"layer": e.layer, "t": round(e.t0, 2), "result": f"{(hit - i0) * DT:.3f}s", "pre_db": round(float(pre), 1)})
    add("speech_open_latency_s", max(lat) if lat else None, f"{len(lat)} onsets measured, {misses} missed", ev_rows)
    add("speech_open_misses", float(misses) if (lat or misses) else None, f"{misses} missed of {len(lat) + misses} measurable onsets")

    # --- instrument-open ------------------------------------------------------
    i_lat: list[float] = []
    rows: list[dict[str, Any]] = []
    for e in ev("instrument_open"):
        i0 = _i(e.params.get("t_onset", e.t0))
        hi = min(len(log.t), i0 + _i(OPEN_WINDOW_S))
        hit = next((k for k in range(i0, hi) if any(log.instrument_open[c][k] for c in e.channels)), None)
        pre = max(float(log.vol_db[c][max(0, i0 - 1)]) for c in e.channels)
        rise = max(float(log.vol_db[c][i0 : min(len(log.t), i0 + _i(1.0))].max()) for c in e.channels) - pre
        if hit is None:
            i_lat.append(math.inf)
            rows.append({"layer": e.layer, "t": round(e.t0, 2), "result": "NEVER OPENED", "rise_db_1s": round(rise, 1)})
        else:
            l = (hit - i0) * DT
            i_lat.append(l if rise >= 10.0 else math.inf)
            rows.append({"layer": e.layer, "t": round(e.t0, 2), "result": f"{l:.3f}s", "rise_db_1s": round(rise, 1)})
    add("instrument_open_latency_s", max(i_lat) if i_lat else None, f"{len(i_lat)} song starts measured", rows)

    # --- false opens + probes -------------------------------------------------
    viol: list[dict[str, Any]] = []
    for e in ev("no_instrument_open"):
        a, b = _clip(log, e.t0, e.t1 + 1.0)
        for c in e.channels:
            if bool(log.instrument_open[c][a:b].any()):
                viol.append({"window": e.layer or e.segment, "channel": c, "t": round(e.t0, 2)})
    n_windows = len(ev("no_instrument_open"))
    add("false_instrument_opens", float(len(viol)) if n_windows else None, f"{len(viol)} violations in {n_windows} windows", viol)
    probes = []
    for e in ev("probe_instrument_open"):
        a, b = _clip(log, e.t0, e.t1 + 1.0)
        probes.append({"window": e.layer, "channel": e.channels[0], "fired": bool(log.instrument_open[e.channels[0]][a:b].any())})
    fired = sum(1 for p in probes if p["fired"])
    add("ch9_single_onset_opens", float(fired) if probes else None, f"{fired}/{len(probes)} single-instrument probes fired the reflex", probes)

    # --- speech over intro cap ------------------------------------------------
    duck = rules.firewall.duck_depth_db
    worship = {ServicePhase.WORSHIP_LOW.value, ServicePhase.WORSHIP_HIGH.value}
    worst = None
    for e in ev("speech_over_intro_cap"):
        a, b = _clip(log, e.t0, e.t1)
        for k in range(a, b):
            if not log.speech_open[k]:
                continue
            for c in e.channels:
                if log.instrument_open[c][k]:
                    ph = PHASES[log.phase[k]]
                    base = rules.phase_baselines_db[ServicePhase(ph if ph in worship else ServicePhase.WORSHIP_LOW.value)]
                    ex = float(log.vol_db[c][k]) - (base[c] + duck)
                    worst = ex if worst is None else max(worst, ex)
    add("speech_over_intro_excess_db", worst, "max excess over (WORSHIP_LOW baseline + duck depth) while open under speech"
        if worst is not None else "instrument_open never fired under the intro speech")

    # --- ducking --------------------------------------------------------------
    drops: list[dict[str, Any]] = []
    for e in ev("ducking"):
        i0 = _i(e.params.get("t_onset", e.t0))
        for c in e.channels:
            pre = float(log.vol_db[c][max(0, i0 - 1)])
            if pre < -20.0:
                continue
            post = float(log.vol_db[c][i0 : min(len(log.t), i0 + _i(2.0))].min())
            drops.append({"layer": e.layer, "channel": c, "pre_db": round(pre, 1), "drop_db": round(pre - post, 1)})
    add("duck_drop_db", min(d["drop_db"] for d in drops) if drops else None, f"{len(drops)} audible channels measured", drops)

    # --- feedback -------------------------------------------------------------
    fb_rows: list[dict[str, Any]] = []
    fb_lat: list[float] = []
    windows: list[tuple[float, float]] = []
    for e in ev("feedback_cut"):
        c = e.channels[0]
        i0 = _i(e.params.get("t_onset", e.t0))
        windows.append((e.t0, e.t1 + FB_RELEASE_TAIL_S))
        hit = next((k for k in range(i0, min(len(log.t), i0 + _i(OPEN_WINDOW_S))) if not np.isnan(log.feedback_ceiling[c][k])), None)
        fb_lat.append(math.inf if hit is None else (hit - i0) * DT)
        fb_rows.append({"layer": e.layer, "channel": c, "result": "NO CUT" if hit is None else f"{(hit - i0) * DT:.3f}s"})
    add("feedback_cut_latency_s", max(fb_lat) if fb_lat else None, f"{len(fb_lat)} howls injected", fb_rows)
    episodes: list[dict[str, Any]] = []
    for c in CHANNELS:
        active = ~np.isnan(log.feedback_ceiling[c])
        starts = np.flatnonzero(active & ~np.concatenate([[False], active[:-1]]))
        for k in starts:
            t = k * DT
            if not any(w0 <= t <= w1 for w0, w1 in windows):
                episodes.append({"channel": c, "t": round(float(t), 2), "segment": tl.segment_at(t).id})
    add("false_feedback_cuts", float(len(episodes)), f"{len(episodes)} cuts outside howl windows", episodes)

    # --- human override -------------------------------------------------------
    drift: list[float] = []
    for e in ev("human_move"):
        c = e.channels[0]
        tgt = log.human_target_db.get(c)
        if tgt is None:
            continue
        a, b = _clip(log, e.t0 + 0.2, e.t1)
        drift.append(float(np.max(np.abs(log.vol_db[c][a:b] - tgt))) if b > a else math.nan)
    add("human_override_drift_db", max(drift) if drift else None, f"{len(drift)} human moves held", [{"drift_db": round(d, 3)} for d in drift])

    # --- transition slew ------------------------------------------------------
    slews: list[float] = []
    for s in tl.segments:
        if s.phase != "TRANSITION":
            continue
        a, b = _clip(log, s.t0 + 0.5, s.t1)
        for c in CHANNELS[1:]:
            if b - a > 2:
                slews.append(float(np.max(np.abs(np.diff(log.vol_db[c][a:b]))) / DT))
    add("transition_max_slew_db_s", max(slews) if slews else None, "worst |d fader/dt| across TRANSITION segments")

    # --- loudness -------------------------------------------------------------
    errs: list[dict[str, Any]] = []
    for s in tl.segments:
        if s.phase in (None, "TRANSITION"):
            continue
        a, b = _clip(log, s.t0 + settle_s + 3.0, s.t1)
        if b - a < 50:
            continue
        med = float(np.median(log.lufs_short[a:b]))
        if med > -69.0:
            errs.append({"segment": s.id, "median_lufs": round(med, 1), "target": rules.phase_target_lufs[ServicePhase(s.phase)],
                         "err_lu": round(abs(med - rules.phase_target_lufs[ServicePhase(s.phase)]), 1)})
    add("lufs_error_lu", max(e["err_lu"] for e in errs) if errs else None, "emulated master, informational", errs)

    # --- clarity --------------------------------------------------------------
    clar: list[dict[str, Any]] = []
    n = len(log.t)
    speech_on = np.zeros(n, dtype=bool)
    band_on = np.zeros(n, dtype=bool)
    for lay in tl.layers:
        a, b = _clip(log, lay.t0, lay.t1)
        if lay.part == "speech":
            speech_on[a:b] = True
        elif lay.part in ("keys", "guitar", "drums", "bass"):
            band_on[a:b] = True
    for s in tl.segments:
        a, b = _clip(log, s.t0 + 1.5, s.t1)
        m = speech_on[a:b] & band_on[a:b]
        if m.sum() * DT < 5.0:
            continue
        sl = slice(a, b)
        mix = 10 * math.log10(log.pastor_e_mix[sl][m].sum() / (log.band_e_mix[sl][m].sum() + 1e-12) + 1e-12)
        raw = 10 * math.log10(log.pastor_e_raw[sl][m].sum() / (log.band_e_raw[sl][m].sum() + 1e-12) + 1e-12)
        clar.append({"segment": s.id, "gain_db": round(mix - raw, 1)})
    add("clarity_gain_db", min(c["gain_db"] for c in clar) if clar else None, f"{len(clar)} speech-over-band segments", clar)

    # --- no speech-open on singing/silence ------------------------------------
    nso = ev("no_speech_open")
    bad_ticks = sum(int(log.speech_open[_clip(log, e.t0, e.t1)[0] : _clip(log, e.t0, e.t1)[1]].sum()) for e in nso)
    add("no_speech_open_violations", float(bad_ticks) if nso else None, f"{bad_ticks} speech_open ticks across {len(nso)} windows")

    # --- loop latency ---------------------------------------------------------
    p99 = float(np.percentile(log.tick_ms, 99))
    add("loop_p99_ms", p99, f"p95 {np.percentile(log.tick_ms, 95):.2f} ms, p99 {p99:.2f} ms on this machine (headless)")
    return out
