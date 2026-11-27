"""Headless closed-loop runner: drives a Timeline through the real AetherMixEngine 50 Hz pipeline.

Taps 1-12 carry the pre-fader stems; the post-fader master (taps 15-16) is rebuilt every tick from the engine's own
fader state, which emulates the MainStage return path (Mode A+ in docs/EVAL_SPEC.md). It exercises perception,
the safety firewall, the offset learner and the LUFS meter exactly as the live engine does, but not MainStage itself.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from engine.core.protocol import AutonomyMode, MixingRules, ServicePhase, ServicePlan, cc_to_db, db_to_cc
from engine.core.vad_worker import VADWorker
from engine.main import DEFAULT_MIXING_RULES_PATH, DEFAULT_SERVICE_PLAN_PATH, AetherMixEngine
from engine.tests.mock_io import MockBridgeClient, MockLLMSupervisor

from evals import BLOCK_FRAMES, SAMPLE_RATE
from evals.timeline import Timeline

CHANNELS = ("ch1_pastor", "ch2_lead_vox", "ch3_bv1", "ch4_bv2", "ch5_keys", "ch7_guitars", "ch9_drums")
BAND_CHANNELS = ("ch5_keys", "ch7_guitars", "ch9_drums")
PHASES = [p.value for p in ServicePhase]
DT = BLOCK_FRAMES / SAMPLE_RATE
SIM_T0 = 1000.0


@dataclass
class TickLog:
    """Per-tick telemetry for one service run (arrays are length n_ticks)."""

    service: str
    profile: str
    t: np.ndarray
    phase: np.ndarray  # index into PHASES
    vol_db: dict[str, np.ndarray]
    speech_open: np.ndarray
    instrument_open: dict[str, np.ndarray]
    feedback_ceiling: dict[str, np.ndarray]  # NaN when none
    lufs_short: np.ndarray
    pastor_e_mix: np.ndarray
    band_e_mix: np.ndarray
    pastor_e_raw: np.ndarray
    band_e_raw: np.ndarray
    tick_ms: np.ndarray
    human_target_db: dict[str, float] = field(default_factory=dict)
    heartbeat: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    masters: tuple[np.ndarray, np.ndarray] | None = None  # (aethermix, static) stereo float32
    vad_backend: str = "unknown"  # "silero" | "dsp"
    vol_db_slew: dict[str, np.ndarray] | None = None  # live only: CC steps de-quantised for the slew gate

    def save_npz(self, path: Path) -> None:
        arrays: dict[str, Any] = {"t": self.t, "phase": self.phase, "speech_open": self.speech_open,
                                  "lufs_short": self.lufs_short, "tick_ms": self.tick_ms,
                                  "pastor_e_mix": self.pastor_e_mix, "band_e_mix": self.band_e_mix,
                                  "pastor_e_raw": self.pastor_e_raw, "band_e_raw": self.band_e_raw}
        for ch in CHANNELS:
            arrays[f"vol_{ch}"] = self.vol_db[ch]
            arrays[f"instr_{ch}"] = self.instrument_open[ch]
            arrays[f"fb_{ch}"] = self.feedback_ceiling[ch]
        np.savez_compressed(path, **arrays)


def _gain(db: float) -> float:
    return 0.0 if db <= -59.5 else 10.0 ** (db / 20.0)


async def run_timeline(
    tl: Timeline,
    plan_path: Path = DEFAULT_SERVICE_PLAN_PATH,
    rules_path: Path = DEFAULT_MIXING_RULES_PATH,
    use_service_plan: bool = False,
    keep_audio: bool = False,
    progress: bool = False,
    vad: str = "auto",
) -> TickLog:
    """vad: "silero" (default model if installed), "dsp" (deterministic fallback), or "auto" = DSP when the
    speech part is a synthetic stand-in (Silero correctly rejects the synthetic glottal tone), else Silero."""
    plan = ServicePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    rules = MixingRules.model_validate_json(rules_path.read_text(encoding="utf-8"))
    base = rules.phase_baselines_db[ServicePhase.SPOKEN_WORD]
    init_cc: dict[str, int] = {}
    for ch in plan.channels:
        init_cc[f"{ch.id}:vol"] = db_to_cc(base[ch.id], ch.cc_taper)
        if ch.reverb_cc is not None:
            init_cc[f"{ch.id}:reverb"] = db_to_cc(rules.phase_reverb_baselines_db[ServicePhase.SPOKEN_WORD][ch.id], ch.reverb_cc_taper)
    n = tl.n_ticks
    log = TickLog(
        service=tl.service.id, profile=tl.profile, t=np.arange(n) * DT, phase=np.zeros(n, dtype=np.int8),
        vol_db={c: np.zeros(n) for c in CHANNELS}, speech_open=np.zeros(n, dtype=bool),
        instrument_open={c: np.zeros(n, dtype=bool) for c in CHANNELS},
        feedback_ceiling={c: np.full(n, np.nan) for c in CHANNELS}, lufs_short=np.full(n, -70.0),
        pastor_e_mix=np.zeros(n), band_e_mix=np.zeros(n), pastor_e_raw=np.zeros(n), band_e_raw=np.zeros(n),
        tick_ms=np.zeros(n),
    )
    if vad == "auto":
        vad = "dsp" if any(p.endswith(":speech") for p in tl.synthetic_parts) else "silero"
    vad_worker = VADWorker(plan, model_path=Path("/nonexistent/silero_vad.onnx")) if vad == "dsp" else None
    static_db = {c: float(np.mean([rules.phase_baselines_db[p][c] for p in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH)]))
                 for c in CHANNELS}
    aether_blocks: list[np.ndarray] = []
    static_blocks: list[np.ndarray] = []

    with tempfile.TemporaryDirectory(prefix="aethermix_eval_") as td:
        td_path = Path(td)
        bridge = MockBridgeClient(plan, initial_cc=init_cc)
        engine = AetherMixEngine(
            plan=plan, rules=rules, rules_path=td_path / "mixing_rules.json", backup_dir=td_path / "backups",
            db_path=td_path / "telemetry.db", token="eval-token", headless=True, enable_clap=False,
            use_service_plan=use_service_plan, bridge_client=bridge, llm_supervisor=MockLLMSupervisor(plan),
            vad_worker=vad_worker,
        )
        await engine.start(start_loop=False)
        log.vad_backend = "silero" if engine.vad_worker.using_onnx else "dsp"
        try:
            for ch in plan.channels:
                engine.safety_firewall.set_channel_armed(ch.id, True)
                engine.safety_firewall.confirm_sync(ch.id)
            engine.safety_firewall.set_mode(AutonomyMode.ACTIVE)
            last_master = np.zeros((BLOCK_FRAMES, 2), dtype=np.float32)
            pending_moves = [e for e in tl.events if e.kind == "human_move"]
            seq = 0
            cur_db = {c: base[c] for c in CHANNELS}
            for k in range(n):
                t = k * DT
                now_s = SIM_T0 + t
                if use_service_plan:
                    step = tl.segment_at(t).step_idx
                    if step is not None:
                        engine.service_step_idx = step
                for ev in list(pending_moves):
                    if t >= ev.t0:
                        pending_moves.remove(ev)
                        ch_id = ev.channels[0]
                        taper = plan.channel_by_id(ch_id).cc_taper
                        cc = db_to_cc(cur_db[ch_id] + float(ev.params["delta_db"]), taper)
                        seq += 1
                        engine._sim_clock_s = now_s
                        await engine._on_bridge_human_cc(
                            {"channel_id": ch_id, "control": "vol", "cc": cc, "human_seq": seq, "source": "hardware"})
                        log.human_target_db[ch_id] = cc_to_db(cc, taper)
                blk = tl.block(k)
                blk[:, 14:16] = last_master
                t0 = time.perf_counter()
                out = await engine.step_once(dt_s=DT, now_s=now_s, audio_block=blk, broadcast=False)
                log.tick_ms[k] = (time.perf_counter() - t0) * 1000.0
                st = out.channels_state
                for c in CHANNELS:
                    cur_db[c] = st[c].vol_current_db
                    log.vol_db[c][k] = cur_db[c]
                    log.instrument_open[c][k] = st[c].instrument_open
                    if st[c].feedback_ceiling_db is not None:
                        log.feedback_ceiling[c][k] = st[c].feedback_ceiling_db
                log.speech_open[k] = st["ch1_pastor"].speech_open
                log.phase[k] = PHASES.index(engine.scene_classifier.active_phase.value)
                m = getattr(engine, "_last_metrics", None)
                if m is not None:
                    log.lufs_short[k] = m.master_lufs_short_term
                for ev_out in out.events:
                    log.events.append({"t": round(t, 3), "category": ev_out.category.value, "summary": ev_out.summary})
                g = {c: _gain(cur_db[c]) for c in CHANNELS}
                left = (blk[:, 0] * g["ch1_pastor"] + blk[:, 1] * g["ch2_lead_vox"] + 0.7 * blk[:, 2] * g["ch3_bv1"]
                        + 0.7 * blk[:, 3] * g["ch4_bv2"] + blk[:, 4] * g["ch5_keys"] + blk[:, 6] * g["ch7_guitars"]
                        + (blk[:, 8] + blk[:, 10]) * g["ch9_drums"]) * 0.35
                right = (blk[:, 0] * g["ch1_pastor"] + blk[:, 1] * g["ch2_lead_vox"] + 0.7 * blk[:, 2] * g["ch3_bv1"]
                         + 0.7 * blk[:, 3] * g["ch4_bv2"] + blk[:, 5] * g["ch5_keys"] + blk[:, 7] * g["ch7_guitars"]
                         + (blk[:, 9] + blk[:, 11]) * g["ch9_drums"]) * 0.35
                last_master = np.clip(np.stack([left, right], axis=1), -0.99, 0.99).astype(np.float32)
                col_e = np.mean(blk.astype(np.float64) ** 2, axis=0)
                grp_gain = [g["ch5_keys"]] * 2 + [g["ch7_guitars"]] * 2 + [g["ch9_drums"]] * 4
                log.pastor_e_raw[k] = float(col_e[0])
                log.band_e_raw[k] = float(col_e[4:12].sum())
                log.pastor_e_mix[k] = float(col_e[0]) * g["ch1_pastor"] ** 2
                log.band_e_mix[k] = float(sum(col_e[4 + i] * grp_gain[i] ** 2 for i in range(8)))
                if keep_audio:
                    aether_blocks.append(last_master)
                    sg = {c: _gain(static_db[c]) for c in CHANNELS}
                    sl = (blk[:, 0] * sg["ch1_pastor"] + blk[:, 1] * sg["ch2_lead_vox"] + 0.7 * (blk[:, 2] * sg["ch3_bv1"] + blk[:, 3] * sg["ch4_bv2"])
                          + blk[:, 4] * sg["ch5_keys"] + blk[:, 6] * sg["ch7_guitars"] + (blk[:, 8] + blk[:, 10]) * sg["ch9_drums"]) * 0.35
                    sr_ = (blk[:, 0] * sg["ch1_pastor"] + blk[:, 1] * sg["ch2_lead_vox"] + 0.7 * (blk[:, 2] * sg["ch3_bv1"] + blk[:, 3] * sg["ch4_bv2"])
                           + blk[:, 5] * sg["ch5_keys"] + blk[:, 7] * sg["ch7_guitars"] + (blk[:, 9] + blk[:, 11]) * sg["ch9_drums"]) * 0.35
                    static_blocks.append(np.clip(np.stack([sl, sr_], axis=1), -0.99, 0.99).astype(np.float32))
                if progress and k % 3000 == 0:
                    print(f"  [{tl.service.id}] {t:6.1f}s / {tl.total_s:.0f}s", flush=True)
            hb = engine.build_heartbeat_payload()
            log.heartbeat = {"loop_p95_ms": float(hb.loop_p95_ms), "loop_p99_ms": float(hb.loop_p99_ms)}
        finally:
            await engine.stop()
    if keep_audio:
        log.masters = (np.concatenate(aether_blocks), np.concatenate(static_blocks))
    return log
