#!/usr/bin/env python3
"""Synthetic Multi-Stem 16ch Church Service Simulator (SPEC §9 Phase 4).

Generates realistic 16-channel 48 kHz multi-stem church service audio:
  SPOKEN_WORD -> WORSHIP_LOW -> WORSHIP_HIGH (+ stage bleed + pastor speech-open interrupt)
  -> PRAYER_UNDERSCORE (+ keys ducking) -> ANTI-FEEDBACK HOWL -> TRANSITION (applause)

Modes:
  --headless  : Runs the complete AetherMixEngine pipeline in-process and prints a
                quantitative PASS/FAIL verification report against SPEC v5.1 gates.
  --blackhole : Streams the 16-channel 48 kHz multi-stem audio directly into BlackHole 16ch
                for live MainStage & AetherMix rehearsal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core.protocol import (
    AutonomyMode,
    MixingRules,
    ServicePhase,
    ServicePlan,
)
from engine.core.vad_worker import VADWorker
from engine.main import (
    DEFAULT_MIXING_RULES_PATH,
    DEFAULT_SERVICE_PLAN_PATH,
    AetherMixEngine,
)
from engine.tests.mock_io import (
    MockBridgeClient,
    MockLLMSupervisor,
    SceneAudioConfig,
    SyntheticServiceAudioGenerator,
)

# A path that never exists: forces VADWorker onto its deterministic DSP backend
# even when ~/.aethermix/models/silero_vad.onnx is installed (mirrors evals/runner.py).
_DSP_ONLY_MODEL_PATH = Path("/nonexistent/silero_vad.onnx")
VAD_BACKENDS = ("dsp", "silero")


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


async def run_headless_simulation(
    plan_path: Path = DEFAULT_SERVICE_PLAN_PATH,
    rules_path: Path = DEFAULT_MIXING_RULES_PATH,
    vad: str = "dsp",
) -> tuple[bool, list[GateCheck], dict[str, Any]]:
    """Run full synthetic service through AetherMixEngine in-process and evaluate SPEC v5.1 gates.

    ``vad="dsp"`` (default) pins the deterministic DSP speech VAD: the synthetic pastor
    voice is a formant/AM model that Silero (trained on real speech) correctly rejects, so
    the speech gates are only meaningful on DSP. ``vad="silero"`` uses the installed
    ONNX model if present (falls back to DSP if not); ``metrics["vad_backend"]`` says which ran.
    """
    if vad not in VAD_BACKENDS:
        raise ValueError(f"vad must be one of {VAD_BACKENDS}, got {vad!r}")
    plan = ServicePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    rules = MixingRules.model_validate_json(rules_path.read_text(encoding="utf-8"))
    gen = SyntheticServiceAudioGenerator(sample_rate=plan.sample_rate, seed=2026)

    with tempfile.TemporaryDirectory(prefix="aethermix_sim_", dir="/tmp") as td:
        td_path = Path(td)
        bridge = MockBridgeClient(plan)
        llm = MockLLMSupervisor(plan)
        vad_worker = VADWorker(plan, model_path=_DSP_ONLY_MODEL_PATH if vad == "dsp" else None)
        engine = AetherMixEngine(
            plan=plan,
            rules=rules,
            rules_path=td_path / "mixing_rules.json",
            backup_dir=td_path / "backups",
            db_path=td_path / "telemetry.db",
            token="sim-token",
            headless=True,
            enable_clap=False,
            use_service_plan=False,
            bridge_client=bridge,
            llm_supervisor=llm,
            vad_worker=vad_worker,
        )
        await engine.start(start_loop=False)
        vad_backend = "silero" if engine.vad_worker.using_onnx else "dsp"
        try:
            # Arm all channels & switch to ACTIVE mode for full simulation
            for ch in plan.channels:
                engine.safety_firewall.set_channel_armed(ch.id, True)
                engine.safety_firewall.confirm_sync(ch.id)
            engine.safety_firewall.set_mode(AutonomyMode.ACTIVE)

            dt = 0.02
            sim_t = 0.0
            checks: list[GateCheck] = []

            async def _run_segment(cfg: SceneAudioConfig, seconds: float) -> Any:
                nonlocal sim_t
                steps = max(1, round(seconds / dt))
                last_out = None
                for _ in range(steps):
                    sim_t += dt
                    blk = gen.generate_block(cfg, frames=960)
                    last_out = await engine.step_once(dt_s=dt, now_s=sim_t, audio_block=blk, broadcast=False)
                return last_out

            # 1. SPOKEN_WORD (pastor talking solo)
            out_spoken = await _run_segment(SceneAudioConfig.for_phase(ServicePhase.SPOKEN_WORD), 5.0)
            checks.append(
                GateCheck(
                    name="phase_spoken_word",
                    passed=engine.scene_classifier.active_phase == ServicePhase.SPOKEN_WORD
                    and out_spoken.channels_state["ch1_pastor"].vol_current_db > -2.0,
                    detail=(
                        f"phase={engine.scene_classifier.active_phase.value}, "
                        f"ch1_pastor={out_spoken.channels_state['ch1_pastor'].vol_current_db:.2f} dB"
                    ),
                )
            )

            # 2. WORSHIP_LOW (lead vocal + keys + guitars; 4s hysteresis + 6s 4dB/s glide)
            out_wlow = await _run_segment(SceneAudioConfig.for_phase(ServicePhase.WORSHIP_LOW), 10.0)
            checks.append(
                GateCheck(
                    name="phase_worship_low",
                    passed=engine.scene_classifier.active_phase == ServicePhase.WORSHIP_LOW
                    and out_wlow.channels_state["ch2_lead_vox"].vol_current_db > -3.0
                    and out_wlow.channels_state["ch5_keys"].vol_current_db > -8.0,
                    detail=(
                        f"phase={engine.scene_classifier.active_phase.value}, "
                        f"lead_vox={out_wlow.channels_state['ch2_lead_vox'].vol_current_db:.2f} dB, "
                        f"keys={out_wlow.channels_state['ch5_keys'].vol_current_db:.2f} dB"
                    ),
                )
            )

            # 3. WORSHIP_HIGH (full band + drums + BVs + stage bleed into pastor mic)
            out_whigh = await _run_segment(SceneAudioConfig.for_phase(ServicePhase.WORSHIP_HIGH), 6.0)
            pastor_bleed_rejected = not out_whigh.channels_state["ch1_pastor"].speech_open
            checks.append(
                GateCheck(
                    name="phase_worship_high_and_bleed_rejection",
                    passed=engine.scene_classifier.active_phase == ServicePhase.WORSHIP_HIGH
                    and pastor_bleed_rejected
                    and out_whigh.channels_state["ch9_drums"].vol_current_db > -5.0,
                    detail=(
                        f"phase={engine.scene_classifier.active_phase.value}, "
                        f"drums={out_whigh.channels_state['ch9_drums'].vol_current_db:.2f} dB, "
                        f"pastor_speech_open_on_bleed={out_whigh.channels_state['ch1_pastor'].speech_open}"
                    ),
                )
            )

            # 4. Pastor Interrupt during WORSHIP_HIGH -> Speech-Open < 1.0s + Band Auto-Ducking
            pre_duck_drums = out_whigh.channels_state["ch9_drums"].vol_current_db
            interrupt_cfg = SceneAudioConfig.for_phase(ServicePhase.WORSHIP_HIGH)
            interrupt_cfg.pastor_speaking = True
            speech_open_latency_s: float | None = None
            t_interrupt_start = sim_t
            out_int = out_whigh
            for _ in range(round(1.5 / dt)):
                sim_t += dt
                blk = gen.generate_block(interrupt_cfg, frames=960)
                out_int = await engine.step_once(dt_s=dt, now_s=sim_t, audio_block=blk, broadcast=False)
                if (
                    speech_open_latency_s is None
                    and out_int.channels_state["ch1_pastor"].speech_open
                    and out_int.channels_state["ch1_pastor"].vol_current_db >= -1.0
                ):
                    speech_open_latency_s = sim_t - t_interrupt_start
            post_duck_drums = out_int.channels_state["ch9_drums"].vol_current_db
            checks.append(
                GateCheck(
                    name="speech_open_under_1s_and_worship_ducking",
                    passed=(speech_open_latency_s is not None and speech_open_latency_s < 1.0)
                    and (post_duck_drums <= pre_duck_drums - 4.0),
                    detail=(
                        f"speech_open_latency={speech_open_latency_s:.3f}s (<1.0s), "
                        f"drums_duck={pre_duck_drums:.1f} -> {post_duck_drums:.1f} dB"
                        if speech_open_latency_s is not None
                        else f"speech_open FAILED; pastor={out_int.channels_state['ch1_pastor'].vol_current_db:.1f} dB"
                    ),
                )
            )

            # 5. PRAYER_UNDERSCORE (pastor talking + warm keys pad -> keys ducked, NO instrument_open)
            out_prayer = await _run_segment(SceneAudioConfig.for_phase(ServicePhase.PRAYER_UNDERSCORE), 6.0)
            checks.append(
                GateCheck(
                    name="phase_prayer_underscore_and_keys_ducking",
                    passed=engine.scene_classifier.active_phase == ServicePhase.PRAYER_UNDERSCORE
                    and not out_prayer.channels_state["ch5_keys"].instrument_open
                    and out_prayer.channels_state["ch5_keys"].vol_current_db <= -14.0,
                    detail=(
                        f"phase={engine.scene_classifier.active_phase.value}, "
                        f"keys={out_prayer.channels_state['ch5_keys'].vol_current_db:.2f} dB, "
                        f"instrument_open={out_prayer.channels_state['ch5_keys'].instrument_open}"
                    ),
                )
            )

            # 6. Anti-Feedback Howl on ch1_pastor (growing narrowband 2500Hz sine while keys pad plays)
            pre_fb_pastor = out_prayer.channels_state["ch1_pastor"].vol_current_db
            fb_triggered = False
            out_fb = out_prayer
            for i in range(round(1.0 / dt)):
                sim_t += dt
                fb_cfg = SceneAudioConfig.for_phase(ServicePhase.PRAYER_UNDERSCORE)
                fb_cfg.pastor_speaking = False
                fb_cfg.feedback_channel_idx = 0
                fb_cfg.feedback_freq_hz = 2500.0
                # Grow amplitude from -16 dBFS to -3 dBFS over 0.6s
                fb_cfg.feedback_amp = min(0.7, 0.16 * (1.0 + 0.12 * i))
                blk = gen.generate_block(fb_cfg, frames=960)
                out_fb = await engine.step_once(dt_s=dt, now_s=sim_t, audio_block=blk, broadcast=False)
                if out_fb.channels_state["ch1_pastor"].feedback_ceiling_db is not None:
                    fb_triggered = True
            checks.append(
                GateCheck(
                    name="anti_feedback_cut_with_pad_bleed_rejection",
                    passed=fb_triggered
                    and out_fb.channels_state["ch1_pastor"].vol_current_db <= pre_fb_pastor - 3.5,
                    detail=(
                        f"fb_triggered={fb_triggered}, pastor={pre_fb_pastor:.1f} -> "
                        f"{out_fb.channels_state['ch1_pastor'].vol_current_db:.1f} dB, "
                        f"ceiling={out_fb.channels_state['ch1_pastor'].feedback_ceiling_db}"
                    ),
                )
            )

            # 7. TRANSITION (applause -> retains last_macro_phase == PRAYER_UNDERSCORE)
            out_trans = await _run_segment(SceneAudioConfig.for_phase(ServicePhase.TRANSITION), 5.0)
            checks.append(
                GateCheck(
                    name="phase_transition_preserves_macro_phase",
                    passed=engine.scene_classifier.active_phase == ServicePhase.TRANSITION
                    and engine.scene_classifier.last_macro_phase == ServicePhase.PRAYER_UNDERSCORE,
                    detail=(
                        f"phase={engine.scene_classifier.active_phase.value}, "
                        f"last_macro={engine.scene_classifier.last_macro_phase.value}"
                    ),
                )
            )

            # 8. Tier-1 50Hz loop latency gate (p99 <= 25 ms)
            hb = engine.build_heartbeat_payload()
            checks.append(
                GateCheck(
                    name="tier1_50hz_loop_latency_gate",
                    passed=hb.loop_p99_ms <= 25.0 and len(bridge.sent_agent_writes) > 0,
                    detail=(
                        f"p95={hb.loop_p95_ms:.2f}ms, p99={hb.loop_p99_ms:.2f}ms (<=25ms), "
                        f"agent_writes={len(bridge.sent_agent_writes)}, heartbeats={bridge.heartbeat_count}"
                    ),
                )
            )

            all_passed = all(c.passed for c in checks)
            metrics = {
                "vad_backend": vad_backend,
                "sim_duration_s": round(sim_t, 2),
                "loop_p95_ms": hb.loop_p95_ms,
                "loop_p99_ms": hb.loop_p99_ms,
                "agent_writes": len(bridge.sent_agent_writes),
                "bridge_heartbeats": bridge.heartbeat_count,
                "last_out_mode": out_trans.autonomy_mode.value,
            }
            return all_passed, checks, metrics
        finally:
            await engine.stop()


def run_blackhole_playback(
    plan_path: Path = DEFAULT_SERVICE_PLAN_PATH,
    segment_seconds: float = 12.0,
) -> int:
    """Stream realistic 16-channel 48 kHz synthetic service stems directly into BlackHole 16ch."""
    import sounddevice as sd  # type: ignore

    plan = ServicePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    devs = sd.query_devices()
    dev_idx = next(
        (
            i
            for i, d in enumerate(devs)
            if plan.audio_device_hint.lower() in d["name"].lower() and d["max_output_channels"] >= 16
        ),
        None,
    )
    if dev_idx is None:
        print(f"ERROR: 16ch output device matching '{plan.audio_device_hint}' not found.", file=sys.stderr)
        return 1

    gen = SyntheticServiceAudioGenerator(sample_rate=plan.sample_rate, seed=2026)
    schedule = [
        ("SPOKEN_WORD (Pastor Opening)", SceneAudioConfig.for_phase(ServicePhase.SPOKEN_WORD)),
        ("WORSHIP_LOW (Acoustic Verse)", SceneAudioConfig.for_phase(ServicePhase.WORSHIP_LOW)),
        ("WORSHIP_HIGH (Full Band Chorus)", SceneAudioConfig.for_phase(ServicePhase.WORSHIP_HIGH)),
        ("PRAYER_UNDERSCORE (Pastor + Pad)", SceneAudioConfig.for_phase(ServicePhase.PRAYER_UNDERSCORE)),
        ("TRANSITION (Applause)", SceneAudioConfig.for_phase(ServicePhase.TRANSITION)),
    ]

    print(f"Streaming 16ch 48kHz service simulation to '{devs[dev_idx]['name']}'...")
    with sd.OutputStream(
        device=dev_idx,
        samplerate=plan.sample_rate,
        channels=16,
        dtype="float32",
        blocksize=960,
    ) as stream:
        for title, cfg in schedule:
            print(f"  -> {title} ({segment_seconds:.1f}s)")
            end_t = time.monotonic() + segment_seconds
            while time.monotonic() < end_t:
                blk = gen.generate_block(cfg, frames=960)
                stream.write(blk)
    print("BlackHole service playback complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherMix 16ch Church Service Simulator")
    parser.add_argument("--service-plan", type=Path, default=DEFAULT_SERVICE_PLAN_PATH)
    parser.add_argument("--mixing-rules", type=Path, default=DEFAULT_MIXING_RULES_PATH)
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument("--headless", action="store_true", help="Run full engine pipeline in-process")
    mode_grp.add_argument("--blackhole", action="store_true", help="Stream 16ch audio out to BlackHole 16ch")
    parser.add_argument("--segment-seconds", type=float, default=12.0, help="Seconds per phase in --blackhole mode")
    parser.add_argument(
        "--vad",
        choices=VAD_BACKENDS,
        default="dsp",
        help="Speech VAD backend for --headless (default dsp; silero rejects the synthetic voice)",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if args.blackhole:
        return run_blackhole_playback(args.service_plan, segment_seconds=args.segment_seconds)

    # Default to headless simulation
    all_passed, checks, metrics = asyncio.run(
        run_headless_simulation(args.service_plan, args.mixing_rules, vad=args.vad)
    )
    if args.json_output:
        print(
            json.dumps(
                {
                    "passed": all_passed,
                    "metrics": metrics,
                    "checks": [asdict(c) for c in checks],
                },
                indent=2,
            )
        )
    else:
        print("=" * 72)
        print("AetherMix v5.1 Headless Service Simulation Report")
        print(f"VAD backend: {metrics['vad_backend']} (requested --vad {args.vad})")
        print("=" * 72)
        for c in checks:
            print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        print("-" * 72)
        print(
            f"OVERALL: {'ALL GATES PASSED' if all_passed else 'FAILED'} "
            f"(sim={metrics['sim_duration_s']}s, p95={metrics['loop_p95_ms']}ms, "
            f"p99={metrics['loop_p99_ms']}ms, writes={metrics['agent_writes']})"
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
