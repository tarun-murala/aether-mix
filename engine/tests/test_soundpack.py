"""End-to-end validation of the packaged 5-tune Multi-Instrument Sound Pack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from engine.core.audio_tap import AudioTap
from engine.core.offset_learner import OffsetLearner
from engine.core.protocol import (
    AutonomyMode,
    BrainCategory,
    BridgeStatePayload,
    MixingRules,
    ServicePhase,
    ServicePlan,
    db_to_cc,
)
from engine.core.rules_manager import RulesManager
from engine.core.safety_firewall import SafetyFirewall
from engine.core.scene_classifier import SceneClassifier
from engine.core.vad_worker import VADWorker
from scripts.play_mainstage_pack import (
    BLOCK_FRAMES,
    SOUNDPACK_DIR,
    compose_16ch_block,
    load_packaged_soundpack,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_soundpack_integrity_and_levels() -> None:
    pack = load_packaged_soundpack(SOUNDPACK_DIR)
    assert len(pack.tunes) == 5

    # Verify speech & applause are non-silent, non-clipping, finite 48 kHz arrays
    for name, arr in [("pastor_speech", pack.pastor_speech), ("room_applause", pack.room_applause)]:
        assert np.all(np.isfinite(arr)), f"{name} contains non-finite samples"
        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(arr**2))) + 1e-12)
        peak = float(np.max(np.abs(arr)))
        assert -34.0 <= rms_db <= -10.0, f"{name} unexpected RMS {rms_db:.2f} dBFS"
        assert peak <= 0.99, f"{name} clips at {peak:.3f}"

    # Verify all 5 tunes have distinct BPM/key and valid stems
    bpms = {t.bpm for t in pack.tunes}
    keys = {t.key for t in pack.tunes}
    assert len(bpms) == 5, "Expected 5 distinct tempos across packaged tunes"
    assert len(keys) == 5, "Expected 5 distinct musical keys across packaged tunes"

    for tune in pack.tunes:
        assert tune.duration_s >= 18.0
        for sname, arr in [
            ("lead_vox", tune.lead_vox),
            ("bv1", tune.bv1),
            ("bv2", tune.bv2),
            ("keys", tune.keys),
            ("guitars", tune.guitars),
            ("drums", tune.drums),
            ("bass", tune.bass),
        ]:
            assert np.all(np.isfinite(arr)), f"{tune.id}/{sname} non-finite"
            rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(arr**2))) + 1e-12)
            assert -38.0 <= rms_db <= -8.0, f"{tune.id}/{sname} unexpected RMS {rms_db:.2f} dBFS"
            assert float(np.max(np.abs(arr))) <= 0.99


def test_soundpack_realtime_ai_mixing_and_zero_false_feedback() -> None:
    """Run all 5 tunes + pastor interrupt + sermon + feedback test through AetherMix."""
    pack = load_packaged_soundpack(SOUNDPACK_DIR)
    plan = ServicePlan.model_validate_json(
        (REPO_ROOT / "engine" / "schemas" / "service_plan.json").read_text(encoding="utf-8")
    )
    rules = MixingRules.model_validate_json(
        (REPO_ROOT / "engine" / "schemas" / "mixing_rules.json").read_text(encoding="utf-8")
    )

    audio_tap = AudioTap(plan, headless=True)
    vad = VADWorker(plan, model_path=Path.home() / ".aethermix" / "models" / "silero_vad.onnx")
    classifier = SceneClassifier(plan)
    learner = OffsetLearner(plan)
    rules_mgr = RulesManager(plan, rules, backup_dir=REPO_ROOT / "engine" / "storage" / "rule_backups")
    fw = SafetyFirewall(plan, rules, rules_mgr, learner)

    # Seed initial bridge state and arm all channels in ACTIVE
    init_cc: dict[str, int] = {}
    for ch in plan.channels:
        init_cc[f"{ch.id}:vol"] = db_to_cc(rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE][ch.id], ch.cc_taper)
        if ch.reverb_cc is not None:
            init_cc[f"{ch.id}:reverb"] = db_to_cc(
                rules.phase_reverb_baselines_db[ServicePhase.PRAYER_UNDERSCORE][ch.id],
                ch.reverb_cc_taper,
            )
    fw.hydrate_from_bridge(BridgeStatePayload(last_sent_cc=init_cc, bypass_latched=False, restored_from_disk_at=None))
    for ch in plan.channels:
        fw.set_channel_armed(ch.id, True)
    fw.set_mode(AutonomyMode.ACTIVE)

    rng = np.random.default_rng(77)
    now_s = 1.0
    last_phase_eval_s = -10.0
    active_phase = ServicePhase.PRAYER_UNDERSCORE
    last_macro = ServicePhase.PRAYER_UNDERSCORE
    cursor = 0
    speech_cursor = 0
    false_fb_cuts: list[str] = []

    def step_ticks(tune_idx: int, seconds: float, *, step_idx: int = 0, **flags: object) -> list[ServicePhase]:
        nonlocal now_s, last_phase_eval_s, active_phase, last_macro, cursor, speech_cursor
        phases: list[ServicePhase] = []
        n_ticks = round(seconds * 50.0)
        for _ in range(n_ticks):
            block = compose_16ch_block(
                pack,
                tune_idx,
                cursor,
                speech_cursor=speech_cursor,
                frames=BLOCK_FRAMES,
                rng=rng,
                **flags,  # type: ignore[arg-type]
            )
            audio_tap.push_block(block)
            metrics = audio_tap.poll_metrics()
            vad_probs, sung_vocals = vad.process_frame(metrics)
            classifier.observe_tick(metrics, vad_probs, sung_vocals)
            if now_s - last_phase_eval_s >= 2.0 - 1e-9:
                decision = classifier.update(
                    metrics=metrics,
                    vad_probs=vad_probs,
                    sung_vocals=sung_vocals,
                    clap_probs=None,
                    service_step_idx=step_idx,
                    use_service_plan=False,
                    now_s=now_s,
                )
                last_phase_eval_s = now_s
                active_phase = decision.active_phase
                last_macro = decision.last_macro_phase
            out = fw.tick(
                metrics=metrics,
                vad_probs=vad_probs,
                sung_vocals=sung_vocals,
                active_phase=active_phase,
                last_macro_phase=last_macro,
                dt_s=0.02,
                now_s=now_s,
                tick_elapsed_ms=1.0,
            )
            for ev in out.events:
                if (
                    ev.category == BrainCategory.SAFETY
                    and "feedback" in ev.summary.lower()
                    and not flags.get("feedback_hz")
                ):
                    false_fb_cuts.append(f"tune={tune_idx+1} {ev.summary}")
            phases.append(active_phase)
            now_s += 0.02
            cursor += BLOCK_FRAMES
            if flags.get("pastor_on"):
                speech_cursor += BLOCK_FRAMES
        return phases

    # 1. Play every single one of the 5 packaged tunes full-band (6s each = 30s total)
    for tune_idx in range(5):
        phases = step_ticks(
            tune_idx, 6.0, lead_on=True, bv_on=True, keys_on=True, guitars_on=True, drums_on=True, bass_on=True
        )
        assert phases[-1] in (ServicePhase.WORSHIP_HIGH, ServicePhase.WORSHIP_LOW), (
            f"Tune {tune_idx+1} ended in unexpected phase {phases[-1]}"
        )

    assert not false_fb_cuts, f"Unexpected false anti-feedback cuts across 5 tunes: {false_fb_cuts[:5]}"

    # 2. Pastor interrupts Tune 2 (Anthem Rise) -> verify speech_open fires within 1.0 s!
    speech_open_ms: float | None = None
    start_s = now_s
    for _ in range(75):  # 1.5 s
        block = compose_16ch_block(
            pack,
            1,
            cursor,
            speech_cursor=speech_cursor,
            frames=BLOCK_FRAMES,
            pastor_on=True,
            lead_on=False,
            bv_on=False,
            keys_on=True,
            guitars_on=True,
            drums_on=True,
            bass_on=True,
            rng=rng,
        )
        block[:, 0] *= 10.0 ** (3.0 / 20.0)  # close-mic pulpit projection
        audio_tap.push_block(block)
        metrics = audio_tap.poll_metrics()
        vad_probs, sung_vocals = vad.process_frame(metrics)
        classifier.observe_tick(metrics, vad_probs, sung_vocals)
        out = fw.tick(
            metrics=metrics,
            vad_probs=vad_probs,
            sung_vocals=sung_vocals,
            active_phase=active_phase,
            last_macro_phase=last_macro,
            dt_s=0.02,
            now_s=now_s,
            tick_elapsed_ms=1.0,
        )
        if out.channels_state["ch1_pastor"].speech_open and speech_open_ms is None:
            speech_open_ms = (now_s - start_s) * 1000.0
        now_s += 0.02
        cursor += BLOCK_FRAMES
        speech_cursor += BLOCK_FRAMES
    assert speech_open_ms is not None and speech_open_ms <= 1000.0, f"speech_open too slow: {speech_open_ms} ms"

    # 3. Sermon solo -> classifies as SPOKEN_WORD
    sermon_phases = step_ticks(
        0, 6.0, pastor_on=True, lead_on=False, bv_on=False, keys_on=False, guitars_on=False, drums_on=False, bass_on=False
    )
    assert sermon_phases[-1] == ServicePhase.SPOKEN_WORD

    # 4. Injected growing 2.48 kHz feedback howl on Lead Vocal during Tune 1 -> caught by anti-feedback!
    fb_caught = False
    for idx in range(75):  # 1.5 s
        fb_amp = 0.05 * (10.0 ** ((idx / 75.0) * 1.0))
        block = compose_16ch_block(
            pack, 0, cursor, frames=BLOCK_FRAMES, lead_on=True, keys_on=True, feedback_hz=2480.0, feedback_gain=fb_amp, rng=rng
        )
        audio_tap.push_block(block)
        metrics = audio_tap.poll_metrics()
        vad_probs, sung_vocals = vad.process_frame(metrics)
        out = fw.tick(
            metrics=metrics,
            vad_probs=vad_probs,
            sung_vocals=sung_vocals,
            active_phase=active_phase,
            last_macro_phase=last_macro,
            dt_s=0.02,
            now_s=now_s,
            tick_elapsed_ms=1.0,
        )
        if any(e.category == BrainCategory.SAFETY and "feedback" in e.summary.lower() for e in out.events):
            fb_caught = True
            break
        now_s += 0.02
        cursor += BLOCK_FRAMES
    assert fb_caught, "Expected anti-feedback to catch injected 2.48 kHz feedback howl"
