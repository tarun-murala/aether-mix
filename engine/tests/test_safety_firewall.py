"""Comprehensive tests for engine/core/safety_firewall.py covering every SPEC v5.1 §9 Phase 2 gate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from engine.core.offset_learner import OffsetLearner
from engine.core.protocol import (
    AutonomyMode,
    BrainCategory,
    BridgeStatePayload,
    MixingRules,
    ServicePhase,
    ServicePlan,
    SyncStatus,
    cc_to_db,
    db_to_cc,
)
from engine.core.rules_manager import RulesManager
from engine.core.safety_firewall import SafetyFirewall

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"


@dataclass
class FakeAudioMetrics:
    """Duck-typed stand-in for AudioFrameMetrics so tests run without audio_tap."""

    ts_ms: int = 0
    rms_db: dict[str, float] = field(default_factory=dict)
    norm_vband_db: dict[str, float] = field(default_factory=dict)
    fft_mag_db: dict[str, np.ndarray] = field(default_factory=dict)
    speech_to_band_snr_db: dict[str, float] = field(default_factory=dict)
    master_lufs_momentary: float = -21.0
    master_lufs_short_term: float = -21.0
    tcc_silence_warning: bool = False


def _make_stack() -> tuple[ServicePlan, MixingRules, RulesManager, OffsetLearner, SafetyFirewall]:
    plan = ServicePlan.model_validate_json((SCHEMAS_DIR / "service_plan.json").read_text(encoding="utf-8"))
    rules = MixingRules.model_validate_json((SCHEMAS_DIR / "mixing_rules.json").read_text(encoding="utf-8"))
    mgr = RulesManager(plan, rules)
    learner = OffsetLearner(plan)
    fw = SafetyFirewall(plan, rules, mgr, learner)
    return plan, rules, mgr, learner, fw


def _arm_all_and_activate(plan: ServicePlan, rules: MixingRules, fw: SafetyFirewall, phase: ServicePhase) -> None:
    """Hydrate all channels at the given phase baseline, arm all, and switch to ACTIVE."""
    last_cc: dict[str, int] = {}
    for ch in plan.channels:
        base = rules.phase_baselines_db[phase][ch.id]
        last_cc[f"{ch.id}:vol"] = db_to_cc(base, ch.cc_taper)
        if ch.reverb_cc is not None:
            rbase = rules.phase_reverb_baselines_db[phase][ch.id]
            last_cc[f"{ch.id}:reverb"] = db_to_cc(rbase, ch.reverb_cc_taper)
    fw.hydrate_from_bridge(BridgeStatePayload(last_sent_cc=last_cc, restored_from_disk_at=None))
    fw.set_armed_channels(plan.channel_ids)
    fw.active_phase = phase
    fw.last_macro_phase = phase
    fw.set_mode(AutonomyMode.ACTIVE)


def _silent_metrics(plan: ServicePlan) -> FakeAudioMetrics:
    return FakeAudioMetrics(
        norm_vband_db={ch.id: -45.0 for ch in plan.channels},
        speech_to_band_snr_db={ch.id: -10.0 for ch in plan.channels},
    )


# ---------------------------------------------------------------------------
# 1. Boot, Bridge Hydration & Disk-Restore Safety Drop
# ---------------------------------------------------------------------------


def test_boot_shadow_observe_disarmed_and_disk_restore_safety_drop() -> None:
    plan, _, _, _, fw = _make_stack()
    assert fw.autonomy_mode == AutonomyMode.SHADOW_OBSERVE
    assert fw.get_armed_channel_ids() == []

    # In SHADOW_OBSERVE, tick emits zero agent writes
    out = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH)
    assert out.agent_writes == []
    assert out.autonomy_mode == AutonomyMode.SHADOW_OBSERVE

    # Live bridge hydration -> SYNCED
    fw.hydrate_from_bridge(
        BridgeStatePayload(last_sent_cc={"ch1_pastor:vol": 96, "ch5_keys:vol": 64}, restored_from_disk_at=None)
    )
    snap = fw.snapshot_channels_state()
    assert snap["ch1_pastor"].sync_status == SyncStatus.SYNCED
    assert snap["ch1_pastor"].vol_current_db == pytest.approx(1.1, abs=0.05)  # CC 96 on MainStage-measured taper

    # Activate, then simulate bridge crash + restart from disk (restored_from_disk_at != None)
    fw.set_mode(AutonomyMode.ACTIVE)
    events = fw.hydrate_from_bridge(
        BridgeStatePayload(
            last_sent_cc={"ch1_pastor:vol": 96, "ch5_keys:vol": 64},
            restored_from_disk_at=1700000000000,
        ),
        was_active=True,
    )
    assert fw.autonomy_mode == AutonomyMode.SHADOW_OBSERVE
    assert any(e.category == BrainCategory.SAFETY for e in events)
    snap_after = fw.snapshot_channels_state()
    assert all(st.sync_status == SyncStatus.UNVERIFIED for st in snap_after.values())


# ---------------------------------------------------------------------------
# 2. Tier-1 Speech-Open Reflex (< 1.0 s from -20 dB -> 0 dB)
# ---------------------------------------------------------------------------


def test_speech_open_worship_under_1s_and_3s_silence_release() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)

    # Pastor starts at -20.0 dB in WORSHIP_HIGH
    assert fw.snapshot_channels_state()["ch1_pastor"].vol_current_db == pytest.approx(-20.0, abs=0.2)

    m_speech = _silent_metrics(plan)
    m_speech.norm_vband_db["ch1_pastor"] = -4.0
    m_speech.speech_to_band_snr_db["ch1_pastor"] = +10.0

    t = 0.0
    dt = 0.02
    reached_spoken_t: float | None = None
    for _ in range(60):  # 1.2 s
        t += dt
        out = fw.tick(m_speech, {"ch1_pastor": 0.92}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
        if out.channels_state["ch1_pastor"].vol_current_db >= -0.2 and reached_spoken_t is None:
            reached_spoken_t = t

    assert reached_spoken_t is not None
    # SPEC §9 Phase 2 hard gate: < 1.0 s (400 ms hold + ~420 ms slew = ~820 ms)
    assert 0.70 <= reached_spoken_t < 1.0
    assert fw.snapshot_channels_state()["ch1_pastor"].speech_open is True

    # Silence for 2.5 s -> speech_open MUST STILL hold
    m_quiet = _silent_metrics(plan)
    for _ in range(125):  # 2.5 s
        t += dt
        out = fw.tick(m_quiet, {"ch1_pastor": 0.05}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
    assert out.channels_state["ch1_pastor"].speech_open is True

    # Another 0.6 s silence (total > 3.0 s) -> releases and glides down at 4.0 dB/s
    for _ in range(40):
        t += dt
        out = fw.tick(m_quiet, {"ch1_pastor": 0.05}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
    assert out.channels_state["ch1_pastor"].speech_open is False
    assert out.channels_state["ch1_pastor"].vol_current_db < -0.5


def test_speech_open_faster_150ms_hold_in_spoken_transition() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)
    fw._channels["ch1_pastor"].vol.current_db = -12.0

    m_speech = _silent_metrics(plan)
    m_speech.speech_to_band_snr_db["ch1_pastor"] = 8.0
    t = 0.0
    for _ in range(8):  # 160 ms
        t += 0.02
        out = fw.tick(m_speech, {"ch1_pastor": 0.85}, active_phase=ServicePhase.TRANSITION, dt_s=0.02, now_s=t)
    assert out.channels_state["ch1_pastor"].speech_open is True


# ---------------------------------------------------------------------------
# 3. Tier-1 Ensemble-Gated Instrument-Open Reflex (< 800 ms, Zero False Opens)
# ---------------------------------------------------------------------------


def test_instrument_open_drums_under_800ms_and_8s_auto_release() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)
    # Segment 3 = Sermon (next worship segment is seg_5 = WORSHIP_HIGH where ch9_drums = -2.0 dB)
    fw.service_step_idx = 3

    t = 0.0
    dt = 0.02
    # 3.2 s of instrument silence (<=-30 dB norm)
    for _ in range(160):
        t += dt
        fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)

    # Drums enter at -6 dB norm -> onset gate fires after 200 ms, slews -16 -> -2 dB at 24 dB/s (583 ms)
    onset_start_t = t
    m_drums = _silent_metrics(plan)
    m_drums.norm_vband_db["ch9_drums"] = -6.0
    reached_t: float | None = None

    for _ in range(45):  # 900 ms
        t += dt
        out = fw.tick(m_drums, {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
        if out.channels_state["ch9_drums"].vol_current_db >= -2.1 and reached_t is None:
            reached_t = t - onset_start_t

    assert reached_t is not None
    # SPEC §9 Phase 2 hard gate: -16 dB -> -2 dB in < 800 ms!
    assert reached_t <= 0.80 + 1e-6
    assert out.channels_state["ch9_drums"].instrument_open is True

    # Stay in SPOKEN_WORD for > 8.0 s -> auto-releases and glides back toward -16 dB at 4 dB/s
    for _ in range(420):  # 8.4 s
        t += dt
        out = fw.tick(m_drums, {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)

    assert out.channels_state["ch9_drums"].instrument_open is False
    assert out.channels_state["ch9_drums"].vol_current_db < -3.0


def test_instrument_open_zero_false_triggers_in_prayer_and_solo_guitar() -> None:
    plan, rules, _, _, fw = _make_stack()

    # Case A: PRAYER_UNDERSCORE -> NEVER fires even with full band onset
    _arm_all_and_activate(plan, rules, fw, ServicePhase.PRAYER_UNDERSCORE)
    t = 0.0
    dt = 0.02
    for _ in range(160):  # 3.2 s silent
        t += dt
        fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.PRAYER_UNDERSCORE, dt_s=dt, now_s=t)

    m_full = _silent_metrics(plan)
    m_full.norm_vband_db["ch5_keys"] = -4.0
    m_full.norm_vband_db["ch7_guitars"] = -4.0
    m_full.norm_vband_db["ch9_drums"] = -4.0
    for _ in range(30):
        t += dt
        out = fw.tick(m_full, {}, active_phase=ServicePhase.PRAYER_UNDERSCORE, dt_s=dt, now_s=t)
        assert out.channels_state["ch5_keys"].instrument_open is False
        assert out.channels_state["ch7_guitars"].instrument_open is False
        assert out.channels_state["ch9_drums"].instrument_open is False

    # Case B: Solo guitar tuning in SPOKEN_WORD -> 0 false opens
    _, _, _, _, fw2 = _make_stack()
    _arm_all_and_activate(plan, rules, fw2, ServicePhase.SPOKEN_WORD)
    t = 0.0
    for _ in range(160):
        t += dt
        fw2.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)

    m_solo_gtr = _silent_metrics(plan)
    m_solo_gtr.norm_vband_db["ch7_guitars"] = -5.0
    for _ in range(50):  # 1.0 s solo guitar
        t += dt
        out2 = fw2.tick(m_solo_gtr, {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
        assert out2.channels_state["ch7_guitars"].instrument_open is False
        assert out2.channels_state["ch7_guitars"].vol_current_db == pytest.approx(-16.0, abs=0.1)


def test_instrument_open_two_roles_speech_over_intro_duck_cap_and_prestage() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)
    fw.service_step_idx = 3  # Next worship segment = WORSHIP_HIGH (keys=-3, guitars=-4)

    t = 0.0
    dt = 0.02
    # 3.2 s silence while pastor is speaking (speech_open = True)
    m_pastor = _silent_metrics(plan)
    m_pastor.norm_vband_db["ch1_pastor"] = -4.0
    m_pastor.speech_to_band_snr_db["ch1_pastor"] = +12.0
    for _ in range(160):
        t += dt
        fw.tick(m_pastor, {"ch1_pastor": 0.95}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
    assert fw.snapshot_channels_state()["ch1_pastor"].speech_open is True

    # Keys + Guitars (2 roles, no drums) enter within 1.0 s while pastor speaks over intro
    m_intro = _silent_metrics(plan)
    m_intro.norm_vband_db["ch1_pastor"] = -4.0
    m_intro.speech_to_band_snr_db["ch1_pastor"] = +12.0
    m_intro.norm_vband_db["ch5_keys"] = -6.0
    m_intro.norm_vband_db["ch7_guitars"] = -6.0
    for _ in range(60):  # 1.2 s
        t += dt
        out = fw.tick(m_intro, {"ch1_pastor": 0.95}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)

    # Ensemble gate fired, but targets are capped at worship_target + duck_depth_db (-6 dB)!
    assert out.channels_state["ch5_keys"].instrument_open is True
    assert out.channels_state["ch7_guitars"].instrument_open is True
    assert out.channels_state["ch5_keys"].vol_current_db == pytest.approx(-3.0 - 6.0, abs=0.2)
    assert out.channels_state["ch7_guitars"].vol_current_db == pytest.approx(-4.0 - 6.0, abs=0.2)

    # Segment advance pre-staging test
    _, _, _, _, fw_ps = _make_stack()
    _arm_all_and_activate(plan, rules, fw_ps, ServicePhase.SPOKEN_WORD)
    start_keys_db = fw_ps.snapshot_channels_state()["ch5_keys"].vol_current_db
    fw_ps.on_segment_advance(4, now_s=1.0)  # seg_5 = WORSHIP_HIGH
    out_ps = fw_ps.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=0.5, now_s=1.5)
    # Pre-staged toward WORSHIP_LOW at 4.0 dB/s -> +2.0 dB in 0.5 s
    assert out_ps.channels_state["ch5_keys"].vol_current_db == pytest.approx(start_keys_db + 2.0, abs=0.05)


# ---------------------------------------------------------------------------
# 4. Human Override Zero-Drift Invariance, Pre-Pickup Touch, >hard_max_db & Smooth Clear
# ---------------------------------------------------------------------------


def test_human_override_zero_drift_conditional_suspend_and_smooth_4db_s_clear() -> None:
    plan, rules, _, learner, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)

    # Pre-train learner so live_learned_db is non-zero
    learner.record_human_correction(
        ServicePhase.WORSHIP_HIGH, "ch3_bv1", "vol", np.array([0.0, 0.0, 1.0, 0.0, 1.0]), +2.5
    )

    # Operator moves ch3_bv1 vol to -2.0 dB (also has rule_002 conditional_attenuate -24 dB on pastor speech!)
    ch3 = plan.channel_by_id("ch3_bv1")
    target_cc = db_to_cc(-2.0, ch3.cc_taper)
    expected_db = cc_to_db(target_cc, ch3.cc_taper)
    fw.on_human_cc("ch3_bv1", "vol", target_cc, human_seq=1, now_s=0.0)

    # Also move ch2_lead_vox reverb to -6.0 dB (verifies per-control independence)
    ch2 = plan.channel_by_id("ch2_lead_vox")
    rev_cc = db_to_cc(-6.0, ch2.reverb_cc_taper)
    expected_rev_db = cc_to_db(rev_cc, ch2.reverb_cc_taper)
    fw.on_human_cc("ch2_lead_vox", "reverb", rev_cc, human_seq=1, now_s=0.0)

    # Run 20 seconds (past 15 s touch lock!) with pastor speech_open active (triggering rule_002)
    # AND wildly varying audio phi AND extra learner weight updates -> MUST drift 0.00 dB!
    t = 0.0
    dt = 0.02
    for step in range(1000):  # 20.0 s
        t += dt
        m = _silent_metrics(plan)
        m.norm_vband_db["ch1_pastor"] = -2.0
        m.speech_to_band_snr_db["ch1_pastor"] = +15.0
        m.norm_vband_db["ch3_bv1"] = -15.0 + 10.0 * np.sin(step * 0.1)
        m.master_lufs_short_term = -13.0 + 3.0 * np.cos(step * 0.07)
        if step == 400:
            learner.record_human_correction(
                ServicePhase.WORSHIP_HIGH, "ch3_bv1", "vol", np.array([0.0, 0.0, 1.0, 0.0, 1.0]), -3.5
            )
        out = fw.tick(m, {"ch1_pastor": 0.95}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
        assert out.channels_state["ch3_bv1"].vol_current_db == pytest.approx(expected_db, abs=1e-4)
        assert out.channels_state["ch2_lead_vox"].reverb_current_db == pytest.approx(expected_rev_db, abs=1e-4)

    assert out.channels_state["ch1_pastor"].speech_open is True  # rule_002 was indeed triggered & suspended!

    # Now transition to SPOKEN_WORD -> offset clears smoothly at 4.0 dB/s (NO instantaneous jump!)
    t += dt
    out_next = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
    delta_1tick = abs(out_next.channels_state["ch3_bv1"].vol_current_db - expected_db)
    assert delta_1tick == pytest.approx(4.0 * dt, abs=1e-3)


def test_pre_pickup_touch_lock_and_human_above_hard_max_db() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)

    # Pre-pickup touch on ch5_keys immediately blocks phase glide into WORSHIP_HIGH
    start_keys_db = fw.snapshot_channels_state()["ch5_keys"].vol_current_db
    fw.on_human_touch("ch5_keys", "vol", human_seq=5, now_s=0.0)
    out = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.5, now_s=0.5)
    assert out.channels_state["ch5_keys"].vol_current_db == pytest.approx(start_keys_db, abs=1e-4)
    assert out.channels_state["ch5_keys"].vol_touch_lock_s > 14.0

    # Human pushes ch1_pastor to +5.0 dB (above hard_max_db = +3.0 dB) -> respected until released!
    ch1 = plan.channel_by_id("ch1_pastor")
    cc_plus5 = db_to_cc(5.0, ch1.cc_taper)
    db_plus5 = cc_to_db(cc_plus5, ch1.cc_taper)
    assert db_plus5 > ch1.hard_max_db
    fw.on_human_cc("ch1_pastor", "vol", cc_plus5, human_seq=2, now_s=1.0)
    out_hold = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.02, now_s=17.0)
    assert out_hold.channels_state["ch1_pastor"].vol_current_db == pytest.approx(db_plus5, abs=1e-4)

    # Release human offset -> glides smoothly down toward baseline at 4.0 dB/s
    fw.release_human_offset("ch1_pastor", "vol", now_s=17.0)
    out_rel = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.25, now_s=17.25)
    assert out_rel.channels_state["ch1_pastor"].vol_current_db == pytest.approx(db_plus5 - 1.0, abs=0.05)


# ---------------------------------------------------------------------------
# 5. Role-Configurable Auto-Ducking (48 dB/s Down, 8 dB/s Up)
# ---------------------------------------------------------------------------


def test_role_configurable_auto_ducking_spoken_vs_worship() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.PRAYER_UNDERSCORE)

    # In PRAYER_UNDERSCORE, only instrument_pad (ch5_keys) ducks; guitars/drums do NOT duck
    m_talk = _silent_metrics(plan)
    m_talk.speech_to_band_snr_db["ch1_pastor"] = +8.0
    t = 0.0
    for _ in range(25):  # 500 ms (150ms hold + 125ms attack)
        t += 0.02
        out = fw.tick(m_talk, {"ch1_pastor": 0.90}, active_phase=ServicePhase.PRAYER_UNDERSCORE, dt_s=0.02, now_s=t)

    # ch5_keys baseline (-10) + rule_001 (-3) + duck (-6) = -19.0 dB
    assert out.channels_state["ch5_keys"].vol_current_db == pytest.approx(-19.0, abs=0.15)
    # ch7_guitars stays at -16.0 dB (not in PRAYER_UNDERSCORE duck targets)
    assert out.channels_state["ch7_guitars"].vol_current_db == pytest.approx(-16.0, abs=0.1)

    # Stop talking -> duck releases at 8.0 dB/s (takes 0.75 s for +6 dB)
    # After 3.0 s speech_open also releases, check release rate directly right after duck clears
    _, _, _, _, fw_w = _make_stack()
    _arm_all_and_activate(plan, rules, fw_w, ServicePhase.WORSHIP_HIGH)
    t = 0.0
    for _ in range(35):  # 700 ms (400 ms worship speech_open hold + 125 ms duck attack)
        t += 0.02
        out_w = fw_w.tick(m_talk, {"ch1_pastor": 0.92}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.02, now_s=t)
    # During worship Pastor Interrupt, ALL band roles duck -6 dB!
    assert out_w.channels_state["ch5_keys"].vol_current_db == pytest.approx(-3.0 - 6.0, abs=0.2)
    assert out_w.channels_state["ch7_guitars"].vol_current_db == pytest.approx(-4.0 - 6.0, abs=0.2)
    assert out_w.channels_state["ch9_drums"].vol_current_db == pytest.approx(-2.0 - 6.0, abs=0.2)


# ---------------------------------------------------------------------------
# 6. Bleed-Rejected, Escalating Anti-Feedback (Overrides Touch Lock!)
# ---------------------------------------------------------------------------


def test_antifeedback_cuts_under_400ms_overrides_touch_lock_escalates_and_lifts() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)

    # Engage touch lock on ch1_pastor at 0.0 dB!
    ch1 = plan.channel_by_id("ch1_pastor")
    fw.on_human_cc("ch1_pastor", "vol", db_to_cc(0.0, ch1.cc_taper), human_seq=1, now_s=0.0)

    # Stationary howl at bin 60 (~2.8 kHz), prominence +22 dB above -60 dB floor
    howl_spec = np.full(513, -60.0, dtype=np.float64)
    howl_spec[60] = -38.0  # +22 dB above local median floor
    quiet_spec = np.full(513, -60.0, dtype=np.float64)

    m_howl = _silent_metrics(plan)
    m_howl.fft_mag_db = {ch.id: quiet_spec.copy() for ch in plan.channels}
    m_howl.fft_mag_db["ch1_pastor"] = howl_spec

    t = 0.0
    dt = 0.02
    first_cut_t: float | None = None
    safety_events = []

    # Run 1.6 s of sustained howl -> should cut -4 dB in < 400 ms and escalate to -16 dB!
    for _ in range(80):
        t += dt
        out = fw.tick(m_howl, {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
        safety_events.extend(e for e in out.events if e.category == BrainCategory.SAFETY)
        if out.channels_state["ch1_pastor"].vol_current_db <= -3.9 and first_cut_t is None:
            first_cut_t = t

    assert first_cut_t is not None
    # SPEC §9 Phase 2 hard gate: cuts stationary howl < 400 ms!
    assert first_cut_t < 0.40
    # Overrode touch lock and announced via SAFETY BrainEvent
    assert any("OVERRIDING TOUCH LOCK" in e.summary for e in safety_events)
    # Escalated all the way to -16 dB
    assert out.channels_state["ch1_pastor"].feedback_ceiling_db == pytest.approx(-16.0, abs=0.3)  # CC quantization on the measured taper
    assert out.channels_state["ch1_pastor"].vol_current_db == pytest.approx(-16.0, abs=0.3)  # CC quantization on the measured taper

    # Clear howl -> sticky ceiling holds for 5.0 s, then lifts +1.0 dB/s
    m_clear = _silent_metrics(plan)
    m_clear.fft_mag_db = {ch.id: quiet_spec.copy() for ch in plan.channels}
    for _ in range(240):  # 4.8 s clear -> still sticky at -16.0 dB
        t += dt
        out = fw.tick(m_clear, {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
    assert out.channels_state["ch1_pastor"].feedback_ceiling_db == pytest.approx(-16.0, abs=0.3)  # CC quantization on the measured taper

    for _ in range(110):  # +2.2 s (2.0 s of lift at +1 dB/s -> ceiling ~ -14.0 dB)
        t += dt
        out = fw.tick(m_clear, {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)
    assert out.channels_state["ch1_pastor"].feedback_ceiling_db == pytest.approx(-14.0, abs=0.15)


def test_antifeedback_rejects_synth_pad_bleed_and_clean_harmonic_vocal() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)

    # 1. Synth pad bleed: strong narrowband note on ch5_keys at bin 40 (+28 dB) bleeding into ch1_pastor (+22 dB)
    keys_spec = np.full(513, -60.0, dtype=np.float64)
    keys_spec[40] = -32.0  # +28 dB prominence
    mic_bleed = np.full(513, -60.0, dtype=np.float64)
    mic_bleed[41] = -38.0  # +22 dB prominence within ±1 bin and within 6 dB of keys!

    # 2. Clean sung harmonic vocal on ch2_lead_vox (fundamental bin 14 + harmonics 28, 42, 56, static amplitude)
    vocal_spec = np.full(513, -60.0, dtype=np.float64)
    for b in (14, 28, 42, 56):
        vocal_spec[b] = -38.0  # +22 dB harmonic series, no growth

    m = _silent_metrics(plan)
    m.fft_mag_db = {ch.id: np.full(513, -60.0, dtype=np.float64) for ch in plan.channels}
    m.fft_mag_db["ch5_keys"] = keys_spec
    m.fft_mag_db["ch1_pastor"] = mic_bleed
    m.fft_mag_db["ch2_lead_vox"] = vocal_spec

    t = 0.0
    for _ in range(100):  # 2.0 s sustained
        t += 0.02
        out = fw.tick(m, {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.02, now_s=t)
        assert out.channels_state["ch1_pastor"].feedback_ceiling_db is None
        assert out.channels_state["ch2_lead_vox"].feedback_ceiling_db is None


def _bv2_tone_metrics(plan: ServicePlan, bin_k: int, mag_db: float, floor_db: float = -70.0) -> FakeAudioMetrics:
    """Single harmonic-poor partial on ch4_bv2 (worst case: a sine-like 'ooh', or a pure howl)."""
    m = _silent_metrics(plan)
    m.fft_mag_db = {ch.id: np.full(513, floor_db, dtype=np.float64) for ch in plan.channels}
    m.fft_mag_db["ch4_bv2"][bin_k] = mag_db
    return m


def test_antifeedback_ignores_held_sung_note_attack_swell_and_dip() -> None:
    """Regression (re-rendered soundpack tune 3, ch4_bv2): a held, harmonic-poor sung note must not
    be cut. The candidate used to latch its baseline on the note's attack frame, so the jump from
    attack to sustain looked like feedback growth (-4 -> -8 -> -12 dB false cuts)."""
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)
    sung = {"ch4_bv2": True}
    # ~420 Hz note: 2-tick attack to sustain, gentle 1.5 dB/s crescendo, then a 4 dB
    # expression dip and recovery (all values taken from the measured bv2 stem envelope)
    envelope = [-28.0, -17.0] + [-15.5 + 0.03 * i for i in range(40)]
    envelope += [-15.0, -16.5, -18.5, -20.4, -20.0, -19.0, -17.5, -16.5, -16.0, -15.5, -15.0, -14.6]
    envelope += [-14.5] * 30
    t = 0.0
    for mag in envelope:
        t += 0.02
        out = fw.tick(
            _bv2_tone_metrics(plan, 9, mag), {}, sung_vocals=sung,
            active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.02, now_s=t,
        )
        assert not any("FEEDBACK" in e.summary for e in out.events), f"false cut at t={t:.2f}s mag={mag}"
        assert out.channels_state["ch4_bv2"].feedback_ceiling_db is None


def test_antifeedback_still_cuts_ringing_howl_on_singing_bv_fundamental() -> None:
    """A real howl at a vocal-fundamental frequency on a singing BV mic rings up continuously
    (here 20 dB/s from a -42 dB onset) and must still be cut promptly."""
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)
    sung = {"ch4_bv2": True}
    t = 0.0
    first_cut_t: float | None = None
    for i in range(60):
        t += 0.02
        mag = min(-8.0, -42.0 + 20.0 * 0.02 * i)
        out = fw.tick(
            _bv2_tone_metrics(plan, 9, mag), {}, sung_vocals=sung,
            active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.02, now_s=t,
        )
        if first_cut_t is None and any("ANTI-FEEDBACK CUT" in e.summary for e in out.events):
            first_cut_t = t
    assert first_cut_t is not None, "ringing howl on a sung fundamental was not cut"
    assert first_cut_t < 0.40
    assert out.channels_state["ch4_bv2"].feedback_ceiling_db is not None


# ---------------------------------------------------------------------------
# 7. Watchdog (>50 ms x 5 Consecutive Late Ticks) & Smart Thumbs-Down Rollback
# ---------------------------------------------------------------------------


def test_watchdog_5_late_ticks_drops_to_shadow_without_zeroing_midi() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)

    t = 0.0
    for i in range(4):
        t += 0.02
        out = fw.tick(_silent_metrics(plan), {}, dt_s=0.02, now_s=t, tick_elapsed_ms=60.0)
        assert out.watchdog_tripped is False
        assert out.autonomy_mode == AutonomyMode.ACTIVE

    # 5th consecutive late tick -> trips watchdog!
    t += 0.02
    out5 = fw.tick(_silent_metrics(plan), {}, dt_s=0.02, now_s=t, tick_elapsed_ms=55.0)
    assert out5.watchdog_tripped is True
    assert out5.autonomy_mode == AutonomyMode.SHADOW_OBSERVE
    assert out5.agent_writes == []  # Zero writes; NEVER sends CC 0!
    assert any(e.summary == "WATCHDOG_DROP_TO_SHADOW" for e in out5.events)


def test_smart_thumbs_down_rollback_and_thumbs_up_skipping_touch_locked() -> None:
    plan, rules, _, learner, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)

    t = 0.0
    dt = 0.1
    # Run 10 s in SPOKEN_WORD (ch5_keys = -16 dB)
    for _ in range(100):
        t += dt
        fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=dt, now_s=t)

    # Transition to WORSHIP_HIGH for 6 s so keys glide up toward -3 dB
    for _ in range(60):
        t += dt
        fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
    keys_before_down = fw.snapshot_channels_state()["ch5_keys"].vol_current_db
    assert keys_before_down > -5.0

    # Touch-lock ch7_guitars so thumbs_down MUST skip it!
    ch7 = plan.channel_by_id("ch7_guitars")
    fw.on_human_cc("ch7_guitars", "vol", db_to_cc(-4.0, ch7.cc_taper), human_seq=1, now_s=t)
    gtr_locked_db = fw.snapshot_channels_state()["ch7_guitars"].vol_current_db

    # First thumbs_down -> rolls back non-touch-locked controls to T-5s and holds 10 s
    samples1, evts1 = fw.handle_thumbs_down(now_s=t)
    assert len(samples1) > 0
    assert all(s["channel_id"] != "ch7_guitars" for s in samples1)
    assert "T-5s" in evts1[0].summary

    # Step 2 s -> keys rolled back & holding, ch7_guitars untouched
    for _ in range(20):
        t += dt
        out = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
    assert out.channels_state["ch5_keys"].vol_current_db < keys_before_down - 2.0
    assert out.channels_state["ch7_guitars"].vol_current_db == pytest.approx(gtr_locked_db, abs=1e-4)

    # Second thumbs_down within 5 s -> rolls back further to pre-transition / T-15s (-16 dB)!
    _samples2, evts2 = fw.handle_thumbs_down(now_s=t)
    assert "pre-transition" in evts2[0].summary or "T-15s" in evts2[0].summary
    for _ in range(40):
        t += dt
        out2 = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=dt, now_s=t)
    assert out2.channels_state["ch5_keys"].vol_current_db == pytest.approx(-16.0, abs=0.5)

    # Thumbs up reinforces armed controls with alpha=1.0
    up_samples, _ = fw.handle_thumbs_up(now_s=t)
    assert len(up_samples) >= len(plan.channels)
    assert all(s["alpha"] == 1.0 for s in up_samples)
    assert learner.sample_count() > 0


# ---------------------------------------------------------------------------
# 8. Tier-1 50 Hz Loop Latency Performance Gate (p99 <= 25 ms)
# ---------------------------------------------------------------------------


def test_tier1_tick_latency_well_below_25ms_p99_gate() -> None:
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)

    rng = np.random.default_rng(42)
    m = _silent_metrics(plan)
    m.fft_mag_db = {ch.id: rng.normal(-60.0, 2.0, size=513) for ch in plan.channels}

    latencies_ms: list[float] = []
    t = 0.0
    for _ in range(300):
        t += 0.02
        t0 = time.perf_counter()
        fw.tick(m, {"ch1_pastor": 0.2, "ch2_lead_vox": 0.8}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.02, now_s=t)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    assert p95 < 5.0
    assert p99 <= 25.0


# ---------------------------------------------------------------------------
# 9. QA Iteration 1 Regression Tests (Defects #1, #2, #11, #14)
# ---------------------------------------------------------------------------


def test_qa1_incremental_bridge_state_does_not_retrigger_disk_restore_or_clobber_glides() -> None:
    """Defect #1: Same restored_from_disk_at on incremental bridge_state must not drop ACTIVE or clobber glides."""
    plan, rules, _, _, fw = _make_stack()

    # 1. Boot with a bridge that restored from disk (restored_from_disk_at = 1700000000000)
    init_state = BridgeStatePayload(
        last_sent_cc={f"{ch.id}:vol": db_to_cc(rules.phase_baselines_db[ServicePhase.SPOKEN_WORD][ch.id], ch.cc_taper) for ch in plan.channels},
        restored_from_disk_at=1700000000000,
    )
    fw.hydrate_from_bridge(init_state, was_active=False, is_full_sync=True)
    assert all(st.sync_status == SyncStatus.UNVERIFIED for st in fw.snapshot_channels_state().values())

    # 2. Operator confirms sync on all channels, arms, and enters ACTIVE
    for ch in plan.channels:
        fw.confirm_sync(ch.id)
    fw.set_armed_channels(plan.channel_ids)
    fw.set_mode(AutonomyMode.ACTIVE)

    # 3. Start gliding ch5_keys toward WORSHIP_HIGH (-16 -> -3 dB)
    out1 = fw.tick(_silent_metrics(plan), {}, active_phase=ServicePhase.WORSHIP_HIGH, dt_s=0.5, now_s=0.5)
    mid_glide_db = out1.channels_state["ch5_keys"].vol_current_db
    assert mid_glide_db > -15.5  # Gliding upward

    # 4. Incremental bridge_state broadcast (e.g. pickup_pending change) with SAME restored_from_disk_at
    inc_state = BridgeStatePayload(
        last_sent_cc=init_state.last_sent_cc,
        human_seq={"ch5_keys:vol": 3},
        pickup_pending={"ch5_keys:vol": -1},
        restored_from_disk_at=1700000000000,
    )
    evts = fw.hydrate_from_bridge(inc_state, was_active=True, is_full_sync=False)
    assert evts == []
    assert fw.autonomy_mode == AutonomyMode.ACTIVE  # MUST stay ACTIVE!
    snap = fw.snapshot_channels_state()
    assert snap["ch5_keys"].sync_status == SyncStatus.SYNCED  # MUST stay SYNCED!
    assert snap["ch5_keys"].vol_current_db == pytest.approx(mid_glide_db, abs=1e-4)  # NOT clobbered!
    assert snap["ch5_keys"].vol_pickup_dir == -1

    # 5. A NEW bridge crash-restart produces a NEW restored_from_disk_at -> MUST drop ACTIVE -> SHADOW_OBSERVE!
    restart_state = BridgeStatePayload(
        last_sent_cc=init_state.last_sent_cc,
        restored_from_disk_at=1700000099999,
    )
    evts2 = fw.hydrate_from_bridge(restart_state, was_active=True, is_full_sync=True)
    assert fw.autonomy_mode == AutonomyMode.SHADOW_OBSERVE
    assert any(e.category == BrainCategory.SAFETY for e in evts2)
    assert all(st.sync_status == SyncStatus.UNVERIFIED for st in fw.snapshot_channels_state().values())


def test_qa2_speech_open_preserves_vol_offset_rules_learned_offsets_and_human_zero_drift() -> None:
    """Defect #2: speech_open must preserve continuous vol_db_offset, learned offsets, and human zero-drift."""
    from engine.core.protocol import MixingRule, RuleParameter, RulePhase

    plan, rules, mgr, learner, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)

    # Add +2.5 dB continuous rule on ch1_pastor in SPOKEN_WORD
    mgr.add_permanent_rule(
        MixingRule(
            id="pastor_boost",
            description="Boost pastor +2.5 dB",
            phase=RulePhase.SPOKEN_WORD,
            channel_id="ch1_pastor",
            parameter=RuleParameter.VOL_DB_OFFSET,
            value=2.5,
        )
    )
    # Also train a small positive learned offset on ch1_pastor
    learner.record_human_correction(
        ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", np.array([0.0, 0.0, 1.0, 0.0, 1.0]), +1.0
    )

    # Let pastor glide to boosted target while silent
    t = 0.0
    for _ in range(50):
        t += 0.02
        out_silent = fw.tick(_silent_metrics(plan), {"ch1_pastor": 0.0}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=0.02, now_s=t)
    boosted_db = out_silent.channels_state["ch1_pastor"].vol_current_db
    assert boosted_db >= 2.4

    # Pastor starts speaking (speech_open engages after 150 ms) -> MUST NOT pull pastor down to 0.0 dB!
    m_speech = _silent_metrics(plan)
    m_speech.norm_vband_db["ch1_pastor"] = -4.0
    m_speech.speech_to_band_snr_db["ch1_pastor"] = +12.0
    for _ in range(25):  # 500 ms
        t += 0.02
        out_sp = fw.tick(m_speech, {"ch1_pastor": 0.95}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=0.02, now_s=t)
    assert out_sp.channels_state["ch1_pastor"].speech_open is True
    assert out_sp.channels_state["ch1_pastor"].vol_current_db >= 2.4


def test_qa11_bypass_unlatch_resets_autonomy_mode_to_shadow_observe() -> None:
    """Defect #11: Unlatching bypass via bridge_state resets EMERGENCY_BYPASS -> SHADOW_OBSERVE."""
    plan, rules, _, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.WORSHIP_HIGH)

    # Latch emergency bypass via bridge_state
    fw.hydrate_from_bridge(BridgeStatePayload(bypass_latched=True), was_active=True, is_full_sync=False)
    assert fw.bypass_latched is True
    assert fw.autonomy_mode == AutonomyMode.EMERGENCY_BYPASS

    # Unlatch emergency bypass via bridge_state -> drops cleanly to SHADOW_OBSERVE (never stuck in EMERGENCY_BYPASS)
    fw.hydrate_from_bridge(BridgeStatePayload(bypass_latched=False), was_active=False, is_full_sync=False)
    assert fw.bypass_latched is False
    assert fw.autonomy_mode == AutonomyMode.SHADOW_OBSERVE


def test_qa14_conditional_mute_write_uses_mute_human_seq() -> None:
    """Defect #14: Conditional mute AgentCCWrite uses human_seq[(ch_id, 'mute')], not 'vol'."""
    from engine.core.protocol import (
        ConditionPredicate,
        MixingRule,
        RuleCondition,
        RuleParameter,
        RulePhase,
    )

    plan, rules, mgr, _, fw = _make_stack()
    _arm_all_and_activate(plan, rules, fw, ServicePhase.SPOKEN_WORD)

    mgr.add_permanent_rule(
        MixingRule(
            id="mute_bv2_on_pastor",
            description="Mute BV2 when pastor speaks",
            phase=RulePhase.ANY,
            channel_id="ch4_bv2",
            parameter=RuleParameter.CONDITIONAL_MUTE,
            value=1.0,
            condition=RuleCondition(when_channel="ch1_pastor", predicate=ConditionPredicate.SPEECH_OPEN),
        )
    )
    fw.hydrate_from_bridge(
        BridgeStatePayload(human_seq={"ch4_bv2:vol": 11, "ch4_bv2:mute": 4}),
        was_active=True,
        is_full_sync=False,
    )

    m_speech = _silent_metrics(plan)
    m_speech.speech_to_band_snr_db["ch1_pastor"] = +12.0
    mute_writes = []
    t = 0.0
    for _ in range(12):  # 240 ms (> 150 ms spoken hold)
        t += 0.02
        out = fw.tick(m_speech, {"ch1_pastor": 0.95}, active_phase=ServicePhase.SPOKEN_WORD, dt_s=0.02, now_s=t)
        mute_writes.extend(w for w in out.agent_writes if w.channel_id == "ch4_bv2" and w.control == "mute")

    assert len(mute_writes) == 1
    assert mute_writes[0].cc == 127
    assert mute_writes[0].ack_human_seq == 4

