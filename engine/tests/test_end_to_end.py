"""End-to-End integration tests for AetherMix v5.1 Engine, TelemetryStore, WS Protocol & Simulator."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from engine.core.protocol import (
    PROTOCOL_VERSION,
    AutonomyMode,
    MixingRules,
    ProposedPatch,
    RuleExpiry,
    RuleParameter,
    RulePhase,
    RuleScope,
    ServicePhase,
    ServicePlan,
)
from engine.core.telemetry_store import TelemetryStore
from engine.main import (
    DEFAULT_MIXING_RULES_PATH,
    DEFAULT_SERVICE_PLAN_PATH,
    AetherMixEngine,
    create_app,
)
from engine.tests.mock_io import (
    MockBridgeClient,
    MockLLMSupervisor,
    SceneAudioConfig,
    SyntheticServiceAudioGenerator,
)
from fastapi.testclient import TestClient
from scripts.simulate_service import run_headless_simulation
from starlette.websockets import WebSocketDisconnect


def _load_schemas() -> tuple[ServicePlan, MixingRules]:
    plan = ServicePlan.model_validate_json(DEFAULT_SERVICE_PLAN_PATH.read_text(encoding="utf-8"))
    rules = MixingRules.model_validate_json(DEFAULT_MIXING_RULES_PATH.read_text(encoding="utf-8"))
    return plan, rules


def _make_test_engine(tmp_path: Path, *, headless: bool = True) -> tuple[AetherMixEngine, MockBridgeClient, MockLLMSupervisor]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan, rules = _load_schemas()
    plan_path = tmp_path / "service_plan.json"
    plan_path.write_text(DEFAULT_SERVICE_PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    rules_path = tmp_path / "mixing_rules.json"
    rules_path.write_text(DEFAULT_MIXING_RULES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    bridge = MockBridgeClient(plan)
    llm = MockLLMSupervisor(plan)
    engine = AetherMixEngine(
        plan=plan,
        rules=rules,
        plan_path=plan_path,
        rules_path=rules_path,
        backup_dir=tmp_path / "rule_backups",
        db_path=tmp_path / "telemetry.db",
        token="test-secret-token-v51",
        headless=headless,
        enable_clap=False,
        use_service_plan=False,
        bridge_client=bridge,
        llm_supervisor=llm,
    )
    return engine, bridge, llm


def _ws_rpc(ws: Any, msg_type: str, payload: dict[str, Any], req_id: str) -> dict[str, Any]:
    """Send a client RPC and read frames until the correlated ack/error arrives."""
    ws.send_json({"type": msg_type, "req_id": req_id, "payload": payload})
    for _ in range(25):
        msg = ws.receive_json()
        if msg.get("req_id") == req_id and msg.get("type") in ("ack", "error"):
            return msg
    raise AssertionError(f"Did not receive correlated ack/error for req_id={req_id} ({msg_type})")


# ---------------------------------------------------------------------------
# 1. TelemetryStore WAL & Persistence
# ---------------------------------------------------------------------------


def test_telemetry_store_wal_and_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry.db"
    store = TelemetryStore(db_path=db_path)
    store.start()
    try:
        store.log_1hz_summary(
            active_phase="WORSHIP_HIGH",
            autonomy_mode="ACTIVE",
            master_lufs=-13.4,
            loop_p95_ms=1.6,
            loop_p99_ms=3.2,
            summary={"armed_count": 5},
        )
        store.log_event("SAFETY", {"summary": "Test safety event"})
        store.save_last_known_state(
            armed_ids=["ch1_pastor", "ch5_keys"],
            channels_cc={"ch1_pastor:vol": 96, "ch5_keys:vol": 64},
        )
        store.save_policy_snapshot(version=3, state_dict={"version": 3, "models": {}})
        store.queue_review_sample({"phase": "WORSHIP_LOW", "channel_id": "ch5_keys", "control": "vol", "phi": [0, 0, 1, 0, 1], "target_offset_db": -1.5, "alpha": 2.0})
        store.queue_review_sample({"phase": "WORSHIP_LOW", "channel_id": "ch7_guitars", "control": "vol", "phi": [0, 0, 1, 0, 1], "target_offset_db": 1.0, "alpha": 1.0})

        assert store.flush(timeout=2.0)

        # Verify WAL mode active on SQLite file
        conn = sqlite3.connect(str(db_path))
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert str(mode).lower() == "wal"
        finally:
            conn.close()

        last_state = store.load_last_known_state()
        assert last_state["armed_ids"] == ["ch1_pastor", "ch5_keys"]
        assert last_state["channels_cc"]["ch1_pastor:vol"] == 96

        snap = store.load_latest_policy_snapshot()
        assert snap is not None
        assert snap[0] == 3
        assert snap[1]["version"] == 3

        queued = store.list_queued_reviews()
        assert len(queued) == 2
        assert queued[0]["channel_id"] == "ch5_keys"

        store.clear_queued_reviews()
        assert store.list_queued_reviews() == []

        assert len(store.list_recent_telemetry(limit=10)) == 1
        assert len(store.list_recent_events(limit=10)) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. WebSocket Bearer Token Auth & Protocol v5.1 Handshake
# ---------------------------------------------------------------------------


def test_ws_auth_and_protocol_handshake(tmp_path: Path) -> None:
    engine, _bridge, _llm = _make_test_engine(tmp_path)
    app = create_app(engine, auto_start_loop=False)

    with TestClient(app) as client:
        # Missing Bearer token rejected
        with pytest.raises(WebSocketDisconnect) as exc_missing, client.websocket_connect("/ws"):
            pass
        assert exc_missing.value.code == 4401

        # Wrong Bearer token rejected
        with (
            pytest.raises(WebSocketDisconnect) as exc_bad,
            client.websocket_connect("/ws", headers={"Authorization": "Bearer wrong-token"}),
        ):
            pass
        assert exc_bad.value.code == 4401

        # Bad protocol version rejected on hello
        with client.websocket_connect(
            "/ws", headers={"Authorization": f"Bearer {engine.token}"}
        ) as ws:
            ws.send_json({"type": "hello", "req_id": "h-bad", "payload": {"protocol_version": "1.0"}})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["req_id"] == "h-bad"
            assert err["payload"]["code"] == "HANDSHAKE_FAILED"

        # Valid Bearer token + v5.1 hello returns correlated ack + state_snapshot + queued_reviews
        with client.websocket_connect(
            "/ws", headers={"Authorization": f"Bearer {engine.token}"}
        ) as ws:
            ws.send_json(
                {"type": "hello", "req_id": "h-ok", "payload": {"protocol_version": PROTOCOL_VERSION}}
            )
            ack = ws.receive_json()
            assert ack["type"] == "ack"
            assert ack["req_id"] == "h-ok"
            assert ack["payload"]["ok"] is True

            snap = ws.receive_json()
            assert snap["type"] == "state_snapshot"
            payload = snap["payload"]
            assert payload["autonomy_mode"] == "SHADOW_OBSERVE"
            assert payload["channel_ids"] == engine.plan.channel_ids
            assert all(ch["armed"] is False for ch in payload["channels"].values())

            qr = ws.receive_json()
            assert qr["type"] == "queued_reviews"
            assert qr["payload"]["samples"] == []


# ---------------------------------------------------------------------------
# 3. All v5.1 Client RPCs, Critique Preview/Promote, Rollback & Safe-Update Lock
# ---------------------------------------------------------------------------


def test_all_rpcs_critique_preview_promote_and_safe_update_lock(tmp_path: Path) -> None:
    engine, bridge, _llm = _make_test_engine(tmp_path)
    app = create_app(engine, auto_start_loop=False)

    # Seed calibration audio blocks
    gen = SyntheticServiceAudioGenerator(seed=99)
    for _ in range(10):
        engine.push_audio_block(gen.generate_block(SceneAudioConfig.for_phase(ServicePhase.SPOKEN_WORD)))

    with (
        TestClient(app) as client,
        client.websocket_connect("/ws", headers={"Authorization": f"Bearer {engine.token}"}) as ws,
    ):
        ws.send_json(
            {"type": "hello", "req_id": "h1", "payload": {"protocol_version": PROTOCOL_VERSION}}
        )
        assert ws.receive_json()["type"] == "ack"
        assert ws.receive_json()["type"] == "state_snapshot"
        assert ws.receive_json()["type"] == "queued_reviews"

        # Safe-update allowed in SHADOW_OBSERVE
        r = _ws_rpc(ws, "apply_update", {}, "req-upd-shadow")
        assert r["type"] == "ack" and r["payload"]["ok"] is True

        # Arm channels + persist + restore_previous_arming
        r = _ws_rpc(ws, "set_channel_armed", {"channel_id": "ch1_pastor", "armed": True}, "req-arm1")
        assert r["type"] == "ack"
        r = _ws_rpc(ws, "set_channel_armed", {"channel_id": "ch5_keys", "armed": True}, "req-arm2")
        assert r["type"] == "ack"

        # Disarm in memory, then verify restore_previous_arming restores saved bitmask
        engine.safety_firewall.set_armed_channels([])
        assert engine.safety_firewall.get_armed_channel_ids() == []
        r = _ws_rpc(ws, "restore_previous_arming", {}, "req-restore")
        assert r["type"] == "ack"
        assert set(r["payload"]["data"]["restored_armed_ids"]) == {"ch1_pastor", "ch5_keys"}

        # Confirm sync & enter ACTIVE mode
        r = _ws_rpc(ws, "sync_channel", {"channel_id": "ch5_keys"}, "req-sync")
        assert r["type"] == "ack"
        r = _ws_rpc(ws, "set_mode", {"mode": "ACTIVE"}, "req-active")
        assert r["type"] == "ack"

        # Safe-Update Lock MUST block while autonomy_mode == ACTIVE
        r = _ws_rpc(ws, "apply_update", {}, "req-upd-active")
        assert r["type"] == "error"
        assert r["payload"]["code"] == "SAFE_UPDATE_BLOCKED"

        # Release human offset, override phase, advance service step
        r = _ws_rpc(ws, "release_human_offset", {"channel_id": "ch5_keys", "control": "vol"}, "req-rel")
        assert r["type"] == "ack"
        r = _ws_rpc(ws, "override_phase", {"phase": "WORSHIP_LOW", "hold_seconds": 30.0}, "req-ovr")
        assert r["type"] == "ack"
        assert engine.scene_classifier.active_phase == ServicePhase.WORSHIP_LOW
        r = _ws_rpc(ws, "advance_service_step", {"step_idx": 2}, "req-step")
        assert r["type"] == "ack"
        assert engine.service_step_idx == 2

        # Freeze learning -> thumbs_up queues review samples -> selective commit_queued_reviews (Defect #10)
        r = _ws_rpc(ws, "set_freeze_learning", {"freeze": True}, "req-frz-on")
        assert r["type"] == "ack"
        assert engine.freeze_learning is True

        r = _ws_rpc(ws, "feedback_thumb", {"direction": "up"}, "req-tup-frz")
        assert r["type"] == "ack"
        assert r["payload"]["data"]["queued"] is True
        queued_all = engine.telemetry_store.list_queued_reviews()
        assert len(queued_all) >= 2
        first_id = queued_all[0]["id"]

        # Selective commit by sample_ids leaves remaining samples untouched
        r = _ws_rpc(ws, "commit_queued_reviews", {"action": "commit", "sample_ids": [first_id]}, "req-commit-one")
        assert r["type"] == "ack"
        assert r["payload"]["data"]["committed"] == 1
        assert r["payload"]["data"]["remaining"] == len(queued_all) - 1

        # Commit remaining samples
        r = _ws_rpc(ws, "commit_queued_reviews", {"action": "commit"}, "req-commit")
        assert r["type"] == "ack"
        assert engine.telemetry_store.list_queued_reviews() == []

        r = _ws_rpc(ws, "set_freeze_learning", {"freeze": False}, "req-frz-off")
        assert r["type"] == "ack"

        # Critique 1: small single-channel tweak -> auto-applied ephemeral, then promote to permanent
        r = _ws_rpc(ws, "submit_critique", {"text": "make guitars slightly louder"}, "req-crit1")
        assert r["type"] == "ack"
        cards1 = r["payload"]["data"]["pending_rules"]
        assert len(cards1) == 1
        assert cards1[0]["requires_preview"] is False
        assert cards1[0]["applied_in_session"] is True
        rule_id_1 = cards1[0]["rule_id"]

        r = _ws_rpc(ws, "resolve_pending_rule", {"rule_id": rule_id_1, "action": "promote"}, "req-prom1")
        assert r["type"] == "ack"
        # Verify atomic backup created when saving permanent rule
        assert list((tmp_path / "rule_backups").glob("mixing_rules_*.json"))

        # Toggle & delete the promoted permanent rule
        r = _ws_rpc(ws, "toggle_rule", {"rule_id": rule_id_1, "enabled": False}, "req-tog")
        assert r["type"] == "ack" and r["payload"]["data"]["enabled"] is False
        r = _ws_rpc(ws, "delete_rule", {"rule_id": rule_id_1}, "req-del")
        assert r["type"] == "ack" and r["payload"]["data"]["deleted"] is True

        # Critique 2: conditional mute / large patch -> requires_preview=True, NOT applied yet
        r = _ws_rpc(
            ws, "submit_critique", {"text": "mute backup vocals when pastor speaks"}, "req-crit2"
        )
        assert r["type"] == "ack"
        cards2 = r["payload"]["data"]["pending_rules"]
        assert len(cards2) == 1
        assert cards2[0]["requires_preview"] is True
        assert cards2[0]["applied_in_session"] is False
        rule_id_2 = cards2[0]["rule_id"]

        r = _ws_rpc(
            ws, "resolve_pending_rule", {"rule_id": rule_id_2, "action": "apply_session"}, "req-app2"
        )
        assert r["type"] == "ack"
        r = _ws_rpc(
            ws, "resolve_pending_rule", {"rule_id": rule_id_2, "action": "dismiss"}, "req-dis2"
        )
        assert r["type"] == "ack"

        # Reset learning
        r = _ws_rpc(ws, "reset_learning", {"phase": "WORSHIP_LOW"}, "req-rst")
        assert r["type"] == "ack" and r["payload"]["data"]["reset_models"] > 0

        # 4-step calibration RPCs + atomic service_plan.json persistence + quick_derive_others (Defect #4)
        assert _ws_rpc(ws, "capture_room_silence", {"duration_s": 0.2}, "req-cal1")["type"] == "ack"
        assert _ws_rpc(ws, "capture_input_nominals", {"group": "spoken", "duration_s": 0.2}, "req-cal2")["type"] == "ack"
        assert _ws_rpc(ws, "capture_input_nominals", {"group": "band", "duration_s": 0.2}, "req-cal3")["type"] == "ack"
        assert _ws_rpc(ws, "capture_bleed_floor", {"duration_s": 0.2}, "req-cal4")["type"] == "ack"

        bridge._state.last_sent_cc["ch9_drums:vol"] = 96  # 0.0 dB
        r_base = _ws_rpc(
            ws, "capture_baseline", {"phase": "WORSHIP_HIGH", "quick_derive_others": True}, "req-cal5"
        )
        assert r_base["type"] == "ack" and r_base["payload"]["data"]["calibrated_at"] is not None
        assert engine.rules.phase_baselines_db[ServicePhase.WORSHIP_HIGH]["ch9_drums"] == 0.0
        assert engine.rules.phase_baselines_db[ServicePhase.WORSHIP_LOW]["ch9_drums"] == -10.0

        assert _ws_rpc(ws, "midi_learn_pulse", {"channel_id": "ch5_keys", "control": "vol"}, "req-mlp")["type"] == "ack"
        assert ("ch5_keys", "vol") in bridge.midi_learn_pulses

        # Mute polarity forbidden on spoken_primary, accepts both active_high & mute_active_high on vocal_lead (Defect #3)
        r_m_bad = _ws_rpc(ws, "verify_mute_polarity", {"channel_id": "ch1_pastor", "active_high": True}, "req-mp1")
        assert r_m_bad["type"] == "error"
        r_m_ok = _ws_rpc(ws, "verify_mute_polarity", {"channel_id": "ch2_lead_vox", "active_high": False}, "req-mp2")
        assert r_m_ok["type"] == "ack"
        assert r_m_ok["payload"]["data"]["mute_active_high"] is False
        assert r_m_ok["payload"]["data"]["sent_cc"] == 0
        # Persisted to service_plan.json
        reloaded_plan = ServicePlan.model_validate_json((tmp_path / "service_plan.json").read_text(encoding="utf-8"))
        assert reloaded_plan.channel_by_id("ch2_lead_vox").mute_active_high is False


# ---------------------------------------------------------------------------
# 4. Full Pipeline: MeterFrame, Thumbs Down Rollback & Direct Bridge Bypass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_meter_frame_thumbs_down_rollback_and_bridge_bypass(tmp_path: Path) -> None:
    engine, bridge, _llm = _make_test_engine(tmp_path)
    gen = SyntheticServiceAudioGenerator(seed=5150)
    await engine.start(start_loop=False)
    try:
        for ch in engine.plan.channels:
            engine.safety_firewall.set_channel_armed(ch.id, True)
            engine.safety_firewall.confirm_sync(ch.id)
        engine.safety_firewall.set_mode(AutonomyMode.ACTIVE)

        # Run 6 seconds in WORSHIP_LOW so history has T-5s snapshots
        t = 0.0
        dt = 0.02
        for _ in range(300):
            t += dt
            out = await engine.step_once(
                dt_s=dt,
                now_s=t,
                audio_block=gen.generate_block(SceneAudioConfig.for_phase(ServicePhase.WORSHIP_LOW)),
                broadcast=False,
            )

        assert engine._last_metrics is not None
        mf = engine.build_meter_frame(engine._last_metrics, out.channels_state, engine._last_vad_probs)
        n = len(engine.plan.channel_ids)
        assert len(mf.rms_db) == n
        assert len(mf.vol_target_cc) == n
        assert len(mf.vol_actual_cc) == n
        assert len(mf.reverb_target_cc) == n
        assert len(mf.reverb_actual_cc) == n
        assert mf.reverb_actual_cc[0] is not None  # ch1_pastor has reverb
        assert mf.reverb_actual_cc[4] is None      # ch5_keys has no reverb

        # Record keys level in WORSHIP_LOW (~ -4 dB), then switch to WORSHIP_HIGH and push keys up
        keys_before_db = out.channels_state["ch5_keys"].vol_current_db
        assert keys_before_db < -3.5

        # Override to WORSHIP_HIGH for 2 seconds so keys climbs toward -3 dB and drums climbs toward -2 dB
        engine.scene_classifier.override_phase(ServicePhase.WORSHIP_HIGH, hold_seconds=30.0, now_s=t)
        for _ in range(100):
            t += dt
            out = await engine.step_once(
                dt_s=dt,
                now_s=t,
                audio_block=gen.generate_block(SceneAudioConfig.for_phase(ServicePhase.WORSHIP_HIGH)),
                broadcast=False,
            )
        drums_after_high = out.channels_state["ch9_drums"].vol_current_db
        assert drums_after_high > -5.0

        # Hardware thumbs_down button triggers T-5s rollback!
        await bridge.simulate_action_button("thumbs_down")
        for _ in range(125):  # 2.5s glide back toward T-5s WORSHIP_LOW snapshot
            t += dt
            out = await engine.step_once(
                dt_s=dt,
                now_s=t,
                audio_block=gen.generate_block(SceneAudioConfig.for_phase(ServicePhase.WORSHIP_HIGH)),
                broadcast=False,
            )
        assert out.channels_state["ch9_drums"].vol_current_db < drums_after_high - 5.0

        # Hardware emergency_bypass latches directly on bridge & blocks all subsequent agent writes
        writes_before_bypass = len(bridge.sent_agent_writes)
        await bridge.simulate_action_button("emergency_bypass")
        assert bridge.state.bypass_latched is True
        assert engine.safety_firewall.autonomy_mode == AutonomyMode.EMERGENCY_BYPASS

        for _ in range(25):
            t += dt
            await engine.step_once(
                dt_s=dt,
                now_s=t,
                audio_block=gen.generate_block(SceneAudioConfig.for_phase(ServicePhase.SPOKEN_WORD)),
                broadcast=False,
            )
        assert len(bridge.sent_agent_writes) == writes_before_bypass

        # Human manual CC still passes straight through the bridge while bypassed!
        await bridge.simulate_human_cc("ch5_keys", "vol", 77, source="hardware")
        assert bridge.state.last_sent_cc["ch5_keys:vol"] == 77
    finally:
        await engine.stop()


# ---------------------------------------------------------------------------
# 5. Full Headless Service Simulator Gate Verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_headless_service_simulator_all_gates_pass() -> None:
    all_passed, checks, metrics = await run_headless_simulation()
    failed = [c for c in checks if not c.passed]
    assert all_passed, f"Simulator gates failed: {failed}"
    assert metrics["loop_p99_ms"] <= 25.0
    assert metrics["agent_writes"] > 0


# ---------------------------------------------------------------------------
# 6. QA Iteration 1 Regression Tests (#1, #5, #6, #7, #13)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qa_regressions_reconnect_stall_watchdog_signature_and_override_expiry(tmp_path: Path) -> None:
    # Defects #1 & #5: incremental bridge_state never drops ACTIVE or clobbers live glides,
    # whereas bridge reconnect with a new restored_from_disk_at drops ACTIVE -> SHADOW_OBSERVE once.
    engine, bridge, _llm = _make_test_engine(tmp_path, headless=True)
    bridge._state.restored_from_disk_at = 111111
    await engine.start(start_loop=False)
    try:
        for ch in engine.plan.channels:
            engine.safety_firewall.set_channel_armed(ch.id, True)
            engine.safety_firewall.confirm_sync(ch.id)
        engine.safety_firewall.set_mode(AutonomyMode.ACTIVE)

        # Incremental bridge_state broadcast with the SAME restored_from_disk_at must NOT drop ACTIVE!
        bridge._state.pickup_pending["ch5_keys:vol"] = -1
        await bridge.simulate_state_broadcast()
        assert engine.safety_firewall.autonomy_mode == AutonomyMode.ACTIVE
        assert engine._last_channels_state["ch5_keys"].vol_pickup_dir == -1

        # Snapshot signature changes when pickup_dir or phase_confidence changes (Defect #7)
        sig_a = engine._snapshot_signature()
        engine.phase_confidence = 0.85
        sig_b = engine._snapshot_signature()
        assert sig_a != sig_b

        # New bridge crash-restart reconnect with a NEW restored_from_disk_at drops ACTIVE -> SHADOW_OBSERVE
        await bridge.simulate_reconnect(restored_from_disk_at=222222)
        assert engine.safety_firewall.autonomy_mode == AutonomyMode.SHADOW_OBSERVE

        # Defect #13: override_phase expires PHASE_CHANGE ephemeral rules
        cards = engine.rules_manager.ingest_proposed_patches(
            [
                ProposedPatch(
                    description="Temp phase tweak",
                    scope=RuleScope.EPHEMERAL_SESSION,
                    expires_on=RuleExpiry.PHASE_CHANGE,
                    phase=RulePhase.ANY,
                    channel_id="ch5_keys",
                    parameter=RuleParameter.VOL_DB_OFFSET,
                    value=-1.5,
                )
            ]
        )
        assert len(cards) == 1
        assert len(engine.rules_manager.ephemeral_rules) == 1
        await engine.handle_rpc("override_phase", {"phase": "WORSHIP_HIGH"}, req_id="ovr-exp")
        assert len(engine.rules_manager.ephemeral_rules) == 0
    finally:
        await engine.stop()

    # Defect #6: live audio-stall watchdog trips when no new audio blocks arrive for > 500ms
    live_engine, _b2, _l2 = _make_test_engine(tmp_path / "live", headless=False)
    live_engine.audio_tap.headless = True  # avoid PortAudio device open while testing non-headless engine stall
    await live_engine.start(start_loop=False)
    try:
        live_engine.safety_firewall.set_mode(AutonomyMode.ACTIVE)
        gen = SyntheticServiceAudioGenerator(seed=7)
        # Deliver one initial block at t=0.02
        out1 = await live_engine.step_once(
            dt_s=0.02,
            now_s=0.02,
            audio_block=gen.generate_block(SceneAudioConfig.for_phase(ServicePhase.SPOKEN_WORD)),
            broadcast=False,
        )
        assert out1.watchdog_tripped is False
        # Advance 0.6s with NO new audio blocks -> audio_stale_s > 0.5 trips watchdog!
        out2 = await live_engine.step_once(dt_s=0.02, now_s=0.62, audio_block=None, broadcast=False)
        assert out2.watchdog_tripped is True
        assert out2.autonomy_mode == AutonomyMode.SHADOW_OBSERVE
    finally:
        await live_engine.stop()
