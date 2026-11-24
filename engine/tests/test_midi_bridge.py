"""Comprehensive headless test suite for AetherMix Standalone MIDI Bridge (SPEC v5.1)."""

from __future__ import annotations

import ast
import asyncio
import errno
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from bridge.midi_bridge_daemon import BridgeConfig, MidiBridgeDaemon, MockMidiBackend
from engine.core.bridge_client import BridgeClient
from engine.core.protocol import BridgeStatePayload

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_PLAN_PATH = REPO_ROOT / "engine" / "schemas" / "service_plan.json"
BRIDGE_DAEMON_PATH = REPO_ROOT / "bridge" / "midi_bridge_daemon.py"
PLIST_PATH = REPO_ROOT / "scripts" / "com.aethermix.midibridge.plist"


def _load_plan_dict(**midi_overrides: Any) -> dict[str, Any]:
    raw = json.loads(SERVICE_PLAN_PATH.read_text(encoding="utf-8"))
    raw["midi"].update(midi_overrides)
    return raw


@pytest.fixture
def short_tmp_dir() -> Path:
    """Short directory inside workspace to stay under macOS AF_UNIX 104-byte path limit."""
    base = REPO_ROOT / "engine" / ".pytest_cache"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="s_", dir=str(base)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


async def _open_bridge_stream(daemon: MidiBridgeDaemon, sock: Path) -> tuple[asyncio.StreamReader, Any]:
    """Prefer real AF_UNIX connect; fall back to in-process NDJSON loopback only on EPERM sandboxes."""
    try:
        return await asyncio.open_unix_connection(str(sock))
    except PermissionError as exc:
        if exc.errno == errno.EPERM:
            return daemon.open_inprocess_connection()
        raise


def _make_client(daemon: MidiBridgeDaemon, sock: Path, **kwargs: Any) -> BridgeClient:
    return BridgeClient(
        socket_path=sock,
        stream_opener=lambda: _open_bridge_stream(daemon, sock),
        **kwargs,
    )


async def _wait_until(predicate: Any, timeout: float = 2.0, step: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(step)
    raise AssertionError("Timed out waiting for condition")


# ---------------------------------------------------------------------------
# 1. Zero ML / Heavy Imports (AST + Subprocess Runtime Isolation)
# ---------------------------------------------------------------------------


def test_zero_ml_or_heavy_imports_ast_and_runtime() -> None:
    forbidden = {
        "torch",
        "numpy",
        "scipy",
        "fastapi",
        "pydantic",
        "transformers",
        "onnxruntime",
        "sounddevice",
        "uvicorn",
        "httpx",
        "engine",
    }
    tree = ast.parse(BRIDGE_DAEMON_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    offenders = imported_roots & forbidden
    assert not offenders, f"Forbidden imports in midi_bridge_daemon.py: {sorted(offenders)}"

    # Verify in a clean subprocess that importing midi_bridge_daemon loads zero heavy deps
    code = (
        "import json, sys; "
        "import bridge.midi_bridge_daemon; "
        f"bad = sorted(m for m in {sorted(forbidden)!r} if m in sys.modules); "
        "print(json.dumps(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    loaded_forbidden = json.loads(proc.stdout.strip())
    assert loaded_forbidden == [], f"Heavy modules loaded at runtime: {loaded_forbidden}"


# ---------------------------------------------------------------------------
# 2. macOS LaunchAgent Plist Validation
# ---------------------------------------------------------------------------


def test_launchd_plist_manifest_valid() -> None:
    raw_text = PLIST_PATH.read_text(encoding="utf-8")
    for placeholder in ("__REPO_ROOT__", "__PYTHON__", "__HOME__"):
        assert placeholder in raw_text, f"Missing placeholder {placeholder} in launchd plist"

    parsed = plistlib.loads(raw_text.encode("utf-8"))
    assert parsed["Label"] == "com.aethermix.midibridge"
    assert parsed["KeepAlive"] is True
    assert parsed["RunAtLoad"] is True
    assert parsed["ThrottleInterval"] == 1
    assert parsed["ProcessType"] == "Interactive"
    assert "__PYTHON__" in parsed["ProgramArguments"][0]
    assert "__REPO_ROOT__/bridge/midi_bridge_daemon.py" in parsed["ProgramArguments"]


# ---------------------------------------------------------------------------
# 3. Socket Permissions, Auth, and Dual Concurrent Clients
# ---------------------------------------------------------------------------


async def test_socket_permissions_and_token_auth(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    token = "secret-session-token-v51"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token=token,
        state_path=state_file,
        midi_backend=backend,
    )
    await daemon.start()
    try:
        mode = stat.S_IMODE(os.stat(sock).st_mode)
        assert mode == 0o600, f"Expected 0600 socket perms, got {oct(mode)}"

        # Bad token must be rejected
        bad_client = _make_client(daemon, sock, token="wrong-token", role="engine")
        with pytest.raises(PermissionError, match="AUTH_FAILED"):
            await bad_client.connect()

        # Wrong protocol_version must be rejected
        reader, writer = await _open_bridge_stream(daemon, sock)
        try:
            bad_hello = {
                "type": "hello",
                "role": "ui",
                "token": token,
                "protocol_version": "4.0",
            }
            writer.write((json.dumps(bad_hello) + "\n").encode("utf-8"))
            await writer.drain()
            resp = json.loads((await asyncio.wait_for(reader.readline(), timeout=2.0)).decode("utf-8"))
            assert resp["type"] == "error"
            assert resp["code"] == "VERSION_MISMATCH"
        finally:
            writer.close()
            await writer.wait_closed()

        # Concurrent engine + ui connections
        ui_states: list[BridgeStatePayload] = []
        engine_client = _make_client(daemon, sock, token=token, role="engine")
        ui_client = _make_client(
            daemon,
            sock,
            token=token,
            role="ui",
            on_state_change=ui_states.append,
        )
        await ui_client.connect()
        assert ui_client.connected
        assert ui_client.state.engine_connected is False
        assert ui_client.state.controller_connected is True

        await engine_client.connect()
        assert engine_client.connected
        assert engine_client.state.engine_connected is True

        await _wait_until(lambda: any(s.engine_connected for s in ui_states))
        assert ui_client.state.engine_connected is True

        await engine_client.close()
        await ui_client.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 4. Always-On <2ms Hardware Passthrough & fader_map Translation
# ---------------------------------------------------------------------------


async def test_hardware_fader_map_translation_and_latency_under_2ms(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
    )
    await daemon.start()
    try:
        # Engine is completely disconnected!
        assert daemon.engine_connected is False

        t0 = time.perf_counter()
        # hw_ch=1, hw_cc=5 -> ch5_keys:vol -> MainStage ch=1, cc=24
        backend.inject_cc(1, 5, 77)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 2.0, f"Hardware passthrough took {elapsed_ms:.3f} ms (limit <2ms)"
        assert len(backend.sent_messages) == 1
        sent = backend.sent_messages[0]
        assert (sent.midi_channel, sent.cc_num, sent.value) == (1, 24, 77)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 77

        # hw_ch=1, hw_cc=11 -> ch1_pastor:reverb -> MainStage ch=1, cc=30
        backend.inject_cc(1, 11, 52)
        assert len(backend.sent_messages) == 2
        sent_rev = backend.sent_messages[1]
        assert (sent_rev.midi_channel, sent_rev.cc_num, sent_rev.value) == (1, 30, 52)
        assert daemon.last_sent_cc["ch1_pastor:reverb"] == 52
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 5. Hardware Action Bindings & Bridge-Latched Emergency Bypass
# ---------------------------------------------------------------------------


async def test_hardware_action_bindings_and_latched_emergency_bypass(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
    )
    await daemon.start()
    try:
        actions: list[dict[str, Any]] = []
        engine = _make_client(
            daemon,
            sock,
            token="tok",
            role="engine",
            on_action_button=actions.append,
        )
        await engine.connect()

        # Thumbs up & thumbs down
        backend.inject_cc(1, 102, 127)
        backend.inject_cc(1, 103, 127)
        await _wait_until(lambda: len(actions) == 2)
        assert [a["action"] for a in actions] == ["thumbs_up", "thumbs_down"]

        # Hardware emergency_bypass latches ON and drops all agent writes
        backend.inject_cc(1, 104, 127)
        await _wait_until(lambda: len(actions) == 3 and engine.state.bypass_latched is True)
        assert actions[2]["action"] == "emergency_bypass"
        assert daemon.bypass_latched is True

        # Releasing momentary button (value=0) must NOT unlatch bypass
        backend.inject_cc(1, 104, 0)
        await asyncio.sleep(0.02)
        assert daemon.bypass_latched is True

        # Agent writes must be dropped immediately while bypassed
        await engine.send_agent_cc("ch5_keys", "vol", 90, ack_human_seq=0)
        await asyncio.sleep(0.03)
        assert len(backend.sent_messages) == 0

        # Hardware fader MUST still pass through while bypass_latched=True
        backend.inject_cc(1, 5, 65)
        assert len(backend.sent_messages) == 1
        assert (backend.sent_messages[0].cc_num, backend.sent_messages[0].value) == (24, 65)

        await engine.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 6. Bridge-Level In-Flight Race Guard (1ms drop + human_seq fence)
# ---------------------------------------------------------------------------


async def test_inflight_race_guard_1ms_and_stale_human_seq(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
        race_guard_ms=250.0,
    )
    await daemon.start()
    try:
        human_events: list[dict[str, Any]] = []
        engine = _make_client(
            daemon,
            sock,
            token="tok",
            role="engine",
            on_human_cc=human_events.append,
        )
        await engine.connect()

        # Human moves ch5_keys:vol to CC 70
        backend.inject_cc(1, 5, 70)
        await _wait_until(lambda: len(human_events) == 1)
        seq_after_move = human_events[0]["human_seq"]
        assert seq_after_move == 1
        assert engine.human_seq["ch5_keys:vol"] == 1
        assert len(backend.sent_messages) == 1

        # 1ms after human move: agent write arrives EVEN WITH matching ack_human_seq -> DROPPED by 250ms window!
        await asyncio.sleep(0.001)
        await engine.send_agent_cc("ch5_keys", "vol", 40, ack_human_seq=seq_after_move)
        await asyncio.sleep(0.03)
        assert len(backend.sent_messages) == 1
        assert daemon.last_sent_cc["ch5_keys:vol"] == 70

        # Wait past the 250ms guard window
        await asyncio.sleep(0.25)

        # Agent write with stale ack_human_seq < current_human_seq -> DROPPED by human_seq fence!
        await engine.send_agent_cc("ch5_keys", "vol", 42, ack_human_seq=seq_after_move - 1)
        await asyncio.sleep(0.03)
        assert len(backend.sent_messages) == 1
        assert daemon.last_sent_cc["ch5_keys:vol"] == 70

        # Agent write with current ack_human_seq AFTER 250ms window -> ALLOWED!
        await engine.send_agent_cc("ch5_keys", "vol", 45, ack_human_seq=seq_after_move)
        await _wait_until(lambda: len(backend.sent_messages) == 2)
        assert (backend.sent_messages[-1].cc_num, backend.sent_messages[-1].value) == (24, 45)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 45

        await engine.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 7. Soft-Takeover pickup_catchup: Pre-Pickup Touch Lock + ±2 CC Crossover
# ---------------------------------------------------------------------------


async def test_pickup_catchup_pre_pickup_touch_lock_and_crossover(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict(takeover_mode="pickup_catchup")),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
        race_guard_ms=250.0,
    )
    await daemon.start()
    try:
        touches: list[dict[str, Any]] = []
        cc_events: list[dict[str, Any]] = []
        engine = _make_client(
            daemon,
            sock,
            token="tok",
            role="engine",
            on_human_touch=touches.append,
            on_human_cc=cc_events.append,
        )
        await engine.connect()

        # Agent moves ch5_keys:vol to CC 80
        await engine.send_agent_cc("ch5_keys", "vol", 80, ack_human_seq=0)
        await _wait_until(lambda: len(backend.sent_messages) == 1)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 80

        # Physical fader moves at CC 20 (below 80): must emit human_touch_event(picked_up=False),
        # bump human_seq, engage 250ms guard, set pickup_pending=+1, and NOT transmit MIDI!
        backend.inject_cc(1, 5, 20)
        await _wait_until(lambda: len(touches) == 1)
        assert touches[0]["channel_id"] == "ch5_keys"
        assert touches[0]["control"] == "vol"
        assert touches[0]["picked_up"] is False
        assert touches[0]["source"] == "hardware"
        assert touches[0]["human_seq"] == 1
        assert daemon.pickup_pending["ch5_keys:vol"] == 1
        assert len(backend.sent_messages) == 1  # No fader jump to 20!
        assert len(cc_events) == 0

        # Agent tries to keep gliding while operator is approaching pickup -> MUST BE BLOCKED!
        await engine.send_agent_cc("ch5_keys", "vol", 85, ack_human_seq=0)
        await engine.send_agent_cc("ch5_keys", "vol", 86, ack_human_seq=1)
        await asyncio.sleep(0.03)
        assert len(backend.sent_messages) == 1
        assert daemon.last_sent_cc["ch5_keys:vol"] == 80

        # Physical fader moves closer (CC 55) -> still pre-pickup (+1)
        backend.inject_cc(1, 5, 55)
        await _wait_until(lambda: len(touches) == 2)
        assert touches[1]["picked_up"] is False
        assert touches[1]["human_seq"] == 2
        assert daemon.pickup_pending["ch5_keys:vol"] == 1
        assert len(backend.sent_messages) == 1

        # Physical fader enters ±2 CC window (CC 79 vs 80) -> picks up and transmits!
        backend.inject_cc(1, 5, 79)
        await _wait_until(lambda: len(cc_events) == 1)
        assert touches[-1]["picked_up"] is True
        assert cc_events[0]["cc"] == 79
        assert cc_events[0]["human_seq"] == 3
        assert daemon.pickup_pending["ch5_keys:vol"] == 0
        assert len(backend.sent_messages) == 2
        assert backend.sent_messages[-1].value == 79

        # Now test negative pickup direction (-1) when physical fader starts ABOVE last_sent_cc
        await asyncio.sleep(0.27)
        await engine.send_agent_cc("ch5_keys", "vol", 50, ack_human_seq=3)
        await _wait_until(lambda: daemon.last_sent_cc["ch5_keys:vol"] == 50)

        backend.inject_cc(1, 5, 100)
        await _wait_until(lambda: daemon.pickup_pending.get("ch5_keys:vol") == -1)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 50  # Still held at 50

        # Fast sweep across 50 from 100 -> 40 crosses ref and picks up immediately!
        backend.inject_cc(1, 5, 40)
        await _wait_until(lambda: daemon.last_sent_cc["ch5_keys:vol"] == 40)
        assert daemon.pickup_pending["ch5_keys:vol"] == 0

        await engine.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 8. Soft-Takeover Modes: immediate & relative_encoder
# ---------------------------------------------------------------------------


async def test_takeover_modes_immediate_and_relative_encoder(short_tmp_dir: Path) -> None:
    # Mode A: immediate
    backend_imm = MockMidiBackend()
    daemon_imm = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict(takeover_mode="immediate")),
        socket_path=short_tmp_dir / "imm.sock",
        token="tok",
        state_path=short_tmp_dir / "imm_state.json",
        midi_backend=backend_imm,
    )
    await daemon_imm.start()
    try:
        daemon_imm.last_sent_cc["ch5_keys:vol"] = 100
        backend_imm.inject_cc(1, 5, 15)
        assert daemon_imm.last_sent_cc["ch5_keys:vol"] == 15
        assert backend_imm.sent_messages[-1].value == 15
    finally:
        await daemon_imm.stop()

    # Mode B: relative_encoder
    backend_rel = MockMidiBackend()
    daemon_rel = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict(takeover_mode="relative_encoder")),
        socket_path=short_tmp_dir / "rel.sock",
        token="tok",
        state_path=short_tmp_dir / "rel_state.json",
        midi_backend=backend_rel,
    )
    await daemon_rel.start()
    try:
        daemon_rel.last_sent_cc["ch5_keys:vol"] = 60
        backend_rel.inject_cc(1, 5, 4)  # +4 -> 64
        assert daemon_rel.last_sent_cc["ch5_keys:vol"] == 64
        backend_rel.inject_cc(1, 5, 126)  # -2 -> 62
        assert daemon_rel.last_sent_cc["ch5_keys:vol"] == 62
        backend_rel.inject_cc(1, 5, 100)  # -28 -> 34
        assert daemon_rel.last_sent_cc["ch5_keys:vol"] == 34
    finally:
        await daemon_rel.stop()


# ---------------------------------------------------------------------------
# 9. Dead-Engine 500ms Watchdog & EOF Hold (NEVER Zero MIDI)
# ---------------------------------------------------------------------------


async def test_dead_engine_500ms_watchdog_and_eof_hold_without_zeroing(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
        engine_timeout_ms=500.0,
    )
    await daemon.start()
    try:
        ui_states: list[BridgeStatePayload] = []
        ui = _make_client(daemon, sock, token="tok", role="ui", on_state_change=ui_states.append)
        engine = _make_client(daemon, sock, token="tok", role="engine")

        await ui.connect()
        await engine.connect()
        await _wait_until(lambda: ui.state.engine_connected is True)

        # Engine writes CC 96 to ch1_pastor:vol and sends heartbeat
        await engine.send_agent_cc("ch1_pastor", "vol", 96, ack_human_seq=0)
        await engine.send_heartbeat()
        await _wait_until(lambda: len(backend.sent_messages) == 1)
        assert daemon.last_sent_cc["ch1_pastor:vol"] == 96

        # Engine goes completely silent for >500ms -> dead-engine watchdog trips!
        await _wait_until(lambda: ui.state.engine_connected is False, timeout=1.5)
        assert daemon.engine_connected is False
        assert daemon.last_sent_cc["ch1_pastor:vol"] == 96
        # Verify ZERO zero-CC messages were sent!
        assert all(m.value != 0 for m in backend.sent_messages)
        assert len(backend.sent_messages) == 1

        # Reconnect fresh engine, then drop via EOF -> immediate engine_connected=False, still no zeros!
        await engine.close()
        engine2 = _make_client(daemon, sock, token="tok", role="engine")
        await engine2.connect()
        await _wait_until(lambda: ui.state.engine_connected is True)

        await engine2.close()
        await _wait_until(lambda: ui.state.engine_connected is False, timeout=1.0)
        assert daemon.last_sent_cc["ch1_pastor:vol"] == 96
        assert len(backend.sent_messages) == 1

        await ui.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 10. UI Direct Manual CC & set_bypass While Engine Disconnected
# ---------------------------------------------------------------------------


async def test_ui_direct_manual_cc_and_set_bypass_while_engine_disconnected(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
        race_guard_ms=100.0,
    )
    await daemon.start()
    try:
        ui = _make_client(daemon, sock, token="tok", role="ui")
        await ui.connect()
        assert ui.state.engine_connected is False

        # Direct UI manual CC works with NO engine connected
        await ui.send_ui_manual_cc("ch5_keys", "vol", 72)
        await _wait_until(lambda: len(backend.sent_messages) == 1)
        assert (backend.sent_messages[-1].midi_channel, backend.sent_messages[-1].cc_num, backend.sent_messages[-1].value) == (
            1,
            24,
            72,
        )
        fresh = await ui.get_state()
        assert fresh.last_sent_cc["ch5_keys:vol"] == 72
        assert fresh.human_seq["ch5_keys:vol"] == 1

        # UI latches bypass while engine is disconnected
        await ui.send_set_bypass(True)
        await _wait_until(lambda: ui.state.bypass_latched is True)

        # Engine connects later; its writes are dropped while bypass_latched=True
        engine = _make_client(daemon, sock, token="tok", role="engine")
        await engine.connect()
        assert engine.state.bypass_latched is True
        await asyncio.sleep(0.12)
        await engine.send_agent_cc("ch5_keys", "vol", 30, ack_human_seq=1)
        await asyncio.sleep(0.03)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 72

        # UI manual CC STILL works while bypass_latched=True!
        await ui.send_ui_manual_cc("ch5_keys", "vol", 84)
        await _wait_until(lambda: daemon.last_sent_cc["ch5_keys:vol"] == 84)

        # UI unlatches bypass; after race guard expires, agent writes resume
        await ui.send_set_bypass(False)
        await _wait_until(lambda: engine.state.bypass_latched is False)
        await asyncio.sleep(0.12)
        await engine.send_agent_cc("ch5_keys", "vol", 60, ack_human_seq=engine.human_seq["ch5_keys:vol"])
        await _wait_until(lambda: daemon.last_sent_cc["ch5_keys:vol"] == 60)

        await engine.close()
        await ui.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 11. Atomic Disk Persistence (bridge_state.json) & Restart Recovery
# ---------------------------------------------------------------------------


async def test_atomic_bridge_state_persistence_and_restart_recovery(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"

    # First run: cold boot -> restored_from_disk_at is None
    backend1 = MockMidiBackend()
    daemon1 = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend1,
        persist_debounce_ms=100.0,
    )
    await daemon1.start()
    assert daemon1.restored_from_disk_at is None

    # Trigger CC writes and wait for 100ms debounced atomic flush
    backend1.inject_cc(1, 1, 96)  # ch1_pastor:vol = 96
    backend1.inject_cc(1, 5, 64)  # ch5_keys:vol = 64
    await _wait_until(lambda: state_file.exists(), timeout=1.0)

    # Verify no leftover .tmp files and valid JSON payload with 0600 perms
    leftovers = list(short_tmp_dir.glob("*.tmp*"))
    assert leftovers == []
    assert stat.S_IMODE(os.stat(state_file).st_mode) == 0o600
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["last_sent_cc"]["ch1_pastor:vol"] == 96
    assert saved["last_sent_cc"]["ch5_keys:vol"] == 64

    # Write another CC and immediately stop() before 100ms debounce fires -> must flush immediately!
    backend1.inject_cc(1, 2, 88)  # ch2_lead_vox:vol = 88
    msg_count_before_stop = len(backend1.sent_messages)
    await daemon1.stop()
    # Shutdown must NEVER emit zero MIDI CCs
    assert len(backend1.sent_messages) == msg_count_before_stop

    # Second run: restart daemon pointing at same state_file
    backend2 = MockMidiBackend()
    daemon2 = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend2,
    )
    await daemon2.start()
    try:
        assert isinstance(daemon2.restored_from_disk_at, int)
        assert daemon2.restored_from_disk_at > 0

        engine = _make_client(daemon2, sock, token="tok", role="engine")
        hydrated = await engine.connect()
        assert hydrated.restored_from_disk_at == daemon2.restored_from_disk_at
        assert hydrated.last_sent_cc == {
            "ch1_pastor:vol": 96,
            "ch5_keys:vol": 64,
            "ch2_lead_vox:vol": 88,
        }

        # Restored controls must enter pickup_catchup (physical fader at 10 must NOT jump pastor from 96!)
        backend2.inject_cc(1, 1, 10)
        assert daemon2.last_sent_cc["ch1_pastor:vol"] == 96
        assert daemon2.pickup_pending["ch1_pastor:vol"] == 1
        assert len(backend2.sent_messages) == 0

        await engine.close()
    finally:
        await daemon2.stop()


# ---------------------------------------------------------------------------
# 12. Change Deduplication, midi_learn_pulse, and spoken_primary Mute Guard
# ---------------------------------------------------------------------------


async def test_change_deduplication_midi_learn_pulse_and_spoken_mute_guard(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    backend = MockMidiBackend()
    plan = _load_plan_dict()
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(plan),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
    )
    await daemon.start()
    try:
        engine = _make_client(daemon, sock, token="tok", role="engine")
        await engine.connect()

        # First write sends MIDI; duplicate identical CC is deduplicated
        await engine.send_agent_cc("ch1_pastor", "vol", 96, ack_human_seq=0)
        await _wait_until(lambda: len(backend.sent_messages) == 1)
        await engine.send_agent_cc("ch1_pastor", "vol", 96, ack_human_seq=0)
        await asyncio.sleep(0.03)
        assert len(backend.sent_messages) == 1

        # midi_learn_pulse bypasses deduplication and emits a MIDI CC pulse
        await engine.send_midi_learn_pulse("ch1_pastor", "vol")
        await _wait_until(lambda: len(backend.sent_messages) == 2)
        assert (backend.sent_messages[-1].midi_channel, backend.sent_messages[-1].cc_num) == (1, 20)

        # Agent mute on spoken_primary must never transmit even if forcefully targeted
        await engine.send_agent_cc("ch1_pastor", "mute", 127, ack_human_seq=0)
        await asyncio.sleep(0.03)
        assert len(backend.sent_messages) == 2

        await engine.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 13. USB-MIDI Hot-Plug Polling & virtual_port vs iac_bus Support
# ---------------------------------------------------------------------------


async def test_usb_hotplug_reattach_in_pickup_and_iac_bus_mode(short_tmp_dir: Path) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"
    # Start with controller unplugged and iac_bus output mode
    backend = MockMidiBackend(
        input_ports=[],
        output_ports=["IAC Driver AetherMix Bus 1"],
    )
    plan = _load_plan_dict(
        mainstage_out_mode="iac_bus",
        mainstage_out_port="IAC Driver AetherMix Bus 1",
    )
    daemon = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(plan),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend,
        usb_poll_interval_s=0.05,
    )
    await daemon.start()
    try:
        assert backend.opened_output == ("IAC Driver AetherMix Bus 1", "iac_bus")
        assert daemon.controller_connected is False

        ui = _make_client(daemon, sock, token="tok", role="ui")
        await ui.connect()
        assert ui.state.controller_connected is False

        # Plug in controller -> hot-plug poll connects it
        backend.input_ports.append("X-Touch Mini")
        await _wait_until(lambda: ui.state.controller_connected is True, timeout=1.0)
        backend.inject_cc(1, 5, 90)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 90

        # Unplug controller -> hot-plug poll detects disconnect, UI direct fader still works
        backend.input_ports.clear()
        await _wait_until(lambda: ui.state.controller_connected is False, timeout=1.0)
        await ui.send_ui_manual_cc("ch5_keys", "vol", 40)
        await _wait_until(lambda: daemon.last_sent_cc["ch5_keys:vol"] == 40)

        # Replug controller -> re-attaches in pickup_catchup mode (physical fader at 90 does NOT jump from 40!)
        backend.input_ports.append("X-Touch Mini")
        await _wait_until(lambda: ui.state.controller_connected is True, timeout=1.0)
        backend.inject_cc(1, 5, 90)
        assert daemon.last_sent_cc["ch5_keys:vol"] == 40
        assert daemon.pickup_pending["ch5_keys:vol"] == -1

        await ui.close()
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# 14. Regression Test (Defect #5): BridgeClient Auto-Reconnect on Bridge Restart
# ---------------------------------------------------------------------------


async def test_bridge_client_auto_reconnect_after_bridge_restart_delivers_restored_from_disk_at(
    short_tmp_dir: Path,
) -> None:
    sock = short_tmp_dir / "bridge.sock"
    state_file = short_tmp_dir / "bridge_state.json"

    backend1 = MockMidiBackend()
    daemon1 = MidiBridgeDaemon(
        BridgeConfig.from_service_plan_dict(_load_plan_dict()),
        socket_path=sock,
        token="tok",
        state_path=state_file,
        midi_backend=backend1,
    )
    await daemon1.start()

    active_daemon: list[MidiBridgeDaemon | None] = [daemon1]

    async def _dynamic_opener() -> tuple[asyncio.StreamReader, Any]:
        current = active_daemon[0]
        if current is None:
            raise ConnectionRefusedError("Bridge daemon is currently stopped")
        return await _open_bridge_stream(current, sock)

    disconnected: list[bool] = []
    reconnected_states: list[BridgeStatePayload] = []

    engine = BridgeClient(
        socket_path=sock,
        token="tok",
        role="engine",
        stream_opener=_dynamic_opener,
        auto_reconnect=True,
        reconnect_interval_s=0.05,
        on_disconnect=lambda: disconnected.append(True),
        on_reconnect=reconnected_states.append,
    )

    daemon2: MidiBridgeDaemon | None = None
    try:
        initial = await engine.connect()
        assert engine.connected is True
        assert initial.restored_from_disk_at is None

        # Write state through daemon1 before simulated crash/restart
        await engine.send_agent_cc("ch1_pastor", "vol", 96, ack_human_seq=0)
        await engine.send_agent_cc("ch5_keys", "vol", 64, ack_human_seq=0)
        await _wait_until(lambda: len(backend1.sent_messages) == 2)

        # Simulate bridge daemon crash/stop (launchd restarts it shortly after)
        active_daemon[0] = None
        await daemon1.stop()

        await _wait_until(lambda: engine.connected is False and len(disconnected) == 1, timeout=1.0)

        # Stay down briefly so reconnect loop polls while bridge is unavailable
        await asyncio.sleep(0.12)
        assert engine.connected is False
        assert len(reconnected_states) == 0

        # Start fresh daemon2 on the same socket & persisted state_file
        backend2 = MockMidiBackend()
        daemon2 = MidiBridgeDaemon(
            BridgeConfig.from_service_plan_dict(_load_plan_dict()),
            socket_path=sock,
            token="tok",
            state_path=state_file,
            midi_backend=backend2,
        )
        await daemon2.start()
        active_daemon[0] = daemon2

        # Client must automatically reconnect, re-send hello + get_bridge_state, and deliver restored_from_disk_at!
        await _wait_until(
            lambda: engine.connected is True and len(reconnected_states) == 1,
            timeout=2.0,
        )
        rehydrated = reconnected_states[0]
        assert rehydrated.restored_from_disk_at is not None
        assert rehydrated.restored_from_disk_at > 0
        assert rehydrated.last_sent_cc == {"ch1_pastor:vol": 96, "ch5_keys:vol": 64}
        assert engine.state.restored_from_disk_at == rehydrated.restored_from_disk_at

        # Verify live IPC works again immediately after reconnect
        await engine.send_heartbeat()
        await engine.send_agent_cc("ch5_keys", "vol", 72, ack_human_seq=0)
        await _wait_until(lambda: daemon2.last_sent_cc.get("ch5_keys:vol") == 72, timeout=1.0)
        assert daemon2.engine_connected is True

        # Explicit close() must disable auto-reconnect
        await engine.close()
        assert engine.connected is False
        await asyncio.sleep(0.12)
        assert engine.connected is False
    finally:
        await engine.close()
        if daemon2 is not None:
            await daemon2.stop()
        else:
            await daemon1.stop()
