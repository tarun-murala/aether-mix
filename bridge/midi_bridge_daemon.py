"""Standalone zero-ML MIDI Bridge daemon for AetherMix (SPEC v5.1).

Design invariants:
- Strictly Python stdlib + optional python-rtmidi ONLY (zero ML/heavy imports).
- Dual concurrent NDJSON AF_UNIX server (0600) for role="engine" and role="ui",
  authenticated with ~/.aethermix/session.token and protocol_version="5.1".
- Always-on <2ms human passthrough (hardware fader_map + UI ui_manual_cc) even
  when the AI engine is dead, disconnected, or emergency_bypass is latched.
- Bridge-level in-flight race guard: per-control monotonic human_seq fence +
  250ms drop window for agent_cc_write.
- Soft-takeover (pickup_catchup, immediate, relative_encoder) with pre-pickup
  human_touch_event so the engine freezes glides before ±2 CC crossover.
- Dead-engine watchdog (>500ms silence or EOF) sets engine_connected=False and
  holds last_sent_cc without ever zeroing MIDI outputs.
- Atomic disk persistence (tmp + fsync + os.replace) of last_sent_cc debounced
  100ms on change and flushed immediately on SIGTERM/shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import secrets
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

try:
    import rtmidi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional in headless environments
    rtmidi = None

PROTOCOL_VERSION = "5.1"
SCHEMA_VERSION = 5

DEFAULT_AETHERMIX_DIR = Path.home() / ".aethermix"
DEFAULT_SOCKET_PATH = DEFAULT_AETHERMIX_DIR / "bridge.sock"
DEFAULT_TOKEN_PATH = DEFAULT_AETHERMIX_DIR / "session.token"
DEFAULT_STATE_PATH = DEFAULT_AETHERMIX_DIR / "bridge_state.json"

logger = logging.getLogger("aethermix.midi_bridge")


def _now_ms() -> int:
    return int(time.time() * 1000)


def ensure_session_token(token_path: Path = DEFAULT_TOKEN_PATH) -> str:
    """Create (0600) or load the shared local session token."""
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if token_path.exists():
        tok = token_path.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    tmp_path = token_path.with_suffix(f".tmp.{os.getpid()}")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (tok + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, token_path)
    os.chmod(token_path, 0o600)
    return tok


def control_key(channel_id: str, control: str) -> str:
    return f"{channel_id}:{control}"


# ---------------------------------------------------------------------------
# MIDI Backend Abstraction (RtMidi + Headless Mock)
# ---------------------------------------------------------------------------


class MidiBackend(Protocol):
    def list_input_ports(self) -> list[str]: ...

    def list_output_ports(self) -> list[str]: ...

    def open_output(self, port_name: str, mode: str) -> bool: ...

    def open_input(self, port_name: str, callback: Callable[[list[int]], None]) -> bool: ...

    def close_input(self) -> None: ...

    def send_cc(self, midi_channel: int, cc_num: int, value: int) -> None: ...

    def close(self) -> None: ...


class RtMidiBackend:
    """Real CoreMIDI / python-rtmidi backend."""

    def __init__(self) -> None:
        self._midi_in: Any = None
        self._midi_out: Any = None
        self._in_open = False
        self._out_open = False

    def _require_rtmidi(self) -> Any:
        if rtmidi is None:
            raise RuntimeError("python-rtmidi is not installed; use MockMidiBackend or --headless")
        return rtmidi

    def list_input_ports(self) -> list[str]:
        if rtmidi is None:
            return []
        probe = rtmidi.MidiIn()
        try:
            return list(probe.get_ports())
        finally:
            probe.delete()

    def list_output_ports(self) -> list[str]:
        if rtmidi is None:
            return []
        probe = rtmidi.MidiOut()
        try:
            return list(probe.get_ports())
        finally:
            probe.delete()

    def open_output(self, port_name: str, mode: str) -> bool:
        mod = self._require_rtmidi()
        if self._midi_out is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self._midi_out.close_port()
            self._midi_out = None
            self._out_open = False

        out = mod.MidiOut()
        if mode == "virtual_port":
            out.open_virtual_port(port_name)
            self._midi_out = out
            self._out_open = True
            return True

        # mode == "iac_bus": attach to existing OS-owned IAC Driver bus
        ports = list(out.get_ports())
        target_idx: int | None = None
        needle = port_name.lower()
        for idx, name in enumerate(ports):
            if name.lower() == needle or needle in name.lower():
                target_idx = idx
                break
        if target_idx is None:
            out.delete()
            return False
        out.open_port(target_idx)
        self._midi_out = out
        self._out_open = True
        return True

    def open_input(self, port_name: str, callback: Callable[[list[int]], None]) -> bool:
        if rtmidi is None:
            return False
        self.close_input()
        midi_in = rtmidi.MidiIn()
        ports = list(midi_in.get_ports())
        needle = port_name.lower()
        target_idx: int | None = None
        for idx, name in enumerate(ports):
            if name.lower() == needle or needle in name.lower():
                target_idx = idx
                break
        if target_idx is None:
            midi_in.delete()
            return False

        def _rt_cb(event: tuple[list[int], float], _data: Any = None) -> None:
            msg, _delta = event
            if msg:
                callback(list(msg))

        midi_in.open_port(target_idx)
        midi_in.ignore_types(sysex=True, timing=True, active_sense=True)
        midi_in.set_callback(_rt_cb)
        self._midi_in = midi_in
        self._in_open = True
        return True

    def close_input(self) -> None:
        if self._midi_in is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self._midi_in.cancel_callback()
                self._midi_in.close_port()
            self._midi_in = None
        self._in_open = False

    def send_cc(self, midi_channel: int, cc_num: int, value: int) -> None:
        if not self._out_open or self._midi_out is None:
            return
        status = 0xB0 | ((int(midi_channel) - 1) & 0x0F)
        self._midi_out.send_message([status, int(cc_num) & 0x7F, int(value) & 0x7F])

    def close(self) -> None:
        self.close_input()
        if self._midi_out is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self._midi_out.close_port()
            self._midi_out = None
        self._out_open = False


@dataclass
class SentMidiCC:
    midi_channel: int
    cc_num: int
    value: int
    ts_mono: float


class MockMidiBackend:
    """Deterministic in-memory MIDI backend for headless testing & simulation."""

    def __init__(
        self,
        input_ports: list[str] | None = None,
        output_ports: list[str] | None = None,
    ) -> None:
        self.input_ports: list[str] = list(input_ports) if input_ports is not None else ["X-Touch Mini"]
        self.output_ports: list[str] = (
            list(output_ports) if output_ports is not None else ["IAC Driver AetherMix Bus 1"]
        )
        self.opened_output: tuple[str, str] | None = None
        self.opened_input: str | None = None
        self.sent_messages: list[SentMidiCC] = []
        self._input_callback: Callable[[list[int]], None] | None = None
        self.closed: bool = False

    def list_input_ports(self) -> list[str]:
        return list(self.input_ports)

    def list_output_ports(self) -> list[str]:
        return list(self.output_ports)

    def open_output(self, port_name: str, mode: str) -> bool:
        if mode == "virtual_port":
            self.opened_output = (port_name, mode)
            return True
        needle = port_name.lower()
        for name in self.output_ports:
            if name.lower() == needle or needle in name.lower():
                self.opened_output = (name, mode)
                return True
        return False

    def open_input(self, port_name: str, callback: Callable[[list[int]], None]) -> bool:
        needle = port_name.lower()
        for name in self.input_ports:
            if name.lower() == needle or needle in name.lower():
                self.opened_input = name
                self._input_callback = callback
                return True
        return False

    def close_input(self) -> None:
        self.opened_input = None
        self._input_callback = None

    def send_cc(self, midi_channel: int, cc_num: int, value: int) -> None:
        self.sent_messages.append(
            SentMidiCC(
                midi_channel=int(midi_channel),
                cc_num=int(cc_num) & 0x7F,
                value=int(value) & 0x7F,
                ts_mono=time.monotonic(),
            )
        )

    def inject_cc(self, hw_channel: int, hw_cc: int, value: int) -> None:
        """Simulate an incoming hardware controller MIDI CC message."""
        if self._input_callback is None:
            raise RuntimeError("MockMidiBackend has no active input listener")
        status = 0xB0 | ((int(hw_channel) - 1) & 0x0F)
        self._input_callback([status, int(hw_cc) & 0x7F, int(value) & 0x7F])

    def close(self) -> None:
        self.close_input()
        self.opened_output = None
        self.closed = True


# ---------------------------------------------------------------------------
# Bridge Configuration
# ---------------------------------------------------------------------------


@dataclass
class TargetControlSpec:
    channel_id: str
    control: str
    midi_channel: int
    target_cc: int
    role: str


@dataclass
class BridgeConfig:
    hardware_in_port: str | None = None
    takeover_mode: str = "pickup_catchup"
    mainstage_out_mode: str = "virtual_port"
    mainstage_out_port: str = "AetherMix To MainStage"
    mainstage_feedback_in_port: str | None = None
    # (hw_channel, hw_cc) -> (channel_id, control)
    fader_map: dict[tuple[int, int], tuple[str, str]] = field(default_factory=dict)
    # (hw_channel, hw_cc) -> action_name ("thumbs_up" | "thumbs_down" | "emergency_bypass")
    action_bindings: dict[tuple[int, int], str] = field(default_factory=dict)
    # "channel_id:control" -> TargetControlSpec
    targets: dict[str, TargetControlSpec] = field(default_factory=dict)

    @classmethod
    def from_service_plan_dict(cls, data: dict[str, Any]) -> BridgeConfig:
        midi = data.get("midi", {})
        fader_map: dict[tuple[int, int], tuple[str, str]] = {}
        for entry in midi.get("fader_map", []):
            hw_ch = int(entry["hw_channel"])
            hw_cc = int(entry["hw_cc"])
            fader_map[(hw_ch, hw_cc)] = (str(entry["channel_id"]), str(entry.get("control", "vol")))

        action_bindings: dict[tuple[int, int], str] = {}
        raw_actions = midi.get("action_bindings") or {}
        for action_name in ("thumbs_up", "thumbs_down", "emergency_bypass"):
            binding = raw_actions.get(action_name)
            if binding:
                action_bindings[(int(binding["hw_channel"]), int(binding["hw_cc"]))] = action_name

        targets: dict[str, TargetControlSpec] = {}
        for ch in data.get("channels", []):
            ch_id = str(ch["id"])
            midi_ch = int(ch.get("midi_channel", 1))
            role = str(ch.get("role", ""))
            if ch.get("vol_cc") is not None:
                targets[control_key(ch_id, "vol")] = TargetControlSpec(
                    channel_id=ch_id,
                    control="vol",
                    midi_channel=midi_ch,
                    target_cc=int(ch["vol_cc"]),
                    role=role,
                )
            if ch.get("reverb_cc") is not None:
                targets[control_key(ch_id, "reverb")] = TargetControlSpec(
                    channel_id=ch_id,
                    control="reverb",
                    midi_channel=midi_ch,
                    target_cc=int(ch["reverb_cc"]),
                    role=role,
                )
            if ch.get("mute_cc") is not None:
                targets[control_key(ch_id, "mute")] = TargetControlSpec(
                    channel_id=ch_id,
                    control="mute",
                    midi_channel=midi_ch,
                    target_cc=int(ch["mute_cc"]),
                    role=role,
                )

        takeover_mode = str(midi.get("takeover_mode", "pickup_catchup"))
        if takeover_mode not in ("pickup_catchup", "immediate", "relative_encoder"):
            raise ValueError(f"Unsupported takeover_mode: {takeover_mode}")
        out_mode = str(midi.get("mainstage_out_mode", "virtual_port"))
        if out_mode not in ("virtual_port", "iac_bus"):
            raise ValueError(f"Unsupported mainstage_out_mode: {out_mode}")

        return cls(
            hardware_in_port=midi.get("hardware_in_port"),
            takeover_mode=takeover_mode,
            mainstage_out_mode=out_mode,
            mainstage_out_port=str(midi.get("mainstage_out_port", "AetherMix To MainStage")),
            mainstage_feedback_in_port=midi.get("mainstage_feedback_in_port"),
            fader_map=fader_map,
            action_bindings=action_bindings,
            targets=targets,
        )

    @classmethod
    def from_service_plan_file(cls, path: Path) -> BridgeConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_service_plan_dict(raw)


# ---------------------------------------------------------------------------
# Connected Client Session
# ---------------------------------------------------------------------------


@dataclass
class _ClientConn:
    client_id: int
    role: str  # "engine" | "ui"
    reader: asyncio.StreamReader
    writer: Any  # asyncio.StreamWriter or compatible loopback writer


class _LoopbackWriter:
    """In-process asyncio.StreamWriter-compatible peer for sandboxed/headless tests."""

    def __init__(self, peer_reader: asyncio.StreamReader, peer_closed_ref: list[bool]) -> None:
        self._peer_reader = peer_reader
        self._peer_closed_ref = peer_closed_ref
        self._closing = False

    def is_closing(self) -> bool:
        return self._closing

    def write(self, data: bytes) -> None:
        if self._closing or self._peer_closed_ref[0]:
            raise BrokenPipeError("Loopback stream is closed")
        self._peer_reader.feed_data(data)

    async def drain(self) -> None:
        if self._closing:
            raise BrokenPipeError("Loopback stream is closed")
        await asyncio.sleep(0)

    def close(self) -> None:
        if not self._closing:
            self._closing = True
            self._peer_reader.feed_eof()

    async def wait_closed(self) -> None:
        await asyncio.sleep(0)


def _create_loopback_stream_pair() -> tuple[
    tuple[asyncio.StreamReader, _LoopbackWriter],
    tuple[asyncio.StreamReader, _LoopbackWriter],
]:
    c2s_reader = asyncio.StreamReader()
    s2c_reader = asyncio.StreamReader()
    client_closed = [False]
    server_closed = [False]

    class _TrackedLoopbackWriter(_LoopbackWriter):
        def __init__(
            self,
            peer_reader: asyncio.StreamReader,
            own_closed_ref: list[bool],
            peer_closed_ref: list[bool],
        ) -> None:
            super().__init__(peer_reader, peer_closed_ref)
            self._own_closed_ref = own_closed_ref

        def close(self) -> None:
            self._own_closed_ref[0] = True
            super().close()

    client_writer = _TrackedLoopbackWriter(c2s_reader, client_closed, server_closed)
    server_writer = _TrackedLoopbackWriter(s2c_reader, server_closed, client_closed)
    return (s2c_reader, client_writer), (c2s_reader, server_writer)


# ---------------------------------------------------------------------------
# Standalone MIDI Bridge Daemon
# ---------------------------------------------------------------------------


class MidiBridgeDaemon:
    """Zero-ML standalone MIDI router, takeover engine, race guard, and AF_UNIX server."""

    def __init__(
        self,
        config: BridgeConfig,
        *,
        socket_path: Path = DEFAULT_SOCKET_PATH,
        token: str | None = None,
        token_path: Path = DEFAULT_TOKEN_PATH,
        state_path: Path = DEFAULT_STATE_PATH,
        midi_backend: MidiBackend | None = None,
        race_guard_ms: float = 250.0,
        engine_timeout_ms: float = 500.0,
        persist_debounce_ms: float = 100.0,
        usb_poll_interval_s: float = 1.0,
        pickup_tolerance_cc: int = 2,
    ) -> None:
        self.config = config
        self.socket_path = Path(socket_path)
        self.token_path = Path(token_path)
        self.token = token if token is not None else ensure_session_token(self.token_path)
        self.state_path = Path(state_path)
        self.midi_backend: MidiBackend = midi_backend if midi_backend is not None else RtMidiBackend()

        self.race_guard_ms = float(race_guard_ms)
        self.engine_timeout_ms = float(engine_timeout_ms)
        self.persist_debounce_ms = float(persist_debounce_ms)
        self.usb_poll_interval_s = float(usb_poll_interval_s)
        self.pickup_tolerance_cc = int(pickup_tolerance_cc)

        # Authoritative bridge state
        self.last_sent_cc: dict[str, int] = {}
        self.human_seq: dict[str, int] = {}
        self.bypass_latched: bool = False
        self.controller_connected: bool = False
        self.engine_connected: bool = False
        self.pickup_pending: dict[str, int] = {}
        self.restored_from_disk_at: int | None = None

        # Internal timing & soft-takeover state
        self._last_human_touch_mono: dict[str, float] = {}
        self._hw_picked_up: dict[str, bool] = {}
        self._last_hw_cc: dict[str, int] = {}
        self._prev_unpicked_hw_cc: dict[str, int] = {}
        self._last_engine_activity_mono: float = 0.0
        self._output_ready: bool = False

        # Disk persistence debounce state
        self._state_dirty: bool = False
        self._persist_handle: asyncio.TimerHandle | None = None

        # Asyncio server & background tasks
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._server: asyncio.AbstractServer | None = None
        self._clients: dict[int, _ClientConn] = {}
        self._engine_client_id: int | None = None
        self._next_client_id: int = 1
        self._watchdog_task: asyncio.Task[None] | None = None
        self._usb_poll_task: asyncio.Task[None] | None = None
        self._inprocess_tasks: set[asyncio.Task[None]] = set()
        self._running: bool = False

        # Load persisted state immediately on construction
        self._load_persisted_state()

    # ------------------------------------------------------------------
    # State Snapshot & Atomic Persistence
    # ------------------------------------------------------------------

    def build_state_message(self) -> dict[str, Any]:
        return {
            "type": "bridge_state",
            "last_sent_cc": dict(self.last_sent_cc),
            "human_seq": dict(self.human_seq),
            "bypass_latched": self.bypass_latched,
            "controller_connected": self.controller_connected,
            "engine_connected": self.engine_connected,
            "pickup_pending": dict(self.pickup_pending),
            "restored_from_disk_at": self.restored_from_disk_at,
            "ts_ms": _now_ms(),
        }

    def _load_persisted_state(self) -> None:
        if not self.state_path.exists():
            self.restored_from_disk_at = None
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to read persisted bridge state %s: %s", self.state_path, exc)
            self.restored_from_disk_at = None
            return

        saved_cc = raw.get("last_sent_cc")
        if isinstance(saved_cc, dict):
            restored: dict[str, int] = {}
            for key, val in saved_cc.items():
                if isinstance(key, str) and isinstance(val, (int, float)):
                    restored[key] = max(0, min(127, int(val)))
            if restored:
                self.last_sent_cc = restored
                self.restored_from_disk_at = _now_ms()
                # All restored controls require soft-takeover pickup until crossed
                self._hw_picked_up.clear()
                self._prev_unpicked_hw_cc.clear()

    def _schedule_persist(self) -> None:
        self._state_dirty = True
        if self._loop is None or self._loop.is_closed():
            return
        if self._persist_handle is not None:
            return
        delay_s = max(0.0, self.persist_debounce_ms / 1000.0)
        self._persist_handle = self._loop.call_later(delay_s, self._on_persist_timer)

    def _on_persist_timer(self) -> None:
        self._persist_handle = None
        if self._state_dirty:
            self.flush_state_to_disk()

    def flush_state_to_disk(self, *, force: bool = False) -> None:
        """Atomically persist last_sent_cc via tmp + fsync + os.replace."""
        if self._persist_handle is not None:
            self._persist_handle.cancel()
            self._persist_handle = None
        if not force and not self._state_dirty:
            return
        if not self.last_sent_cc and not self._state_dirty:
            return

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at_ms": _now_ms(),
            "last_sent_cc": dict(self.last_sent_cc),
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        tmp_path = self.state_path.with_name(f"{self.state_path.name}.tmp.{os.getpid()}")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, self.state_path)
        os.chmod(self.state_path, 0o600)
        self._state_dirty = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._running = True

        # Open MainStage MIDI output port
        try:
            self._output_ready = self.midi_backend.open_output(
                self.config.mainstage_out_port,
                self.config.mainstage_out_mode,
            )
            if not self._output_ready:
                logger.warning(
                    "MainStage output port %r (%s) not available yet; will retry",
                    self.config.mainstage_out_port,
                    self.config.mainstage_out_mode,
                )
        except (OSError, RuntimeError) as exc:
            logger.error("Error opening MIDI output port: %s", exc)
            self._output_ready = False

        # Try initial hardware controller attachment
        self._poll_midi_ports_once(broadcast_on_change=False)

        # Prepare AF_UNIX socket with 0600 permissions
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )
        os.chmod(self.socket_path, 0o600)

        # Start watchdog & 1Hz USB-MIDI hot-plug polling tasks
        self._watchdog_task = asyncio.create_task(self._engine_watchdog_loop(), name="bridge-watchdog")
        self._usb_poll_task = asyncio.create_task(self._usb_hotplug_loop(), name="bridge-usb-poll")
        logger.info("MIDI Bridge listening on %s (mode=%s)", self.socket_path, self.config.mainstage_out_mode)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        for task in (self._watchdog_task, self._usb_poll_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._watchdog_task = None
        self._usb_poll_task = None

        # Close all connected clients
        for conn in list(self._clients.values()):
            with contextlib.suppress(OSError, RuntimeError):
                conn.writer.close()
        self._clients.clear()
        self._engine_client_id = None
        self.engine_connected = False

        if self._inprocess_tasks:
            for t in list(self._inprocess_tasks):
                t.cancel()
            await asyncio.gather(*self._inprocess_tasks, return_exceptions=True)
            self._inprocess_tasks.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Flush state atomically BEFORE closing MIDI (and NEVER zero MIDI!)
        self.flush_state_to_disk(force=bool(self.last_sent_cc))

        try:
            self.midi_backend.close()
        except (OSError, RuntimeError) as exc:
            logger.warning("Error closing MIDI backend: %s", exc)

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # MIDI Transmission (with Change Deduplication)
    # ------------------------------------------------------------------

    def _transmit_cc(
        self,
        spec: TargetControlSpec,
        cc_value: int,
        *,
        force: bool = False,
        update_state: bool = True,
    ) -> bool:
        key = control_key(spec.channel_id, spec.control)
        quantized = max(0, min(127, int(cc_value)))
        prev = self.last_sent_cc.get(key)
        if not force and prev == quantized:
            return False

        self.midi_backend.send_cc(spec.midi_channel, spec.target_cc, quantized)
        if update_state:
            self.last_sent_cc[key] = quantized
            self._schedule_persist()
        return True

    # ------------------------------------------------------------------
    # Hardware MIDI Input Handling (<2ms Passthrough + Pickup + Buttons)
    # ------------------------------------------------------------------

    def _on_raw_midi(self, msg: list[int]) -> None:
        """Thread-safe entry point from RtMidi callback or MockMidiBackend."""
        if not msg:
            return
        if self._loop is None or self._loop.is_closed():
            return
        if threading.get_ident() == self._loop_thread_id:
            self.handle_hardware_midi_message(msg)
        else:
            self._loop.call_soon_threadsafe(self.handle_hardware_midi_message, msg)

    def handle_hardware_midi_message(self, msg: list[int]) -> None:
        """Process a single raw MIDI message on the event-loop thread."""
        if len(msg) < 3:
            return
        status = msg[0] & 0xF0
        if status != 0xB0:
            return  # Only Control Change messages are mapped

        hw_channel = (msg[0] & 0x0F) + 1
        hw_cc = msg[1] & 0x7F
        value = msg[2] & 0x7F
        now_mono = time.monotonic()
        ts_ms = _now_ms()

        # 1. Check action button bindings first
        action = self.config.action_bindings.get((hw_channel, hw_cc))
        if action is not None:
            if value >= 64:
                self._handle_action_button(action, ts_ms)
            return

        # 2. Check fader / encoder mapping
        mapped = self.config.fader_map.get((hw_channel, hw_cc))
        if mapped is None:
            return
        channel_id, control = mapped
        key = control_key(channel_id, control)
        spec = self.config.targets.get(key)
        if spec is None:
            return

        mode = self.config.takeover_mode
        if mode == "immediate":
            self._handle_hardware_immediate(spec, key, value, now_mono, ts_ms)
        elif mode == "relative_encoder":
            self._handle_hardware_relative(spec, key, value, now_mono, ts_ms)
        else:
            self._handle_hardware_pickup(spec, key, value, now_mono, ts_ms)

    def _handle_action_button(self, action: str, ts_ms: int) -> None:
        if action == "emergency_bypass":
            # Bridge-latched emergency bypass: engages immediately regardless of engine health
            self.bypass_latched = True
            self._broadcast(self.build_state_message())

        event = {
            "type": "action_button_event",
            "action": action,
            "ts_ms": ts_ms,
        }
        self._broadcast(event)

    def _bump_human_seq(self, key: str, now_mono: float) -> int:
        seq = self.human_seq.get(key, 0) + 1
        self.human_seq[key] = seq
        self._last_human_touch_mono[key] = now_mono
        return seq

    def _handle_hardware_immediate(
        self,
        spec: TargetControlSpec,
        key: str,
        value: int,
        now_mono: float,
        ts_ms: int,
    ) -> None:
        seq = self._bump_human_seq(key, now_mono)
        self._last_hw_cc[key] = value
        self._hw_picked_up[key] = True
        self.pickup_pending[key] = 0
        self._transmit_cc(spec, value)
        self._broadcast(
            {
                "type": "human_cc_event",
                "channel_id": spec.channel_id,
                "control": spec.control,
                "cc": value,
                "human_seq": seq,
                "source": "hardware",
                "ts_ms": ts_ms,
            }
        )

    def _handle_hardware_relative(
        self,
        spec: TargetControlSpec,
        key: str,
        raw_value: int,
        now_mono: float,
        ts_ms: int,
    ) -> None:
        # Standard 7-bit 2's complement relative encoder:
        # 1..63 -> +1..+63; 65..127 -> -(128 - raw_value); 0 or 64 -> 0
        if 1 <= raw_value <= 63:
            delta = raw_value
        elif 65 <= raw_value <= 127:
            delta = raw_value - 128
        else:
            return

        base = self.last_sent_cc.get(key, 0)
        new_cc = max(0, min(127, base + delta))
        seq = self._bump_human_seq(key, now_mono)
        self._hw_picked_up[key] = True
        self.pickup_pending[key] = 0
        self._transmit_cc(spec, new_cc)
        self._broadcast(
            {
                "type": "human_cc_event",
                "channel_id": spec.channel_id,
                "control": spec.control,
                "cc": new_cc,
                "human_seq": seq,
                "source": "hardware",
                "ts_ms": ts_ms,
            }
        )

    def _handle_hardware_pickup(
        self,
        spec: TargetControlSpec,
        key: str,
        value: int,
        now_mono: float,
        ts_ms: int,
    ) -> None:
        # ALWAYS increment human_seq and engage 250ms race guard on ANY physical movement!
        seq = self._bump_human_seq(key, now_mono)
        self._last_hw_cc[key] = value

        already_picked_up = self._hw_picked_up.get(key, False)
        has_ref = key in self.last_sent_cc

        if not has_ref:
            # No reference value yet: pick up immediately
            self._hw_picked_up[key] = True
            self._prev_unpicked_hw_cc.pop(key, None)
            prev_dir = self.pickup_pending.get(key, 0)
            self.pickup_pending[key] = 0
            self._transmit_cc(spec, value)
            self._broadcast(
                {
                    "type": "human_cc_event",
                    "channel_id": spec.channel_id,
                    "control": spec.control,
                    "cc": value,
                    "human_seq": seq,
                    "source": "hardware",
                    "ts_ms": ts_ms,
                }
            )
            if prev_dir != 0:
                self._broadcast(self.build_state_message())
            return

        if not already_picked_up:
            ref = self.last_sent_cc[key]
            prev_hw = self._prev_unpicked_hw_cc.get(key)
            within_window = abs(value - ref) <= self.pickup_tolerance_cc
            crossed = prev_hw is not None and (
                (prev_hw < ref < value) or (value < ref < prev_hw)
            )

            if not (within_window or crossed):
                # Pre-pickup movement: DO NOT send MIDI CC, emit human_touch_event immediately
                direction = 1 if value < ref else -1
                prev_dir = self.pickup_pending.get(key, 0)
                self.pickup_pending[key] = direction
                self._prev_unpicked_hw_cc[key] = value
                self._broadcast(
                    {
                        "type": "human_touch_event",
                        "channel_id": spec.channel_id,
                        "control": spec.control,
                        "human_seq": seq,
                        "picked_up": False,
                        "pickup_dir": direction,
                        "source": "hardware",
                        "ts_ms": ts_ms,
                    }
                )
                if prev_dir != direction:
                    self._broadcast(self.build_state_message())
                return

            # Crossover achieved! Transition to picked up.
            self._hw_picked_up[key] = True
            self._prev_unpicked_hw_cc.pop(key, None)
            prev_dir = self.pickup_pending.get(key, 0)
            self.pickup_pending[key] = 0
            self._broadcast(
                {
                    "type": "human_touch_event",
                    "channel_id": spec.channel_id,
                    "control": spec.control,
                    "human_seq": seq,
                    "picked_up": True,
                    "pickup_dir": 0,
                    "source": "hardware",
                    "ts_ms": ts_ms,
                }
            )
            self._transmit_cc(spec, value)
            self._broadcast(
                {
                    "type": "human_cc_event",
                    "channel_id": spec.channel_id,
                    "control": spec.control,
                    "cc": value,
                    "human_seq": seq,
                    "source": "hardware",
                    "ts_ms": ts_ms,
                }
            )
            if prev_dir != 0:
                self._broadcast(self.build_state_message())
            return

        # Already picked up: pass straight through
        self._transmit_cc(spec, value)
        self._broadcast(
            {
                "type": "human_cc_event",
                "channel_id": spec.channel_id,
                "control": spec.control,
                "cc": value,
                "human_seq": seq,
                "source": "hardware",
                "ts_ms": ts_ms,
            }
        )

    # ------------------------------------------------------------------
    # Socket Client Handling & Message Dispatch
    # ------------------------------------------------------------------

    def open_inprocess_connection(self) -> tuple[asyncio.StreamReader, Any]:
        """Open an in-process NDJSON stream pair directly attached to _handle_client."""
        if not self._running:
            raise ConnectionError("MidiBridgeDaemon is not running")
        (client_reader, client_writer), (server_reader, server_writer) = _create_loopback_stream_pair()
        task = asyncio.create_task(self._handle_client(server_reader, server_writer))
        self._inprocess_tasks.add(task)
        task.add_done_callback(self._inprocess_tasks.discard)
        return client_reader, client_writer

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: Any,
    ) -> None:
        conn: _ClientConn | None = None
        try:
            # 1. Authenticate via hello frame
            try:
                raw_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            except TimeoutError:
                self._write_json(writer, {"type": "error", "code": "HANDSHAKE_TIMEOUT", "message": "Expected hello"})
                return

            if not raw_line:
                return

            try:
                hello = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write_json(writer, {"type": "error", "code": "INVALID_JSON", "message": "Malformed hello"})
                return

            if not isinstance(hello, dict) or hello.get("type") != "hello":
                self._write_json(writer, {"type": "error", "code": "AUTH_FAILED", "message": "Expected hello frame"})
                return

            role = hello.get("role")
            tok = str(hello.get("token", ""))
            proto_ver = str(hello.get("protocol_version", ""))

            if role not in ("engine", "ui"):
                self._write_json(writer, {"type": "error", "code": "INVALID_ROLE", "message": f"Unknown role: {role}"})
                return
            if not secrets.compare_digest(tok, self.token):
                self._write_json(writer, {"type": "error", "code": "AUTH_FAILED", "message": "Invalid session token"})
                return
            if proto_ver != PROTOCOL_VERSION:
                self._write_json(
                    writer,
                    {
                        "type": "error",
                        "code": "VERSION_MISMATCH",
                        "message": f"Expected protocol_version {PROTOCOL_VERSION}, got {proto_ver}",
                    },
                )
                return

            # Register authenticated client
            client_id = self._next_client_id
            self._next_client_id += 1
            conn = _ClientConn(client_id=client_id, role=role, reader=reader, writer=writer)

            if role == "engine":
                # Replace any stale engine connection
                old_id = self._engine_client_id
                if old_id is not None and old_id in self._clients:
                    old_conn = self._clients.pop(old_id)
                    with contextlib.suppress(OSError, RuntimeError):
                        old_conn.writer.close()
                self._engine_client_id = client_id
                self._clients[client_id] = conn
                self.engine_connected = True
                self._last_engine_activity_mono = time.monotonic()
                # Broadcast updated state (engine_connected=True) to all clients including this one
                self._broadcast(self.build_state_message())
            else:
                self._clients[client_id] = conn
                # Send initial authoritative bridge state to the newly connected UI client
                self._write_json(writer, self.build_state_message())

            # 2. NDJSON message loop
            while self._running:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._write_json(writer, {"type": "error", "code": "INVALID_JSON", "message": "Malformed frame"})
                    continue
                if not isinstance(msg, dict):
                    continue
                self._dispatch_client_message(conn, msg)

        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Client handler error: %s", exc)
        finally:
            if conn is not None:
                self._clients.pop(conn.client_id, None)
                if conn.role == "engine" and self._engine_client_id == conn.client_id:
                    self._engine_client_id = None
                    if self.engine_connected:
                        self.engine_connected = False
                        # NEVER zero MIDI on engine disconnect — hold last_sent_cc!
                        self._broadcast(self.build_state_message())
            with contextlib.suppress(OSError, RuntimeError):
                writer.close()
                await writer.wait_closed()

    def _dispatch_client_message(self, conn: _ClientConn, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        now_mono = time.monotonic()

        if msg_type == "get_bridge_state":
            self._write_json(conn.writer, self.build_state_message())
            return

        if conn.role == "engine":
            if msg_type == "engine_heartbeat":
                self._last_engine_activity_mono = now_mono
                if not self.engine_connected:
                    self.engine_connected = True
                    self._broadcast(self.build_state_message())
                return

            if msg_type == "agent_cc_write":
                self._last_engine_activity_mono = now_mono
                if not self.engine_connected:
                    self.engine_connected = True
                    self._broadcast(self.build_state_message())
                self._handle_agent_cc_write(msg, now_mono)
                return

            if msg_type == "midi_learn_pulse":
                self._last_engine_activity_mono = now_mono
                self._handle_midi_learn_pulse(msg)
                return

            self._write_json(
                conn.writer,
                {"type": "error", "code": "UNKNOWN_TYPE", "message": f"Unsupported engine message: {msg_type}"},
            )
            return

        # role == "ui"
        if msg_type == "ui_manual_cc":
            self._handle_ui_manual_cc(conn, msg, now_mono)
            return

        if msg_type == "set_bypass":
            latched = bool(msg.get("latched", False))
            if self.bypass_latched != latched:
                self.bypass_latched = latched
            self._broadcast(self.build_state_message())
            return

        if msg_type == "midi_learn_pulse":
            self._handle_midi_learn_pulse(msg)
            return

        self._write_json(
            conn.writer,
            {"type": "error", "code": "UNKNOWN_TYPE", "message": f"Unsupported ui message: {msg_type}"},
        )

    def _handle_agent_cc_write(self, msg: dict[str, Any], now_mono: float) -> None:
        # 1. Emergency bypass drops ALL agent writes immediately
        if self.bypass_latched:
            return

        channel_id = str(msg.get("channel_id", ""))
        control = str(msg.get("control", "vol"))
        key = control_key(channel_id, control)
        spec = self.config.targets.get(key)
        if spec is None:
            return

        # 2. Hard safety rule: auto-mute is strictly forbidden on spoken_primary
        if spec.control == "mute" and spec.role == "spoken_primary":
            return

        # 3. Bridge-level in-flight race guard: human_seq fence + 250ms drop window
        current_seq = self.human_seq.get(key, 0)
        try:
            ack_seq = int(msg.get("ack_human_seq", 0))
        except (TypeError, ValueError):
            ack_seq = -1
        if ack_seq < current_seq:
            return

        last_touch = self._last_human_touch_mono.get(key)
        if last_touch is not None and (now_mono - last_touch) * 1000.0 < self.race_guard_ms:
            return

        try:
            cc_val = round(float(msg["cc"]))
        except (KeyError, TypeError, ValueError):
            return
        cc_val = max(0, min(127, cc_val))

        changed = self._transmit_cc(spec, cc_val)
        if changed and self.config.takeover_mode == "pickup_catchup":
            # Agent moved the virtual fader; physical non-motorized fader must pick up again
            last_hw = self._last_hw_cc.get(key)
            if last_hw is None or abs(last_hw - cc_val) > self.pickup_tolerance_cc:
                self._hw_picked_up[key] = False
                self._prev_unpicked_hw_cc.pop(key, None)

    def _handle_midi_learn_pulse(self, msg: dict[str, Any]) -> None:
        if self.bypass_latched:
            return
        channel_id = str(msg.get("channel_id", ""))
        control = str(msg.get("control", "vol"))
        key = control_key(channel_id, control)
        spec = self.config.targets.get(key)
        if spec is None:
            return
        # Bypass change deduplication so MainStage MIDI Learn always receives a CC pulse
        pulse_cc = self.last_sent_cc.get(key, 64)
        self._transmit_cc(spec, pulse_cc, force=True, update_state=False)

    def _handle_ui_manual_cc(self, conn: _ClientConn, msg: dict[str, Any], now_mono: float) -> None:
        channel_id = str(msg.get("channel_id", ""))
        control = str(msg.get("control", "vol"))
        key = control_key(channel_id, control)
        spec = self.config.targets.get(key)
        if spec is None:
            self._write_json(
                conn.writer,
                {"type": "error", "code": "UNKNOWN_CONTROL", "message": f"Unknown control: {key}"},
            )
            return

        try:
            cc_val = max(0, min(127, round(float(msg["cc"]))))
        except (KeyError, TypeError, ValueError):
            self._write_json(conn.writer, {"type": "error", "code": "INVALID_CC", "message": "Invalid cc value"})
            return

        # Always passes through (even when engine is dead or bypass_latched=True!)
        seq = self._bump_human_seq(key, now_mono)
        self._transmit_cc(spec, cc_val)

        if self.config.takeover_mode == "pickup_catchup":
            last_hw = self._last_hw_cc.get(key)
            if last_hw is None or abs(last_hw - cc_val) > self.pickup_tolerance_cc:
                self._hw_picked_up[key] = False
                self._prev_unpicked_hw_cc.pop(key, None)
            self.pickup_pending[key] = 0

        self._broadcast(
            {
                "type": "human_cc_event",
                "channel_id": spec.channel_id,
                "control": spec.control,
                "cc": cc_val,
                "human_seq": seq,
                "source": "ui",
                "ts_ms": _now_ms(),
            }
        )

    # ------------------------------------------------------------------
    # Background Loops (Dead-Engine Watchdog & 1Hz USB Hot-Plug Poll)
    # ------------------------------------------------------------------

    async def _engine_watchdog_loop(self) -> None:
        check_interval_s = min(0.05, max(0.005, (self.engine_timeout_ms / 1000.0) / 10.0))
        while self._running:
            await asyncio.sleep(check_interval_s)
            if not self.engine_connected:
                continue
            elapsed_ms = (time.monotonic() - self._last_engine_activity_mono) * 1000.0
            if elapsed_ms > self.engine_timeout_ms:
                logger.warning(
                    "Dead-engine watchdog tripped (%.1f ms > %.1f ms); holding last_sent_cc",
                    elapsed_ms,
                    self.engine_timeout_ms,
                )
                self.engine_connected = False
                # NEVER zero MIDI — hold last_sent_cc and broadcast state to UI
                self._broadcast(self.build_state_message())

    async def _usb_hotplug_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.usb_poll_interval_s)
            self._poll_midi_ports_once(broadcast_on_change=True)

    def _poll_midi_ports_once(self, *, broadcast_on_change: bool) -> None:
        # Retry MainStage IAC bus attachment if not yet open
        if not self._output_ready:
            try:
                self._output_ready = self.midi_backend.open_output(
                    self.config.mainstage_out_port,
                    self.config.mainstage_out_mode,
                )
            except (OSError, RuntimeError):
                self._output_ready = False

        hw_port = self.config.hardware_in_port
        if not hw_port:
            return

        try:
            ports = self.midi_backend.list_input_ports()
        except (OSError, RuntimeError):
            ports = []

        needle = hw_port.lower()
        present = any(p.lower() == needle or needle in p.lower() for p in ports)

        if self.controller_connected and not present:
            logger.warning("Hardware MIDI controller %r disconnected", hw_port)
            self.midi_backend.close_input()
            self.controller_connected = False
            self._hw_picked_up.clear()
            self._prev_unpicked_hw_cc.clear()
            self.pickup_pending.clear()
            if broadcast_on_change:
                self._broadcast(self.build_state_message())
        elif not self.controller_connected and present:
            opened = False
            try:
                opened = self.midi_backend.open_input(hw_port, self._on_raw_midi)
            except (OSError, RuntimeError) as exc:
                logger.warning("Failed to open hardware MIDI controller %r: %s", hw_port, exc)
            if opened:
                logger.info("Hardware MIDI controller %r connected (takeover=%s)", hw_port, self.config.takeover_mode)
                self.controller_connected = True
                # Re-attach in pickup_catchup mode so replugging never causes fader jumps
                self._hw_picked_up.clear()
                self._prev_unpicked_hw_cc.clear()
                self.pickup_pending.clear()
                if broadcast_on_change:
                    self._broadcast(self.build_state_message())

    # ------------------------------------------------------------------
    # Wire Helpers
    # ------------------------------------------------------------------

    def _write_json(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        if writer.is_closing():
            return
        with contextlib.suppress(OSError, RuntimeError):
            data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            writer.write(data)

    def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        for conn in list(self._clients.values()):
            if conn.writer.is_closing():
                continue
            with contextlib.suppress(OSError, RuntimeError):
                conn.writer.write(data)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _default_service_plan_path() -> Path:
    return Path(__file__).resolve().parent.parent / "engine" / "schemas" / "service_plan.json"


async def _run_daemon(args: argparse.Namespace) -> None:
    config = BridgeConfig.from_service_plan_file(Path(args.service_plan))
    backend: MidiBackend = MockMidiBackend() if args.headless else RtMidiBackend()
    daemon = MidiBridgeDaemon(
        config,
        socket_path=Path(args.socket),
        token_path=Path(args.token_path),
        state_path=Path(args.state_path),
        midi_backend=backend,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        daemon.flush_state_to_disk(force=bool(daemon.last_sent_cc))
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover
            pass

    await daemon.start()
    try:
        await stop_event.wait()
    finally:
        await daemon.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherMix Standalone Zero-ML MIDI Bridge Daemon (v5.1)")
    parser.add_argument(
        "--service-plan",
        type=str,
        default=str(_default_service_plan_path()),
        help="Path to engine/schemas/service_plan.json",
    )
    parser.add_argument("--socket", type=str, default=str(DEFAULT_SOCKET_PATH), help="AF_UNIX socket path")
    parser.add_argument("--token-path", type=str, default=str(DEFAULT_TOKEN_PATH), help="Session token path")
    parser.add_argument("--state-path", type=str, default=str(DEFAULT_STATE_PATH), help="Persisted bridge state path")
    parser.add_argument("--headless", action="store_true", help="Run with in-memory MockMidiBackend (no CoreMIDI)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(_run_daemon(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
