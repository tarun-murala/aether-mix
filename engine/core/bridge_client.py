"""Async Unix-socket client for AetherMix's standalone midi_bridge_daemon (SPEC v5.1)."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from engine.core.protocol import (
    DEFAULT_BRIDGE_SOCKET_PATH,
    DEFAULT_TOKEN_PATH,
    PROTOCOL_VERSION,
    BridgeStatePayload,
    ensure_session_token,
)

logger = logging.getLogger("aethermix.bridge_client")


def _control_key(channel_id: str, control: str) -> str:
    return f"{channel_id}:{control}"


class BridgeClient:
    """Async NDJSON AF_UNIX client for midi_bridge_daemon (role='engine' or role='ui')."""

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_BRIDGE_SOCKET_PATH,
        token: str | None = None,
        on_human_touch: Callable[[dict[str, Any]], Any] | None = None,
        on_human_cc: Callable[[dict[str, Any]], Any] | None = None,
        on_action_button: Callable[[dict[str, Any]], Any] | None = None,
        on_state_change: Callable[[BridgeStatePayload], Any] | None = None,
        *,
        role: Literal["engine", "ui"] = "engine",
        token_path: Path = DEFAULT_TOKEN_PATH,
        stream_opener: Callable[[], Any] | None = None,
        auto_reconnect: bool = True,
        reconnect_interval_s: float = 0.25,
        on_reconnect: Callable[[BridgeStatePayload], Any] | None = None,
        on_disconnect: Callable[[], Any] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self._token = token if token is not None else ensure_session_token(token_path)
        self._role: Literal["engine", "ui"] = role
        self._stream_opener = stream_opener
        self._auto_reconnect = bool(auto_reconnect)
        self._reconnect_interval_s = max(0.02, float(reconnect_interval_s))
        self._on_human_touch = on_human_touch
        self._on_human_cc = on_human_cc
        self._on_action_button = on_action_button
        self._on_state_change = on_state_change
        self._on_reconnect = on_reconnect
        self._on_disconnect = on_disconnect

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._connected: bool = False
        self._closed: bool = False
        self._agent_seq: int = 0

        self._state: BridgeStatePayload = BridgeStatePayload()
        self._human_seq: dict[str, int] = {}
        self._pending_state_futures: list[asyncio.Future[BridgeStatePayload]] = []

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> BridgeStatePayload:
        return self._state

    @property
    def human_seq(self) -> dict[str, int]:
        return dict(self._human_seq)

    async def _connect_once(
        self,
        *,
        timeout: float = 5.0,
        fetch_state: bool = False,
    ) -> BridgeStatePayload:
        if self._stream_opener is not None:
            opened = self._stream_opener()
            if inspect.isawaitable(opened):
                reader, writer = await asyncio.wait_for(opened, timeout=timeout)
            else:
                reader, writer = opened
        else:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.socket_path)),
                timeout=timeout,
            )
        self._reader = reader
        self._writer = writer

        hello = {
            "type": "hello",
            "role": self._role,
            "token": self._token,
            "protocol_version": PROTOCOL_VERSION,
        }
        try:
            await self._send_json(hello)
            raw_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not raw_line:
                raise ConnectionError("Bridge closed connection during handshake")
            frame = json.loads(raw_line.decode("utf-8"))
            if frame.get("type") == "error":
                code = frame.get("code", "BRIDGE_ERROR")
                msg = frame.get("message", "Bridge rejected connection")
                if code in ("AUTH_FAILED", "INVALID_ROLE", "VERSION_MISMATCH"):
                    raise PermissionError(f"{code}: {msg}")
                raise RuntimeError(f"{code}: {msg}")
            if frame.get("type") != "bridge_state":
                raise RuntimeError(f"Expected initial bridge_state frame, got {frame.get('type')!r}")

            self._apply_bridge_state(frame)

            if fetch_state:
                await self._send_json({"type": "get_bridge_state"})
                state_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
                if not state_line:
                    raise ConnectionError("Bridge closed connection during state hydration")
                state_frame = json.loads(state_line.decode("utf-8"))
                if state_frame.get("type") == "bridge_state":
                    self._apply_bridge_state(state_frame)

            self._connected = True
        except Exception:
            await self._cleanup_transport()
            raise

        return self._state

    async def connect(self, timeout: float = 5.0) -> BridgeStatePayload:
        """Connect to midi_bridge_daemon, authenticate via hello, and hydrate bridge_state."""
        if self._connected:
            return self._state

        self._closed = False
        await self._connect_once(timeout=timeout, fetch_state=False)
        await self._invoke_callback(self._on_state_change, self._state)
        self._reader_task = asyncio.create_task(self._read_loop(), name=f"bridge-client-{self._role}")
        return self._state

    def start_reconnect(self) -> None:
        """Start background reconnect polling if disconnected and not explicitly closed."""
        if self._closed or self._connected:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(
            self._reconnect_loop(),
            name=f"bridge-reconnect-{self._role}",
        )

    async def close(self) -> None:
        """Close the bridge connection and stop background reader/reconnect tasks."""
        self._closed = True
        self._connected = False
        for task in (self._reconnect_task, self._reader_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._reconnect_task = None
        self._reader_task = None
        await self._cleanup_transport()
        self._fail_pending_state_futures(ConnectionError("BridgeClient closed"))

    async def send_agent_cc(
        self,
        channel_id: str,
        control: str,
        cc: int,
        ack_human_seq: int | None = None,
    ) -> None:
        """Send an agent_cc_write command fenced by ack_human_seq."""
        self._require_connected()
        key = _control_key(channel_id, control)
        ack_seq = self._human_seq.get(key, 0) if ack_human_seq is None else int(ack_human_seq)
        self._agent_seq += 1
        await self._send_json(
            {
                "type": "agent_cc_write",
                "channel_id": channel_id,
                "control": control,
                "cc": max(0, min(127, int(cc))),
                "seq": self._agent_seq,
                "ack_human_seq": ack_seq,
            }
        )

    async def send_midi_learn_pulse(self, channel_id: str, control: str) -> None:
        """Trigger a deduplication-bypassing MIDI CC pulse for MainStage MIDI Learn."""
        self._require_connected()
        await self._send_json(
            {
                "type": "midi_learn_pulse",
                "channel_id": channel_id,
                "control": control,
            }
        )

    async def send_heartbeat(self) -> None:
        """Send an engine_heartbeat frame (call at >=20Hz to satisfy 500ms bridge watchdog)."""
        self._require_connected()
        await self._send_json(
            {
                "type": "engine_heartbeat",
                "ts_ms": int(time.time() * 1000),
            }
        )

    async def send_ui_manual_cc(self, channel_id: str, control: str, cc: int) -> None:
        """Send a direct UI manual CC write (for role='ui' clients)."""
        self._require_connected()
        await self._send_json(
            {
                "type": "ui_manual_cc",
                "channel_id": channel_id,
                "control": control,
                "cc": max(0, min(127, int(cc))),
            }
        )

    async def send_set_bypass(self, latched: bool) -> None:
        """Latch or release emergency_bypass directly on the bridge (for role='ui' clients)."""
        self._require_connected()
        await self._send_json(
            {
                "type": "set_bypass",
                "latched": bool(latched),
            }
        )

    async def get_state(self, timeout: float = 2.0) -> BridgeStatePayload:
        """Request and await a fresh authoritative bridge_state snapshot."""
        self._require_connected()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[BridgeStatePayload] = loop.create_future()
        self._pending_state_futures.append(fut)
        try:
            await self._send_json({"type": "get_bridge_state"})
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            if fut in self._pending_state_futures:
                self._pending_state_futures.remove(fut)

    # ------------------------------------------------------------------
    # Internal Transport, Reconnect Loop & Frame Dispatch
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected or self._writer is None or self._writer.is_closing():
            raise ConnectionError("BridgeClient is not connected")

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._writer is None or self._writer.is_closing():
            raise ConnectionError("BridgeClient socket is closed")
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self._writer.write(data)
        await self._writer.drain()

    async def _cleanup_transport(self) -> None:
        writer = self._writer
        self._writer = None
        self._reader = None
        if writer is not None:
            with contextlib.suppress(OSError, RuntimeError):
                writer.close()
                await writer.wait_closed()

    def _fail_pending_state_futures(self, exc: Exception) -> None:
        while self._pending_state_futures:
            fut = self._pending_state_futures.pop(0)
            if not fut.done():
                fut.set_exception(exc)

    def _apply_bridge_state(self, frame: dict[str, Any]) -> None:
        parsed = BridgeStatePayload(
            last_sent_cc=dict(frame.get("last_sent_cc", {})),
            human_seq=dict(frame.get("human_seq", {})),
            bypass_latched=bool(frame.get("bypass_latched", False)),
            controller_connected=bool(frame.get("controller_connected", False)),
            engine_connected=bool(frame.get("engine_connected", False)),
            pickup_pending=dict(frame.get("pickup_pending", {})),
            restored_from_disk_at=frame.get("restored_from_disk_at"),
        )
        self._state = parsed
        self._human_seq = dict(parsed.human_seq)

    async def _invoke_callback(self, cb: Callable[[Any], Any] | None, arg: Any) -> None:
        if cb is None:
            return
        try:
            res = cb(arg)
            if inspect.isawaitable(res):
                await res
        except Exception as exc:  # noqa: BLE001 - isolate user callback failures from IPC loop
            logger.warning("BridgeClient callback error: %s", exc)

    async def _invoke_zero_arg_callback(self, cb: Callable[[], Any] | None) -> None:
        if cb is None:
            return
        try:
            res = cb()
            if inspect.isawaitable(res):
                await res
        except Exception as exc:  # noqa: BLE001 - isolate user callback failures from IPC loop
            logger.warning("BridgeClient callback error: %s", exc)

    async def _reconnect_loop(self) -> None:
        while not self._closed and not self._connected:
            await asyncio.sleep(self._reconnect_interval_s)
            if self._closed or self._connected:
                return
            try:
                await self._connect_once(timeout=2.0, fetch_state=True)
            except PermissionError as exc:
                logger.error("Bridge reconnect rejected (%s); stopping reconnect loop", exc)
                return
            except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                logger.debug("Bridge reconnect attempt failed (%s); retrying", exc)
                continue

            logger.info("BridgeClient (%s) reconnected to %s", self._role, self.socket_path)
            self._reader_task = asyncio.create_task(
                self._read_loop(),
                name=f"bridge-client-{self._role}",
            )
            if self._on_reconnect is not None:
                await self._invoke_callback(self._on_reconnect, self._state)
            else:
                await self._invoke_callback(self._on_state_change, self._state)
            return

    async def _read_loop(self) -> None:
        assert self._reader is not None
        reader = self._reader
        try:
            while self._connected:
                line = await reader.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logger.warning("BridgeClient received malformed JSON line")
                    continue
                if not isinstance(frame, dict):
                    continue
                await self._handle_incoming_frame(frame)
        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, BrokenPipeError):
            pass
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("BridgeClient read loop error: %s", exc)
        finally:
            self._connected = False
            await self._cleanup_transport()
            self._fail_pending_state_futures(ConnectionError("Bridge disconnected"))
            if not self._closed:
                await self._invoke_zero_arg_callback(self._on_disconnect)
                if self._auto_reconnect:
                    self.start_reconnect()

    async def _handle_incoming_frame(self, frame: dict[str, Any]) -> None:
        msg_type = frame.get("type")

        if msg_type == "bridge_state":
            self._apply_bridge_state(frame)
            while self._pending_state_futures:
                fut = self._pending_state_futures.pop(0)
                if not fut.done():
                    fut.set_result(self._state)
            await self._invoke_callback(self._on_state_change, self._state)
            return

        if msg_type == "human_touch_event":
            ch_id = str(frame.get("channel_id", ""))
            ctrl = str(frame.get("control", "vol"))
            key = _control_key(ch_id, ctrl)
            seq = int(frame.get("human_seq", 0))
            if seq > self._human_seq.get(key, 0):
                self._human_seq[key] = seq
                self._state.human_seq[key] = seq
            if "pickup_dir" in frame:
                self._state.pickup_pending[key] = int(frame["pickup_dir"])
            await self._invoke_callback(self._on_human_touch, frame)
            return

        if msg_type == "human_cc_event":
            ch_id = str(frame.get("channel_id", ""))
            ctrl = str(frame.get("control", "vol"))
            key = _control_key(ch_id, ctrl)
            seq = int(frame.get("human_seq", 0))
            cc_val = int(frame.get("cc", 0))
            if seq > self._human_seq.get(key, 0):
                self._human_seq[key] = seq
                self._state.human_seq[key] = seq
            self._state.last_sent_cc[key] = cc_val
            await self._invoke_callback(self._on_human_cc, frame)
            return

        if msg_type == "action_button_event":
            if frame.get("action") == "emergency_bypass":
                self._state.bypass_latched = True
            await self._invoke_callback(self._on_action_button, frame)
            return

        if msg_type == "error":
            logger.warning("Bridge error frame: %s", frame)
