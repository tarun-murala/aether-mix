"""Mode B live capture: play a Timeline through MainStage, record the running engine's WebSocket stream, score it.

The engine runs as its own process (closed-loop plan, taps from BlackHole 64ch), MainStage moves the real faders,
and this module only listens. Per-tick state is rebuilt from the engine's own broadcast:

- ``meter_frame`` (every tick): pre-fader tap levels, ``vol_actual_cc`` (the fader position the bridge last confirmed
  to MainStage) and the momentary master LUFS measured on MainStage's post-fader master tap (64ch 15-16).
- ``state_snapshot`` (on change): active phase, speech_open, instrument_open, feedback ceilings.
- ``heartbeat`` (1 Hz): loop p95/p99. ``brain_event``: engine actions.

Timing: the player stamps the wall clock at which sample 0 reached the DAC; the residual (MainStage I/O latency and
buffering) is measured by cross-correlating the expected pastor tap level with the recorded ch1 ``rms_db``.
The resulting TickLog is scored by the same ``score_run`` as headless runs.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
from engine.core.protocol import PROTOCOL_VERSION, ServicePlan, cc_to_db

from evals import BLOCK_FRAMES, SAMPLE_RATE
from evals.runner import CHANNELS, DT, PHASES, TickLog
from evals.timeline import Timeline

MAX_LAG_S = 1.5  # search window for the audio-path alignment


@dataclass
class LiveRecording:
    service: str
    profile: str
    play_t0: float  # wall-clock seconds at which sample 0 of the timeline reached the output DAC
    channel_ids: list[str]
    frames: list[dict[str, Any]] = field(default_factory=list)  # {"rx": wall_s, "type": str, "ts_ms": int, "payload": {...}}
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: Path) -> LiveRecording:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls(**json.load(fh))


# ---------------------------------------------------------------------------
# Rebuilding a TickLog from the recorded stream
# ---------------------------------------------------------------------------


def _frame_t(f: dict[str, Any]) -> float:
    """Engine-side timestamp (envelope ts_ms) when present, else receive time."""
    ts = f.get("ts_ms")
    return float(ts) / 1000.0 if ts else float(f["rx"])


def _hold(ts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the last sample at or before each grid time (-1 when none yet)."""
    return np.searchsorted(ts, grid, side="right") - 1


def expected_pastor_db(tl: Timeline) -> np.ndarray:
    out = np.full(tl.n_ticks, -120.0)
    for k in range(tl.n_ticks):
        e = float(np.mean(tl.block(k)[:, 0].astype(np.float64) ** 2))
        out[k] = 10.0 * np.log10(e + 1e-12)
    return out


def estimate_offset_s(rec: LiveRecording, tl: Timeline, expected_db: np.ndarray | None = None) -> tuple[float, float]:
    """(offset_s, correlation): extra delay between the DAC stamp and the engine hearing ch1 on its tap."""
    meters = [f for f in rec.frames if f["type"] == "meter_frame"]
    if len(meters) < 50:
        return 0.0, 0.0
    idx = rec.channel_ids.index("ch1_pastor")
    ts = np.array([_frame_t(f) for f in meters])
    lv = np.array([float(f["payload"]["rms_db"][idx]) for f in meters])
    exp = expected_db if expected_db is not None else expected_pastor_db(tl)
    exp = np.clip(exp, -80.0, 0.0)
    grid = rec.play_t0 + np.arange(tl.n_ticks) * DT
    best = (0.0, -2.0)
    for lag_k in range(-int(MAX_LAG_S / DT), int(MAX_LAG_S / DT) + 1):
        h = _hold(ts, grid + lag_k * DT)
        ok = h >= 0
        if ok.sum() < 100:
            continue
        got = np.clip(lv[h[ok]], -80.0, 0.0)
        a, b = exp[ok] - exp[ok].mean(), got - got.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        r = float((a * b).sum() / den) if den > 0 else 0.0
        if r > best[1]:
            best = (lag_k * DT, r)
    return best


def dequantize_steps(v: np.ndarray) -> np.ndarray:
    """Piecewise-linear fader through the instants the CC changed (one 7-bit step is up to ~3 dB low on the taper,
    so the held staircase has no meaningful per-tick slope; a steady ramp keeps its true slope)."""
    ch = np.flatnonzero(np.diff(v) != 0) + 1
    if len(ch) == 0:
        return v.astype(float).copy()
    knots = np.concatenate(([0], ch, [len(v) - 1]))
    vals = np.concatenate(([v[0]], v[ch], [v[-1]]))
    return np.interp(np.arange(len(v)), knots, vals)


def ticklog_from_recording(rec: LiveRecording, tl: Timeline, plan: ServicePlan, offset_s: float = 0.0) -> TickLog:
    n = tl.n_ticks
    log = TickLog(
        service=tl.service.id, profile=tl.profile, t=np.arange(n) * DT, phase=np.zeros(n, dtype=np.int8),
        vol_db={c: np.full(n, -60.0) for c in CHANNELS}, speech_open=np.zeros(n, dtype=bool),
        instrument_open={c: np.zeros(n, dtype=bool) for c in CHANNELS},
        feedback_ceiling={c: np.full(n, np.nan) for c in CHANNELS}, lufs_short=np.full(n, -70.0),
        pastor_e_mix=np.zeros(n), band_e_mix=np.zeros(n), pastor_e_raw=np.zeros(n), band_e_raw=np.zeros(n),
        tick_ms=np.zeros(n), vad_backend=str(rec.meta.get("vad_backend", "live")),
    )
    grid = rec.play_t0 + offset_s + log.t
    ids = rec.channel_ids
    tapers = {c: plan.channel_by_id(c).cc_taper for c in CHANNELS}

    meters = [f for f in rec.frames if f["type"] == "meter_frame"]
    if meters:
        ts = np.array([_frame_t(f) for f in meters])
        h = _hold(ts, grid)
        for c in CHANNELS:
            i = ids.index(c)
            cc = np.array([int(f["payload"]["vol_actual_cc"][i]) for f in meters])
            db = np.array([max(-60.0, cc_to_db(int(v), tapers[c])) for v in cc])
            log.vol_db[c] = np.where(h >= 0, db[np.maximum(h, 0)], db[0])
        log.vol_db_slew = {c: dequantize_steps(log.vol_db[c]) for c in CHANNELS}
        lufs = np.array([float(f["payload"].get("master_lufs", -70.0)) for f in meters])
        log.lufs_short = np.where(h >= 0, lufs[np.maximum(h, 0)], -70.0)

    snaps = [f for f in rec.frames if f["type"] == "state_snapshot" and f["payload"].get("channels")]
    if snaps:
        ts = np.array([_frame_t(f) for f in snaps])
        h = np.maximum(_hold(ts, grid), 0)
        ph = np.array([PHASES.index(f["payload"]["active_phase"]) for f in snaps], dtype=np.int8)
        log.phase = ph[h]
        log.speech_open = np.array([bool(f["payload"]["channels"]["ch1_pastor"]["speech_open"]) for f in snaps])[h]
        for c in CHANNELS:
            log.instrument_open[c] = np.array([bool(f["payload"]["channels"][c]["instrument_open"]) for f in snaps])[h]
            fb = np.array([f["payload"]["channels"][c].get("feedback_ceiling_db") for f in snaps], dtype=object)
            log.feedback_ceiling[c] = np.array([np.nan if v is None else float(v) for v in fb[h]])

    hbs = [f for f in rec.frames if f["type"] == "heartbeat"]
    if hbs:
        ts = np.array([_frame_t(f) for f in hbs])
        p99 = np.array([float(f["payload"]["loop_p99_ms"]) for f in hbs])
        log.tick_ms = p99[np.maximum(_hold(ts, grid), 0)]
        log.heartbeat = {"loop_p95_ms": float(max(f["payload"]["loop_p95_ms"] for f in hbs)), "loop_p99_ms": float(p99.max())}

    for m in rec.meta.get("human_moves", []):
        log.human_target_db[m["channel"]] = float(m["target_db"])

    t_rel = lambda f: _frame_t(f) - rec.play_t0 - offset_s
    log.events = [{"t": round(t_rel(f), 3), "category": f["payload"].get("category"), "summary": f["payload"].get("summary")}
                  for f in rec.frames if f["type"] == "brain_event"]

    # Fader-implied speech/band energies: pre-fader stems (known exactly) times the faders MainStage held.
    for k in range(n):
        blk = tl.block(k)
        col_e = np.mean(blk.astype(np.float64) ** 2, axis=0)
        g = {c: 0.0 if log.vol_db[c][k] <= -59.5 else 10.0 ** (log.vol_db[c][k] / 20.0) for c in CHANNELS}
        grp = [g["ch5_keys"]] * 2 + [g["ch7_guitars"]] * 2 + [g["ch9_drums"]] * 4
        log.pastor_e_raw[k] = float(col_e[0])
        log.band_e_raw[k] = float(col_e[4:12].sum())
        log.pastor_e_mix[k] = float(col_e[0]) * g["ch1_pastor"] ** 2
        log.band_e_mix[k] = float(sum(col_e[4 + i] * grp[i] ** 2 for i in range(8)))
    return log


# ---------------------------------------------------------------------------
# Capture: WebSocket listener + 16-channel player
# ---------------------------------------------------------------------------


class _Player:
    """Streams the timeline to a 16-channel device from a callback and stamps when sample 0 hit the DAC."""

    def __init__(self, tl: Timeline, device: int) -> None:
        self.tl, self.device = tl, device
        self.k = 0
        self.t0_wall: float | None = None
        self.done = threading.Event()
        self.underruns = 0

    def _cb(self, outdata: np.ndarray, frames: int, tinfo: Any, status: Any) -> None:
        if status.output_underflow:
            self.underruns += 1
        if self.t0_wall is None:
            # stream clock -> wall clock at this instant, then forward to the DAC time of this first buffer
            self.t0_wall = time.time() + (tinfo.outputBufferDacTime - tinfo.currentTime)
        if self.k >= self.tl.n_ticks:  # keep the stream alive with silence; run() aborts it (CallbackStop can hang close)
            outdata.fill(0.0)
            self.done.set()
            return
        outdata[:] = np.clip(self.tl.block(self.k), -1.0, 1.0)
        self.k += 1

    def run(self) -> None:
        import sounddevice as sd  # type: ignore[import-untyped]

        stream = sd.OutputStream(device=self.device, samplerate=SAMPLE_RATE, channels=16, dtype="float32",
                                 blocksize=BLOCK_FRAMES, callback=self._cb, latency="high")
        stream.start()
        try:
            self.done.wait(timeout=self.tl.total_s + 30.0)
        finally:
            stream.abort()
            stream.close()


async def _rpc(ws: Any, rid: list[int], msg_type: str, payload: dict[str, Any]) -> None:
    rid[0] += 1
    await ws.send(json.dumps({"type": msg_type, "req_id": f"eval-{rid[0]}", "payload": payload}))


class BridgeUI:
    """Talks to the MIDI bridge as a `ui` client: `ui_manual_cc` moves the MainStage fader (and reaches the engine
    as a human move), exactly like the AetherMix UI fader."""

    def __init__(self, socket_path: Path, token: str) -> None:
        self.socket_path, self.token = socket_path, token
        self.reader: Any = None
        self.writer: Any = None

    async def __aenter__(self) -> Self:
        self.reader, self.writer = await asyncio.open_unix_connection(str(self.socket_path))
        self._send({"type": "hello", "role": "ui", "token": self.token, "protocol_version": PROTOCOL_VERSION})
        await self.writer.drain()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.writer.close()

    def _send(self, msg: dict[str, Any]) -> None:
        self.writer.write((json.dumps(msg) + "\n").encode("utf-8"))

    async def set_cc(self, channel_id: str, cc: int, control: str = "vol") -> None:
        self._send({"type": "ui_manual_cc", "channel_id": channel_id, "control": control, "cc": int(cc)})
        await self.writer.drain()

    async def state(self) -> dict[str, Any]:
        self._send({"type": "get_bridge_state"})
        await self.writer.drain()
        while True:
            msg = json.loads(await asyncio.wait_for(self.reader.readline(), timeout=5.0))
            if msg.get("type") == "bridge_state":
                return msg


async def capture(tl: Timeline, device: int, token: str, url: str = "ws://127.0.0.1:8765/ws", tail_s: float = 4.0,
                  arm: bool = True, progress: bool = True, bridge_socket: Path | None = None,
                  plan: ServicePlan | None = None) -> LiveRecording:
    """Arm + ACTIVE the running engine, play the timeline, record every broadcast, then restore the engine's mode.

    With `bridge_socket` and `plan`, the timeline's scripted human fader moves are sent as UI fader moves."""
    import contextlib

    import websockets

    frames: list[dict[str, Any]] = []
    moves_done: list[dict[str, Any]] = []
    async with contextlib.AsyncExitStack() as stack:
        ws = await stack.enter_async_context(
            websockets.connect(url, additional_headers={"Authorization": f"Bearer {token}"}, max_size=2**24))
        ui = await stack.enter_async_context(BridgeUI(bridge_socket, token)) if bridge_socket and plan else None
        rid = [0]
        await _rpc(ws, rid, "hello", {"protocol_version": PROTOCOL_VERSION, "client": "aethermix-eval"})
        first: dict[str, Any] = {}
        while "channels" not in first:
            env = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if env.get("type") == "error":
                raise RuntimeError(f"engine refused the eval client: {env.get('payload')}")
            if env.get("type") == "state_snapshot":
                first = env["payload"]
        ids = list(first["channel_ids"])
        prev_mode = first["autonomy_mode"]
        prev_armed = {c: bool(s["armed"]) for c, s in first["channels"].items()}
        stop = asyncio.Event()

        async def reader() -> None:
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except TimeoutError:
                    continue
                env = json.loads(raw)
                frames.append({"rx": time.time(), "type": env.get("type"), "ts_ms": env.get("ts_ms"), "payload": env.get("payload") or {}})

        rd = asyncio.create_task(reader())
        player = _Player(tl, device)
        try:
            if arm:
                for c in ids:
                    await _rpc(ws, rid, "set_channel_armed", {"channel_id": c, "armed": True})
                    await _rpc(ws, rid, "sync_channel", {"channel_id": c})
                await _rpc(ws, rid, "set_mode", {"mode": "ACTIVE"})
            await asyncio.sleep(1.0)  # let ACTIVE settle before sample 0
            pending = sorted((e for e in tl.events if e.kind == "human_move"), key=lambda e: e.t0) if ui else []
            play = asyncio.create_task(asyncio.to_thread(player.run))
            last_print = time.time()
            while not play.done():
                await asyncio.sleep(DT)
                t_now = player.k * DT
                while pending and t_now >= pending[0].t0 and plan is not None and ui is not None:
                    ev = pending.pop(0)
                    ch = ev.channels[0]
                    meter = next((f for f in reversed(frames) if f["type"] == "meter_frame"), None)
                    taper = plan.channel_by_id(ch).cc_taper
                    cur = cc_to_db(int(meter["payload"]["vol_actual_cc"][ids.index(ch)]), taper) if meter else 0.0
                    from engine.core.protocol import db_to_cc

                    cc = db_to_cc(cur + float(ev.params["delta_db"]), taper)
                    await ui.set_cc(ch, cc)
                    moves_done.append({"channel": ch, "t": round(t_now, 3), "cc": cc, "target_db": cc_to_db(cc, taper)})
                if progress and time.time() - last_print >= 5.0:
                    last_print = time.time()
                    print(f"  [{tl.service.id}] live {t_now:6.1f}s / {tl.total_s:.0f}s, {len(frames)} frames", flush=True)
            await play
            await asyncio.sleep(tail_s)
        finally:
            stop.set()
            await rd
            await _rpc(ws, rid, "set_mode", {"mode": prev_mode})
            for c, a in prev_armed.items():
                await _rpc(ws, rid, "set_channel_armed", {"channel_id": c, "armed": a})
            await asyncio.sleep(0.5)
    if player.t0_wall is None:
        raise RuntimeError("player never started")
    return LiveRecording(tl.service.id, tl.profile, player.t0_wall, ids, frames,
                         {"underruns": player.underruns, "prev_mode": prev_mode, "vad_backend": "live engine",
                          "human_moves": moves_done})


# ---------------------------------------------------------------------------
# Pre-fader invariance (gate mainstage_prefader_invariance_db)
# ---------------------------------------------------------------------------


async def prefader_check(out_dev: int, in_dev: int, token: str, bridge_socket: Path, channel_id: str = "ch1_pastor",
                         in_col: int = 0, tap: int = 0, master: tuple[int, int] = (14, 15), sweep_s: float = 8.0) -> dict[str, Any]:
    """Play a steady 1 kHz sine into one MainStage strip, sweep its fader CC 1..127 through the bridge, and measure the
    BlackHole 64ch pre-fader tap (must not move) and the post-fader master tap (must follow). Restores the fader."""
    import sounddevice as sd  # type: ignore[import-untyped]

    amp = 0.14  # 1 kHz sine, ~ -20 dBFS RMS: exact RMS per window, so any tap movement is the fader, not the signal
    phase = [0]
    rec: list[np.ndarray] = []
    marks: list[tuple[float, int]] = []

    def out_cb(outdata: np.ndarray, frames: int, tinfo: Any, status: Any) -> None:
        outdata.fill(0.0)
        n0 = phase[0]
        outdata[:, in_col] = (amp * np.sin(2 * np.pi * 1000.0 * (n0 + np.arange(frames)) / SAMPLE_RATE)).astype(np.float32)
        phase[0] = n0 + frames

    def in_cb(indata: np.ndarray, frames: int, tinfo: Any, status: Any) -> None:
        rec.append(indata.copy())

    async with BridgeUI(bridge_socket, token) as ui:
        st = await ui.state()
        orig = int(st.get("last_sent_cc", {}).get(f"{channel_id}:vol", 96))
        n_in = max(tap, *master) + 1
        with sd.OutputStream(device=out_dev, samplerate=SAMPLE_RATE, channels=16, dtype="float32", blocksize=BLOCK_FRAMES, callback=out_cb), \
             sd.InputStream(device=in_dev, samplerate=SAMPLE_RATE, channels=n_in, dtype="float32", blocksize=BLOCK_FRAMES, callback=in_cb):
            t_start = time.time()
            await ui.set_cc(channel_id, 96)
            await asyncio.sleep(2.0)
            steps = list(range(1, 128, 2))
            for cc in steps:
                await ui.set_cc(channel_id, cc)
                marks.append((time.time() - t_start, cc))
                await asyncio.sleep(sweep_s / len(steps))
            await asyncio.sleep(1.0)
            await ui.set_cc(channel_id, orig)
            await asyncio.sleep(0.5)
    x = np.concatenate(rec) if rec else np.zeros((0, n_in))
    win = int(0.1 * SAMPLE_RATE)
    nwin = len(x) // win
    lev = lambda col: np.array([10 * np.log10(np.mean(x[i * win:(i + 1) * win, col].astype(np.float64) ** 2) + 1e-12) for i in range(nwin)])
    pre, mst = lev(tap), np.maximum(lev(master[0]), lev(master[1]))
    a, b = int(1.0 / 0.1), int((2.0 + sweep_s) / 0.1)  # settled hold through the end of the sweep
    seg_pre, seg_mst = pre[a:b], mst[a:b]
    return {
        "channel_id": channel_id, "fader_restored_cc": orig,
        "prefader_tap_db_range": round(float(seg_pre.max() - seg_pre.min()), 3) if len(seg_pre) else None,
        "prefader_tap_mean_db": round(float(seg_pre.mean()), 2) if len(seg_pre) else None,
        "master_tap_db_range": round(float(seg_mst.max() - seg_mst.min()), 2) if len(seg_mst) else None,
        "master_tap_db_min_max": [round(float(seg_mst.min()), 1), round(float(seg_mst.max()), 1)] if len(seg_mst) else None,
        "pre_curve_db": [round(float(v), 2) for v in seg_pre], "master_curve_db": [round(float(v), 2) for v in seg_mst],
    }
