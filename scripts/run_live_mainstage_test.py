#!/usr/bin/env python3
"""Live MainStage + BlackHole 16ch + Real CoreMIDI + Real MedleyDB Stems Runner.

Wires the complete real hardware/virtual-bus loop on macOS:
  1. Streams real downloaded MedleyDB studio multitracks + real male/female speech
     into BlackHole 16ch outputs (Ch 1-12) at 48 kHz in real time.
  2. Runs standalone MidiBridgeDaemon with REAL RtMidiBackend(), creating the live
     CoreMIDI virtual port 'AetherMix To MainStage' (visible in Apple MainStage).
  3. Runs AetherMixEngine with REAL PortAudio CoreAudio input tapping BlackHole 16ch
     + REAL Silero VAD ONNX + authenticated WebSocket on ws://127.0.0.1:8765/ws.

Inside Apple MainStage:
  - Audio Input:  BlackHole 16ch
  - Audio Output: MacBook Pro Speakers (or your PA audio interface)
  - Concert-Level Audio Strips: Input 1 (Pastor), Input 2 (Lead Vox), Input 3-4 (BVs),
    Input 5-6 (Keys/Pad), Input 7-8 (Guitars), Input 9-12 (Drums/Band)
  - Use AetherMix Setup Wizard -> 'MIDI Learn Pulse' to map CC 20-26 in 1 click!
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridge.midi_bridge_daemon import BridgeConfig, MidiBridgeDaemon, RtMidiBackend
from engine.core.protocol import (
    DEFAULT_BRIDGE_SOCKET_PATH,
    DEFAULT_BRIDGE_STATE_PATH,
    DEFAULT_TOKEN_PATH,
    ServicePhase,
    db_to_cc,
    ensure_session_token,
)
from engine.main import (
    DEFAULT_MIXING_RULES_PATH,
    DEFAULT_SERVICE_PLAN_PATH,
    build_engine_from_files,
    create_app,
)
from scripts.validate_real_stems import (
    BLOCK_FRAMES,
    SAMPLE_RATE,
    RealStemPack,
    compose_stage_block,
    discover_real_stems,
)

logger = logging.getLogger("aethermix.live_mainstage")

LIVE_SCENES: list[tuple[str, float, dict[str, bool]]] = [
    ("Pastoral Prayer Underscore (Ch1 Speech + Ch5-6 Pad)", 14.0, {"pastor_on": True, "keys_on": True}),
    (
        "Acoustic Worship Verse (WORSHIP_LOW)",
        14.0,
        {"lead_on": True, "keys_on": True, "guitars_on": True},
    ),
    (
        "Full Band Chorus (WORSHIP_HIGH)",
        16.0,
        {"lead_on": True, "bv_on": True, "keys_on": True, "guitars_on": True, "drums_on": True},
    ),
    (
        "Pastor Interrupt Over Full Band",
        10.0,
        {"pastor_on": True, "keys_on": True, "guitars_on": True, "drums_on": True},
    ),
    ("Sermon (SPOKEN_WORD)", 12.0, {"pastor_on": True}),
    ("Applause Transition", 8.0, {"applause_on": True}),
]


def find_blackhole_device() -> int:
    for idx, dev in enumerate(sd.query_devices()):
        name = str(dev.get("name", ""))
        if "blackhole" in name.lower() and int(dev.get("max_output_channels", 0)) >= 16:
            return idx
    raise RuntimeError(
        "BlackHole 16ch audio device not found! Run `brew reinstall --cask blackhole-16ch` first."
    )


class BlackHoleStemStreamer:
    """Streams real downloaded multitrack WAV stems into BlackHole 16ch in real time."""

    def __init__(self, pack: RealStemPack, device_idx: int) -> None:
        self.pack = pack
        self.device_idx = device_idx
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="blackhole-stem-streamer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        rng = np.random.default_rng(108)
        cursor = 0
        scene_idx = 0
        scene_start = time.monotonic()
        logger.info("Streaming real stems to BlackHole 16ch (device=%d): %s", self.device_idx, LIVE_SCENES[0][0])

        with sd.OutputStream(
            device=self.device_idx,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_FRAMES,
            channels=16,
            dtype="float32",
        ) as stream:
            self._stream = stream
            while not self._stop_evt.is_set():
                title, dur_s, kw = LIVE_SCENES[scene_idx]
                now = time.monotonic()
                if now - scene_start >= dur_s:
                    scene_idx = (scene_idx + 1) % len(LIVE_SCENES)
                    scene_start = now
                    title, dur_s, kw = LIVE_SCENES[scene_idx]
                    logger.info("Live BlackHole scene -> %s", title)

                block = compose_stage_block(self.pack, cursor, rng=rng, **kw)
                if title.startswith("Pastor Interrupt"):
                    block[:, 0] *= 10.0 ** (3.0 / 20.0)
                stream.write(block)
                cursor += BLOCK_FRAMES


async def run_live_mainstage() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    find_blackhole_device()
    token = ensure_session_token(DEFAULT_TOKEN_PATH)

    # 1. Start real-stem 16ch audio stream in an ISOLATED child process so macOS
    # CoreAudio/PortAudio never opens dual AUHAL streams on BlackHole inside one process!
    streamer_proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--streamer-only",
        cwd=str(REPO_ROOT),
    )
    await asyncio.sleep(0.6)

    # 2. Start standalone MidiBridgeDaemon with REAL CoreMIDI Virtual Port ('AetherMix To MainStage')
    bridge_cfg = BridgeConfig.from_service_plan_file(DEFAULT_SERVICE_PLAN_PATH)
    bridge_daemon = MidiBridgeDaemon(
        bridge_cfg,
        socket_path=DEFAULT_BRIDGE_SOCKET_PATH,
        state_path=DEFAULT_BRIDGE_STATE_PATH,
        token_path=DEFAULT_TOKEN_PATH,
        midi_backend=RtMidiBackend(),
    )
    await bridge_daemon.start()
    logger.info(
        "REAL CoreMIDI Virtual Port '%s' is LIVE (visible in Apple MainStage!)",
        bridge_cfg.mainstage_out_port,
    )

    # 3. Start AetherMixEngine tapping REAL BlackHole 16ch CoreAudio input + REAL Silero VAD ONNX
    engine = build_engine_from_files(
        service_plan_path=DEFAULT_SERVICE_PLAN_PATH,
        mixing_rules_path=DEFAULT_MIXING_RULES_PATH,
        headless=False,
        bridge_socket_path=DEFAULT_BRIDGE_SOCKET_PATH,
        token_path=DEFAULT_TOKEN_PATH,
    )
    for ch in engine.plan.channels:
        base_db = engine.rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE][ch.id]
        bridge_daemon.last_sent_cc[f"{ch.id}:vol"] = db_to_cc(base_db, ch.cc_taper)
        if ch.reverb_cc is not None:
            rev_db = engine.rules.phase_reverb_baselines_db[ServicePhase.PRAYER_UNDERSCORE][ch.id]
            bridge_daemon.last_sent_cc[f"{ch.id}:reverb"] = db_to_cc(rev_db, ch.reverb_cc_taper)

    app = create_app(engine, auto_start_loop=True)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="info"))

    try:
        logger.info("AetherMix Engine live on ws://127.0.0.1:8765/ws (token=%s...)", token[:8])
        await server.serve()
    finally:
        streamer_proc.terminate()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(streamer_proc.wait(), timeout=3.0)
        await bridge_daemon.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live MainStage + BlackHole 16ch Runner")
    parser.add_argument("--streamer-only", action="store_true", help="Run only the BlackHole 16ch stem player")
    args = parser.parse_args()
    if args.streamer_only:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        bh_idx = find_blackhole_device()
        pack = discover_real_stems()
        streamer = BlackHoleStemStreamer(pack, device_idx=bh_idx)
        streamer._run()
        return 0
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_live_mainstage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
