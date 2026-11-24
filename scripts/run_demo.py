#!/usr/bin/env python3
"""Launch a live interactive AetherMix v5.1 demo backend for the native SwiftUI app.

Starts:
  1. Standalone MidiBridgeDaemon (mock MIDI backend) on ~/.aethermix/bridge.sock
  2. AetherMixEngine FastAPI WebSocket server on ws://127.0.0.1:8765/ws
  3. A looping 50 Hz synthetic multi-stem church service audio feeder so the
     native macOS Mission Control UI displays live meters, VAD LEDs, phase
     transitions, dual-ring knob glides, and Agent Brain events without
     needing physical MainStage / BlackHole hardware attached.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridge.midi_bridge_daemon import BridgeConfig, MidiBridgeDaemon, MockMidiBackend
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
    AetherMixEngine,
    build_engine_from_files,
    create_app,
)
from engine.tests.mock_io import SceneAudioConfig, SyntheticServiceAudioGenerator

logger = logging.getLogger("aethermix.demo")

DEMO_SCENES: list[tuple[str, float, SceneAudioConfig]] = [
    ("Pastoral Prayer Underscore", 12.0, SceneAudioConfig.for_phase(ServicePhase.PRAYER_UNDERSCORE)),
    ("Acoustic Worship Verse (WORSHIP_LOW)", 12.0, SceneAudioConfig.for_phase(ServicePhase.WORSHIP_LOW)),
    ("Full Band Chorus (WORSHIP_HIGH)", 14.0, SceneAudioConfig.for_phase(ServicePhase.WORSHIP_HIGH)),
    (
        "Pastor Interrupt Over Band",
        8.0,
        SceneAudioConfig(
            phase=ServicePhase.WORSHIP_HIGH,
            pastor_speaking=True,
            lead_singing=False,
            bv_singing=False,
            keys_playing=True,
            guitars_playing=True,
            drums_playing=True,
        ),
    ),
    ("Sermon (SPOKEN_WORD)", 12.0, SceneAudioConfig.for_phase(ServicePhase.SPOKEN_WORD)),
    ("Applause Transition", 6.0, SceneAudioConfig.for_phase(ServicePhase.TRANSITION)),
]


async def feed_demo_audio(engine: AetherMixEngine) -> None:
    """Continuously feed 50 Hz realistic multi-stem audio into the running engine."""
    gen = SyntheticServiceAudioGenerator(sample_rate=engine.plan.sample_rate)
    scene_idx = 0
    scene_start = time.monotonic()

    # Wait briefly for engine.start() inside lifespan, then mark channels synced for demo
    await asyncio.sleep(0.5)
    for ch in engine.plan.channels:
        engine.safety_firewall.confirm_sync(ch.id)

    while True:
        title, duration_s, cfg = DEMO_SCENES[scene_idx]
        now = time.monotonic()
        if now - scene_start >= duration_s:
            scene_idx = (scene_idx + 1) % len(DEMO_SCENES)
            scene_start = now
            title, duration_s, cfg = DEMO_SCENES[scene_idx]
            logger.info("Demo scene -> %s", title)

        block = gen.generate_block(cfg, frames=960)
        engine.push_audio_block(block)
        await asyncio.sleep(0.02)


async def run_demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    token = ensure_session_token(DEFAULT_TOKEN_PATH)

    # 1. Start standalone MIDI bridge daemon on ~/.aethermix/bridge.sock
    bridge_cfg = BridgeConfig.from_service_plan_file(DEFAULT_SERVICE_PLAN_PATH)
    bridge_daemon = MidiBridgeDaemon(
        bridge_cfg,
        socket_path=DEFAULT_BRIDGE_SOCKET_PATH,
        state_path=DEFAULT_BRIDGE_STATE_PATH,
        token_path=DEFAULT_TOKEN_PATH,
        midi_backend=MockMidiBackend(),
    )
    await bridge_daemon.start()
    logger.info("Demo bridge daemon listening on %s", DEFAULT_BRIDGE_SOCKET_PATH)

    # 2. Seed initial Prayer Underscore CCs so UI shows real synced knobs immediately
    engine = build_engine_from_files(
        service_plan_path=DEFAULT_SERVICE_PLAN_PATH,
        mixing_rules_path=DEFAULT_MIXING_RULES_PATH,
        headless=True,
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
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="info")
    server = uvicorn.Server(config)

    feeder_task = asyncio.create_task(feed_demo_audio(engine), name="demo-audio-feeder")
    try:
        logger.info("Demo WS server starting on ws://127.0.0.1:8765/ws (token=%s...)", token[:8])
        await server.serve()
    finally:
        feeder_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feeder_task
        await bridge_daemon.stop()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_demo())
