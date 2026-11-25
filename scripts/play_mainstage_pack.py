#!/usr/bin/env python3
"""Real-Time Multi-Instrument Sound Pack Player & MainStage Live Runner for AetherMix.

Streams the packaged 5-tune multi-instrument Sound Pack (`assets/soundpack/`)
into Apple MainStage via BlackHole 16ch (Ch 1-16 at 48 kHz) with:
  - Isolated BlackHole 16ch audio child process (avoids macOS AUHAL err=-50)
  - Standalone CoreMIDI Bridge (`AetherMix To MainStage` virtual port)
  - Live AetherMixEngine (`ws://127.0.0.1:8765/ws`) with Real Silero VAD ONNX
  - Optional interactive keyboard control to switch tunes/scenes on the fly!

Channel routing into MainStage (BlackHole 16ch):
  Ch  1     : Pastor Mic (real recorded male & female speech)
  Ch  2     : Lead Vocal
  Ch  3     : Backup Vocal 1
  Ch  4     : Backup Vocal 2
  Ch  5-6   : Keys / Piano / Rhodes / Organ / Pad (Stereo)
  Ch  7-8   : Acoustic & Electric Guitars (Stereo)
  Ch  9-10  : Drums / Percussion (Stereo)
  Ch 11-12  : Bass (Stereo)
  Ch 13-14  : Room Mics / Congregation Applause (Stereo)
  Ch 15-16  : Master Return Tap (silent from streamer; MainStage can route Main Out here)

Examples:
  # Full live MainStage + AetherMix stack cycling through a realistic 5-tune service:
  engine/.venv/bin/python scripts/play_mainstage_pack.py

  # Interactive jukebox (press 1-5 to switch tunes, p=pastor, a=applause, f=feedback):
  engine/.venv/bin/python scripts/play_mainstage_pack.py --mode jukebox

  # Stream only one specific tune (e.g. Tune 2 Anthem Rise Pop-Rock) into BlackHole 16ch:
  engine/.venv/bin/python scripts/play_mainstage_pack.py --tune 2

  # Audition the packaged tunes directly on your Mac speakers (stereo mix, no BlackHole):
  engine/.venv/bin/python scripts/play_mainstage_pack.py --audition --tune 2
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
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

SOUNDPACK_DIR = REPO_ROOT / "assets" / "soundpack"
SAMPLE_RATE = 48000
BLOCK_FRAMES = 960  # 20 ms = 50 Hz

logger = logging.getLogger("aethermix.soundpack")


@dataclass
class LoadedTune:
    id: str
    title: str
    genre: str
    bpm: float
    key: str
    duration_s: float
    recommended_phase: str
    lead_vox: np.ndarray  # (N,) mono
    bv1: np.ndarray  # (N,) mono
    bv2: np.ndarray  # (N,) mono
    keys: np.ndarray  # (N, 2) stereo
    guitars: np.ndarray  # (N, 2) stereo
    drums: np.ndarray  # (N, 2) stereo
    bass: np.ndarray  # (N, 2) stereo


@dataclass
class PackagedSoundPack:
    tunes: list[LoadedTune]
    pastor_speech: np.ndarray  # (N,) mono
    room_applause: np.ndarray  # (N, 2) stereo
    manifest: dict[str, Any]


def _read_stem(path: Path, *, stereo: bool) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz stem at {path}, got {sr} Hz")
    if stereo:
        if data.shape[1] == 1:
            return np.repeat(data, 2, axis=1).astype(np.float32)
        return data[:, :2].astype(np.float32)
    return np.mean(data, axis=1).astype(np.float32)


def load_packaged_soundpack(pack_dir: Path = SOUNDPACK_DIR) -> PackagedSoundPack:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Sound pack manifest missing at {manifest_path}. Run `make soundpack` to build it."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pastor_speech = _read_stem(pack_dir / manifest["common"]["pastor_speech"], stereo=False)
    room_applause = _read_stem(pack_dir / manifest["common"]["room_applause"], stereo=True)

    tunes: list[LoadedTune] = []
    for entry in manifest["tunes"]:
        stems = entry["stems"]
        tunes.append(
            LoadedTune(
                id=entry["id"],
                title=entry["title"],
                genre=entry["genre"],
                bpm=float(entry["bpm"]),
                key=entry["key"],
                duration_s=float(entry["duration_s"]),
                recommended_phase=entry["recommended_phase"],
                lead_vox=_read_stem(pack_dir / stems["lead_vox"], stereo=False),
                bv1=_read_stem(pack_dir / stems["bv1"], stereo=False),
                bv2=_read_stem(pack_dir / stems["bv2"], stereo=False),
                keys=_read_stem(pack_dir / stems["keys"], stereo=True),
                guitars=_read_stem(pack_dir / stems["guitars"], stereo=True),
                drums=_read_stem(pack_dir / stems["drums"], stereo=True),
                bass=_read_stem(pack_dir / stems["bass"], stereo=True),
            )
        )
    return PackagedSoundPack(
        tunes=tunes,
        pastor_speech=pastor_speech,
        room_applause=room_applause,
        manifest=manifest,
    )


def _loop_slice(arr: np.ndarray, cursor: int, frames: int) -> np.ndarray:
    n = arr.shape[0]
    if n <= 0:
        return np.zeros((frames,) + arr.shape[1:], dtype=np.float32)
    idx = (np.arange(frames) + cursor) % n
    return arr[idx]


def compose_16ch_block(
    pack: PackagedSoundPack,
    tune_idx: int,
    cursor: int,
    *,
    speech_cursor: int | None = None,
    frames: int = BLOCK_FRAMES,
    pastor_on: bool = False,
    lead_on: bool = True,
    bv_on: bool = True,
    keys_on: bool = True,
    guitars_on: bool = True,
    drums_on: bool = True,
    bass_on: bool = True,
    applause_on: bool = False,
    feedback_hz: float | None = None,
    feedback_gain: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compose a realistic 16-channel 48 kHz pre-fader stage block with subtle stage bleed."""
    tune = pack.tunes[tune_idx % len(pack.tunes)]
    if rng is None:
        rng = np.random.default_rng(42)
    sc = cursor if speech_cursor is None else speech_cursor

    block = np.zeros((frames, 16), dtype=np.float32)
    # Realistic quiet room/preamp noise floor (-66 dBFS)
    block[:, :14] = rng.normal(0.0, 5e-4, size=(frames, 14)).astype(np.float32)

    keys = _loop_slice(tune.keys, cursor, frames) if keys_on else np.zeros((frames, 2), dtype=np.float32)
    guitars = _loop_slice(tune.guitars, cursor, frames) if guitars_on else np.zeros((frames, 2), dtype=np.float32)
    drums = _loop_slice(tune.drums, cursor, frames) if drums_on else np.zeros((frames, 2), dtype=np.float32)
    bass = _loop_slice(tune.bass, cursor, frames) if bass_on else np.zeros((frames, 2), dtype=np.float32)

    block[:, 4:6] += keys
    block[:, 6:8] += guitars
    block[:, 8:10] += drums
    block[:, 10:12] += bass

    # Realistic stage bleed into open vocal/pastor mics (~ -24 to -26 dB)
    stage_mono = 0.25 * (
        np.mean(keys, axis=1) + np.mean(guitars, axis=1) + np.mean(drums, axis=1) + np.mean(bass, axis=1)
    )
    block[:, 0] += stage_mono * 0.05
    block[:, 1] += stage_mono * 0.08
    block[:, 2] += stage_mono * 0.08
    block[:, 3] += stage_mono * 0.08

    if pastor_on:
        block[:, 0] += _loop_slice(pack.pastor_speech, sc, frames)
    if lead_on:
        block[:, 1] += _loop_slice(tune.lead_vox, cursor, frames)
    if bv_on:
        block[:, 2] += _loop_slice(tune.bv1, cursor, frames)
        block[:, 3] += _loop_slice(tune.bv2, cursor, frames)

    if applause_on:
        block[:, 12:14] += _loop_slice(pack.room_applause, cursor, frames)
    else:
        # Quiet room mic pickup of stage
        block[:, 12] += stage_mono * 0.04
        block[:, 13] += stage_mono * 0.04

    if feedback_hz is not None and feedback_gain > 0.0:
        t = (np.arange(frames, dtype=np.float32) + cursor) / float(SAMPLE_RATE)
        block[:, 1] += (feedback_gain * np.sin(2.0 * math.pi * feedback_hz * t)).astype(np.float32)

    return np.clip(block, -0.99, 0.99)


@dataclass
class ServiceScene:
    label: str
    tune_idx: int
    duration_s: float
    flags: dict[str, bool]


SERVICE_PLAYLIST: list[ServiceScene] = [
    ServiceScene(
        "Prayer Underscore · Still Waters (Pad + Pastor)",
        4,
        14.0,
        {"pastor_on": True, "lead_on": False, "bv_on": False, "keys_on": True, "guitars_on": False, "drums_on": False, "bass_on": False},
    ),
    ServiceScene(
        "Acoustic Worship · Morning Grace (Ballad)",
        0,
        18.0,
        {"lead_on": True, "bv_on": False, "keys_on": True, "guitars_on": True, "drums_on": False, "bass_on": True},
    ),
    ServiceScene(
        "Full Band Anthem · Anthem Rise (Pop-Rock)",
        1,
        20.0,
        {"lead_on": True, "bv_on": True, "keys_on": True, "guitars_on": True, "drums_on": True, "bass_on": True},
    ),
    ServiceScene(
        "Pastor Interrupt Over Full Band · Anthem Rise",
        1,
        10.0,
        {"pastor_on": True, "lead_on": False, "bv_on": False, "keys_on": True, "guitars_on": True, "drums_on": True, "bass_on": True},
    ),
    ServiceScene(
        "Gospel Groove · Sunday Soul (Rhodes + Organ + Full Band)",
        2,
        18.0,
        {"lead_on": True, "bv_on": True, "keys_on": True, "guitars_on": True, "drums_on": True, "bass_on": True},
    ),
    ServiceScene(
        "Applause Transition",
        2,
        7.0,
        {"lead_on": False, "bv_on": False, "keys_on": False, "guitars_on": False, "drums_on": False, "bass_on": False, "applause_on": True},
    ),
    ServiceScene(
        "Sermon (Spoken Word Solo)",
        0,
        14.0,
        {"pastor_on": True, "lead_on": False, "bv_on": False, "keys_on": False, "guitars_on": False, "drums_on": False, "bass_on": False},
    ),
    ServiceScene(
        "Unplugged Folk · Acoustic Campfire",
        3,
        18.0,
        {"lead_on": True, "bv_on": True, "keys_on": True, "guitars_on": True, "drums_on": True, "bass_on": True},
    ),
]


def find_output_device(hint: str | None, *, min_channels: int) -> int | None:
    if hint is None or hint.lower() == "default":
        return None
    for idx, dev in enumerate(sd.query_devices()):
        name = str(dev.get("name", ""))
        if hint.lower() in name.lower() and int(dev.get("max_output_channels", 0)) >= min_channels:
            return idx
    raise RuntimeError(
        f"Audio output device matching '{hint}' with >= {min_channels} channels not found! "
        "Run `brew reinstall --cask blackhole-16ch` or pass --audition for stereo speakers."
    )


def _downmix_to_stereo(block16: np.ndarray) -> np.ndarray:
    """Create a pleasant stereo audition mix from a 16ch pre-fader block."""
    left = (
        0.75 * block16[:, 0]
        + 0.72 * block16[:, 1]
        + 0.45 * block16[:, 2]
        + 0.25 * block16[:, 3]
        + 0.75 * block16[:, 4]
        + 0.72 * block16[:, 6]
        + 0.78 * block16[:, 8]
        + 0.75 * block16[:, 10]
        + 0.45 * block16[:, 12]
    )
    right = (
        0.75 * block16[:, 0]
        + 0.72 * block16[:, 1]
        + 0.25 * block16[:, 2]
        + 0.45 * block16[:, 3]
        + 0.75 * block16[:, 5]
        + 0.72 * block16[:, 7]
        + 0.78 * block16[:, 9]
        + 0.75 * block16[:, 11]
        + 0.45 * block16[:, 13]
    )
    stereo = np.stack([left, right], axis=1) * 0.42
    return np.tanh(stereo).astype(np.float32)


class SoundPackStreamer:
    """Real-time 16-channel BlackHole (or 2-channel audition) Sound Pack streamer."""

    def __init__(
        self,
        pack: PackagedSoundPack,
        *,
        device_idx: int | None,
        audition_stereo: bool = False,
        mode: str = "service",
        fixed_tune: int | None = None,
        jukebox_sec: float = 24.0,
    ) -> None:
        self.pack = pack
        self.device_idx = device_idx
        self.audition_stereo = audition_stereo
        self.mode = mode
        self.fixed_tune = fixed_tune
        self.jukebox_sec = jukebox_sec

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tune_idx = (fixed_tune - 1) % len(pack.tunes) if fixed_tune is not None else 0
        self._pastor_on = False
        self._applause_until = 0.0
        self._feedback_until = 0.0
        self._sermon_solo = False
        self._acoustic_only = False
        self._manual_override = fixed_tune is not None

    def handle_key(self, ch: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if ch in ("q", "\x03", "\x04"):
                self._stop.set()
                return False
            if ch in "12345":
                idx = int(ch) - 1
                if idx < len(self.pack.tunes):
                    self._tune_idx = idx
                    self._manual_override = True
                    self._sermon_solo = False
                    t = self.pack.tunes[idx]
                    logger.info("[jukebox] Switched to Tune %d: %s (%s · %.0f BPM · %s)", idx + 1, t.title, t.genre, t.bpm, t.key)
            elif ch == "0":
                self._manual_override = False
                logger.info("[jukebox] Resumed auto-cycle mode (%s)", self.mode)
            elif ch == "p":
                self._pastor_on = not self._pastor_on
                self._manual_override = True
                logger.info("[jukebox] Pastor Mic (Ch 1): %s", "ON" if self._pastor_on else "OFF")
            elif ch == "s":
                self._sermon_solo = not self._sermon_solo
                self._pastor_on = self._sermon_solo
                self._manual_override = True
                logger.info("[jukebox] Sermon Solo: %s", "ON" if self._sermon_solo else "OFF")
            elif ch == "b":
                self._acoustic_only = not self._acoustic_only
                self._manual_override = True
                logger.info("[jukebox] Band Mode: %s", "ACOUSTIC (WORSHIP_LOW)" if self._acoustic_only else "FULL BAND (WORSHIP_HIGH)")
            elif ch == "a":
                self._applause_until = now + 5.0
                logger.info("[jukebox] Room Applause burst (Ch 13-14) for 5s")
            elif ch == "f":
                self._feedback_until = now + 2.5
                logger.info("[jukebox] Injecting 2.48 kHz feedback howl on Lead Vocal (Ch 2) for 2.5s!")
        return True

    def run_blocking(self, *, interactive: bool = False) -> None:
        rng = np.random.default_rng(2026)
        cursor = 0
        speech_cursor = 0
        scene_idx = 0
        scene_start = time.monotonic()
        jukebox_start = scene_start

        channels = 2 if self.audition_stereo else 16
        if interactive and sys.stdin.isatty():
            print(
                "\n=== AetherMix Live Sound Pack Controls ===\n"
                "  [1-5] Switch Tune   [p] Toggle Pastor Mic   [b] Acoustic/Full Band\n"
                "  [s]   Sermon Solo   [a] Applause Burst      [f] 2.48 kHz Feedback Test\n"
                "  [0]   Auto-Cycle    [q] Quit\n",
                flush=True,
            )

        old_term = None
        if interactive and sys.stdin.isatty():
            old_term = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

        try:
            if self.mode == "service" and not self._manual_override:
                logger.info("Live BlackHole scene -> %s", SERVICE_PLAYLIST[0].label)
            else:
                t = self.pack.tunes[self._tune_idx]
                logger.info("Streaming Tune %d -> %s (%s · %.0f BPM · %s)", self._tune_idx + 1, t.title, t.genre, t.bpm, t.key)

            with sd.OutputStream(
                device=self.device_idx,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_FRAMES,
                channels=channels,
                dtype="float32",
            ) as stream:
                while not self._stop.is_set():
                    if interactive and sys.stdin.isatty():
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if rlist:
                            key = sys.stdin.read(1).lower()
                            if not self.handle_key(key):
                                break

                    now = time.monotonic()
                    with self._lock:
                        applause_burst = now < self._applause_until
                        fb_active = now < self._feedback_until
                        manual = self._manual_override
                        tune_idx = self._tune_idx
                        pastor_on = self._pastor_on
                        sermon_solo = self._sermon_solo
                        acoustic_only = self._acoustic_only

                    if manual:
                        flags: dict[str, bool] = {
                            "pastor_on": pastor_on,
                            "lead_on": not sermon_solo,
                            "bv_on": (not sermon_solo) and (not acoustic_only),
                            "keys_on": not sermon_solo,
                            "guitars_on": not sermon_solo,
                            "drums_on": (not sermon_solo) and (not acoustic_only),
                            "bass_on": not sermon_solo,
                            "applause_on": applause_burst,
                        }
                    elif self.mode == "jukebox":
                        if now - jukebox_start >= self.jukebox_sec:
                            jukebox_start = now
                            with self._lock:
                                self._tune_idx = (self._tune_idx + 1) % len(self.pack.tunes)
                                tune_idx = self._tune_idx
                            t = self.pack.tunes[tune_idx]
                            logger.info("Jukebox -> Tune %d: %s (%s · %.0f BPM · %s)", tune_idx + 1, t.title, t.genre, t.bpm, t.key)
                        flags = {
                            "pastor_on": False,
                            "lead_on": True,
                            "bv_on": True,
                            "keys_on": True,
                            "guitars_on": True,
                            "drums_on": True,
                            "bass_on": True,
                            "applause_on": applause_burst,
                        }
                    else:
                        sc = SERVICE_PLAYLIST[scene_idx]
                        if now - scene_start >= sc.duration_s:
                            scene_idx = (scene_idx + 1) % len(SERVICE_PLAYLIST)
                            scene_start = now
                            sc = SERVICE_PLAYLIST[scene_idx]
                            logger.info("Live BlackHole scene -> %s", sc.label)
                        tune_idx = sc.tune_idx
                        flags = dict(sc.flags)
                        if applause_burst:
                            flags["applause_on"] = True

                    block16 = compose_16ch_block(
                        self.pack,
                        tune_idx,
                        cursor,
                        speech_cursor=speech_cursor,
                        frames=BLOCK_FRAMES,
                        feedback_hz=2480.0 if fb_active else None,
                        feedback_gain=0.22 if fb_active else 0.0,
                        rng=rng,
                        **flags,
                    )
                    if self.audition_stereo:
                        stream.write(_downmix_to_stereo(block16))
                    else:
                        stream.write(block16)
                    cursor += BLOCK_FRAMES
                    if flags.get("pastor_on"):
                        speech_cursor += BLOCK_FRAMES
        finally:
            if old_term is not None and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)


async def run_full_live_stack(args: argparse.Namespace) -> None:
    find_output_device(args.output_device, min_channels=16)
    token = ensure_session_token(DEFAULT_TOKEN_PATH)

    # 1. Spawn BlackHole 16ch streamer in an ISOLATED child process (avoids macOS AUHAL err=-50)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--player-only",
        "--mode",
        args.mode,
        "--output-device",
        args.output_device,
        "--jukebox-sec",
        str(args.jukebox_sec),
    ]
    if args.tune is not None:
        cmd.extend(["--tune", str(args.tune)])

    streamer_proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(REPO_ROOT))
    await asyncio.sleep(0.7)

    # 2. Standalone MidiBridgeDaemon with REAL CoreMIDI Virtual Port ('AetherMix To MainStage')
    bridge_cfg = BridgeConfig.from_service_plan_file(DEFAULT_SERVICE_PLAN_PATH)
    bridge_daemon = MidiBridgeDaemon(
        bridge_cfg,
        socket_path=DEFAULT_BRIDGE_SOCKET_PATH,
        state_path=DEFAULT_BRIDGE_STATE_PATH,
        token_path=DEFAULT_TOKEN_PATH,
        midi_backend=RtMidiBackend(),
    )
    await bridge_daemon.start()
    logger.info("REAL CoreMIDI Virtual Port '%s' is LIVE (visible in Apple MainStage!)", bridge_cfg.mainstage_out_port)

    # 3. AetherMixEngine tapping REAL BlackHole 16ch CoreAudio input + REAL Silero VAD ONNX
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
        logger.info("AetherMix Engine LIVE on ws://127.0.0.1:8765/ws (token=%s...)", token[:8])
        await server.serve()
    finally:
        streamer_proc.terminate()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(streamer_proc.wait(), timeout=3.0)
        await bridge_daemon.stop()


def print_catalog(pack: PackagedSoundPack) -> None:
    print("\n=== AetherMix Packaged Multi-Instrument Sound Pack ===")
    for i, t in enumerate(pack.tunes, start=1):
        print(f"  [{i}] {t.title:<20s} | {t.genre:<30s} | {t.bpm:5.0f} BPM | {t.key:<8s} | {t.duration_s:5.1f}s loop")
    print("  + Ch 1:  Real human speech (male & female)   | Ch 13-14: Stereo room applause")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Play AetherMix Packaged Multi-Instrument Sound Pack into MainStage")
    parser.add_argument("--list", action="store_true", help="List packaged tunes and exit")
    parser.add_argument("--mode", choices=["service", "jukebox"], default="service", help="Playback mode (default: service)")
    parser.add_argument("--tune", type=int, choices=[1, 2, 3, 4, 5], default=None, help="Lock to a single tune (1..5)")
    parser.add_argument("--jukebox-sec", type=float, default=24.0, help="Seconds per tune in jukebox auto-cycle mode")
    parser.add_argument("--output-device", default="BlackHole 16ch", help="CoreAudio output device name hint")
    parser.add_argument("--player-only", action="store_true", help="Stream audio only (do not start AetherMix engine/bridge)")
    parser.add_argument("--audition", action="store_true", help="Preview stereo mix on default Mac speakers (no BlackHole)")
    parser.add_argument("--interactive", action="store_true", help="Enable live keyboard controls (with --player-only or --audition)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pack = load_packaged_soundpack()

    if args.list:
        print_catalog(pack)
        return 0

    if args.audition or args.player_only:
        dev_idx = None if args.audition else find_output_device(args.output_device, min_channels=16)
        streamer = SoundPackStreamer(
            pack,
            device_idx=dev_idx,
            audition_stereo=args.audition,
            mode=args.mode,
            fixed_tune=args.tune,
            jukebox_sec=args.jukebox_sec,
        )
        with contextlib.suppress(KeyboardInterrupt):
            streamer.run_blocking(interactive=args.interactive or args.audition)
        return 0

    print_catalog(pack)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_full_live_stack(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
