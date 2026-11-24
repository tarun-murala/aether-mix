#!/usr/bin/env python3
"""Record 16ch pre-fader church service audio + JSON phase timeline annotations in SHADOW mode (SPEC §9 Phase 1)."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core.protocol import ServicePhase, ServicePlan

PHASE_HOTKEYS: dict[str, ServicePhase] = {
    "1": ServicePhase.SPOKEN_WORD,
    "2": ServicePhase.PRAYER_UNDERSCORE,
    "3": ServicePhase.WORSHIP_LOW,
    "4": ServicePhase.WORSHIP_HIGH,
    "5": ServicePhase.TRANSITION,
}


def _write_wav_16ch(path: Path, sample_rate: int, blocks: list[np.ndarray]) -> int:
    """Write float32 16ch blocks to 16-bit PCM WAV (or FLAC via soundfile if suffix is .flac)."""
    data = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 16), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".flac":
        try:
            import soundfile as sf  # type: ignore

            sf.write(str(path), data, sample_rate, format="FLAC", subtype="PCM_16")
            return int(data.shape[0])
        except (ImportError, OSError, RuntimeError):
            path = path.with_suffix(".wav")

    clipped = np.clip(data, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(data.shape[1] if data.ndim == 2 else 16)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return int(data.shape[0])


def record_shadow_session(
    plan: ServicePlan,
    out_dir: Path,
    duration_s: float = 10.0,
    initial_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    audio_format: str = "wav",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Capture 16ch pre-fader audio + phase annotations into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("shadow_%Y%m%d_%H%M%S")
    audio_path = out_dir / f"{session_id}.{audio_format}"
    meta_path = out_dir / f"{session_id}_annotations.json"

    annotations: list[dict[str, Any]] = [
        {"offset_s": 0.0, "phase": initial_phase.value, "note": "session_start"}
    ]
    blocks: list[np.ndarray] = []
    start_ts_ms = int(time.time() * 1000)

    if dry_run:
        sr = plan.sample_rate
        total_frames = int(sr * duration_s)
        t = np.arange(total_frames, dtype=np.float32) / float(sr)
        buf = np.zeros((total_frames, 16), dtype=np.float32)
        # Synthesize gentle spoken mic + keys pad + master mix for smoke test
        buf[:, 0] = 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
        buf[:, 4] = 0.03 * np.sin(2.0 * np.pi * 330.0 * t)
        buf[:, 5] = 0.03 * np.sin(2.0 * np.pi * 440.0 * t)
        buf[:, 14] = buf[:, 0] + buf[:, 4]
        buf[:, 15] = buf[:, 0] + buf[:, 5]
        blocks.append(buf)
        if duration_s >= 2.0:
            annotations.append(
                {
                    "offset_s": round(duration_s / 2.0, 2),
                    "phase": ServicePhase.PRAYER_UNDERSCORE.value,
                    "note": "dry_run_marker",
                }
            )
    else:
        import sounddevice as sd  # type: ignore

        devs = sd.query_devices()
        dev_idx = next(
            (
                i
                for i, d in enumerate(devs)
                if plan.audio_device_hint.lower() in d["name"].lower() and d["max_input_channels"] >= 16
            ),
            None,
        )
        if dev_idx is None:
            raise RuntimeError(f"16ch audio device '{plan.audio_device_hint}' not found.")

        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        stop_requested = False

        def _on_sigint(_signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        prev_handler = signal.signal(signal.SIGINT, _on_sigint)

        def _callback(indata: np.ndarray, _frames: int, _time_info: Any, _status: Any) -> None:
            try:
                audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass

        t0 = time.monotonic()
        try:
            with sd.InputStream(
                device=dev_idx,
                channels=16,
                samplerate=plan.sample_rate,
                blocksize=960,
                dtype="float32",
                callback=_callback,
            ):
                print(f"Recording 16ch SHADOW session to {audio_path} (Ctrl+C to stop early)...")
                while not stop_requested:
                    elapsed = time.monotonic() - t0
                    if duration_s > 0 and elapsed >= duration_s:
                        break
                    try:
                        blk = audio_q.get(timeout=0.1)
                        blocks.append(blk)
                    except queue.Empty:
                        continue
        finally:
            signal.signal(signal.SIGINT, prev_handler)

    frames_written = _write_wav_16ch(audio_path, plan.sample_rate, blocks)
    manifest = {
        "session_id": session_id,
        "mode": "SHADOW_OBSERVE",
        "started_at_ms": start_ts_ms,
        "sample_rate": plan.sample_rate,
        "channels": 16,
        "frames_written": frames_written,
        "duration_s": round(frames_written / float(plan.sample_rate), 3),
        "audio_file": audio_path.name,
        "channel_map": {ch.id: ch.tap_channels for ch in plan.channels},
        "annotations": annotations,
    }
    meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record 16ch pre-fader SHADOW service audio + annotations")
    parser.add_argument(
        "--service-plan",
        type=Path,
        default=REPO_ROOT / "engine" / "schemas" / "service_plan.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "engine" / "storage" / "shadow_recordings",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (0 = until Ctrl+C)")
    parser.add_argument("--format", choices=["wav", "flac"], default="wav")
    parser.add_argument("--initial-phase", choices=[p.value for p in ServicePhase], default="SPOKEN_WORD")
    parser.add_argument("--dry-run", action="store_true", help="Synthesize 16ch audio without BlackHole device")
    args = parser.parse_args(argv)

    plan = ServicePlan.model_validate_json(args.service_plan.read_text(encoding="utf-8"))
    manifest = record_shadow_session(
        plan=plan,
        out_dir=args.out_dir,
        duration_s=args.duration,
        initial_phase=ServicePhase(args.initial_phase),
        audio_format=args.format,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
