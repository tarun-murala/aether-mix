"""Isolated Background-QoS 1-thread CPU CLAP semantic scene classifier subprocess (SPEC v5.1).

Runs `laion/clap-htsat-unfused` in a dedicated `multiprocessing('spawn')` subprocess with:
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`
- Background QoS (`os.nice(10)` + macOS `PRIO_DARWIN_BG` best-effort)

Gracefully falls back to disabled/unloaded (`latest_probs() -> (None, False)`) without crashing
when `enabled=False`, or when `torch`/`transformers` or cached model weights are absent.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import multiprocessing
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from engine.core.protocol import ServicePhase

logger = logging.getLogger(__name__)

DEFAULT_CLAP_MODEL_DIR = Path.home() / ".aethermix" / "models" / "clap"

# Exact SPEC v5.1 §4 CLAP semantic prompts
CLAP_PROMPTS: dict[ServicePhase, str] = {
    ServicePhase.SPOKEN_WORD: "pastor or speaker talking alone without music",
    ServicePhase.PRAYER_UNDERSCORE: "soft ambient keyboard pad underneath spoken prayer",
    ServicePhase.WORSHIP_LOW: "acoustic guitar or piano gentle worship verse",
    ServicePhase.WORSHIP_HIGH: "full church band with drums electric guitar and loud singing",
    ServicePhase.TRANSITION: "silence applause or stage transition between songs",
}

_PHASE_ORDER: tuple[ServicePhase, ...] = (
    ServicePhase.SPOKEN_WORD,
    ServicePhase.PRAYER_UNDERSCORE,
    ServicePhase.WORSHIP_LOW,
    ServicePhase.WORSHIP_HIGH,
    ServicePhase.TRANSITION,
)


def _apply_background_qos() -> None:
    """Lower subprocess priority so CLAP never preempts MainStage AudioWorkgroup threads."""
    try:
        os.nice(10)
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL("libSystem.B.dylib")
            # PRIO_DARWIN_PROCESS = 4, PRIO_DARWIN_BG = 0x1000
            libc.setpriority(4, 0, 0x1000)
        except (OSError, AttributeError) as exc:
            logger.debug("Unable to set PRIO_DARWIN_BG: %s", exc)


def _clap_subprocess_main(
    model_dir_str: str,
    in_queue: Any,
    out_queue: Any,
    ready_event: Any,
) -> None:
    """Entry point executed inside the isolated `spawn` subprocess."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _apply_background_qos()

    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except (ImportError, OSError, RuntimeError) as exc:
        out_queue.put({"status": "error", "error": f"import_failed: {exc}"})
        return

    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        processor = ClapProcessor.from_pretrained(model_dir_str, local_files_only=True)
        model = ClapModel.from_pretrained(model_dir_str, local_files_only=True)
        model.eval()
        model.to("cpu")

        prompts = [CLAP_PROMPTS[p] for p in _PHASE_ORDER]
        with torch.no_grad():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            text_embeds = model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        ready_event.set()
        out_queue.put({"status": "ready"})
    except (OSError, RuntimeError, ValueError) as exc:
        out_queue.put({"status": "error", "error": f"model_load_failed: {exc}"})
        return

    while True:
        try:
            item = in_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        if item is None:
            break

        audio_np, sample_rate = item
        try:
            mono = np.asarray(audio_np, dtype=np.float32).reshape(-1)
            # CLAP HTSAT expects 48 kHz mono audio
            if sample_rate != 48000 and mono.size > 0:
                from scipy import signal as sp_signal

                n_out = max(1, round(mono.size * 48000.0 / float(sample_rate)))
                mono = sp_signal.resample(mono, n_out).astype(np.float32)

            with torch.no_grad():
                audio_inputs = processor(
                    audios=[mono], sampling_rate=48000, return_tensors="pt", padding=True
                )
                audio_embeds = model.get_audio_features(**audio_inputs)
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                logit_scale = model.logit_scale_a.exp().clamp(max=100.0)
                logits = (audio_embeds @ text_embeds.T) * logit_scale
                probs_tensor = torch.softmax(logits[0], dim=-1).cpu().numpy()

            probs = {
                phase.value: float(probs_tensor[i]) for i, phase in enumerate(_PHASE_ORDER)
            }
            # Keep only the latest inference result in out_queue
            while True:
                try:
                    out_queue.get_nowait()
                except queue.Empty:
                    break
            out_queue.put({"status": "ok", "probs": probs, "ts_ms": int(time.time() * 1000)})
        except (RuntimeError, ValueError, OSError) as exc:
            out_queue.put({"status": "error", "error": f"inference_failed: {exc}"})


class CLAPWorker:
    """Manages the isolated 1-thread Background-QoS CLAP subprocess with graceful fallback."""

    def __init__(
        self,
        enabled: bool = True,
        model_dir: Path | None = None,
        min_submit_interval_s: float = 2.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_CLAP_MODEL_DIR
        self._min_submit_interval_s = float(min_submit_interval_s)

        self._ctx = multiprocessing.get_context("spawn")
        self._proc: Any = None
        self._in_queue: Any = None
        self._out_queue: Any = None
        self._ready_event: Any = None

        self._latest_probs: dict[ServicePhase, float] | None = None
        self._healthy: bool = False
        self._started: bool = False
        self._last_submit_s: float = 0.0
        self._last_error: str | None = None

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @staticmethod
    def _dependencies_available() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )

    def _weights_available(self) -> bool:
        return self.model_dir.exists() and (self.model_dir / "config.json").exists()

    def start(self) -> None:
        """Start the isolated CLAP subprocess if enabled and dependencies + weights exist."""
        if self._started:
            return
        self._started = True
        self._healthy = False
        self._latest_probs = None
        self._last_error = None

        if not self.enabled:
            logger.info("CLAPWorker disabled by configuration")
            return

        if not self._dependencies_available():
            logger.info("torch/transformers not installed; CLAPWorker running in graceful unloaded mode")
            self._last_error = "ml_dependencies_missing"
            return

        if not self._weights_available():
            logger.info(
                "CLAP weights not found in %s; CLAPWorker running in graceful unloaded mode",
                self.model_dir,
            )
            self._last_error = "model_weights_missing"
            return

        try:
            self._in_queue = self._ctx.Queue(maxsize=2)
            self._out_queue = self._ctx.Queue(maxsize=4)
            self._ready_event = self._ctx.Event()
            self._proc = self._ctx.Process(
                target=_clap_subprocess_main,
                args=(str(self.model_dir), self._in_queue, self._out_queue, self._ready_event),
                name="aethermix-clap-worker",
                daemon=True,
            )
            self._proc.start()
            logger.info("Started isolated CLAP subprocess (pid=%s)", self._proc.pid)
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to spawn CLAP subprocess (%s); falling back to disabled", exc)
            self._last_error = str(exc)
            self._cleanup_process()

    def stop(self) -> None:
        """Stop the isolated CLAP subprocess cleanly."""
        self._started = False
        self._healthy = False
        self._cleanup_process()

    def _cleanup_process(self) -> None:
        if self._in_queue is not None:
            try:
                self._in_queue.put_nowait(None)
            except (queue.Full, OSError, ValueError) as exc:
                logger.debug("Ignoring queue sentinel error during CLAP stop: %s", exc)
        if self._proc is not None:
            try:
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
                if self._proc.is_alive():
                    self._proc.kill()
            except (OSError, ValueError) as exc:
                logger.debug("Ignoring process cleanup error during CLAP stop: %s", exc)
            finally:
                self._proc = None
        for q in (self._in_queue, self._out_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except (OSError, ValueError) as exc:
                    logger.debug("Ignoring queue close error during CLAP stop: %s", exc)
        self._in_queue = None
        self._out_queue = None
        self._ready_event = None

    def submit_audio(self, mono_3s: np.ndarray, sample_rate: int = 48000) -> None:
        """Non-blocking submission of a 3-second pre-fader mono buffer at 0.5 Hz cadence."""
        if not self._started or self._proc is None or self._in_queue is None:
            return
        if not self._proc.is_alive():
            self._healthy = False
            return

        now = time.monotonic()
        if (now - self._last_submit_s) < self._min_submit_interval_s:
            return
        self._last_submit_s = now

        # Drop any stale queued item so inference always runs on freshest 3s window
        while True:
            try:
                self._in_queue.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break

        try:
            self._in_queue.put_nowait((np.asarray(mono_3s, dtype=np.float32).copy(), int(sample_rate)))
        except queue.Full:
            pass
        except (OSError, ValueError) as exc:
            logger.debug("CLAP submit_audio failed: %s", exc)

    def latest_probs(self) -> tuple[dict[ServicePhase, float] | None, bool]:
        """Return `(probs_by_phase | None, is_healthy)` without blocking."""
        if not self._started or self._proc is None or self._out_queue is None:
            return None, False

        if not self._proc.is_alive():
            self._healthy = False
            return None, False

        if self._ready_event is not None and self._ready_event.is_set():
            self._healthy = True

        while True:
            try:
                msg = self._out_queue.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break

            status = msg.get("status")
            if status == "ready":
                self._healthy = True
                self._last_error = None
            elif status == "ok":
                raw_probs = msg.get("probs", {})
                parsed = {
                    phase: float(raw_probs.get(phase.value, 0.0)) for phase in _PHASE_ORDER
                }
                total = sum(parsed.values())
                if total > 0.0:
                    parsed = {k: v / total for k, v in parsed.items()}
                self._latest_probs = parsed
                self._healthy = True
                self._last_error = None
            elif status == "error":
                self._last_error = str(msg.get("error", "unknown"))
                logger.warning("CLAP subprocess error: %s", self._last_error)

        if not self._healthy:
            return None, False
        return (dict(self._latest_probs) if self._latest_probs is not None else None), True
