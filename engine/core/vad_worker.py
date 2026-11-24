"""Voice Activity Detection (1-thread ONNX Silero VAD + DSP fallback) & Sung-Vocal Detector.

Implements SPEC v5.1 §4 & §5.3:
- Speech VAD on `vad_enabled` channels (`ch1_pastor`, `ch2_lead_vox`):
  Uses `onnxruntime` Silero VAD (`intra_op_num_threads=1`, `inter_op_num_threads=1`) at
  16 kHz / 512-sample chunks when `onnxruntime` and `silero_vad.onnx` are available,
  falling back cleanly to a deterministic DSP speech-envelope + spectral-centroid + ZCR
  detector when running without ONNX weights.
- Sung-vocal activity on `vocal_lead` & `vocal_bg` channels (`Ch 2-4`):
  Uses gain-normalized 200-4000 Hz RMS + normalized autocorrelation harmonicity salience
  above bleed floor (`speech_to_band_snr_db >= +3.0 dB`), NOT Silero VAD.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from engine.core.audio_tap import DB_FLOOR, VOCAL_16K_SR, AudioFrameMetrics
from engine.core.protocol import ChannelConfig, ChannelRole, ServicePlan

logger = logging.getLogger(__name__)

DEFAULT_SILERO_MODEL_PATH = Path.home() / ".aethermix" / "models" / "silero_vad.onnx"

SUNG_VOCAL_ROLES: frozenset[ChannelRole] = frozenset(
    {
        ChannelRole.VOCAL_LEAD,
        ChannelRole.VOCAL_BG,
    }
)

# Sung vocal pitch range: 80 Hz (E2) to 1000 Hz (C6) at 16 kHz
_HARMONICITY_MIN_LAG = int(VOCAL_16K_SR / 1000.0)  # 16 samples
_HARMONICITY_MAX_LAG = int(VOCAL_16K_SR / 80.0)  # 200 samples
_HARMONICITY_THRESHOLD = 0.45
_SUNG_VOCAL_SNR_MIN_DB = 3.0
_SUNG_VOCAL_NORM_MIN_DB = -24.0
_SUNG_VOCAL_ABS_MIN_DB = -55.0

_SILERO_CHUNK_SAMPLES = 512


def _power_to_db(power: float) -> float:
    if power <= 1e-12:
        return DB_FLOOR
    return max(DB_FLOOR, 10.0 * math.log10(power))


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_autocorr_harmonicity(
    pcm_16k: np.ndarray,
    min_lag: int = _HARMONICITY_MIN_LAG,
    max_lag: int = _HARMONICITY_MAX_LAG,
) -> float:
    """Normalized autocorrelation peak salience in [0.0, 1.0] over human vocal pitch lags."""
    n = pcm_16k.shape[0]
    if n <= min_lag + 4:
        return 0.0
    x = pcm_16k.astype(np.float64) - float(np.mean(pcm_16k))
    sq = x * x
    total_energy = float(np.sum(sq))
    if total_energy <= 1e-10:
        return 0.0

    # O(N log N) exact autocorrelation via zero-padded FFT
    n_fft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, n=n_fft)
    ac = np.fft.irfft(spec.real * spec.real + spec.imag * spec.imag, n=n_fft)[:n]

    # Exact energy normalization for each lag L: sqrt(sum(x[:n-L]^2) * sum(x[L:]^2))
    csum = np.empty(n + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(sq, out=csum[1:])

    hi_lag = min(max_lag, n // 2)
    if hi_lag <= min_lag:
        return 0.0
    lags = np.arange(min_lag, hi_lag + 1)
    e_left = csum[n - lags]
    e_right = csum[n] - csum[lags]
    denom = np.sqrt(np.maximum(e_left * e_right, 0.0)) + 1e-12
    nac = ac[lags] / denom
    return float(np.clip(np.max(nac), 0.0, 1.0))


class VADWorker:
    """Speech VAD (ONNX Silero or deterministic DSP fallback) + Sung-Vocal Harmonicity Detector."""

    def __init__(self, plan: ServicePlan, model_path: Path | None = None) -> None:
        self.plan = plan
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_SILERO_MODEL_PATH

        self._vad_channels: list[ChannelConfig] = [ch for ch in plan.channels if ch.vad_enabled]
        self._sung_channels: list[ChannelConfig] = [
            ch for ch in plan.channels if ch.role in SUNG_VOCAL_ROLES
        ]

        # 16 kHz SOS bandpass filters for speech (300-3400 Hz) and sung vocals (200-4000 Hz)
        nyq_16k = 0.5 * VOCAL_16K_SR
        self._speech_bp_sos = signal.butter(
            4, [300.0 / nyq_16k, 3400.0 / nyq_16k], btype="bandpass", output="sos"
        )
        self._sung_bp_sos = signal.butter(
            4, [200.0 / nyq_16k, 4000.0 / nyq_16k], btype="bandpass", output="sos"
        )

        # ONNX Silero VAD session & per-channel non-overlapping 512-sample FIFO / RNN state
        self._ort_session: Any = None
        self._silero_v5: bool = True
        self._silero_state: dict[str, Any] = {}
        self._silero_context: dict[str, np.ndarray] = {
            ch.id: np.zeros((1, 64), dtype=np.float32) for ch in self._vad_channels
        }
        self._silero_fifo: dict[str, np.ndarray] = {
            ch.id: np.empty(0, dtype=np.float32) for ch in self._vad_channels
        }
        self._last_silero_prob: dict[str, float] = {ch.id: 0.0 for ch in self._vad_channels}
        self._last_onnx_ts_ms: dict[str, int] = {ch.id: 0 for ch in self._vad_channels}
        self._using_onnx: bool = False

        # Smoothers & diagnostics
        self._vad_ema: dict[str, float] = {ch.id: 0.0 for ch in plan.channels}
        self.last_harmonicity: dict[str, float] = {ch.id: 0.0 for ch in self._sung_channels}
        self._running: bool = False

    @property
    def using_onnx(self) -> bool:
        return self._using_onnx

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._try_load_onnx()

    def stop(self) -> None:
        self._running = False
        self._ort_session = None
        self._silero_state.clear()
        self._silero_fifo = {ch.id: np.empty(0, dtype=np.float32) for ch in self._vad_channels}
        self._last_silero_prob = {ch.id: 0.0 for ch in self._vad_channels}
        self._last_onnx_ts_ms = {ch.id: 0 for ch in self._vad_channels}
        self._using_onnx = False

    def _try_load_onnx(self) -> None:
        try:
            exists = self.model_path.exists()
        except OSError as exc:
            logger.info("Silero VAD ONNX path inaccessible (%s); using DSP fallback", exc)
            self._using_onnx = False
            return
        if not exists:
            logger.info("Silero VAD ONNX model not found at %s; using DSP fallback", self.model_path)
            self._using_onnx = False
            return
        try:
            import onnxruntime as ort
        except ImportError:
            logger.info("onnxruntime not installed; using deterministic DSP VAD fallback")
            self._using_onnx = False
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            input_names = {inp.name for inp in session.get_inputs()}
            self._silero_v5 = "state" in input_names
            self._ort_session = session
            self._reset_silero_states()
            self._using_onnx = True
            logger.info("Loaded 1-thread ONNX Silero VAD from %s (v5=%s)", self.model_path, self._silero_v5)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to initialize ONNX Silero VAD (%s); using DSP fallback", exc)
            self._ort_session = None
            self._using_onnx = False

    def _reset_silero_states(self) -> None:
        self._silero_state.clear()
        for ch in self._vad_channels:
            self._silero_fifo[ch.id] = np.empty(0, dtype=np.float32)
            self._silero_context[ch.id] = np.zeros((1, 64), dtype=np.float32)
            self._last_silero_prob[ch.id] = 0.0
            self._last_onnx_ts_ms[ch.id] = 0
            if self._silero_v5:
                self._silero_state[ch.id] = np.zeros((2, 1, 128), dtype=np.float32)
            else:
                self._silero_state[ch.id] = (
                    np.zeros((2, 1, 64), dtype=np.float32),
                    np.zeros((2, 1, 64), dtype=np.float32),
                )

    # ------------------------------------------------------------------
    # Frame Processing
    # ------------------------------------------------------------------

    def process_frame(
        self, metrics: AudioFrameMetrics
    ) -> tuple[dict[str, float], dict[str, bool]]:
        """Return `(vad_probs, sung_vocal_active)` for the current AudioFrameMetrics frame."""
        vad_probs: dict[str, float] = {ch.id: 0.0 for ch in self.plan.channels}
        sung_vocal_active: dict[str, bool] = {}

        # 1. Speech VAD on vad_enabled channels
        new_pcm_map = getattr(metrics, "vocal_new_pcm_16k", None)
        for ch in self._vad_channels:
            pcm = metrics.vocal_pcm_16k.get(ch.id)
            if pcm is None or pcm.size == 0:
                self._vad_ema[ch.id] = 0.0
                vad_probs[ch.id] = 0.0
                continue

            if self._using_onnx and self._ort_session is not None:
                new_pcm = new_pcm_map.get(ch.id) if isinstance(new_pcm_map, dict) else None
                prob = self._infer_silero_onnx(ch.id, pcm, new_pcm, metrics.ts_ms)
            else:
                prob = self._infer_dsp_vad(ch, pcm, metrics)
            vad_probs[ch.id] = prob

        # 2. Sung-vocal activity on Ch 2-4 (vocal_lead / vocal_bg) via 200-4000Hz RMS + harmonicity
        for ch in self._sung_channels:
            pcm = metrics.vocal_pcm_16k.get(ch.id)
            active, harm = self._detect_sung_vocal(ch, pcm, metrics)
            self.last_harmonicity[ch.id] = harm
            sung_vocal_active[ch.id] = active

        return vad_probs, sung_vocal_active

    def _run_silero_chunk(self, channel_id: str, chunk_512: np.ndarray) -> float:
        """Execute one non-overlapping 512-sample (32 ms) Silero ONNX RNN step.

        Silero VAD v5 requires prepending the trailing 64 samples of the previous
        chunk (context window), producing an input tensor of shape (1, 576).
        """
        x_512 = chunk_512.astype(np.float32, copy=False)[np.newaxis, :]  # (1, 512)
        sr = np.array(VOCAL_16K_SR, dtype=np.int64)
        if self._silero_v5:
            ctx = self._silero_context.setdefault(channel_id, np.zeros((1, 64), dtype=np.float32))
            x_v5 = np.concatenate([ctx, x_512], axis=1)  # (1, 576)
            self._silero_context[channel_id] = x_v5[:, -64:].copy()
            state = self._silero_state.setdefault(channel_id, np.zeros((2, 1, 128), dtype=np.float32))
            out, new_state = self._ort_session.run(None, {"input": x_v5, "state": state, "sr": sr})
            self._silero_state[channel_id] = new_state
        else:
            h, c = self._silero_state.setdefault(
                channel_id,
                (np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)),
            )
            out, new_h, new_c = self._ort_session.run(
                None, {"input": x_512, "h": h, "c": c, "sr": sr}
            )
            self._silero_state[channel_id] = (new_h, new_c)
        return float(np.clip(np.asarray(out).reshape(-1)[0], 0.0, 1.0))

    def _infer_silero_onnx(
        self,
        channel_id: str,
        pcm_16k: np.ndarray,
        new_pcm_16k: np.ndarray | None,
        ts_ms: int,
    ) -> float:
        """Feed strictly non-overlapping 512-sample (32 ms / ~31 Hz) hops into Silero ONNX RNN."""
        if new_pcm_16k is not None:
            fifo = self._silero_fifo.get(channel_id, np.empty(0, dtype=np.float32))
            if new_pcm_16k.size > 0:
                new_arr = np.asarray(new_pcm_16k, dtype=np.float32).reshape(-1)
                fifo = new_arr if fifo.size == 0 else np.concatenate([fifo, new_arr])

            prob = self._last_silero_prob.get(channel_id, 0.0)
            offset = 0
            total = fifo.shape[0]
            while total - offset >= _SILERO_CHUNK_SAMPLES:
                chunk = fifo[offset : offset + _SILERO_CHUNK_SAMPLES]
                prob = self._run_silero_chunk(channel_id, chunk)
                offset += _SILERO_CHUNK_SAMPLES

            self._silero_fifo[channel_id] = fifo[offset:].copy() if offset > 0 else fifo
            self._last_silero_prob[channel_id] = prob
            self._last_onnx_ts_ms[channel_id] = ts_ms
            return prob

        # Fallback for synthetic one-shot AudioFrameMetrics without `vocal_new_pcm_16k`
        if ts_ms > 0 and ts_ms == self._last_onnx_ts_ms.get(channel_id, 0):
            return self._last_silero_prob.get(channel_id, 0.0)
        self._last_onnx_ts_ms[channel_id] = ts_ms

        if pcm_16k.shape[0] < _SILERO_CHUNK_SAMPLES:
            chunk = np.zeros(_SILERO_CHUNK_SAMPLES, dtype=np.float32)
            chunk[-pcm_16k.shape[0] :] = pcm_16k
        else:
            chunk = pcm_16k[-_SILERO_CHUNK_SAMPLES:].astype(np.float32, copy=False)
        prob = self._run_silero_chunk(channel_id, chunk)
        self._last_silero_prob[channel_id] = prob
        return prob

    def _infer_dsp_vad(
        self, ch: ChannelConfig, pcm_16k: np.ndarray, metrics: AudioFrameMetrics
    ) -> float:
        """Deterministic DSP speech-envelope + spectral-centroid + ZCR VAD fallback."""
        x = pcm_16k.astype(np.float64)
        total_power = float(np.mean(x * x))
        total_db = _power_to_db(total_power)
        if total_db <= -70.0:
            self._vad_ema[ch.id] = 0.0
            return 0.0
        if total_db <= -60.0:
            self._vad_ema[ch.id] *= 0.25
            return 0.0 if self._vad_ema[ch.id] < 0.02 else float(self._vad_ema[ch.id])

        # Voice-band (300-3400 Hz) energy & ratio
        vband = signal.sosfilt(self._speech_bp_sos, x)
        vband_power = float(np.mean(vband * vband))
        vband_db = _power_to_db(vband_power)
        vband_ratio = vband_power / (total_power + 1e-12)

        # Normalized level relative to channel nominal
        norm_vb = metrics.norm_vband_db.get(ch.id, vband_db - ch.nominal_vband_rms_db)
        energy_score = _sigmoid((norm_vb + 24.0) / 3.5) * _sigmoid((vband_db + 52.0) / 3.0)

        # Voice-band concentration score (rejects pure sub-bass < 200Hz and pure HF hiss > 4kHz)
        ratio_score = float(np.clip((vband_ratio - 0.18) / 0.42, 0.0, 1.0))

        # Spectral centroid on Hann-windowed 16 kHz frame
        win = np.hanning(x.shape[0])
        mag = np.abs(np.fft.rfft(x * win))
        freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / VOCAL_16K_SR)
        mag_sum = float(np.sum(mag))
        centroid_hz = float(np.sum(freqs * mag) / mag_sum) if mag_sum > 1e-12 else 0.0

        if 250.0 <= centroid_hz <= 2800.0:
            centroid_score = 1.0
        elif centroid_hz < 250.0:
            centroid_score = float(np.clip((centroid_hz - 100.0) / 150.0, 0.0, 1.0))
        else:
            centroid_score = float(np.clip((4500.0 - centroid_hz) / 1700.0, 0.0, 1.0))

        # Zero-crossing rate (speech ~0.012-0.30; broadband white noise ~0.50; sub-bass <0.01)
        zc = float(np.mean((x[:-1] * x[1:]) < 0.0)) if x.shape[0] > 1 else 0.0
        if 0.012 <= zc <= 0.30:
            zcr_score = 1.0
        elif zc < 0.012:
            zcr_score = float(np.clip(zc / 0.012, 0.0, 1.0))
        else:
            zcr_score = float(np.clip((0.46 - zc) / 0.16, 0.0, 1.0))

        # Broadband spectral flatness penalty (white noise ~0.7+, speech/formants <0.45)
        psd = mag * mag
        arith = float(np.mean(psd))
        flatness = (
            float(np.exp(np.mean(np.log(np.maximum(psd, 1e-15)))) / arith) if arith > 1e-15 else 1.0
        )
        tonality_score = float(np.clip((0.68 - flatness) / 0.28, 0.1, 1.0))

        raw_prob = float(
            np.clip(
                energy_score * (0.35 * ratio_score + 0.30 * centroid_score + 0.25 * zcr_score + 0.10 * tonality_score),
                0.0,
                1.0,
            )
        )
        # Boost clearly speech-like frames so sustained speech sits solidly in [0.80, 0.98]
        if energy_score > 0.65 and ratio_score > 0.65 and centroid_score > 0.7 and zcr_score > 0.7:
            raw_prob = min(0.99, raw_prob ** 0.5)

        prev = self._vad_ema.get(ch.id, 0.0)
        alpha = 0.85 if raw_prob >= prev else 0.40
        smoothed = alpha * raw_prob + (1.0 - alpha) * prev
        self._vad_ema[ch.id] = smoothed
        return float(np.clip(smoothed, 0.0, 1.0))

    def _detect_sung_vocal(
        self, ch: ChannelConfig, pcm_16k: np.ndarray | None, metrics: AudioFrameMetrics
    ) -> tuple[bool, float]:
        """Detect pitched sung vocals via 200-4000Hz gain-normalized RMS + autocorrelation harmonicity."""
        if pcm_16k is None or pcm_16k.size < _HARMONICITY_MIN_LAG * 4:
            return False, 0.0

        # 1. Bandpass 200-4000 Hz at 16 kHz
        y = signal.sosfilt(self._sung_bp_sos, pcm_16k.astype(np.float64))
        rms_db = _power_to_db(float(np.mean(y * y)))
        norm_db = rms_db - ch.nominal_vband_rms_db
        snr_db = metrics.speech_to_band_snr_db.get(ch.id, 0.0)

        # 2. Reject pure high-frequency whistles (>1800 Hz feedback) lacking vocal F0/F1-F2 energy
        spec_y = np.abs(np.fft.rfft(y * np.hanning(y.shape[0]))) ** 2
        freqs_y = np.fft.rfftfreq(y.shape[0], d=1.0 / VOCAL_16K_SR)
        low_vocal_ratio = float(np.sum(spec_y[(freqs_y >= 150.0) & (freqs_y <= 1600.0)])) / (
            float(np.sum(spec_y)) + 1e-12
        )

        # 3. Autocorrelation harmonicity salience over 80-1000 Hz vocal pitch range
        harmonicity = compute_autocorr_harmonicity(y)

        active = (
            rms_db > _SUNG_VOCAL_ABS_MIN_DB
            and norm_db >= _SUNG_VOCAL_NORM_MIN_DB
            and snr_db >= _SUNG_VOCAL_SNR_MIN_DB
            and low_vocal_ratio >= 0.20
            and harmonicity >= _HARMONICITY_THRESHOLD
        )
        return active, harmonicity
