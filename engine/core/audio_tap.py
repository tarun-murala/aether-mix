"""Decoupled 16-channel real-time Audio Tap & DSP feature extractor (SPEC v5.1).

Responsibilities:
- Lock-free preallocated SPSC ring buffer (zero ML and zero allocations in PortAudio callback).
- 50 Hz per-channel wideband RMS, voice-band (300-3400 Hz Butterworth SOS) RMS,
  gain-normalized vband RMS, true peak dBFS, clip flags (>= -0.5 dBFS), and 1024-pt Hann FFT.
- Gain- & bleed-normalized speech-to-band SNR using 3-step sequential soundcheck calibration
  (room silence -> isolated spoken/band nominals -> band-only bleed floor).
- Internal gain-normalized unity-gain pre-fader sum (Ch 1-12 ONLY) in a 3-second rolling
  buffer for the isolated CLAP subprocess (strictly invariant to Master Ch 15-16 fader moves).
- 10 Hz ITU-R BS.1770-4 K-weighted momentary (400 ms) and short-term (3 s) LUFS strictly
  on post-fader Master Ch 15-16.
- Room ambience (Ch 13-14) broadband spectral flatness (> 0.55 applause/transition cue).
- > 2.0 s bitwise all-zero TCC microphone permission silence guard.
- Continuous 16 kHz anti-aliased downsampling for vocal/spoken channels (Silero VAD + harmonicity).
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import signal

from engine.core.protocol import BAND_ROLES, ChannelConfig, ChannelRole, ServicePlan

logger = logging.getLogger(__name__)

DB_FLOOR = -120.0
LUFS_FLOOR = -70.0
CLIP_THRESHOLD_DB = -0.5
NUM_TAP_CHANNELS = 16
FFT_SIZE = 1024
VOCAL_16K_SR = 16000
VOCAL_16K_WINDOW_SAMPLES = 1024  # 64 ms at 16 kHz (>= 512 for Silero VAD & >= 5 cycles at 80 Hz)
TCC_SILENCE_SECONDS = 2.0

VOCAL_ROLES: frozenset[ChannelRole] = frozenset(
    {
        ChannelRole.SPOKEN_PRIMARY,
        ChannelRole.VOCAL_LEAD,
        ChannelRole.VOCAL_BG,
    }
)


@dataclass
class AudioFrameMetrics:
    ts_ms: int
    rms_db: dict[str, float]
    vband_rms_db: dict[str, float]
    norm_vband_db: dict[str, float]
    peak_db: dict[str, float]
    clip: dict[str, bool]
    fft_mag_db: dict[str, np.ndarray]
    speech_to_band_snr_db: dict[str, float]
    room_flatness: float
    master_lufs_momentary: float
    master_lufs_short_term: float
    tcc_silence_warning: bool
    pre_fader_sum_3s: np.ndarray
    vocal_pcm_16k: dict[str, np.ndarray]
    vocal_new_pcm_16k: dict[str, np.ndarray] = field(default_factory=dict)


def _power_to_db(power: float) -> float:
    if power <= 1e-12:
        return DB_FLOOR
    return max(DB_FLOOR, 10.0 * math.log10(power))


def _amp_to_db(amp: float) -> float:
    if amp <= 1e-6:
        return DB_FLOOR
    return max(DB_FLOOR, 20.0 * math.log10(amp))


def _build_k_weighting_sos(sample_rate: int) -> np.ndarray:
    """Return ITU-R BS.1770-4 two-stage K-weighting filter as SOS array."""
    if sample_rate == 48000:
        # Exact reference coefficients from ITU-R BS.1770-4 Table 1 & Table 2
        b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
        a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
        b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
        a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)
    else:
        # Generalized bilinear transform from BS.1770-4 analog prototypes
        fs = float(sample_rate)
        # Stage 1: pre-filter high shelf (+3.999843853973347 dB, Q=0.7071752369554196, f0=1681.974450955533 Hz)
        db_gain = 3.999843853973347
        f0 = 1681.974450955533
        q = 0.7071752369554196
        k = math.tan(math.pi * f0 / fs)
        vh = 10.0 ** (db_gain / 20.0)
        vb = vh ** 0.4996667741545416
        a0 = 1.0 + k / q + k * k
        b1 = np.array(
            [
                (vh + vb * k / q + k * k) / a0,
                2.0 * (k * k - vh) / a0,
                (vh - vb * k / q + k * k) / a0,
            ],
            dtype=np.float64,
        )
        a1 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0], dtype=np.float64)
        # Stage 2: RLB high-pass (f0=38.13547087602444 Hz, Q=0.5003270373238773)
        f0_hp = 38.13547087602444
        q_hp = 0.5003270373238773
        k_hp = math.tan(math.pi * f0_hp / fs)
        a0_hp = 1.0 + k_hp / q_hp + k_hp * k_hp
        b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64) / a0_hp
        a2 = np.array(
            [1.0, 2.0 * (k_hp * k_hp - 1.0) / a0_hp, (1.0 - k_hp / q_hp + k_hp * k_hp) / a0_hp],
            dtype=np.float64,
        )
    sos = np.vstack([np.concatenate([b1, a1]), np.concatenate([b2, a2])])
    return sos


class _LockFreeRingBuffer:
    """Single-producer single-consumer preallocated circular buffer for 16ch float32 audio."""

    def __init__(self, capacity_frames: int, num_channels: int = NUM_TAP_CHANNELS) -> None:
        self.capacity = int(capacity_frames)
        self.num_channels = int(num_channels)
        self._buf = np.zeros((self.capacity, self.num_channels), dtype=np.float32)
        self._write_pos: int = 0
        self._read_pos: int = 0

    def push(self, block: np.ndarray) -> None:
        """Copy incoming `(frames, ch)` float32 block into preallocated ring without allocating."""
        n = block.shape[0]
        if n == 0:
            return
        ch = min(block.shape[1], self.num_channels)
        if n >= self.capacity:
            # Keep only the most recent capacity frames
            tail = block[-self.capacity :, :ch]
            self._buf[:, :ch] = tail
            if ch < self.num_channels:
                self._buf[:, ch:] = 0.0
            new_write = self._write_pos + n
            self._read_pos = max(self._read_pos, new_write - self.capacity)
            self._write_pos = new_write
            return

        w_mod = self._write_pos % self.capacity
        first = min(n, self.capacity - w_mod)
        self._buf[w_mod : w_mod + first, :ch] = block[:first, :ch]
        if ch < self.num_channels:
            self._buf[w_mod : w_mod + first, ch:] = 0.0
        rem = n - first
        if rem > 0:
            self._buf[0:rem, :ch] = block[first:, :ch]
            if ch < self.num_channels:
                self._buf[0:rem, ch:] = 0.0
        new_write = self._write_pos + n
        if new_write - self._read_pos > self.capacity:
            self._read_pos = new_write - self.capacity
        self._write_pos = new_write

    def peek_unread(self) -> np.ndarray:
        """Non-destructively read unread frames currently in the ring without advancing read_pos."""
        w = self._write_pos
        r = self._read_pos
        avail = w - r
        if avail <= 0:
            return np.empty((0, self.num_channels), dtype=np.float32)
        if avail > self.capacity:
            r = w - self.capacity
            avail = self.capacity
        r_mod = r % self.capacity
        first = min(avail, self.capacity - r_mod)
        if first == avail:
            return self._buf[r_mod : r_mod + avail].copy()
        out = np.empty((avail, self.num_channels), dtype=np.float32)
        out[:first] = self._buf[r_mod : r_mod + first]
        out[first:] = self._buf[0 : avail - first]
        return out

    def drain(self) -> np.ndarray:
        """Read all unread frames since last drain."""
        w = self._write_pos
        r = self._read_pos
        avail = w - r
        if avail <= 0:
            return np.empty((0, self.num_channels), dtype=np.float32)
        if avail > self.capacity:
            r = w - self.capacity
            avail = self.capacity
        r_mod = r % self.capacity
        first = min(avail, self.capacity - r_mod)
        if first == avail:
            out = self._buf[r_mod : r_mod + avail].copy()
        else:
            out = np.empty((avail, self.num_channels), dtype=np.float32)
            out[:first] = self._buf[r_mod : r_mod + first]
            out[first:] = self._buf[0 : avail - first]
        self._read_pos = w
        return out


class AudioTap:
    """Real-time 16ch BlackHole / headless audio tap and DSP perception front-end."""

    def __init__(self, plan: ServicePlan, headless: bool = False) -> None:
        self.plan = plan
        self.headless = headless
        self.sample_rate = int(plan.sample_rate)
        self._block_frames = max(1, self.sample_rate // 50)  # 960 samples @ 48 kHz (20 ms)

        # Lock-free ring buffer (holds 4 seconds of 16ch float32 audio)
        self._ring = _LockFreeRingBuffer(capacity_frames=self.sample_rate * 4, num_channels=NUM_TAP_CHANNELS)
        self._recent_raw_blocks: deque[np.ndarray] = deque()
        self._recent_raw_samples: int = 0
        self._recent_raw_max_samples: int = self.sample_rate * 20  # up to 20s for live calibration
        self._last_block_ts_ms: int = 0
        self._stream: Any = None
        self._running = False

        # Channel index maps (1-based in ServicePlan -> 0-based column indices)
        self._ch_tap_idx: dict[str, list[int]] = {
            ch.id: [t - 1 for t in ch.tap_channels if 1 <= t <= NUM_TAP_CHANNELS]
            for ch in plan.channels
        }
        self._room_idx: list[int] = [t - 1 for t in plan.room_tap_channels if 1 <= t <= NUM_TAP_CHANNELS]
        self._master_idx: list[int] = [t - 1 for t in plan.master_tap_channels if 1 <= t <= NUM_TAP_CHANNELS]
        self._band_channel_ids: list[str] = [ch.id for ch in plan.channels if ch.role in BAND_ROLES]
        self._vocal_channel_ids: list[str] = [
            ch.id for ch in plan.channels if ch.vad_enabled or ch.role in VOCAL_ROLES
        ]
        # Pre-fader Ch 1-12 channels ONLY (explicitly exclude 13-16)
        self._prefader_channels: list[ChannelConfig] = [
            ch for ch in plan.channels if all(1 <= t <= 12 for t in ch.tap_channels)
        ]

        # Voice-band (300-3400 Hz) Butterworth SOS filter & persistent state for all 16 raw channels
        nyq = 0.5 * self.sample_rate
        self._vband_sos = signal.butter(
            4,
            [300.0 / nyq, min(3400.0, nyq * 0.95) / nyq],
            btype="bandpass",
            output="sos",
        )
        self._vband_zi = np.zeros((self._vband_sos.shape[0], 2, NUM_TAP_CHANNELS), dtype=np.float64)

        # Anti-aliasing LPF for 48 kHz -> 16 kHz decimation (persistent state per vocal channel)
        self._decim_factor = self.sample_rate // VOCAL_16K_SR if self.sample_rate % VOCAL_16K_SR == 0 else 0
        self._aa_sos = signal.butter(4, 7200.0 / nyq, btype="lowpass", output="sos")
        self._aa_zi: dict[str, np.ndarray] = {
            cid: np.zeros((self._aa_sos.shape[0], 2), dtype=np.float64) for cid in self._vocal_channel_ids
        }
        self._decim_phase: dict[str, int] = {cid: 0 for cid in self._vocal_channel_ids}
        self._vocal_16k_buf: dict[str, np.ndarray] = {
            cid: np.zeros(VOCAL_16K_WINDOW_SAMPLES, dtype=np.float32) for cid in self._vocal_channel_ids
        }

        # FFT window & rolling 1024-sample history per raw channel
        self._hann = np.hanning(FFT_SIZE).astype(np.float64)
        self._hann_norm = float(np.sum(self._hann)) / 2.0  # coherent gain for single-sided rfft
        self._fft_hist = np.zeros((FFT_SIZE, NUM_TAP_CHANNELS), dtype=np.float32)
        fft_freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / self.sample_rate)
        self._room_flat_mask = (fft_freqs >= 200.0) & (fft_freqs <= 12000.0)
        self._room_psd_ema: np.ndarray | None = None
        self._room_noise_floor_db: float = -48.0

        # Pre-fader 3-second mono rolling buffer for CLAP (strictly Ch 1-12)
        self._prefader_3s_len = self.sample_rate * 3
        self._prefader_3s = np.zeros(self._prefader_3s_len, dtype=np.float32)

        # ITU-R BS.1770-4 K-weighting filter on post-fader Master Ch 15-16 (10 Hz updates)
        self._kw_sos = _build_k_weighting_sos(self.sample_rate)
        master_ch_count = max(1, len(self._master_idx))
        self._kw_zi = np.zeros((self._kw_sos.shape[0], 2, master_ch_count), dtype=np.float64)
        self._lufs_step_samples = max(1, self.sample_rate // 10)  # 100 ms = 10 Hz update cadence
        self._lufs_accum_power_sum: float = 0.0  # accumulated sum of (y_L^2 + y_R^2) in current 100ms bin
        self._lufs_accum_count: int = 0
        self._lufs_100ms_bins = np.zeros(30, dtype=np.float64)  # 30 x 100ms = 3.0 s
        self._lufs_bin_pos: int = 0
        self._lufs_bins_filled: int = 0
        self._master_lufs_momentary: float = LUFS_FLOOR
        self._master_lufs_short_term: float = LUFS_FLOOR

        # TCC bitwise all-zero guard
        self._consecutive_zero_samples: int = 0

        # Calibration storage (ambient voice-band noise floor in raw dBFS)
        self._mic_noise_floor_vband_db: dict[str, float] = {
            ch.id: ch.nominal_vband_rms_db + ch.mic_noise_floor_norm_db for ch in plan.channels
        }

        # Latest cached metrics snapshot
        self._last_metrics: AudioFrameMetrics = self._make_empty_metrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.headless:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            logger.warning("sounddevice not installed; running AudioTap in headless mode: %s", exc)
            return

        device_idx = self._find_input_device(sd, self.plan.audio_device_hint)
        self._stream = sd.InputStream(
            device=device_idx,
            samplerate=self.sample_rate,
            blocksize=self._block_frames,
            channels=NUM_TAP_CHANNELS,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("AudioTap stream started (device=%s, sr=%d)", device_idx, self.sample_rate)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error closing sounddevice stream")
            finally:
                self._stream = None

    @staticmethod
    def _find_input_device(sd: Any, hint: str) -> int | None:
        if not hint:
            return None
        hint_lower = hint.lower()
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) >= NUM_TAP_CHANNELS and hint_lower in str(
                dev.get("name", "")
            ).lower():
                return idx
        return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Real-time PortAudio callback: lock-free ring write ONLY (zero ML, zero locking)."""
        self._ring.push(indata)

    # ------------------------------------------------------------------
    # Ingestion & 50 Hz Polling
    # ------------------------------------------------------------------

    def push_block(self, block: np.ndarray) -> None:
        """Push a `(frames, 16)` float32 audio block at `plan.sample_rate` into the lock-free ring."""
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[0] == 0:
            return
        self._ring.push(arr)

    @property
    def last_block_ts_ms(self) -> int:
        """Monotonic timestamp (ms) of the last real audio block drained; 0 before first block."""
        return self._last_block_ts_ms

    def recent_raw_blocks(self, duration_s: float = 5.0) -> list[np.ndarray]:
        """Return recent 16-ch raw blocks spanning up to `duration_s` seconds (drained + unread)."""
        want_samples = max(1, round(max(0.0, float(duration_s)) * self.sample_rate))
        sources: list[np.ndarray] = list(self._recent_raw_blocks)
        unread = self._ring.peek_unread()
        if unread.shape[0] > 0:
            sources.append(unread)
        if not sources:
            return []
        collected: list[np.ndarray] = []
        acc = 0
        for blk in reversed(sources):
            collected.append(blk)
            acc += blk.shape[0]
            if acc >= want_samples:
                break
        collected.reverse()
        if acc > want_samples and collected:
            excess = acc - want_samples
            collected[0] = collected[0][excess:]
        return collected

    def poll_metrics(self) -> AudioFrameMetrics:
        """Drain pending audio from the ring buffer and compute current AudioFrameMetrics."""
        data = self._ring.drain()
        if data.shape[0] == 0:
            if any(arr.size > 0 for arr in self._last_metrics.vocal_new_pcm_16k.values()):
                self._last_metrics = AudioFrameMetrics(
                    ts_ms=self._last_metrics.ts_ms,
                    rms_db=self._last_metrics.rms_db,
                    vband_rms_db=self._last_metrics.vband_rms_db,
                    norm_vband_db=self._last_metrics.norm_vband_db,
                    peak_db=self._last_metrics.peak_db,
                    clip=self._last_metrics.clip,
                    fft_mag_db=self._last_metrics.fft_mag_db,
                    speech_to_band_snr_db=self._last_metrics.speech_to_band_snr_db,
                    room_flatness=self._last_metrics.room_flatness,
                    master_lufs_momentary=self._last_metrics.master_lufs_momentary,
                    master_lufs_short_term=self._last_metrics.master_lufs_short_term,
                    tcc_silence_warning=self._last_metrics.tcc_silence_warning,
                    pre_fader_sum_3s=self._last_metrics.pre_fader_sum_3s,
                    vocal_pcm_16k=self._last_metrics.vocal_pcm_16k,
                    vocal_new_pcm_16k={
                        cid: np.empty(0, dtype=np.float32) for cid in self._vocal_channel_ids
                    },
                )
            return self._last_metrics

        n_frames = data.shape[0]
        now_ms = int(time.time() * 1000)
        ts_ms = max(now_ms, self._last_block_ts_ms + 1)
        self._last_block_ts_ms = ts_ms

        # Record drained raw block in bounded 20s rolling history for live calibration
        self._recent_raw_blocks.append(data)
        self._recent_raw_samples += n_frames
        while self._recent_raw_samples > self._recent_raw_max_samples and len(self._recent_raw_blocks) > 1:
            dropped = self._recent_raw_blocks.popleft()
            self._recent_raw_samples -= dropped.shape[0]

        # 1. TCC bitwise all-zero guard (exact bitwise equality to 0.0 across all channels)
        nonzero_rows = np.any(data != 0.0, axis=1)
        if not np.any(nonzero_rows):
            self._consecutive_zero_samples += n_frames
        else:
            last_nz = int(np.flatnonzero(nonzero_rows)[-1])
            self._consecutive_zero_samples = n_frames - 1 - last_nz
        tcc_silence_warning = (self._consecutive_zero_samples / float(self.sample_rate)) > TCC_SILENCE_SECONDS

        # 2. Voice-band (300-3400 Hz) Butterworth SOS filter across all 16 channels (stateful)
        vband_data, self._vband_zi = signal.sosfilt(
            self._vband_sos, data.astype(np.float64), axis=0, zi=self._vband_zi
        )

        # 3. Update rolling 1024-sample FFT history buffer
        if n_frames >= FFT_SIZE:
            self._fft_hist[:, :] = data[-FFT_SIZE:, :]
        else:
            self._fft_hist[:-n_frames, :] = self._fft_hist[n_frames:, :]
            self._fft_hist[-n_frames:, :] = data

        # Compute 1024-pt Hann FFT for all 16 raw channels in one vectorized rfft call
        windowed = self._fft_hist.astype(np.float64) * self._hann[:, np.newaxis]
        fft_mag = np.abs(np.fft.rfft(windowed, axis=0)) / self._hann_norm  # (513, 16)
        fft_mag_db_raw = np.maximum(DB_FLOOR, 20.0 * np.log10(np.maximum(fft_mag, 1e-6)))

        # 4. Per-logical-channel RMS, voice-band RMS, norm_vband_db, peak, clip, FFT
        rms_db: dict[str, float] = {}
        vband_rms_db: dict[str, float] = {}
        norm_vband_db: dict[str, float] = {}
        peak_db: dict[str, float] = {}
        clip: dict[str, bool] = {}
        fft_mag_db: dict[str, np.ndarray] = {}

        # Use at most the most recent 20ms-to-n_frames window for instantaneous metering;
        # when called at 50 Hz n_frames == 960.
        meter_slice = data[-self._block_frames :] if n_frames > self._block_frames else data
        vband_slice = vband_data[-self._block_frames :] if n_frames > self._block_frames else vband_data

        for ch in self.plan.channels:
            cols = self._ch_tap_idx[ch.id]
            if not cols:
                rms_db[ch.id] = DB_FLOOR
                vband_rms_db[ch.id] = DB_FLOOR
                norm_vband_db[ch.id] = DB_FLOOR - ch.nominal_vband_rms_db
                peak_db[ch.id] = DB_FLOOR
                clip[ch.id] = False
                fft_mag_db[ch.id] = np.full(FFT_SIZE // 2 + 1, DB_FLOOR, dtype=np.float32)
                continue

            sub = meter_slice[:, cols]
            sub_vb = vband_slice[:, cols]
            rms_power = float(np.mean(sub.astype(np.float64) ** 2))
            vb_power = float(np.mean(sub_vb ** 2))
            pk = float(np.max(np.abs(sub)))

            r_db = _power_to_db(rms_power)
            vb_db = _power_to_db(vb_power)
            pk_db = _amp_to_db(pk)

            rms_db[ch.id] = r_db
            vband_rms_db[ch.id] = vb_db
            norm_vband_db[ch.id] = vb_db - ch.nominal_vband_rms_db
            peak_db[ch.id] = pk_db
            clip[ch.id] = pk_db >= CLIP_THRESHOLD_DB

            # Max across tap channels preserves exact tone level for mono bleed cross-checks
            if len(cols) == 1:
                fft_mag_db[ch.id] = fft_mag_db_raw[:, cols[0]].astype(np.float32)
            else:
                fft_mag_db[ch.id] = np.max(fft_mag_db_raw[:, cols], axis=1).astype(np.float32)

        # 5. Acoustic power-sum bleed-calibrated speech_to_band_snr_db (immune to single-instrument nominal outliers)
        band_vband_power = sum(
            (10.0 ** (vband_rms_db[cid] / 10.0) for cid in self._band_channel_ids if vband_rms_db[cid] > DB_FLOOR + 1.0),
            0.0,
        )
        band_sum_vband_db = _power_to_db(band_vband_power)
        ref_band_nom_db = max(
            (ch.nominal_vband_rms_db for ch in self.plan.channels if ch.id in self._band_channel_ids),
            default=-18.0,
        )
        band_sum_norm_db = band_sum_vband_db - ref_band_nom_db
        speech_to_band_snr_db: dict[str, float] = {}
        for ch in self.plan.channels:
            expected_bleed_norm = max(
                band_sum_norm_db + ch.bleed_offset_db,
                ch.mic_noise_floor_norm_db,
            )
            speech_to_band_snr_db[ch.id] = norm_vband_db[ch.id] - expected_bleed_norm

        # 6. Internal gain-normalized unity-gain pre-fader sum (Ch 1-12 ONLY) for CLAP
        self._update_prefader_sum(data)

        # 7. 10 Hz ITU-R BS.1770-4 K-weighted Master LUFS strictly on post-fader Ch 15-16
        self._update_master_lufs(data)

        # 8. Room ambience (Ch 13-14) broadband spectral flatness (> 0.55 applause cue)
        room_flatness = self._compute_room_flatness(meter_slice, fft_mag)

        # 9. 16 kHz anti-aliased downsampled vocal PCM (for VADWorker)
        vocal_pcm_16k, vocal_new_pcm_16k = self._update_vocal_pcm_16k(data)

        metrics = AudioFrameMetrics(
            ts_ms=ts_ms,
            rms_db=rms_db,
            vband_rms_db=vband_rms_db,
            norm_vband_db=norm_vband_db,
            peak_db=peak_db,
            clip=clip,
            fft_mag_db=fft_mag_db,
            speech_to_band_snr_db=speech_to_band_snr_db,
            room_flatness=room_flatness,
            master_lufs_momentary=self._master_lufs_momentary,
            master_lufs_short_term=self._master_lufs_short_term,
            tcc_silence_warning=tcc_silence_warning,
            pre_fader_sum_3s=self._prefader_3s.copy(),
            vocal_pcm_16k=vocal_pcm_16k,
            vocal_new_pcm_16k=vocal_new_pcm_16k,
        )
        self._last_metrics = metrics
        return metrics

    def _update_prefader_sum(self, data: np.ndarray) -> None:
        """Compute gain-normalized unity sum of pre-fader Ch 1-12 into 3s rolling buffer."""
        n = data.shape[0]
        if not self._prefader_channels:
            return
        # Normalize each pre-fader strip so its nominal_rms_db maps to -24 dBFS, then sum
        target_nominal_db = -24.0
        mix = np.zeros(n, dtype=np.float64)
        for ch in self._prefader_channels:
            cols = self._ch_tap_idx[ch.id]
            if not cols:
                continue
            gain = 10.0 ** (np.clip(target_nominal_db - ch.nominal_rms_db, -24.0, 24.0) / 20.0)
            if len(cols) == 1:
                strip_mono = data[:, cols[0]].astype(np.float64)
            else:
                strip_mono = np.mean(data[:, cols].astype(np.float64), axis=1)
            mix += strip_mono * gain
        mix_f32 = np.clip(mix, -1.0, 1.0).astype(np.float32)

        buf_len = self._prefader_3s_len
        if n >= buf_len:
            self._prefader_3s[:] = mix_f32[-buf_len:]
        else:
            self._prefader_3s[:-n] = self._prefader_3s[n:]
            self._prefader_3s[-n:] = mix_f32

    def _update_master_lufs(self, data: np.ndarray) -> None:
        """ITU-R BS.1770-4 K-weighted momentary (400 ms) & short-term (3 s) LUFS on Ch 15-16."""
        if not self._master_idx:
            return
        master = data[:, self._master_idx].astype(np.float64)
        kw_out, self._kw_zi = signal.sosfilt(self._kw_sos, master, axis=0, zi=self._kw_zi)
        # Per-sample stereo sum of squared K-weighted channels (g_L = g_R = 1.0)
        sample_energy = np.sum(kw_out ** 2, axis=1)

        idx = 0
        total = sample_energy.shape[0]
        step = self._lufs_step_samples
        while idx < total:
            need = step - self._lufs_accum_count
            take = min(need, total - idx)
            self._lufs_accum_power_sum += float(np.sum(sample_energy[idx : idx + take]))
            self._lufs_accum_count += take
            idx += take

            if self._lufs_accum_count >= step:
                mean_energy = self._lufs_accum_power_sum / float(self._lufs_accum_count)
                self._lufs_100ms_bins[self._lufs_bin_pos] = mean_energy
                self._lufs_bin_pos = (self._lufs_bin_pos + 1) % 30
                self._lufs_bins_filled = min(30, self._lufs_bins_filled + 1)
                self._lufs_accum_power_sum = 0.0
                self._lufs_accum_count = 0

                # Recompute momentary (last 4 x 100ms = 400 ms) and short-term (last 30 x 100ms = 3.0 s)
                m_count = min(4, self._lufs_bins_filled)
                m_indices = [(self._lufs_bin_pos - 1 - k) % 30 for k in range(m_count)]
                m_energy = float(np.mean(self._lufs_100ms_bins[m_indices]))
                self._master_lufs_momentary = (
                    max(LUFS_FLOOR, -0.691 + 10.0 * math.log10(m_energy))
                    if m_energy > 1e-12
                    else LUFS_FLOOR
                )

                st_count = self._lufs_bins_filled
                st_indices = [(self._lufs_bin_pos - 1 - k) % 30 for k in range(st_count)]
                st_energy = float(np.mean(self._lufs_100ms_bins[st_indices]))
                self._master_lufs_short_term = (
                    max(LUFS_FLOOR, -0.691 + 10.0 * math.log10(st_energy))
                    if st_energy > 1e-12
                    else LUFS_FLOOR
                )

    def _compute_room_flatness(self, meter_slice: np.ndarray, fft_mag: np.ndarray) -> float:
        """Compute Wiener entropy (spectral flatness) on room tap Ch 13-14 over 200-12000 Hz.

        Gated by room RMS above ambient HVAC/preamp noise floor so quiet thermal hiss
        (-54 dBFS) never triggers false applause detection.
        """
        if not self._room_idx:
            return 0.0
        room_sub = meter_slice[:, self._room_idx]
        room_rms_db = _power_to_db(float(np.mean(room_sub.astype(np.float64) ** 2)))
        min_applause_db = max(-38.0, self._room_noise_floor_db + 10.0)
        if room_rms_db < min_applause_db:
            self._room_psd_ema = None
            return 0.0

        # Power spectrum averaged across room tap channels, smoothed briefly (~100ms EMA)
        # so single-frame FFT exponential variance doesn't depress true broadband noise below 0.55.
        psd = np.mean(fft_mag[self._room_flat_mask][:, self._room_idx] ** 2, axis=1)
        if self._room_psd_ema is None or self._room_psd_ema.shape != psd.shape:
            self._room_psd_ema = psd.copy()
        else:
            self._room_psd_ema = 0.25 * psd + 0.75 * self._room_psd_ema

        arith_mean = float(np.mean(self._room_psd_ema))
        if arith_mean <= 1e-15:
            return 0.0
        geo_mean = float(np.exp(np.mean(np.log(np.maximum(self._room_psd_ema, 1e-15)))))
        return float(np.clip(geo_mean / arith_mean, 0.0, 1.0))

    def _update_vocal_pcm_16k(
        self, data: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Downsample vocal/spoken channels to 16 kHz, returning (rolling 64ms, newly drained)."""
        out: dict[str, np.ndarray] = {}
        out_new: dict[str, np.ndarray] = {}
        for cid in self._vocal_channel_ids:
            cols = self._ch_tap_idx[cid]
            if not cols:
                out[cid] = self._vocal_16k_buf[cid].copy()
                out_new[cid] = np.empty(0, dtype=np.float32)
                continue
            mono = (
                data[:, cols[0]].astype(np.float64)
                if len(cols) == 1
                else np.mean(data[:, cols].astype(np.float64), axis=1)
            )
            if self._decim_factor > 0:
                filt, self._aa_zi[cid] = signal.sosfilt(self._aa_sos, mono, zi=self._aa_zi[cid])
                phase = self._decim_phase[cid]
                dec = filt[phase :: self._decim_factor].astype(np.float32)
                self._decim_phase[cid] = (phase - len(filt)) % self._decim_factor
            else:
                g = math.gcd(VOCAL_16K_SR, self.sample_rate)
                dec = signal.resample_poly(mono, VOCAL_16K_SR // g, self.sample_rate // g).astype(
                    np.float32
                )

            buf = self._vocal_16k_buf[cid]
            m = dec.shape[0]
            if m >= VOCAL_16K_WINDOW_SAMPLES:
                buf[:] = dec[-VOCAL_16K_WINDOW_SAMPLES:]
            elif m > 0:
                buf[:-m] = buf[m:]
                buf[-m:] = dec
            out[cid] = buf.copy()
            out_new[cid] = dec
        return out, out_new

    # ------------------------------------------------------------------
    # 3-Step Sequential Soundcheck Calibration (SPEC §5.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_blocks(blocks: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(blocks, np.ndarray):
            arr = blocks.astype(np.float32, copy=False)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            return arr
        block_list = [np.asarray(b, dtype=np.float32) for b in blocks if np.asarray(b).size > 0]
        if not block_list:
            return np.empty((0, NUM_TAP_CHANNELS), dtype=np.float32)
        return np.vstack(block_list)

    def _filter_vband_offline(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run clean voice-band filter on calibration audio, trimming initial filter transient."""
        vband = signal.sosfilt(self._vband_sos, audio.astype(np.float64), axis=0)
        trim = min(480, audio.shape[0] // 10) if audio.shape[0] >= 960 else 0
        return audio[trim:], vband[trim:]

    def capture_room_silence(self, blocks: Sequence[np.ndarray] | np.ndarray) -> dict[str, float]:
        """Step A1: Measure ambient voice-band noise floor (dBFS) for each channel."""
        audio = self._concat_blocks(blocks)
        if audio.shape[0] == 0:
            return dict(self._mic_noise_floor_vband_db)

        _, vband = self._filter_vband_offline(audio)
        if self._room_idx:
            room_db = _power_to_db(float(np.mean(audio[:, self._room_idx].astype(np.float64) ** 2)))
            self._room_noise_floor_db = float(np.clip(room_db, -80.0, -30.0))
        floors_vband_db: dict[str, float] = {}
        for ch in self.plan.channels:
            cols = self._ch_tap_idx[ch.id]
            if not cols:
                continue
            vb_db = _power_to_db(float(np.mean(vband[:, cols] ** 2)))
            self._mic_noise_floor_vband_db[ch.id] = vb_db
            ch.mic_noise_floor_norm_db = float(np.clip(vb_db - ch.nominal_vband_rms_db, -60.0, -12.0))
            floors_vband_db[ch.id] = vb_db
        return floors_vband_db

    def capture_input_nominals(
        self, group: Literal["spoken", "band"], blocks: Sequence[np.ndarray] | np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """Step A2: Measure isolated nominal (wideband_rms_db, vband_rms_db) for spoken or band group."""
        if group not in ("spoken", "band"):
            raise ValueError(f"group must be 'spoken' or 'band', got {group!r}")
        target_roles = VOCAL_ROLES if group == "spoken" else BAND_ROLES
        target_channels = [ch for ch in self.plan.channels if ch.role in target_roles]

        audio = self._concat_blocks(blocks)
        if audio.shape[0] == 0:
            return {ch.id: (ch.nominal_rms_db, ch.nominal_vband_rms_db) for ch in target_channels}

        raw, vband = self._filter_vband_offline(audio)
        frame_len = self._block_frames
        n_frames = max(1, raw.shape[0] // frame_len)

        results: dict[str, tuple[float, float]] = {}
        for ch in target_channels:
            cols = self._ch_tap_idx[ch.id]
            if not cols:
                results[ch.id] = (ch.nominal_rms_db, ch.nominal_vband_rms_db)
                continue

            frame_rms_list: list[float] = []
            frame_vb_list: list[float] = []
            for idx in range(n_frames):
                s0 = idx * frame_len
                s1 = min(raw.shape[0], s0 + frame_len)
                if s1 <= s0:
                    continue
                r_db = _power_to_db(float(np.mean(raw[s0:s1, cols].astype(np.float64) ** 2)))
                vb_db = _power_to_db(float(np.mean(vband[s0:s1, cols] ** 2)))
                frame_rms_list.append(r_db)
                frame_vb_list.append(vb_db)

            rms_arr = np.asarray(frame_rms_list, dtype=np.float64)
            vb_arr = np.asarray(frame_vb_list, dtype=np.float64)

            # Use upper-active frames (within 12 dB of 95th percentile) so pauses/quiet strums don't bias nominal
            p95_vb = float(np.percentile(vb_arr, 95))
            active_mask = vb_arr >= (p95_vb - 12.0)
            nom_rms = float(np.percentile(rms_arr[active_mask], 70)) if np.any(active_mask) else float(np.median(rms_arr))
            nom_vb = float(np.percentile(vb_arr[active_mask], 70)) if np.any(active_mask) else float(np.median(vb_arr))

            # Only overwrite plan nominals when channel actually carried audible soundcheck signal
            if nom_vb > -65.0:
                ch.nominal_rms_db = nom_rms
                ch.nominal_vband_rms_db = nom_vb
                if ch.id in self._mic_noise_floor_vband_db:
                    ch.mic_noise_floor_norm_db = float(
                        np.clip(self._mic_noise_floor_vband_db[ch.id] - nom_vb, -60.0, -12.0)
                    )
            results[ch.id] = (ch.nominal_rms_db, ch.nominal_vband_rms_db)
        return results

    def capture_bleed_floor(self, blocks: Sequence[np.ndarray] | np.ndarray) -> dict[str, float]:
        """Step B: Measure per-mic acoustic bleed transfer ratio while band plays & nobody speaks."""
        mic_channels = [ch for ch in self.plan.channels if ch.role in VOCAL_ROLES]
        band_channels = [ch for ch in self.plan.channels if ch.role in BAND_ROLES]

        audio = self._concat_blocks(blocks)
        if audio.shape[0] == 0 or not band_channels:
            return {ch.id: ch.bleed_offset_db for ch in mic_channels}

        _, vband = self._filter_vband_offline(audio)
        frame_len = self._block_frames
        n_frames = max(1, vband.shape[0] // frame_len)

        diffs_by_mic: dict[str, list[float]] = {ch.id: [] for ch in mic_channels}
        all_diffs_by_mic: dict[str, list[float]] = {ch.id: [] for ch in mic_channels}

        ref_band_nom_db = max((bch.nominal_vband_rms_db for bch in band_channels), default=-18.0)

        for idx in range(n_frames):
            s0 = idx * frame_len
            s1 = min(vband.shape[0], s0 + frame_len)
            if s1 <= s0:
                continue
            band_power = 0.0
            for bch in band_channels:
                bcols = self._ch_tap_idx[bch.id]
                if not bcols:
                    continue
                band_power += float(np.mean(vband[s0:s1, bcols] ** 2))
            if band_power <= 1e-12:
                continue
            band_sum_vb_db = _power_to_db(band_power)
            band_sum_norm = band_sum_vb_db - ref_band_nom_db

            for mch in mic_channels:
                mcols = self._ch_tap_idx[mch.id]
                if not mcols:
                    continue
                m_vb = _power_to_db(float(np.mean(vband[s0:s1, mcols] ** 2)))
                m_norm = m_vb - mch.nominal_vband_rms_db
                diff = m_norm - band_sum_norm
                all_diffs_by_mic[mch.id].append(diff)
                if band_sum_norm > -30.0:
                    diffs_by_mic[mch.id].append(diff)

        bleed_offsets: dict[str, float] = {}
        for mch in mic_channels:
            samples = diffs_by_mic[mch.id] or all_diffs_by_mic[mch.id]
            if samples:
                offset = float(np.clip(np.median(samples), -35.0, -3.0))
                mch.bleed_offset_db = offset
            bleed_offsets[mch.id] = mch.bleed_offset_db
        return bleed_offsets

    def _make_empty_metrics(self) -> AudioFrameMetrics:
        n_bins = FFT_SIZE // 2 + 1
        return AudioFrameMetrics(
            ts_ms=0,
            rms_db={ch.id: DB_FLOOR for ch in self.plan.channels},
            vband_rms_db={ch.id: DB_FLOOR for ch in self.plan.channels},
            norm_vband_db={ch.id: DB_FLOOR - ch.nominal_vband_rms_db for ch in self.plan.channels},
            peak_db={ch.id: DB_FLOOR for ch in self.plan.channels},
            clip={ch.id: False for ch in self.plan.channels},
            fft_mag_db={ch.id: np.full(n_bins, DB_FLOOR, dtype=np.float32) for ch in self.plan.channels},
            speech_to_band_snr_db={ch.id: 0.0 for ch in self.plan.channels},
            room_flatness=0.0,
            master_lufs_momentary=LUFS_FLOOR,
            master_lufs_short_term=LUFS_FLOOR,
            tcc_silence_warning=False,
            pre_fader_sum_3s=self._prefader_3s.copy(),
            vocal_pcm_16k={cid: buf.copy() for cid, buf in self._vocal_16k_buf.items()},
            vocal_new_pcm_16k={},
        )
