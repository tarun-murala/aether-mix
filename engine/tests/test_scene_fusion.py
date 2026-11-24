"""Tests for AudioTap DSP, 3-step soundcheck calibration, VADWorker, CLAPWorker & SceneClassifier."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from engine.core.audio_tap import AudioFrameMetrics, AudioTap
from engine.core.clap_worker import CLAPWorker
from engine.core.protocol import ServicePhase, ServicePlan
from engine.core.scene_classifier import SceneClassifier
from engine.core.vad_worker import VADWorker

SERVICE_PLAN_PATH = Path(__file__).resolve().parents[1] / "schemas" / "service_plan.json"
SR = 48000
BLOCK_FRAMES = 960  # 20 ms @ 48 kHz


def load_plan() -> ServicePlan:
    data = json.loads(SERVICE_PLAN_PATH.read_text(encoding="utf-8"))
    return ServicePlan.model_validate(data)


def make_sine(freq_hz: float, n_samples: int, rms_db: float, sr: int = SR, phase: float = 0.0) -> np.ndarray:
    """Generate a sine wave with exact target RMS in dBFS."""
    t = np.arange(n_samples, dtype=np.float64) / float(sr)
    rms_lin = 10.0 ** (rms_db / 20.0)
    amp = rms_lin * math.sqrt(2.0)
    return (amp * np.sin(2.0 * math.pi * freq_hz * t + phase)).astype(np.float32)


def make_voice_signal(
    n_samples: int, rms_db: float, f0_hz: float = 160.0, sr: int = SR, offset_samples: int = 0
) -> np.ndarray:
    """Generate a pitched voice/speech-like harmonic signal concentrated in 300-2500 Hz."""
    t = (np.arange(n_samples, dtype=np.float64) + offset_samples) / float(sr)
    sig = (
        0.35 * np.sin(2.0 * math.pi * f0_hz * t)
        + 0.65 * np.sin(2.0 * math.pi * (2.0 * f0_hz) * t + 0.3)
        + 0.80 * np.sin(2.0 * math.pi * (3.0 * f0_hz) * t + 0.7)
        + 0.60 * np.sin(2.0 * math.pi * (5.0 * f0_hz) * t + 1.1)
        + 0.35 * np.sin(2.0 * math.pi * (7.0 * f0_hz) * t + 1.7)
    )
    cur_rms = math.sqrt(float(np.mean(sig * sig))) + 1e-12
    target_rms = 10.0 ** (rms_db / 20.0)
    return (sig * (target_rms / cur_rms)).astype(np.float32)


def make_noise(n_samples: int, rms_db: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal(n_samples)
    cur_rms = math.sqrt(float(np.mean(raw * raw))) + 1e-12
    target_rms = 10.0 ** (rms_db / 20.0)
    return (raw * (target_rms / cur_rms)).astype(np.float32)


def push_and_poll(tap: AudioTap, block_16ch: np.ndarray, chunk_size: int = BLOCK_FRAMES) -> AudioFrameMetrics:
    metrics = tap.poll_metrics()
    for start in range(0, block_16ch.shape[0], chunk_size):
        tap.push_block(block_16ch[start : start + chunk_size])
        metrics = tap.poll_metrics()
    return metrics


# ---------------------------------------------------------------------------
# 1. 3-Step Sequential Soundcheck Calibration & Bleed-Normalized Speech SNR
# ---------------------------------------------------------------------------


def test_three_step_calibration_hot_keys_vs_pastor_snr() -> None:
    """3-step calibration produces accurate speech_to_band_snr_db with hot DI keys (-16 dBFS) vs pastor (-18 dBFS)."""
    plan = load_plan()
    # Start with intentionally wrong uncalibrated priors
    plan.channel_by_id("ch1_pastor").nominal_vband_rms_db = -12.0
    plan.channel_by_id("ch1_pastor").bleed_offset_db = -12.0
    plan.channel_by_id("ch5_keys").nominal_vband_rms_db = -24.0

    tap = AudioTap(plan, headless=True)
    rng = np.random.default_rng(42)
    n_cal = SR // 2  # 0.5 s blocks

    # Step A1: Room silence — pastor mic has -54 dBFS ambient voice-band noise
    silence_block = np.zeros((n_cal, 16), dtype=np.float32)
    silence_block[:, 0] = make_sine(1000.0, n_cal, rms_db=-54.0)  # Ch 1 pastor ambient floor
    floors = tap.capture_room_silence([silence_block])
    assert abs(floors["ch1_pastor"] - (-54.0)) < 0.8

    # Step A2 (Part 1): Isolated spoken nominals — pastor at -18 dBFS (band silent)
    spoken_block = np.zeros((n_cal, 16), dtype=np.float32)
    spoken_block[:, 0] = make_voice_signal(n_cal, rms_db=-18.0, f0_hz=175.0)
    spoken_noms = tap.capture_input_nominals("spoken", [spoken_block])
    pastor_nom_rms, pastor_nom_vb = spoken_noms["ch1_pastor"]
    assert abs(pastor_nom_rms - (-18.0)) < 0.8
    assert abs(pastor_nom_vb - (-18.0)) < 1.2
    # Noise floor normalized against newly measured nominal (~ -54 - (-18) = -36 dB)
    assert abs(plan.channel_by_id("ch1_pastor").mic_noise_floor_norm_db - (-36.0)) < 1.5

    # Step A2 (Part 2): Isolated band nominals — hot DI keys at -16 dBFS (spoken mics silent)
    band_block = np.zeros((n_cal, 16), dtype=np.float32)
    keys_sig = make_sine(880.0, n_cal, rms_db=-16.0)
    band_block[:, 4] = keys_sig  # Ch 5 (Keys L)
    band_block[:, 5] = keys_sig  # Ch 6 (Keys R)
    band_noms = tap.capture_input_nominals("band", [band_block])
    keys_nom_rms, keys_nom_vb = band_noms["ch5_keys"]
    assert abs(keys_nom_rms - (-16.0)) < 0.6
    assert abs(keys_nom_vb - (-16.0)) < 0.8

    # Step B: Band-only bleed floor — hot keys play at -16 dBFS, bleeding into pastor mic at -34 dBFS
    # Expected bleed_offset_db = (-34 - (-18)) - (-16 - (-16)) = -16.0 dB
    bleed_block = np.zeros((n_cal, 16), dtype=np.float32)
    bleed_block[:, 4] = make_sine(880.0, n_cal, rms_db=-16.0)
    bleed_block[:, 5] = make_sine(880.0, n_cal, rms_db=-16.0)
    bleed_block[:, 0] = make_sine(880.0, n_cal, rms_db=-34.0)  # -16 dB below pastor nominal
    bleed_offsets = tap.capture_bleed_floor([bleed_block])
    assert abs(bleed_offsets["ch1_pastor"] - (-16.0)) < 1.2

    # Live Case 1: Keys play at nominal (-16 dBFS), nobody speaks (expected bleed only) -> SNR ~ 0 dB
    live_bleed_only = np.zeros((BLOCK_FRAMES * 6, 16), dtype=np.float32)
    live_bleed_only[:, 4] = make_sine(880.0, live_bleed_only.shape[0], rms_db=-16.0)
    live_bleed_only[:, 5] = make_sine(880.0, live_bleed_only.shape[0], rms_db=-16.0)
    live_bleed_only[:, 0] = make_sine(880.0, live_bleed_only.shape[0], rms_db=-34.0)
    m_bleed = push_and_poll(tap, live_bleed_only)
    assert abs(m_bleed.speech_to_band_snr_db["ch1_pastor"]) < 1.5

    # Live Case 2: Keys play +4 dB hotter (-12 dBFS) with +4 dB louder bleed (-30 dBFS) -> SNR STILL ~ 0 dB!
    live_hot_bleed = np.zeros((BLOCK_FRAMES * 6, 16), dtype=np.float32)
    live_hot_bleed[:, 4] = make_sine(880.0, live_hot_bleed.shape[0], rms_db=-12.0)
    live_hot_bleed[:, 5] = make_sine(880.0, live_hot_bleed.shape[0], rms_db=-12.0)
    live_hot_bleed[:, 0] = make_sine(880.0, live_hot_bleed.shape[0], rms_db=-30.0)
    m_hot_bleed = push_and_poll(tap, live_hot_bleed)
    assert abs(m_hot_bleed.speech_to_band_snr_db["ch1_pastor"]) < 1.5

    # Live Case 3: Keys play at -16 dBFS AND pastor speaks at nominal (-18 dBFS) -> SNR ~ +16 dB (>> +6 dB)
    live_speech = np.zeros((BLOCK_FRAMES * 6, 16), dtype=np.float32)
    live_speech[:, 4] = make_sine(880.0, live_speech.shape[0], rms_db=-16.0)
    live_speech[:, 5] = make_sine(880.0, live_speech.shape[0], rms_db=-16.0)
    live_speech[:, 0] = make_voice_signal(live_speech.shape[0], rms_db=-18.0, f0_hz=175.0)
    m_speech = push_and_poll(tap, live_speech)
    assert m_speech.speech_to_band_snr_db["ch1_pastor"] > 12.0

    # Live Case 4: Band silent, pastor mic at ambient noise floor -> SNR clamped around 0 dB (not +80 dB!)
    live_quiet = np.zeros((BLOCK_FRAMES * 6, 16), dtype=np.float32)
    live_quiet[:, 0] = make_sine(1000.0, live_quiet.shape[0], rms_db=-54.0) + make_noise(
        live_quiet.shape[0], rms_db=-70.0, rng=rng
    )
    m_quiet = push_and_poll(tap, live_quiet)
    assert abs(m_quiet.speech_to_band_snr_db["ch1_pastor"]) < 2.0


# ---------------------------------------------------------------------------
# 2. Pre-Fader CLAP Sum Invariance to Master Ch 15-16 Attenuation
# ---------------------------------------------------------------------------


def test_prefader_clap_sum_strictly_invariant_to_master_attenuation() -> None:
    """Pre-fader Ch 1-12 CLAP sum must be strictly invariant to Master Ch 15-16 fader moves/mutes."""
    plan = load_plan()
    tap_full = AudioTap(plan, headless=True)
    tap_attenuated = AudioTap(plan, headless=True)
    tap_muted = AudioTap(plan, headless=True)

    n = SR // 2  # 0.5 s
    base = np.zeros((n, 16), dtype=np.float32)
    base[:, 0] = make_voice_signal(n, rms_db=-18.0)  # Ch 1 pastor
    base[:, 4] = make_sine(440.0, n, rms_db=-16.0)  # Ch 5 keys L
    base[:, 5] = make_sine(440.0, n, rms_db=-16.0)  # Ch 6 keys R
    base[:, 8] = make_sine(120.0, n, rms_db=-14.0)  # Ch 9 drums

    blk_full = base.copy()
    blk_full[:, 14] = make_sine(1000.0, n, rms_db=-6.0)  # Master L loud
    blk_full[:, 15] = make_sine(1000.0, n, rms_db=-6.0)  # Master R loud

    blk_atten = base.copy()
    blk_atten[:, 14] = make_sine(1000.0, n, rms_db=-36.0)  # Master -30 dB attenuated
    blk_atten[:, 15] = make_sine(1000.0, n, rms_db=-36.0)

    blk_muted = base.copy()  # Master Ch 15-16 completely silent (0.0)

    m_full = push_and_poll(tap_full, blk_full)
    m_atten = push_and_poll(tap_attenuated, blk_atten)
    m_muted = push_and_poll(tap_muted, blk_muted)

    assert m_full.pre_fader_sum_3s.shape == (SR * 3,)
    assert float(np.sqrt(np.mean(m_full.pre_fader_sum_3s[-n:] ** 2))) > 0.01
    # Bitwise / exact numerical invariance across Master levels
    np.testing.assert_allclose(m_full.pre_fader_sum_3s, m_atten.pre_fader_sum_3s, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(m_full.pre_fader_sum_3s, m_muted.pre_fader_sum_3s, atol=1e-7, rtol=0.0)
    # Meanwhile Master LUFS MUST track the Master Ch 15-16 difference!
    assert m_full.master_lufs_momentary - m_atten.master_lufs_momentary == pytest.approx(30.0, abs=0.5)
    assert m_muted.master_lufs_momentary <= -69.0


# ---------------------------------------------------------------------------
# 3. Master LUFS ITU-R BS.1770-4 K-Weighting Accuracy on Ch 15-16
# ---------------------------------------------------------------------------


def test_master_lufs_bs1770_k_weighted_accuracy() -> None:
    """ITU-R BS.1770-4 K-weighted momentary (400 ms) & short-term (3 s) LUFS on Ch 15-16."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)

    # 997 Hz stereo sine with -18 dBFS RMS per channel -> z_L + z_R = 2 * 10^(-18/10) -> -15.0 dB
    # Plus BS.1770 K-weighting at 997 Hz (+0.68 dB) and -0.691 offset -> -15.00 LUFS!
    n_3s = SR * 3
    blk = np.zeros((n_3s, 16), dtype=np.float32)
    tone_1k = make_sine(997.0, n_3s, rms_db=-18.0)
    blk[:, 14] = tone_1k
    blk[:, 15] = tone_1k

    metrics = push_and_poll(tap, blk)
    assert metrics.master_lufs_momentary == pytest.approx(-15.0, abs=0.4)
    assert metrics.master_lufs_short_term == pytest.approx(-15.0, abs=0.4)

    # Verify K-weighting high-shelf (+4.0 dB vs DC, which is +3.32 dB vs 997 Hz since 997 Hz is +0.68 dB)
    tap_hf = AudioTap(plan, headless=True)
    n_1s = SR
    blk_hf = np.zeros((n_1s, 16), dtype=np.float32)
    tone_5k = make_sine(5000.0, n_1s, rms_db=-18.0)
    blk_hf[:, 14] = tone_5k
    blk_hf[:, 15] = tone_5k
    m_hf = push_and_poll(tap_hf, blk_hf)
    assert (m_hf.master_lufs_momentary - metrics.master_lufs_momentary) == pytest.approx(3.32, abs=0.25)

    # Verify RLB high-pass attenuates 30 Hz sub-bass relative to 1 kHz
    tap_lf = AudioTap(plan, headless=True)
    blk_lf = np.zeros((n_1s, 16), dtype=np.float32)
    tone_30 = make_sine(30.0, n_1s, rms_db=-18.0)
    blk_lf[:, 14] = tone_30
    blk_lf[:, 15] = tone_30
    m_lf = push_and_poll(tap_lf, blk_lf)
    assert m_lf.master_lufs_momentary < metrics.master_lufs_momentary - 4.0


# ---------------------------------------------------------------------------
# 4. TCC Bitwise All-Zero Guard (> 2.0 s) & Clip / Room Flatness
# ---------------------------------------------------------------------------


def test_tcc_zero_buffer_guard_triggers_and_clears() -> None:
    """TCC silence warning triggers strictly after > 2.0 s of exact zeros and clears on non-zero."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)

    zero_block = np.zeros((BLOCK_FRAMES, 16), dtype=np.float32)
    # 1.96 s (98 x 20 ms) -> NOT triggered yet
    for _ in range(98):
        tap.push_block(zero_block)
        m = tap.poll_metrics()
    assert m.tcc_silence_warning is False

    # Push 3 more blocks -> 2.02 s (> 2.0 s) -> MUST trigger
    for _ in range(3):
        tap.push_block(zero_block)
        m = tap.poll_metrics()
    assert m.tcc_silence_warning is True

    # Single tiny non-zero sample clears warning immediately
    dither_block = np.zeros((BLOCK_FRAMES, 16), dtype=np.float32)
    dither_block[0, 0] = 1e-6
    tap.push_block(dither_block)
    m_clear = tap.poll_metrics()
    assert m_clear.tcc_silence_warning is False


def test_clip_flag_fft_and_room_spectral_flatness() -> None:
    """Verify clip >= -0.5 dBFS, 1024-pt Hann FFT peak bin, and room applause spectral flatness."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)
    rng = np.random.default_rng(7)

    # Silent room must produce flatness == 0.0 (not 1.0!)
    silent = np.zeros((BLOCK_FRAMES * 4, 16), dtype=np.float32)
    m_sil = push_and_poll(tap, silent)
    assert m_sil.room_flatness == 0.0
    assert m_sil.clip["ch1_pastor"] is False

    # Broadband noise on Ch 13-14 (applause) -> room_flatness > 0.55; clipping tone on Ch 1
    n = BLOCK_FRAMES * 12  # 240 ms
    blk = np.zeros((n, 16), dtype=np.float32)
    blk[:, 0] = make_sine(1000.0, n, rms_db=-3.1)  # peak ~ -0.09 dBFS -> clip=True
    blk[:, 4] = make_sine(1000.0, n, rms_db=-10.0)  # peak ~ -7 dBFS -> clip=False
    blk[:, 12] = make_noise(n, rms_db=-24.0, rng=rng)  # Ch 13 room L
    blk[:, 13] = make_noise(n, rms_db=-24.0, rng=rng)  # Ch 14 room R
    m_app = push_and_poll(tap, blk)
    assert m_app.clip["ch1_pastor"] is True
    assert m_app.clip["ch5_keys"] is False
    assert m_app.room_flatness > 0.55

    # FFT bin accuracy for 1 kHz tone
    freqs = np.fft.rfftfreq(1024, d=1.0 / SR)
    peak_bin = int(np.argmax(m_app.fft_mag_db["ch5_keys"]))
    assert abs(freqs[peak_bin] - 1000.0) <= 50.0

    # Pure tonal chord on room mic -> low flatness (< 0.35)
    tap_tonal = AudioTap(plan, headless=True)
    blk_tonal = np.zeros((n, 16), dtype=np.float32)
    blk_tonal[:, 12] = make_sine(440.0, n, rms_db=-24.0) + make_sine(660.0, n, rms_db=-26.0)
    blk_tonal[:, 13] = blk_tonal[:, 12]
    m_tonal = push_and_poll(tap_tonal, blk_tonal)
    assert m_tonal.room_flatness < 0.35


# ---------------------------------------------------------------------------
# 5. VADWorker: Speech VAD & Sung-Vocal Autocorrelation Harmonicity
# ---------------------------------------------------------------------------


def test_vad_worker_speech_and_sung_vocal_harmonicity() -> None:
    """Sung-vocal harmonicity distinguishes pitched vocal tone from unpitched noise/bleed."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)
    vad = VADWorker(plan, model_path=Path("/nonexistent/silero_vad.onnx"))
    vad.start()
    rng = np.random.default_rng(123)

    n = BLOCK_FRAMES * 8  # 160 ms

    # Case A: Pastor speaking + Lead vocal singing pitched melody (330 Hz + harmonics), band silent
    blk_sing = np.zeros((n, 16), dtype=np.float32)
    blk_sing[:, 0] = make_voice_signal(n, rms_db=-18.0, f0_hz=150.0)  # ch1_pastor
    blk_sing[:, 1] = make_voice_signal(n, rms_db=-18.0, f0_hz=330.0)  # ch2_lead_vox (pitched)
    blk_sing[:, 2] = make_noise(n, rms_db=-20.0, rng=rng)  # ch3_bv1 (unpitched noise of SAME level!)
    m_sing = push_and_poll(tap, blk_sing)
    vad_probs, sung_active = vad.process_frame(m_sing)

    assert vad_probs["ch1_pastor"] >= 0.70
    assert sung_active["ch2_lead_vox"] is True
    assert vad.last_harmonicity["ch2_lead_vox"] >= 0.70
    # Unpitched noise at identical RMS level must NOT trigger sung vocal!
    assert sung_active["ch3_bv1"] is False
    assert vad.last_harmonicity["ch3_bv1"] < 0.35

    # Case B: Pitched tone on ch2_lead_vox that is BELOW bleed floor (loud band, low mic level) -> False!
    tap_bleed = AudioTap(plan, headless=True)
    blk_bleed = np.zeros((n, 16), dtype=np.float32)
    blk_bleed[:, 4] = make_sine(440.0, n, rms_db=-14.0)  # Loud keys (+4 dB over nominal)
    blk_bleed[:, 5] = make_sine(440.0, n, rms_db=-14.0)
    # Lead vocal has bleed_offset_db=-14, nominal=-20 -> expected bleed = -30 dBFS; put -32 dBFS (SNR < 0)
    blk_bleed[:, 1] = make_voice_signal(n, rms_db=-32.0, f0_hz=330.0)
    m_bleed = push_and_poll(tap_bleed, blk_bleed)
    _, sung_bleed = vad.process_frame(m_bleed)
    assert sung_bleed["ch2_lead_vox"] is False

    # Case C: Pure silence -> VAD ~ 0, no sung vocals
    tap_sil = AudioTap(plan, headless=True)
    m_sil = push_and_poll(tap_sil, np.zeros((n, 16), dtype=np.float32))
    vad_sil, sung_sil = vad.process_frame(m_sil)
    assert vad_sil["ch1_pastor"] < 0.1
    assert not any(sung_sil.values())

    vad.stop()


# ---------------------------------------------------------------------------
# 6. CLAPWorker Graceful Fallback
# ---------------------------------------------------------------------------


def test_clap_worker_graceful_fallback_without_weights(tmp_path: Path) -> None:
    """CLAPWorker falls back cleanly to (None, False) when disabled or weights/torch are missing."""
    worker = CLAPWorker(enabled=True, model_dir=tmp_path / "missing_clap_dir")
    worker.start()
    worker.submit_audio(np.zeros(SR * 3, dtype=np.float32), sample_rate=SR)
    probs, healthy = worker.latest_probs()
    assert probs is None
    assert healthy is False
    worker.stop()

    disabled_worker = CLAPWorker(enabled=False)
    disabled_worker.start()
    disabled_worker.submit_audio(np.zeros(SR * 3, dtype=np.float32), sample_rate=SR)
    assert disabled_worker.latest_probs() == (None, False)
    disabled_worker.stop()


# ---------------------------------------------------------------------------
# 7. SceneClassifier: Sensor Fusion, 2-Window Hysteresis & TRANSITION Hold
# ---------------------------------------------------------------------------


def _synthetic_metrics(
    plan: ServicePlan,
    norm_vband: dict[str, float],
    snr: dict[str, float] | None = None,
    room_flatness: float = 0.0,
) -> AudioFrameMetrics:
    full_norm = {ch.id: norm_vband.get(ch.id, -50.0) for ch in plan.channels}
    full_snr = {ch.id: (snr or {}).get(ch.id, 0.0) for ch in plan.channels}
    return AudioFrameMetrics(
        ts_ms=int(time.time() * 1000),
        rms_db={ch.id: ch.nominal_rms_db + full_norm[ch.id] for ch in plan.channels},
        vband_rms_db={ch.id: ch.nominal_vband_rms_db + full_norm[ch.id] for ch in plan.channels},
        norm_vband_db=full_norm,
        peak_db={ch.id: -20.0 for ch in plan.channels},
        clip={ch.id: False for ch in plan.channels},
        fft_mag_db={ch.id: np.full(513, -120.0, dtype=np.float32) for ch in plan.channels},
        speech_to_band_snr_db=full_snr,
        room_flatness=room_flatness,
        master_lufs_momentary=-20.0,
        master_lufs_short_term=-20.0,
        tcc_silence_warning=False,
        pre_fader_sum_3s=np.zeros(SR * 3, dtype=np.float32),
        vocal_pcm_16k={},
    )


def test_scene_classifier_hysteresis_and_transition_retains_last_macro_phase() -> None:
    """Hysteresis requires 2 consecutive windows >= 0.70, and TRANSITION retains last_macro_phase."""
    plan = load_plan()
    clf = SceneClassifier(plan, hysteresis_windows=2, confidence_threshold=0.70)
    assert clf.active_phase == ServicePhase.SPOKEN_WORD
    assert clf.last_macro_phase == ServicePhase.SPOKEN_WORD

    # 1. Full band worship high -> Window 1 should NOT transition yet (hysteresis_progress = 0.5)
    m_high = _synthetic_metrics(
        plan,
        norm_vband={"ch5_keys": -2.0, "ch7_guitars": -2.0, "ch9_drums": 0.0, "ch2_lead_vox": -2.0},
    )
    sung_high = {"ch2_lead_vox": True, "ch3_bv1": True, "ch4_bv2": False}
    clap_high = {
        ServicePhase.SPOKEN_WORD: 0.02,
        ServicePhase.PRAYER_UNDERSCORE: 0.02,
        ServicePhase.WORSHIP_LOW: 0.06,
        ServicePhase.WORSHIP_HIGH: 0.88,
        ServicePhase.TRANSITION: 0.02,
    }

    d1 = clf.update(m_high, vad_probs={}, sung_vocals=sung_high, clap_probs=clap_high, service_step_idx=0)
    assert d1.transitioned is False
    assert d1.active_phase == ServicePhase.SPOKEN_WORD
    assert d1.hysteresis_progress == pytest.approx(0.5)

    # Window 2 consecutive -> commits transition to WORSHIP_HIGH!
    d2 = clf.update(m_high, vad_probs={}, sung_vocals=sung_high, clap_probs=clap_high, service_step_idx=0)
    assert d2.transitioned is True
    assert d2.active_phase == ServicePhase.WORSHIP_HIGH
    assert d2.last_macro_phase == ServicePhase.WORSHIP_HIGH
    assert d2.hysteresis_progress == pytest.approx(1.0)
    assert d2.confidence >= 0.70

    # 2. Single stray window of SPOKEN_WORD followed by return to WORSHIP_HIGH must NOT transition
    m_spoken = _synthetic_metrics(plan, norm_vband={"ch1_pastor": 0.0}, snr={"ch1_pastor": 15.0})
    d_stray = clf.update(
        m_spoken,
        vad_probs={"ch1_pastor": 0.95},
        sung_vocals={},
        clap_probs=None,
        service_step_idx=0,
    )
    assert d_stray.transitioned is False
    assert d_stray.active_phase == ServicePhase.WORSHIP_HIGH
    # Back to WORSHIP_HIGH resets candidate
    d_steady = clf.update(m_high, vad_probs={}, sung_vocals=sung_high, clap_probs=clap_high, service_step_idx=0)
    assert d_steady.transitioned is False
    assert d_steady.active_phase == ServicePhase.WORSHIP_HIGH

    # 3. Applause between songs -> TRANSITION after 2 windows, RETAINING last_macro_phase == WORSHIP_HIGH!
    m_applause = _synthetic_metrics(plan, norm_vband={}, room_flatness=0.72)
    d_t1 = clf.update(m_applause, vad_probs={}, sung_vocals={}, clap_probs=None, service_step_idx=0)
    assert d_t1.transitioned is False
    assert d_t1.active_phase == ServicePhase.WORSHIP_HIGH

    d_t2 = clf.update(m_applause, vad_probs={}, sung_vocals={}, clap_probs=None, service_step_idx=0)
    assert d_t2.transitioned is True
    assert d_t2.active_phase == ServicePhase.TRANSITION
    assert d_t2.last_macro_phase == ServicePhase.WORSHIP_HIGH  # Preserved!

    # 4. Pastor praying over soft keys pad (CLAP=None, stale plan step) -> PRAYER_UNDERSCORE
    m_prayer = _synthetic_metrics(
        plan,
        norm_vband={"ch1_pastor": -2.0, "ch5_keys": -6.0},
        snr={"ch1_pastor": 14.0},
    )
    clf.update(m_prayer, vad_probs={"ch1_pastor": 0.92}, sung_vocals={}, clap_probs=None, service_step_idx=0)
    d_pray = clf.update(
        m_prayer, vad_probs={"ch1_pastor": 0.92}, sung_vocals={}, clap_probs=None, service_step_idx=0
    )
    assert d_pray.transitioned is True
    assert d_pray.active_phase == ServicePhase.PRAYER_UNDERSCORE
    assert d_pray.last_macro_phase == ServicePhase.PRAYER_UNDERSCORE

    # 5. Manual phase override pins phase for hold_seconds and releases after expiry
    clf.override_phase(ServicePhase.WORSHIP_LOW, hold_seconds=60.0, now_s=100.0)
    d_ovr = clf.update(
        m_high, vad_probs={}, sung_vocals=sung_high, clap_probs=clap_high, service_step_idx=0, now_s=120.0
    )
    assert d_ovr.active_phase == ServicePhase.WORSHIP_LOW
    assert d_ovr.transitioned is False
    assert "manual_override" in d_ovr.reason

    # After hold expires (t=165s), normal hysteresis resumes
    clf.update(m_high, vad_probs={}, sung_vocals=sung_high, clap_probs=clap_high, service_step_idx=0, now_s=165.0)
    d_post = clf.update(
        m_high, vad_probs={}, sung_vocals=sung_high, clap_probs=clap_high, service_step_idx=0, now_s=167.0
    )
    assert d_post.transitioned is True
    assert d_post.active_phase == ServicePhase.WORSHIP_HIGH


def test_scene_classifier_worship_low_without_service_plan() -> None:
    """Acoustic guitar + lead vocal singing (drums silent) classifies cleanly as WORSHIP_LOW."""
    plan = load_plan()
    clf = SceneClassifier(plan, hysteresis_windows=2, confidence_threshold=0.70)

    m_low = _synthetic_metrics(
        plan,
        norm_vband={"ch2_lead_vox": -4.0, "ch5_keys": -8.0, "ch7_guitars": -6.0},
    )
    sung_low = {"ch2_lead_vox": True, "ch3_bv1": False, "ch4_bv2": False}

    d1 = clf.update(m_low, vad_probs={}, sung_vocals=sung_low, clap_probs=None, use_service_plan=False)
    assert d1.transitioned is False
    d2 = clf.update(m_low, vad_probs={}, sung_vocals=sung_low, clap_probs=None, use_service_plan=False)
    assert d2.transitioned is True
    assert d2.active_phase == ServicePhase.WORSHIP_LOW
    assert d2.last_macro_phase == ServicePhase.WORSHIP_LOW


def test_50hz_dsp_loop_performance_budget() -> None:
    """Per-tick AudioTap + VADWorker latency must stay well below 5 ms p95 budget."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)
    vad = VADWorker(plan, model_path=Path("/nonexistent/silero_vad.onnx"))
    vad.start()

    rng = np.random.default_rng(99)
    block = rng.standard_normal((BLOCK_FRAMES, 16)).astype(np.float32) * 0.05

    # Warm-up
    for _ in range(5):
        tap.push_block(block)
        vad.process_frame(tap.poll_metrics())

    times_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        tap.push_block(block)
        m = tap.poll_metrics()
        vad.process_frame(m)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    p95 = float(np.percentile(times_ms, 95))
    assert p95 < 5.0, f"DSP + VAD p95 tick latency {p95:.2f} ms exceeded 5.0 ms budget"
    vad.stop()


# ---------------------------------------------------------------------------
# 8. Regression Tests: Live Calibration Ring Helper, Monotonic TS & Non-Overlapping ONNX VAD
# ---------------------------------------------------------------------------


def test_recent_raw_blocks_and_monotonic_last_block_ts_ms() -> None:
    """`last_block_ts_ms` only advances on real audio; `recent_raw_blocks` returns real drained/live audio."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)

    # Before any audio arrives: ts_ms == 0, last_block_ts_ms == 0, no recent blocks
    assert tap.last_block_ts_ms == 0
    m0 = tap.poll_metrics()
    assert m0.ts_ms == 0
    assert tap.last_block_ts_ms == 0
    assert tap.recent_raw_blocks(5.0) == []

    # Push 50 x 20ms blocks (1.0 s total) with distinct amplitudes to verify tail ordering
    prev_ts = 0
    for i in range(50):
        blk = np.full((BLOCK_FRAMES, 16), fill_value=float(i + 1) * 0.001, dtype=np.float32)
        tap.push_block(blk)
        m = tap.poll_metrics()
        assert m.ts_ms > prev_ts
        assert tap.last_block_ts_ms == m.ts_ms
        prev_ts = m.ts_ms

    # Empty-ring poll MUST NOT advance ts_ms or last_block_ts_ms
    m_stale = tap.poll_metrics()
    assert m_stale.ts_ms == prev_ts
    assert tap.last_block_ts_ms == prev_ts

    # Requesting 0.25 s of recent blocks returns exact 0.25 * SR = 12,000 samples from tail
    recent = tap.recent_raw_blocks(duration_s=0.25)
    joined = np.vstack(recent)
    assert joined.shape == (SR // 4, 16)
    # Tail sample came from block #50 (value 0.050)
    assert joined[-1, 0] == pytest.approx(0.050, abs=1e-6)
    # First sample of 0.25s slice (last 12.5 blocks -> inside block #38, value 0.038)
    assert joined[0, 0] == pytest.approx(0.038, abs=1e-6)

    # Also captures undrained blocks pushed via PortAudio callback before next poll
    live_blk = np.full((BLOCK_FRAMES, 16), fill_value=0.123, dtype=np.float32)
    tap._audio_callback(live_blk, BLOCK_FRAMES, None, None)
    recent_with_live = np.vstack(tap.recent_raw_blocks(duration_s=0.02))
    assert recent_with_live.shape == (BLOCK_FRAMES, 16)
    assert recent_with_live[-1, 0] == pytest.approx(0.123, abs=1e-6)


def test_onnx_silero_vad_non_overlapping_512_sample_hops() -> None:
    """50 Hz polling (320 samples/tick @ 16 kHz) must feed strictly non-overlapping 512-sample RNN hops."""
    plan = load_plan()
    tap = AudioTap(plan, headless=True)
    vad = VADWorker(plan, model_path=Path("/nonexistent/silero_vad.onnx"))
    vad.start()

    # Inject a mock v5 ONNX session that records every (1, 576) context+chunk fed to the RNN
    class _MockSileroSession:
        def __init__(self) -> None:
            self.chunks_by_call: list[np.ndarray] = []

        def run(self, _output_names: object, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            inp = feed["input"]
            assert inp.shape == (1, 576)
            assert int(feed["sr"]) == 16000
            # Trailing 512 samples are the non-overlapping audio hop (first 64 are v5 context)
            self.chunks_by_call.append(inp[0, 64:].copy())
            return np.array([[0.85]], dtype=np.float32), feed["state"]

    mock_session = _MockSileroSession()
    vad._ort_session = mock_session
    vad._silero_v5 = True
    vad._using_onnx = True
    vad._reset_silero_states()

    # Stream 50 x 20ms blocks (1.0 s = 48,000 samples @ 48 kHz -> 16,000 samples @ 16 kHz)
    # 2 vad_enabled channels (ch1_pastor, ch2_lead_vox) -> 16000 // 512 = 31 non-overlapping hops EACH!
    all_new_pastor: list[np.ndarray] = []
    for i in range(50):
        blk = np.zeros((BLOCK_FRAMES, 16), dtype=np.float32)
        blk[:, 0] = make_voice_signal(BLOCK_FRAMES, rms_db=-18.0, offset_samples=i * BLOCK_FRAMES)
        blk[:, 1] = make_voice_signal(BLOCK_FRAMES, rms_db=-20.0, f0_hz=300.0, offset_samples=i * BLOCK_FRAMES)
        tap.push_block(blk)
        m = tap.poll_metrics()
        all_new_pastor.append(m.vocal_new_pcm_16k["ch1_pastor"])
        probs, _ = vad.process_frame(m)
        if i >= 1:  # After 2nd tick (640 >= 512 samples), ONNX has fired at least once
            assert probs["ch1_pastor"] == pytest.approx(0.85)

    # 31 hops per channel x 2 vad_enabled channels = 62 exact RNN calls (NOT 100!)
    expected_hops_per_ch = 16000 // 512
    assert expected_hops_per_ch == 31
    assert len(mock_session.chunks_by_call) == expected_hops_per_ch * 2

    # Verify ch1_pastor chunks (even indices: 0, 2, 4, ...) match continuous stream with ZERO overlap
    pastor_stream = np.concatenate(all_new_pastor)
    pastor_fed = np.concatenate(mock_session.chunks_by_call[0::2])
    assert pastor_fed.shape == (expected_hops_per_ch * 512,)
    np.testing.assert_allclose(pastor_fed, pastor_stream[: expected_hops_per_ch * 512], atol=1e-7)

    # Empty-ring poll afterwards must trigger ZERO additional ONNX calls
    m_empty = tap.poll_metrics()
    probs_empty, _ = vad.process_frame(m_empty)
    assert len(mock_session.chunks_by_call) == expected_hops_per_ch * 2
    assert probs_empty["ch1_pastor"] == pytest.approx(0.85)

    vad.stop()
