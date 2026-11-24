"""Sensor-Fused Service Phase Classifier with Hysteresis & TRANSITION Preservation (SPEC v5.1).

Combines three evidence streams at Tier-2 cadence (0.5 Hz / 2 s windows):
1. `P_clap`: Semantic audio classification from isolated pre-fader CLAP subprocess (weight 0.40).
2. `P_telemetry`: Quantitative DSP telemetry (calibrated `norm_vband_db`, bleed-normalized
   `speech_to_band_snr_db`, `vad_probs`, sung-vocal harmonicity, and `ch13_room` broadband
   spectral flatness > 0.55 for applause/transitions) (weight 0.40).
3. `P_service_plan`: Order-of-Service segment prior (weight 0.20, optional).

When `clap_probs` is unavailable (`None`), acoustic weight stays 0.80 on `P_telemetry` (+ 0.20
plan) so live audio can still override a stale Order-of-Service step; when `use_service_plan`
is False, acoustic streams re-normalize to `0.50 / 0.50` (or `1.00` on `P_telemetry` alone).

Enforces 2-window hysteresis (`hysteresis_windows=2`, `confidence_threshold=0.70`) and preserves
`last_macro_phase` whenever transitioning into `TRANSITION`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from engine.core.audio_tap import AudioFrameMetrics
from engine.core.protocol import MACRO_PHASES, ChannelRole, ServicePhase, ServicePlan

ALL_PHASES: tuple[ServicePhase, ...] = (
    ServicePhase.SPOKEN_WORD,
    ServicePhase.PRAYER_UNDERSCORE,
    ServicePhase.WORSHIP_LOW,
    ServicePhase.WORSHIP_HIGH,
    ServicePhase.TRANSITION,
)

# Calibrated norm_vband_db thresholds (SPEC §5.3)
ACTIVE_NORM_DB = -18.0
SILENT_NORM_DB = -30.0
APPLAUSE_FLATNESS_THRESHOLD = 0.55


@dataclass
class SceneDecision:
    active_phase: ServicePhase
    last_macro_phase: ServicePhase
    confidence: float
    hysteresis_progress: float
    transitioned: bool
    reason: str
    scores: dict[ServicePhase, float]


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def _norm_activation(norm_db: float) -> float:
    """Linear activation in [0, 1]: 0.0 at <= -30 dB (silent), 1.0 at >= -18 dB (active)."""
    return _clamp01((norm_db - SILENT_NORM_DB) / (ACTIVE_NORM_DB - SILENT_NORM_DB))


def _softmax(logits: dict[ServicePhase, float], beta: float = 2.0) -> dict[ServicePhase, float]:
    max_val = max(logits.values())
    exps = {p: math.exp(beta * (v - max_val)) for p, v in logits.items()}
    total = sum(exps.values())
    if total <= 0.0:
        uniform = 1.0 / len(ALL_PHASES)
        return {p: uniform for p in ALL_PHASES}
    return {p: val / total for p, val in exps.items()}


class SceneClassifier:
    """Sensor-fused church service phase classifier with hysteresis and manual override."""

    def __init__(
        self,
        plan: ServicePlan,
        hysteresis_windows: int = 2,
        confidence_threshold: float = 0.70,
        initial_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    ) -> None:
        self.plan = plan
        self.hysteresis_windows = max(1, int(hysteresis_windows))
        self.confidence_threshold = float(confidence_threshold)

        self.active_phase: ServicePhase = initial_phase
        self.last_macro_phase: ServicePhase = (
            initial_phase if initial_phase in MACRO_PHASES else ServicePhase.SPOKEN_WORD
        )

        self._candidate_phase: ServicePhase | None = None
        self._candidate_count: int = 0

        self._override_phase: ServicePhase | None = None
        self._override_until_s: float = 0.0

        # Pre-index channels by role for O(1) lookups
        self._spoken_ids = [ch.id for ch in plan.channels if ch.role == ChannelRole.SPOKEN_PRIMARY]
        self._vocal_lead_ids = [ch.id for ch in plan.channels if ch.role == ChannelRole.VOCAL_LEAD]
        self._vocal_bg_ids = [ch.id for ch in plan.channels if ch.role == ChannelRole.VOCAL_BG]
        self._pad_ids = [ch.id for ch in plan.channels if ch.role == ChannelRole.INSTRUMENT_PAD]
        self._guitar_ids = [ch.id for ch in plan.channels if ch.role == ChannelRole.INSTRUMENT_GUITAR]
        self._band_ids = [ch.id for ch in plan.channels if ch.role == ChannelRole.INSTRUMENT_BAND]

        # Rolling 2-second observation window (100 ticks @ 50 Hz) so brief speech pauses
        # between real spoken sentences never falsely flip SPOKEN_WORD/PRAYER to TRANSITION.
        self._obs_window: deque[tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, bool], float]] = deque(
            maxlen=100
        )

    def observe_tick(
        self,
        metrics: AudioFrameMetrics,
        vad_probs: dict[str, float],
        sung_vocals: dict[str, bool],
    ) -> None:
        """Record one 50 Hz perception tick into the rolling 2-second window."""
        self._obs_window.append(
            (
                dict(metrics.norm_vband_db),
                dict(metrics.speech_to_band_snr_db),
                dict(vad_probs),
                dict(sung_vocals),
                float(metrics.room_flatness),
            )
        )

    def _window_aggregated_inputs(
        self,
        metrics: AudioFrameMetrics,
        vad_probs: dict[str, float],
        sung_vocals: dict[str, bool],
    ) -> tuple[AudioFrameMetrics, dict[str, float], dict[str, bool]]:
        """Aggregate the rolling 2 s observation window (or pass through if unused)."""
        if len(self._obs_window) < 5:
            return metrics, vad_probs, sung_vocals

        agg_norm: dict[str, float] = {}
        agg_snr: dict[str, float] = {}
        agg_vad: dict[str, float] = {}
        agg_sung: dict[str, bool] = {}

        for ch in self.plan.channels:
            cid = ch.id
            norms = [row[0].get(cid, -120.0) for row in self._obs_window]
            snrs = [row[1].get(cid, -60.0) for row in self._obs_window]
            vads = [row[2].get(cid, 0.0) for row in self._obs_window]
            sungs = [1.0 if row[3].get(cid, False) else 0.0 for row in self._obs_window]

            # Upper-active percentiles ignore natural 200-600 ms breath pauses between phrases
            agg_norm[cid] = float(np.percentile(norms, 80))
            agg_snr[cid] = float(np.percentile(snrs, 85))
            agg_vad[cid] = float(np.percentile(vads, 85))
            agg_sung[cid] = (sum(sungs) / len(sungs)) >= 0.15

        flat_vals = [row[4] for row in self._obs_window]
        agg_flat = float(np.percentile(flat_vals, 75))

        agg_metrics = AudioFrameMetrics(
            ts_ms=metrics.ts_ms,
            rms_db=metrics.rms_db,
            vband_rms_db=metrics.vband_rms_db,
            norm_vband_db=agg_norm,
            peak_db=metrics.peak_db,
            clip=metrics.clip,
            fft_mag_db=metrics.fft_mag_db,
            speech_to_band_snr_db=agg_snr,
            room_flatness=agg_flat,
            master_lufs_momentary=metrics.master_lufs_momentary,
            master_lufs_short_term=metrics.master_lufs_short_term,
            tcc_silence_warning=metrics.tcc_silence_warning,
            pre_fader_sum_3s=metrics.pre_fader_sum_3s,
            vocal_pcm_16k=metrics.vocal_pcm_16k,
        )
        return agg_metrics, agg_vad, agg_sung

    def override_phase(
        self,
        phase: ServicePhase,
        hold_seconds: float = 60.0,
        now_s: float | None = None,
    ) -> None:
        """Manually pin the active phase for `hold_seconds`, resetting hysteresis state."""
        t = time.monotonic() if now_s is None else float(now_s)
        self._override_phase = phase
        self._override_until_s = t + max(0.0, float(hold_seconds))
        self.active_phase = phase
        if phase in MACRO_PHASES:
            self.last_macro_phase = phase
        self._candidate_phase = None
        self._candidate_count = 0

    def compute_telemetry_probs(
        self,
        metrics: AudioFrameMetrics,
        vad_probs: dict[str, float],
        sung_vocals: dict[str, bool],
    ) -> dict[ServicePhase, float]:
        """Compute quantitative `P_telemetry` distribution across all 5 service phases."""
        # 1. Spoken primary activity (requires VAD + bleed-normalized SNR + level)
        spoken_score = 0.0
        for cid in self._spoken_ids:
            vad = _clamp01(float(vad_probs.get(cid, 0.0)))
            snr = float(metrics.speech_to_band_snr_db.get(cid, -20.0))
            norm = float(metrics.norm_vband_db.get(cid, -120.0))
            snr_gate = _clamp01((snr - 1.0) / 5.0)
            lvl_gate = _norm_activation(norm)
            score = vad * snr_gate * lvl_gate
            spoken_score = max(spoken_score, score)

        # 2. Sung-vocal harmonicity activity (Ch 2-4)
        lead_sung = any(bool(sung_vocals.get(cid, False)) for cid in self._vocal_lead_ids)
        bg_sung_count = sum(1 for cid in self._vocal_bg_ids if bool(sung_vocals.get(cid, False)))
        sung_factor = 1.0 if lead_sung else (0.85 if bg_sung_count > 0 else 0.0)
        bv_factor = _clamp01(bg_sung_count / max(1, len(self._vocal_bg_ids))) if self._vocal_bg_ids else 0.0

        # Also support spoken worship leader (talking without singing, instruments silent)
        if spoken_score < 0.3 and not lead_sung:
            for cid in self._vocal_lead_ids:
                vad = _clamp01(float(vad_probs.get(cid, 0.0)))
                snr = float(metrics.speech_to_band_snr_db.get(cid, -20.0))
                norm = float(metrics.norm_vband_db.get(cid, -120.0))
                lead_spoken = vad * _clamp01((snr - 2.0) / 5.0) * _norm_activation(norm)
                if lead_spoken > spoken_score:
                    spoken_score = lead_spoken * 0.9

        # 3. Instrument role activations from calibrated norm_vband_db
        pad_norm = max((metrics.norm_vband_db.get(cid, -120.0) for cid in self._pad_ids), default=-120.0)
        guitar_norm = max((metrics.norm_vband_db.get(cid, -120.0) for cid in self._guitar_ids), default=-120.0)
        drums_norm = max((metrics.norm_vband_db.get(cid, -120.0) for cid in self._band_ids), default=-120.0)

        pad_act = _norm_activation(pad_norm)
        guitar_act = _norm_activation(guitar_norm)
        drums_act = _norm_activation(drums_norm)
        other_band_act = max(guitar_act, drums_act)
        all_instruments_act = max(pad_act, guitar_act, drums_act)

        # 4. Room ambience applause cue (> 0.55 spectral flatness) & stage silence cue
        room_flatness = float(metrics.room_flatness)
        applause_score = _clamp01((room_flatness - 0.45) / (APPLAUSE_FLATNESS_THRESHOLD - 0.40))
        silence_score = (1.0 - spoken_score) * (1.0 - sung_factor) * (1.0 - all_instruments_act)

        # 5. Phase affinity logits
        # SPOKEN_WORD: pastor speaking alone without instruments or sung vocals
        s_spoken = (
            3.3
            * spoken_score
            * (1.0 - 0.85 * pad_act)
            * (1.0 - other_band_act)
            * (1.0 - 0.85 * sung_factor)
            * (1.0 - 0.4 * applause_score)
        )

        # PRAYER_UNDERSCORE: pastor speaking over keys/pad with no drums/guitars/singing
        # (also holds cleanly across brief pastor breath pauses when only keys/pad plays)
        s_prayer = (
            3.5
            * spoken_score
            * pad_act
            * (1.0 - other_band_act)
            * (1.0 - 0.85 * sung_factor)
            * (1.0 - 0.5 * applause_score)
            + 1.85
            * (1.0 - spoken_score)
            * pad_act
            * (1.0 - other_band_act)
            * (1.0 - sung_factor)
            * (1.0 - applause_score)
        )

        # WORSHIP_LOW: gentle acoustic guitar / keys + optional singing, without full drums
        low_inst = max(
            guitar_act,
            0.85 * pad_act * (1.0 if (sung_factor > 0.3 or guitar_act > 0.3) else 0.45),
        )
        s_worship_low = (
            (2.5 * low_inst + 1.3 * sung_factor)
            * (1.0 - 0.85 * drums_act)
            * (1.0 - 0.85 * spoken_score)
            * (1.0 - 0.55 * applause_score)
        )

        # WORSHIP_HIGH: drums/band active with ensemble/vocal support
        ensemble_support = min(
            1.0,
            0.45 + 0.30 * max(pad_act, guitar_act) + 0.25 * sung_factor + 0.20 * bv_factor,
        )
        s_worship_high = (
            3.6 * drums_act * ensemble_support * (1.0 - 0.45 * spoken_score) * (1.0 - 0.4 * applause_score)
        )

        # TRANSITION: applause on room mic (> 0.55 flatness) OR stage silence
        s_transition = (
            3.6 * applause_score * (1.0 - 0.70 * drums_act) * (1.0 - 0.65 * spoken_score)
            + 2.8 * silence_score * (1.0 - 0.4 * applause_score)
        )

        logits = {
            ServicePhase.SPOKEN_WORD: s_spoken,
            ServicePhase.PRAYER_UNDERSCORE: s_prayer,
            ServicePhase.WORSHIP_LOW: s_worship_low,
            ServicePhase.WORSHIP_HIGH: s_worship_high,
            ServicePhase.TRANSITION: s_transition,
        }
        return _softmax(logits, beta=2.0)

    def _compute_service_plan_probs(self, service_step_idx: int) -> dict[ServicePhase, float] | None:
        if not self.plan.order_of_service:
            return None
        if not (0 <= service_step_idx < len(self.plan.order_of_service)):
            return None
        expected = self.plan.order_of_service[service_step_idx].expected_phase
        return {p: (0.60 if p == expected else 0.10) for p in ALL_PHASES}

    @staticmethod
    def _normalize_clap_probs(
        clap_probs: dict[ServicePhase, float] | dict[str, float] | None,
    ) -> dict[ServicePhase, float] | None:
        if not clap_probs:
            return None
        out: dict[ServicePhase, float] = {}
        for phase in ALL_PHASES:
            val = clap_probs.get(phase)  # type: ignore[arg-type]
            if val is None:
                val = clap_probs.get(phase.value, 0.0)  # type: ignore[arg-type]
            out[phase] = max(0.0, float(val))
        total = sum(out.values())
        if total <= 0.0:
            return None
        return {p: v / total for p, v in out.items()}

    def fuse_probabilities(
        self,
        p_telemetry: dict[ServicePhase, float],
        p_clap: dict[ServicePhase, float] | None,
        p_plan: dict[ServicePhase, float] | None,
    ) -> dict[ServicePhase, float]:
        """Fuse CLAP (0.40) + Telemetry (0.40) + ServicePlan (0.20) with graceful re-normalization."""
        if p_clap is not None and p_plan is not None:
            w_clap, w_tel, w_plan = 0.40, 0.40, 0.20
        elif p_clap is not None and p_plan is None:
            w_clap, w_tel, w_plan = 0.50, 0.50, 0.0
        elif p_clap is None and p_plan is not None:
            # Keep total acoustic weight at 0.80 so live audio can override a stale service plan step
            w_clap, w_tel, w_plan = 0.0, 0.80, 0.20
        else:
            w_clap, w_tel, w_plan = 0.0, 1.0, 0.0

        fused: dict[ServicePhase, float] = {}
        for phase in ALL_PHASES:
            score = w_tel * p_telemetry.get(phase, 0.0)
            if p_clap is not None:
                score += w_clap * p_clap.get(phase, 0.0)
            if p_plan is not None:
                score += w_plan * p_plan.get(phase, 0.0)
            fused[phase] = score

        total = sum(fused.values())
        if total > 0.0:
            fused = {p: v / total for p, v in fused.items()}
        return fused

    def update(
        self,
        metrics: AudioFrameMetrics,
        vad_probs: dict[str, float],
        sung_vocals: dict[str, bool],
        clap_probs: dict[ServicePhase, float] | None,
        service_step_idx: int = 0,
        use_service_plan: bool = True,
        now_s: float | None = None,
    ) -> SceneDecision:
        """Evaluate one Tier-2 decision window and update phase hysteresis state."""
        t = time.monotonic() if now_s is None else float(now_s)

        agg_metrics, agg_vad, agg_sung = self._window_aggregated_inputs(metrics, vad_probs, sung_vocals)
        p_telemetry = self.compute_telemetry_probs(agg_metrics, agg_vad, agg_sung)
        p_clap = self._normalize_clap_probs(clap_probs)
        p_plan = self._compute_service_plan_probs(service_step_idx) if use_service_plan else None
        scores = self.fuse_probabilities(p_telemetry, p_clap, p_plan)

        # Check active manual phase override
        if self._override_phase is not None:
            if t < self._override_until_s:
                return SceneDecision(
                    active_phase=self.active_phase,
                    last_macro_phase=self.last_macro_phase,
                    confidence=1.0,
                    hysteresis_progress=1.0,
                    transitioned=False,
                    reason=f"manual_override:{self.active_phase.value}",
                    scores=scores,
                )
            self._override_phase = None

        top_phase = max(ALL_PHASES, key=lambda p: scores[p])
        top_conf = float(scores[top_phase])

        if top_phase == self.active_phase:
            self._candidate_phase = None
            self._candidate_count = 0
            return SceneDecision(
                active_phase=self.active_phase,
                last_macro_phase=self.last_macro_phase,
                confidence=top_conf,
                hysteresis_progress=1.0,
                transitioned=False,
                reason=f"steady:{self.active_phase.value} (conf={top_conf:.2f})",
                scores=scores,
            )

        if top_conf < self.confidence_threshold:
            self._candidate_phase = None
            self._candidate_count = 0
            return SceneDecision(
                active_phase=self.active_phase,
                last_macro_phase=self.last_macro_phase,
                confidence=float(scores[self.active_phase]),
                hysteresis_progress=0.0,
                transitioned=False,
                reason=(
                    f"hold:{self.active_phase.value} "
                    f"(top={top_phase.value} conf={top_conf:.2f} < {self.confidence_threshold:.2f})"
                ),
                scores=scores,
            )

        # Candidate phase meets confidence threshold -> advance hysteresis counter
        if top_phase == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = top_phase
            self._candidate_count = 1

        progress = min(1.0, self._candidate_count / float(self.hysteresis_windows))

        if self._candidate_count >= self.hysteresis_windows:
            prev_phase = self.active_phase
            self.active_phase = top_phase
            # TRANSITION explicitly preserves prior last_macro_phase
            if top_phase in MACRO_PHASES:
                self.last_macro_phase = top_phase
            self._candidate_phase = None
            self._candidate_count = 0
            return SceneDecision(
                active_phase=self.active_phase,
                last_macro_phase=self.last_macro_phase,
                confidence=top_conf,
                hysteresis_progress=1.0,
                transitioned=True,
                reason=f"transition:{prev_phase.value}->{top_phase.value} (conf={top_conf:.2f})",
                scores=scores,
            )

        return SceneDecision(
            active_phase=self.active_phase,
            last_macro_phase=self.last_macro_phase,
            confidence=top_conf,
            hysteresis_progress=progress,
            transitioned=False,
            reason=(
                f"hysteresis:{top_phase.value} "
                f"({self._candidate_count}/{self.hysteresis_windows}, conf={top_conf:.2f})"
            ),
            scores=scores,
        )
