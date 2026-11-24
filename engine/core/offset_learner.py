"""Drift-free per-(phase, channel, control) Ridge Offset Learner (SPEC v5.1 §5.3.2).

Maintains independent Ridge regression models (A_{p,i,c}, b_{p,i,c}) predicting
target dB offsets y in [-4.0, +4.0] from the 5D context vector phi_i(S_t).
All three operator feedback channels (manual fader/reverb moves, thumbs-up,
and thumbs-down) apply exact 1-sample updates — or return queued sample dicts
without mutating weights when freeze=True.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.core.protocol import MACRO_PHASES, ServicePhase, ServicePlan

FEATURE_DIM = 5
OFFSET_MIN_DB = -4.0
OFFSET_MAX_DB = 4.0

ALPHA_HUMAN_CORRECTION = 2.0
ALPHA_THUMBS_UP = 1.0
ALPHA_THUMBS_DOWN = 1.5


def build_context_vector(
    norm_vband_db: float,
    vocal_to_band_ratio_norm: float,
    speech_or_vocal_active: float | bool,
    master_lufs_error_norm: float,
) -> np.ndarray:
    """Build the canonical 5D context vector phi_i(S_t)."""
    return np.array(
        [
            float(np.clip(norm_vband_db / 24.0, -3.0, 2.0)),
            float(np.clip(vocal_to_band_ratio_norm, -3.0, 3.0)),
            1.0 if bool(speech_or_vocal_active) else 0.0,
            float(np.clip(master_lufs_error_norm, -2.0, 2.0)),
            1.0,
        ],
        dtype=np.float64,
    )


def _to_phi(phi: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    if arr.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected {FEATURE_DIM}D context vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Context vector phi contains non-finite values")
    return arr


def _resolve_macro_phase(
    phase: ServicePhase | str,
    fallback_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
) -> ServicePhase:
    p = phase if isinstance(phase, ServicePhase) else ServicePhase(phase)
    if p == ServicePhase.TRANSITION:
        if fallback_macro_phase == ServicePhase.TRANSITION:
            return ServicePhase.SPOKEN_WORD
        return fallback_macro_phase
    return p


@dataclass
class _RidgeModel:
    A: np.ndarray
    b: np.ndarray
    w: np.ndarray
    sample_count: int = 0

    @classmethod
    def fresh(cls, lam: float) -> _RidgeModel:
        return cls(
            A=lam * np.eye(FEATURE_DIM, dtype=np.float64),
            b=np.zeros(FEATURE_DIM, dtype=np.float64),
            w=np.zeros(FEATURE_DIM, dtype=np.float64),
            sample_count=0,
        )

    def update(self, phi: np.ndarray, y: float, alpha: float) -> None:
        self.A += alpha * np.outer(phi, phi)
        self.b += alpha * y * phi
        self.sample_count += 1
        self._recompute_weights()

    def decay(self, gamma: float, lam: float) -> None:
        eye = lam * np.eye(FEATURE_DIM, dtype=np.float64)
        self.A = eye + gamma * (self.A - eye)
        self.b = gamma * self.b
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        self.w = np.linalg.solve(self.A, self.b)


class OffsetLearner:
    """Per-(phase, channel_id, control) Ridge regression offset learner."""

    def __init__(
        self,
        plan: ServicePlan,
        lam: float = 10.0,
        decay_gamma: float = 0.98,
    ) -> None:
        if lam <= 0.0:
            raise ValueError("Ridge regularization lam must be > 0")
        if not (0.0 < decay_gamma <= 1.0):
            raise ValueError("decay_gamma must be in (0, 1]")

        self.plan = plan
        self.lam = float(lam)
        self.decay_gamma = float(decay_gamma)
        self.version: int = 0

        self._channel_ids: set[str] = set(plan.channel_ids)
        self._models: dict[tuple[ServicePhase, str, str], _RidgeModel] = {}

        for phase in MACRO_PHASES:
            for ch in plan.channels:
                self._models[(phase, ch.id, "vol")] = _RidgeModel.fresh(self.lam)
                if ch.reverb_cc is not None:
                    self._models[(phase, ch.id, "reverb")] = _RidgeModel.fresh(self.lam)

    def _get_or_create_model(self, phase: ServicePhase, channel_id: str, control: str) -> _RidgeModel:
        if channel_id not in self._channel_ids:
            raise KeyError(f"Unknown channel_id: {channel_id}")
        if control not in ("vol", "reverb"):
            raise ValueError(f"Unsupported control for OffsetLearner: {control}")
        key = (phase, channel_id, control)
        model = self._models.get(key)
        if model is None:
            model = _RidgeModel.fresh(self.lam)
            self._models[key] = model
        return model

    def predict(
        self,
        phase: ServicePhase | str,
        channel_id: str,
        control: str,
        phi: Sequence[float] | np.ndarray,
        fallback_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    ) -> float:
        """Predict clipped dB offset in [-4.0, +4.0] for (phase, channel_id, control)."""
        macro_phase = _resolve_macro_phase(phase, fallback_macro_phase)
        model = self._get_or_create_model(macro_phase, channel_id, control)
        phi_vec = _to_phi(phi)
        raw = float(np.dot(model.w, phi_vec))
        return float(np.clip(raw, OFFSET_MIN_DB, OFFSET_MAX_DB))

    def get_weights(
        self,
        phase: ServicePhase | str,
        channel_id: str,
        control: str,
        fallback_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    ) -> np.ndarray:
        """Return a copy of the learned 5D weight vector w."""
        macro_phase = _resolve_macro_phase(phase, fallback_macro_phase)
        return self._get_or_create_model(macro_phase, channel_id, control).w.copy()

    def sample_count(
        self,
        phase: ServicePhase | str | None = None,
        channel_id: str | None = None,
        control: str | None = None,
    ) -> int:
        """Return total sample count across matching models."""
        target_phase = None
        if phase is not None:
            target_phase = phase if isinstance(phase, ServicePhase) else ServicePhase(phase)
        total = 0
        for (p, ch_id, ctrl), model in self._models.items():
            if target_phase is not None and p != target_phase:
                continue
            if channel_id is not None and ch_id != channel_id:
                continue
            if control is not None and ctrl != control:
                continue
            total += model.sample_count
        return total

    def _apply_or_queue_sample(
        self,
        phase: ServicePhase,
        channel_id: str,
        control: str,
        phi_vec: np.ndarray,
        target_offset_db: float,
        alpha: float,
        source: str,
        freeze: bool,
    ) -> dict[str, Any]:
        y = float(np.clip(target_offset_db, OFFSET_MIN_DB, OFFSET_MAX_DB))
        # Validate channel/control upfront even when frozen
        model = self._get_or_create_model(phase, channel_id, control)
        sample: dict[str, Any] = {
            "id": f"rev_{uuid.uuid4().hex[:12]}",
            "source": source,
            "phase": phase.value,
            "channel_id": channel_id,
            "control": control,
            "phi": [float(x) for x in phi_vec],
            "target_offset_db": y,
            "alpha": float(alpha),
            "ts_ms": int(time.time() * 1000),
        }
        if not freeze:
            model.update(phi_vec, y, alpha)
            self.version += 1
        return sample

    def record_human_correction(
        self,
        phase: ServicePhase | str,
        channel_id: str,
        control: str,
        phi: Sequence[float] | np.ndarray,
        target_offset_db: float,
        freeze: bool = False,
        fallback_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    ) -> dict[str, Any]:
        """Apply exact 1-sample update (alpha=2.0) for a manual fader/reverb move."""
        macro_phase = _resolve_macro_phase(phase, fallback_macro_phase)
        phi_vec = _to_phi(phi)
        return self._apply_or_queue_sample(
            phase=macro_phase,
            channel_id=channel_id,
            control=control,
            phi_vec=phi_vec,
            target_offset_db=target_offset_db,
            alpha=ALPHA_HUMAN_CORRECTION,
            source="human_correction",
            freeze=freeze,
        )

    def _normalize_batch_targets(
        self,
        targets: Sequence[Any] | Mapping[tuple[str, str], tuple[Sequence[float] | np.ndarray, float]],
    ) -> list[tuple[str, str, np.ndarray, float]]:
        normalized: list[tuple[str, str, np.ndarray, float]] = []
        if isinstance(targets, Mapping):
            for (ch_id, ctrl), (phi, offset_db) in targets.items():
                normalized.append((ch_id, ctrl, _to_phi(phi), float(offset_db)))
            return normalized

        for item in targets:
            if isinstance(item, Mapping):
                ch_id = str(item["channel_id"])
                ctrl = str(item["control"])
                phi_vec = _to_phi(item["phi"])
                raw_y = item.get("target_offset_db", item.get("offset_db"))
                if raw_y is None:
                    raise KeyError("Target dict missing 'target_offset_db' (or 'offset_db')")
                normalized.append((ch_id, ctrl, phi_vec, float(raw_y)))
            elif isinstance(item, (tuple, list)) and len(item) == 4:
                ch_id, ctrl, phi, offset_db = item
                normalized.append((str(ch_id), str(ctrl), _to_phi(phi), float(offset_db)))
            else:
                raise TypeError(
                    "Each target must be a dict with channel_id/control/phi/target_offset_db "
                    "or a 4-tuple (channel_id, control, phi, target_offset_db)"
                )
        return normalized

    def record_thumbs_up(
        self,
        phase: ServicePhase | str,
        armed_targets: Sequence[Any] | Mapping[tuple[str, str], tuple[Sequence[float] | np.ndarray, float]],
        freeze: bool = False,
        fallback_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    ) -> list[dict[str, Any]]:
        """Apply exact 1-sample reinforcement (alpha=1.0) per armed (channel, control)."""
        macro_phase = _resolve_macro_phase(phase, fallback_macro_phase)
        samples: list[dict[str, Any]] = []
        for ch_id, ctrl, phi_vec, offset_db in self._normalize_batch_targets(armed_targets):
            samples.append(
                self._apply_or_queue_sample(
                    phase=macro_phase,
                    channel_id=ch_id,
                    control=ctrl,
                    phi_vec=phi_vec,
                    target_offset_db=offset_db,
                    alpha=ALPHA_THUMBS_UP,
                    source="thumbs_up",
                    freeze=freeze,
                )
            )
        return samples

    def record_thumbs_down(
        self,
        phase: ServicePhase | str,
        rolled_back_targets: Sequence[Any]
        | Mapping[tuple[str, str], tuple[Sequence[float] | np.ndarray, float]],
        freeze: bool = False,
        fallback_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD,
    ) -> list[dict[str, Any]]:
        """Apply exact 1-sample rollback target update (alpha=1.5) per rolled-back control."""
        macro_phase = _resolve_macro_phase(phase, fallback_macro_phase)
        samples: list[dict[str, Any]] = []
        for ch_id, ctrl, phi_vec, offset_db in self._normalize_batch_targets(rolled_back_targets):
            samples.append(
                self._apply_or_queue_sample(
                    phase=macro_phase,
                    channel_id=ch_id,
                    control=ctrl,
                    phi_vec=phi_vec,
                    target_offset_db=offset_db,
                    alpha=ALPHA_THUMBS_DOWN,
                    source="thumbs_down",
                    freeze=freeze,
                )
            )
        return samples

    def commit_queued_samples(self, samples: Sequence[Mapping[str, Any]]) -> int:
        """Commit previously frozen/queued samples into the Ridge models."""
        committed = 0
        for sample in samples:
            phase = _resolve_macro_phase(sample["phase"])
            ch_id = str(sample["channel_id"])
            ctrl = str(sample["control"])
            phi_vec = _to_phi(sample["phi"])
            y = float(np.clip(float(sample["target_offset_db"]), OFFSET_MIN_DB, OFFSET_MAX_DB))
            alpha = float(sample.get("alpha", ALPHA_HUMAN_CORRECTION))
            model = self._get_or_create_model(phase, ch_id, ctrl)
            model.update(phi_vec, y, alpha)
            committed += 1
        if committed > 0:
            self.version += 1
        return committed

    def apply_session_decay(self, gamma: float | None = None) -> None:
        """Apply inter-service forgetting factor gamma (preserving Ridge lam*I prior)."""
        g = self.decay_gamma if gamma is None else float(gamma)
        if not (0.0 < g <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        for model in self._models.values():
            model.decay(g, self.lam)
        self.version += 1

    def reset_learning(
        self,
        phase: ServicePhase | str | None = None,
        channel_id: str | None = None,
        control: str | None = None,
    ) -> int:
        """Reset learned weights back to zero prior for matching (phase, channel_id, control)."""
        target_phase: ServicePhase | None = None
        if phase is not None:
            target_phase = phase if isinstance(phase, ServicePhase) else ServicePhase(phase)

        reset_count = 0
        for key in list(self._models.keys()):
            p, ch_id, ctrl = key
            if target_phase is not None and p != target_phase:
                continue
            if channel_id is not None and ch_id != channel_id:
                continue
            if control is not None and ctrl != control:
                continue
            self._models[key] = _RidgeModel.fresh(self.lam)
            reset_count += 1
        if reset_count > 0:
            self.version += 1
        return reset_count

    def export_state(self) -> dict[str, Any]:
        """Serialize full learner state to a JSON-compatible dictionary."""
        models_out: dict[str, dict[str, Any]] = {}
        for (phase, ch_id, ctrl), model in self._models.items():
            key_str = f"{phase.value}|{ch_id}|{ctrl}"
            models_out[key_str] = {
                "A": model.A.tolist(),
                "b": model.b.tolist(),
                "sample_count": model.sample_count,
            }
        return {
            "schema_version": 5,
            "version": self.version,
            "lam": self.lam,
            "decay_gamma": self.decay_gamma,
            "models": models_out,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore learner state from an exported snapshot dictionary."""
        self.lam = float(state.get("lam", self.lam))
        self.decay_gamma = float(state.get("decay_gamma", self.decay_gamma))
        self.version = int(state.get("version", self.version))

        raw_models = state.get("models", {})
        if not isinstance(raw_models, Mapping):
            raise TypeError("Invalid learner state: 'models' must be a mapping")

        for key_str, payload in raw_models.items():
            parts = str(key_str).split("|")
            if len(parts) != 3:
                continue
            phase_str, ch_id, ctrl = parts
            if ch_id not in self._channel_ids or ctrl not in ("vol", "reverb"):
                continue
            phase = ServicePhase(phase_str)
            if phase == ServicePhase.TRANSITION:
                continue
            A = np.asarray(payload["A"], dtype=np.float64)
            b = np.asarray(payload["b"], dtype=np.float64)
            if A.shape != (FEATURE_DIM, FEATURE_DIM) or b.shape != (FEATURE_DIM,):
                raise ValueError(f"Invalid matrix dimensions for {key_str}")
            w = np.linalg.solve(A, b)
            self._models[(phase, ch_id, ctrl)] = _RidgeModel(
                A=A,
                b=b,
                w=w,
                sample_count=int(payload.get("sample_count", 0)),
            )
