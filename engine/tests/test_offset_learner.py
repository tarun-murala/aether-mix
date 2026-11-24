"""Unit tests for engine/core/offset_learner.py (SPEC v5.1 §5.3.2 & §9 Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from engine.core.offset_learner import OffsetLearner, build_context_vector
from engine.core.protocol import ServicePhase, ServicePlan

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"


@pytest.fixture
def plan() -> ServicePlan:
    return ServicePlan.model_validate_json((SCHEMAS_DIR / "service_plan.json").read_text(encoding="utf-8"))


def test_initial_predictions_are_zero(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0, decay_gamma=0.98)
    phi = build_context_vector(-10.0, 0.5, True, 0.0)
    assert learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi) == 0.0
    assert learner.predict(ServicePhase.WORSHIP_HIGH, "ch2_lead_vox", "reverb", phi) == 0.0
    assert learner.sample_count() == 0


def test_exact_one_sample_ridge_update_and_per_control_isolation(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0)
    phi = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)  # ||phi||^2 = 2.0

    # Closed-form check for 1 sample with alpha=2.0, y=+3.0:
    # A = 10*I + 2*phi*phi^T, b = 2*3*phi = 6*phi
    # By Sherman-Morrison, w^T phi = (alpha * y * ||phi||^2) / (lam + alpha * ||phi||^2)
    # = (2 * 3 * 2) / (10 + 4) = 12 / 14
    sample = learner.record_human_correction(
        ServicePhase.WORSHIP_LOW, "ch2_lead_vox", "vol", phi, target_offset_db=3.0, freeze=False
    )
    assert sample["alpha"] == 2.0
    assert sample["target_offset_db"] == 3.0
    assert learner.sample_count(ServicePhase.WORSHIP_LOW, "ch2_lead_vox", "vol") == 1

    pred_vol = learner.predict(ServicePhase.WORSHIP_LOW, "ch2_lead_vox", "vol", phi)
    assert pred_vol == pytest.approx(12.0 / 14.0, abs=1e-6)

    # Reverb control on same channel & same phase must remain untouched (0.0)
    assert learner.predict(ServicePhase.WORSHIP_LOW, "ch2_lead_vox", "reverb", phi) == 0.0
    # Different phase on same channel & control must remain untouched (0.0)
    assert learner.predict(ServicePhase.WORSHIP_HIGH, "ch2_lead_vox", "vol", phi) == 0.0
    # Different channel must remain untouched (0.0)
    assert learner.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi) == 0.0


def test_transition_uses_fallback_macro_phase(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0)
    phi = np.array([0.1, 0.2, 1.0, 0.0, 1.0], dtype=np.float64)
    learner.record_human_correction(ServicePhase.WORSHIP_HIGH, "ch5_keys", "vol", phi, -2.5)

    pred_wh = learner.predict(ServicePhase.WORSHIP_HIGH, "ch5_keys", "vol", phi)
    pred_tr = learner.predict(
        ServicePhase.TRANSITION, "ch5_keys", "vol", phi, fallback_macro_phase=ServicePhase.WORSHIP_HIGH
    )
    assert pred_tr == pytest.approx(pred_wh, abs=1e-9)
    assert pred_tr < 0.0


def test_output_and_target_clipped_to_plus_minus_4db(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=0.1)
    phi = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for _ in range(10):
        s = learner.record_human_correction(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi, +9.0)
        assert s["target_offset_db"] == 4.0
    pred = learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi)
    assert -4.0 <= pred <= 4.0
    assert pred == pytest.approx(4.0, abs=0.05)


def test_thumbs_up_and_thumbs_down_exact_weights(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0)
    phi = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # ||phi||^2 = 1.0

    up_samples = learner.record_thumbs_up(
        ServicePhase.PRAYER_UNDERSCORE,
        [
            {"channel_id": "ch5_keys", "control": "vol", "phi": phi, "target_offset_db": -2.0},
            {"channel_id": "ch1_pastor", "control": "reverb", "phi": phi, "target_offset_db": +2.0},
        ],
    )
    assert len(up_samples) == 2
    assert all(s["alpha"] == 1.0 for s in up_samples)
    # w^T phi = (1.0 * -2.0 * 1) / (10 + 1) = -2/11
    assert learner.predict(ServicePhase.PRAYER_UNDERSCORE, "ch5_keys", "vol", phi) == pytest.approx(
        -2.0 / 11.0, abs=1e-6
    )
    assert learner.predict(ServicePhase.PRAYER_UNDERSCORE, "ch1_pastor", "reverb", phi) == pytest.approx(
        +2.0 / 11.0, abs=1e-6
    )

    down_samples = learner.record_thumbs_down(
        ServicePhase.WORSHIP_HIGH,
        {("ch7_guitars", "vol"): (phi, -3.0)},
    )
    assert len(down_samples) == 1
    assert down_samples[0]["alpha"] == 1.5
    # w^T phi = (1.5 * -3.0 * 1) / (10 + 1.5) = -4.5 / 11.5
    assert learner.predict(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol", phi) == pytest.approx(
        -4.5 / 11.5, abs=1e-6
    )


def test_freeze_queues_without_mutating_and_commit_applies(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0)
    phi = np.array([0.2, -0.1, 1.0, 0.0, 1.0], dtype=np.float64)

    s1 = learner.record_human_correction(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi, +2.5, freeze=True)
    s2 = learner.record_thumbs_up(
        ServicePhase.SPOKEN_WORD,
        [("ch1_pastor", "reverb", phi, -1.5)],
        freeze=True,
    )
    s3 = learner.record_thumbs_down(
        ServicePhase.WORSHIP_LOW,
        [("ch5_keys", "vol", phi, -2.0)],
        freeze=True,
    )

    # Zero weight mutations while frozen
    assert learner.sample_count() == 0
    assert learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi) == 0.0
    assert learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "reverb", phi) == 0.0
    assert learner.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi) == 0.0

    # Regression #10: every queued sample dict carries a unique non-empty UUID "id"
    all_queued = [s1, *s2, *s3]
    ids = [s["id"] for s in all_queued]
    assert all(isinstance(sid, str) and len(sid) > 0 for sid in ids)
    assert len(set(ids)) == len(all_queued)

    # Commit queued samples -> weights now match non-frozen learning
    committed = learner.commit_queued_samples(all_queued)
    assert committed == 3
    assert learner.sample_count() == 3
    assert learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi) > 0.0
    assert learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "reverb", phi) < 0.0
    assert learner.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi) < 0.0


def test_session_decay_shrinks_weights_and_preserves_regularization(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0, decay_gamma=0.98)
    phi = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for _ in range(5):
        learner.record_human_correction(ServicePhase.WORSHIP_HIGH, "ch9_drums", "vol", phi, -3.0)

    before = abs(learner.predict(ServicePhase.WORSHIP_HIGH, "ch9_drums", "vol", phi))
    assert before > 1.0

    for _ in range(50):
        learner.apply_session_decay()

    after = abs(learner.predict(ServicePhase.WORSHIP_HIGH, "ch9_drums", "vol", phi))
    assert 0.0 < after < before


def test_reset_learning_and_export_load_state_roundtrip(plan: ServicePlan) -> None:
    learner = OffsetLearner(plan, lam=10.0, decay_gamma=0.98)
    phi = np.array([0.5, 0.2, 1.0, -0.3, 1.0], dtype=np.float64)

    learner.record_human_correction(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi, +2.0)
    learner.record_human_correction(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi, -2.0)

    # Export JSON roundtrip
    raw_json = json.dumps(learner.export_state())
    restored = OffsetLearner(plan, lam=5.0)
    restored.load_state(json.loads(raw_json))
    assert restored.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi) == pytest.approx(
        learner.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi), abs=1e-9
    )
    assert restored.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi) == pytest.approx(
        learner.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi), abs=1e-9
    )

    # Scoped reset: only SPOKEN_WORD
    n_reset = restored.reset_learning(phase=ServicePhase.SPOKEN_WORD)
    assert n_reset > 0
    assert restored.predict(ServicePhase.SPOKEN_WORD, "ch1_pastor", "vol", phi) == 0.0
    assert restored.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi) < 0.0

    # Full reset
    restored.reset_learning()
    assert restored.predict(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol", phi) == 0.0
    assert restored.sample_count() == 0
