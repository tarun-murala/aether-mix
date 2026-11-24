"""Unit tests for engine/core/rules_manager.py and engine/core/llm_supervisor.py (SPEC v5.1)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from engine.core.llm_supervisor import LLMSupervisor
from engine.core.protocol import (
    OLLAMA_RESPONSE_JSON_SCHEMA,
    ConditionPredicate,
    MixingRule,
    MixingRules,
    ProposedPatch,
    RuleCondition,
    RuleExpiry,
    RuleParameter,
    RulePhase,
    RuleScope,
    ServicePhase,
    ServicePlan,
)
from engine.core.rules_manager import RulesManager

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"


@pytest.fixture
def plan() -> ServicePlan:
    return ServicePlan.model_validate_json((SCHEMAS_DIR / "service_plan.json").read_text(encoding="utf-8"))


@pytest.fixture
def rules() -> MixingRules:
    return MixingRules.model_validate_json((SCHEMAS_DIR / "mixing_rules.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# RulesManager Tests
# ---------------------------------------------------------------------------


def test_auto_apply_vs_preview_gate_vol_reverb_multichannel_and_conditional(
    plan: ServicePlan, rules: MixingRules
) -> None:
    mgr = RulesManager(plan, rules)

    # 1. Single-channel vol_db_offset within [-3.0, +3.0] -> auto-applied ephemeral
    cards_vol_ok = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Bump lead vocal +2.5 dB",
                scope=RuleScope.EPHEMERAL_SESSION,
                phase=RulePhase.WORSHIP_HIGH,
                channel_id="ch2_lead_vox",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=2.5,
            )
        ]
    )
    assert len(cards_vol_ok) == 1
    assert cards_vol_ok[0].applied_in_session is True
    assert cards_vol_ok[0].requires_preview is False
    assert cards_vol_ok[0].expires_on == RuleExpiry.SEGMENT_CHANGE
    cont, cond, mute = mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch2_lead_vox", "vol")
    assert cont == pytest.approx(2.5)
    assert cond == 0.0 and mute is False

    # 2. Single-channel reverb_db_offset within [-4.0, +4.0] -> auto-applied ephemeral
    cards_rev_ok = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Less reverb on lead vocal",
                scope=RuleScope.EPHEMERAL_SESSION,
                phase=RulePhase.WORSHIP_HIGH,
                channel_id="ch2_lead_vox",
                parameter=RuleParameter.REVERB_DB_OFFSET,
                value=-4.0,
            )
        ]
    )
    assert cards_rev_ok[0].applied_in_session is True
    assert cards_rev_ok[0].requires_preview is False
    cont_r, _, _ = mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch2_lead_vox", "reverb")
    assert cont_r == pytest.approx(-4.0)

    # 3. Single-channel vol_db_offset > +3.0 dB -> PREVIEW REQUIRED, NOT applied yet
    cards_vol_big = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Big guitar cut -4.5 dB",
                scope=RuleScope.EPHEMERAL_SESSION,
                phase=RulePhase.WORSHIP_HIGH,
                channel_id="ch7_guitars",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=-4.5,
            )
        ]
    )
    assert cards_vol_big[0].requires_preview is True
    assert cards_vol_big[0].applied_in_session is False
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol")[0] == 0.0

    # 4. Single-channel reverb_db_offset > 4.0 dB -> PREVIEW REQUIRED
    cards_rev_big = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Big reverb cut -6.0 dB",
                scope=RuleScope.EPHEMERAL_SESSION,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch1_pastor",
                parameter=RuleParameter.REVERB_DB_OFFSET,
                value=-6.0,
            )
        ]
    )
    assert cards_rev_big[0].requires_preview is True
    assert cards_rev_big[0].applied_in_session is False

    # 5. Multi-channel batch (even within ±2 dB each!) -> PREVIEW REQUIRED
    cards_multi = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Lower keys",
                scope=RuleScope.EPHEMERAL_SESSION,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch5_keys",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=-2.0,
            ),
            ProposedPatch(
                description="Lower guitars",
                scope=RuleScope.EPHEMERAL_SESSION,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch7_guitars",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=-2.0,
            ),
        ]
    )
    assert all(c.requires_preview is True and c.applied_in_session is False for c in cards_multi)

    # 6. Conditional attenuate / conditional mute -> PREVIEW REQUIRED
    cards_cond = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Mute BV2 during pastor",
                scope=RuleScope.CANDIDATE_PERMANENT,
                phase=RulePhase.ANY,
                channel_id="ch4_bv2",
                parameter=RuleParameter.CONDITIONAL_MUTE,
                value=1.0,
                condition=RuleCondition(
                    when_channel="ch1_pastor", predicate=ConditionPredicate.SPEECH_OPEN
                ),
            )
        ]
    )
    assert cards_cond[0].requires_preview is True
    assert cards_cond[0].applied_in_session is False


def test_conditional_mute_forbidden_on_spoken_primary(plan: ServicePlan, rules: MixingRules) -> None:
    mgr = RulesManager(plan, rules)
    with pytest.raises(ValueError, match="spoken_primary"):
        mgr.ingest_proposed_patches(
            [
                ProposedPatch(
                    description="Bad mute pastor",
                    scope=RuleScope.EPHEMERAL_SESSION,
                    phase=RulePhase.ANY,
                    channel_id="ch1_pastor",
                    parameter=RuleParameter.CONDITIONAL_MUTE,
                    value=1.0,
                    condition=RuleCondition(when_channel="ch9_drums", predicate=ConditionPredicate.ACTIVE),
                )
            ]
        )


def test_separate_continuous_and_conditional_caps_and_conflict_resolution(
    plan: ServicePlan, rules: MixingRules
) -> None:
    mgr = RulesManager(plan, rules)

    # Stack two continuous rules on different phases (ANY + WORSHIP_HIGH) exceeding -12 dB cap
    mgr.add_permanent_rule(
        MixingRule(
            id="c1",
            description="Cut BV1 any",
            phase=RulePhase.ANY,
            channel_id="ch3_bv1",
            parameter=RuleParameter.VOL_DB_OFFSET,
            value=-8.0,
            created_at_ms=1000,
        )
    )
    mgr.add_permanent_rule(
        MixingRule(
            id="c2",
            description="Cut BV1 worship high",
            phase=RulePhase.WORSHIP_HIGH,
            channel_id="ch3_bv1",
            parameter=RuleParameter.VOL_DB_OFFSET,
            value=-8.0,
            created_at_ms=2000,
        )
    )

    # Note: mixing_rules.json already has rule_002: conditional_attenuate -24.0 dB on ch3_bv1 when ch1_pastor speech_open
    cont_db, cond_db, cond_mute = mgr.evaluate_channel(
        ServicePhase.WORSHIP_HIGH,
        "ch3_bv1",
        "vol",
        speech_open_map={"ch1_pastor": True},
    )
    # Continuous capped at -12.0 SEPARATE from conditional -24.0!
    assert cont_db == pytest.approx(-12.0)
    assert cond_db == pytest.approx(-24.0)
    assert cond_mute is False

    # Most-recent rule wins conflict on identical (phase, channel_id, parameter, condition)
    mgr.add_permanent_rule(
        MixingRule(
            id="c2_newer",
            description="Override BV1 worship high to +4 dB",
            phase=RulePhase.WORSHIP_HIGH,
            channel_id="ch3_bv1",
            parameter=RuleParameter.VOL_DB_OFFSET,
            value=4.0,
            created_at_ms=5000,
        )
    )
    cont_after, cond_after, _ = mgr.evaluate_channel(
        ServicePhase.WORSHIP_HIGH,
        "ch3_bv1",
        "vol",
        speech_open_map={"ch1_pastor": True},
    )
    # ANY (-8.0) + winning WORSHIP_HIGH (+4.0) = -4.0
    assert cont_after == pytest.approx(-4.0)
    assert cond_after == pytest.approx(-24.0)


def test_resolve_toggle_delete_expire_and_atomic_backup(
    plan: ServicePlan, rules: MixingRules, tmp_path: Path
) -> None:
    rules_path = tmp_path / "mixing_rules.json"
    backup_dir = tmp_path / "rule_backups"
    rules_path.write_text(json.dumps(rules.model_dump(mode="json")), encoding="utf-8")

    mgr = RulesManager(plan, rules, rules_path=rules_path, backup_dir=backup_dir)

    # Preview-gated patch -> apply_session -> promote -> atomic write + backup
    cards = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Cut guitars -5 dB",
                scope=RuleScope.CANDIDATE_PERMANENT,
                expires_on=RuleExpiry.SEGMENT_CHANGE,
                phase=RulePhase.WORSHIP_HIGH,
                channel_id="ch7_guitars",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=-5.0,
            )
        ]
    )
    rid = cards[0].rule_id
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol")[0] == 0.0

    # apply_session activates ephemerally without writing permanent file
    mgr.resolve_pending_rule(rid, "apply_session")
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol")[0] == pytest.approx(-5.0)
    assert len(list(backup_dir.glob("*.json"))) == 0

    # promote saves atomically and creates timestamped backup
    perm = mgr.resolve_pending_rule(rid, "promote")
    assert perm is not None and perm.scope == RuleScope.CANDIDATE_PERMANENT
    assert len(list(backup_dir.glob("mixing_rules_*.json"))) == 1
    reloaded = MixingRules.model_validate_json(rules_path.read_text(encoding="utf-8"))
    assert any(r.id == rid for r in reloaded.learned_rules)

    # toggle off and back on
    mgr.toggle_rule(rid, False)
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol")[0] == 0.0
    mgr.toggle_rule(rid, True)
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol")[0] == pytest.approx(-5.0)

    # delete removes permanent rule
    assert mgr.delete_rule(rid) is True
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_HIGH, "ch7_guitars", "vol")[0] == 0.0

    # Auto-applied ephemeral rule dismissed reverts immediately
    auto_card = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="Quick +2 dB keys",
                scope=RuleScope.EPHEMERAL_SESSION,
                expires_on=RuleExpiry.PHASE_CHANGE,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch5_keys",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=2.0,
            )
        ]
    )[0]
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol")[0] == pytest.approx(2.0)
    mgr.resolve_pending_rule(auto_card.rule_id, "dismiss")
    assert mgr.evaluate_channel(ServicePhase.WORSHIP_LOW, "ch5_keys", "vol")[0] == 0.0

    # Ephemeral expiry hierarchy: phase_change < segment_change < end_of_service
    c_phase = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="phase only",
                scope=RuleScope.EPHEMERAL_SESSION,
                expires_on=RuleExpiry.PHASE_CHANGE,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch5_keys",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=1.0,
            )
        ]
    )[0]
    c_seg = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="segment only",
                scope=RuleScope.EPHEMERAL_SESSION,
                expires_on=RuleExpiry.SEGMENT_CHANGE,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch7_guitars",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=1.0,
            )
        ]
    )[0]
    c_eos = mgr.ingest_proposed_patches(
        [
            ProposedPatch(
                description="service only",
                scope=RuleScope.EPHEMERAL_SESSION,
                expires_on=RuleExpiry.END_OF_SERVICE,
                phase=RulePhase.WORSHIP_LOW,
                channel_id="ch9_drums",
                parameter=RuleParameter.VOL_DB_OFFSET,
                value=1.0,
            )
        ]
    )[0]

    assert mgr.expire_ephemeral(RuleExpiry.PHASE_CHANGE) == [c_phase.rule_id]
    assert mgr.expire_ephemeral(RuleExpiry.SEGMENT_CHANGE) == [c_seg.rule_id]
    assert mgr.expire_ephemeral(RuleExpiry.END_OF_SERVICE) == [c_eos.rule_id]


# ---------------------------------------------------------------------------
# LLMSupervisor Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_supervisor_warmup_and_advisory_patch_stripping(plan: ServicePlan) -> None:
    captured_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append(body)
        if "format" not in body:
            return httpx.Response(200, json={"message": {"content": "OK"}})

        # Deliberately return a patch even on phase_transition to prove code strips it!
        llm_json = {
            "thought_summary": "Transitioning into high energy worship.",
            "reasoning": "Drums and vocals are active.",
            "confidence": 0.91,
            "proposed_patches": [
                {
                    "description": "Hallucinated automatic patch",
                    "scope": "ephemeral_session",
                    "phase": "WORSHIP_HIGH",
                    "channel_id": "ch5_keys",
                    "parameter": "vol_db_offset",
                    "value": -2.0,
                }
            ],
        }
        return httpx.Response(200, json={"message": {"content": json.dumps(llm_json)}})

    sup = LLMSupervisor(plan, transport=httpx.MockTransport(handler))
    try:
        assert await sup.warm_up() is True
        assert captured_requests[0]["keep_alive"] == "60m"
        assert captured_requests[0]["think"] is False

        # phase_transition MUST strip proposed_patches in code
        resp_phase = await sup.analyze_event("phase_transition", {"active_phase": "WORSHIP_HIGH"})
        assert captured_requests[1]["format"] == OLLAMA_RESPONSE_JSON_SCHEMA
        assert resp_phase.proposed_patches == []
        assert resp_phase.confidence == pytest.approx(0.91)

        # safety_event MUST also strip proposed_patches in code
        resp_safety = await sup.analyze_event("safety_event", {"event": "feedback_cut"})
        assert resp_safety.proposed_patches == []

        # operator_critique KEEPS valid patches
        resp_crit = await sup.analyze_event(
            "operator_critique", {"active_phase": "WORSHIP_HIGH"}, critique_text="Keys slightly loud"
        )
        assert len(resp_crit.proposed_patches) == 1
        assert resp_crit.proposed_patches[0].channel_id == "ch5_keys"
        assert resp_crit.proposed_patches[0].expires_on == RuleExpiry.SEGMENT_CHANGE
    finally:
        await sup.close()


@pytest.mark.asyncio
async def test_llm_supervisor_sanitizes_forbidden_patches_and_handles_failures(plan: ServicePlan) -> None:
    # 1. Critique containing forbidden spoken_primary mute + unknown channel + 1 valid patch
    def sanitize_handler(request: httpx.Request) -> httpx.Response:
        llm_json = {
            "thought_summary": "Adjusting mix per critique.",
            "reasoning": "Applying requested balance changes.",
            "confidence": 0.85,
            "proposed_patches": [
                {
                    "description": "Forbidden pastor mute",
                    "scope": "ephemeral_session",
                    "phase": "ANY",
                    "channel_id": "ch1_pastor",
                    "parameter": "conditional_mute",
                    "value": 1.0,
                },
                {
                    "description": "Unknown channel",
                    "scope": "ephemeral_session",
                    "phase": "ANY",
                    "channel_id": "ch99_ghost",
                    "parameter": "vol_db_offset",
                    "value": -2.0,
                },
                {
                    "description": "Valid keys cut",
                    "scope": "ephemeral_session",
                    "phase": "PRAYER_UNDERSCORE",
                    "channel_id": "ch5_keys",
                    "parameter": "vol_db_offset",
                    "value": -2.0,
                },
            ],
        }
        return httpx.Response(200, json={"message": {"content": json.dumps(llm_json)}})

    sup = LLMSupervisor(plan, transport=httpx.MockTransport(sanitize_handler))
    try:
        resp = await sup.analyze_event("operator_critique", {}, critique_text="Lower keys")
        assert len(resp.proposed_patches) == 1
        assert resp.proposed_patches[0].channel_id == "ch5_keys"
    finally:
        await sup.close()

    # 2. Timeout fallback
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    sup_timeout = LLMSupervisor(plan, timeout_s=0.1, transport=httpx.MockTransport(timeout_handler))
    try:
        assert await sup_timeout.warm_up() is False
        fallback = await sup_timeout.analyze_event("operator_critique", {}, critique_text="More vocal")
        assert fallback.confidence == 0.0
        assert fallback.proposed_patches == []
        assert sup_timeout.ollama_ok is False
    finally:
        await sup_timeout.close()

    # 3. Malformed / out-of-bounds JSON fallback
    def bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "{not valid json"}})

    sup_bad = LLMSupervisor(plan, transport=httpx.MockTransport(bad_json_handler))
    try:
        fallback = await sup_bad.analyze_event("operator_critique", {}, critique_text="Test")
        assert fallback.confidence == 0.0
        assert fallback.proposed_patches == []
        assert sup_bad.ollama_ok is False
    finally:
        await sup_bad.close()
