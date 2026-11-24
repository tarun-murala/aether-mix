"""Validated Mixing Rules Manager, ±3dB/±4dB Preview Gate & Atomic Backups (SPEC v5.1).

Handles:
- Separate continuous stack cap [-12.0, +6.0] dB vs conditional attenuate [-36.0, 0.0] dB
  and conditional_mute (strictly forbidden on spoken_primary channels).
- Auto-apply gate: only single-channel non-mute vol_db_offset within [-3.0, +3.0] dB or
  reverb_db_offset within [-4.0, +4.0] dB auto-applies as an ephemeral session rule;
  larger, multi-channel, duck_depth, or conditional patches require explicit preview approval.
- Conflict resolution: most-recent rule wins on identical (phase, channel_id, parameter, condition).
- Atomic persistence (tmp + os.replace) with timestamped backups in backup_dir.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from engine.core.protocol import (
    ChannelRole,
    ConditionPredicate,
    MixingRule,
    MixingRules,
    PendingRuleCard,
    ProposedPatch,
    RuleCondition,
    RuleExpiry,
    RuleParameter,
    RulePhase,
    RuleScope,
    ServicePhase,
    ServicePlan,
    cross_validate_plan_and_rules,
)

AUTO_APPLY_VOL_MIN_DB = -3.0
AUTO_APPLY_VOL_MAX_DB = 3.0
AUTO_APPLY_REVERB_MIN_DB = -4.0
AUTO_APPLY_REVERB_MAX_DB = 4.0

CONTINUOUS_STACK_MIN_DB = -12.0
CONTINUOUS_STACK_MAX_DB = 6.0
CONDITIONAL_ATTENUATE_MIN_DB = -36.0
CONDITIONAL_ATTENUATE_MAX_DB = 0.0


class RulesManager:
    """Manages permanent, ephemeral session, and pending preview mixing rules."""

    def __init__(
        self,
        plan: ServicePlan,
        rules: MixingRules,
        rules_path: Path | None = None,
        backup_dir: Path | None = None,
    ) -> None:
        cross_validate_plan_and_rules(plan, rules)
        self.plan = plan
        self.rules = rules
        self.rules_path = Path(rules_path) if rules_path is not None else None
        self.backup_dir = Path(backup_dir) if backup_dir is not None else None

        self._channel_map = {ch.id: ch for ch in plan.channels}
        self.ephemeral_rules: list[MixingRule] = []
        self.pending_rules: dict[str, PendingRuleCard] = {}
        self._order_counter: int = 0
        self._rule_order: dict[str, int] = {}

        for rule in self.rules.learned_rules:
            self._assign_order(rule.id)

    def _assign_order(self, rule_id: str) -> int:
        self._order_counter += 1
        self._rule_order[rule_id] = self._order_counter
        return self._order_counter

    def _validate_target(
        self,
        channel_id: str,
        parameter: RuleParameter,
        condition: RuleCondition | None,
    ) -> None:
        if channel_id not in self._channel_map:
            raise ValueError(f"Unknown channel_id: {channel_id}")
        ch = self._channel_map[channel_id]
        if parameter == RuleParameter.CONDITIONAL_MUTE and ch.role == ChannelRole.SPOKEN_PRIMARY:
            raise ValueError(f"conditional_mute is strictly forbidden on spoken_primary channel {channel_id}")
        if condition is not None and condition.when_channel not in self._channel_map:
            raise ValueError(f"Condition references unknown channel_id: {condition.when_channel}")

    def _can_auto_apply(self, patch: ProposedPatch, is_multi_channel: bool) -> bool:
        if is_multi_channel:
            return False
        if patch.condition is not None:
            return False
        if patch.parameter == RuleParameter.VOL_DB_OFFSET:
            return AUTO_APPLY_VOL_MIN_DB <= patch.value <= AUTO_APPLY_VOL_MAX_DB
        if patch.parameter == RuleParameter.REVERB_DB_OFFSET:
            return AUTO_APPLY_REVERB_MIN_DB <= patch.value <= AUTO_APPLY_REVERB_MAX_DB
        return False

    def ingest_proposed_patches(self, patches: Sequence[ProposedPatch]) -> list[PendingRuleCard]:
        """Validate proposed patches and apply the ±3 dB vol / ±4 dB reverb preview gate."""
        if not patches:
            return []

        # Validate all patches before mutating state
        for patch in patches:
            self._validate_target(patch.channel_id, patch.parameter, patch.condition)

        distinct_channels = {p.channel_id for p in patches}
        is_multi_channel = len(distinct_channels) > 1

        # Also check if multiple patches in the same batch target the same continuous parameter
        # on the same channel with a combined delta exceeding auto-apply bounds
        per_param_sum: dict[tuple[str, RuleParameter], float] = {}
        for p in patches:
            if p.condition is None and p.parameter in (
                RuleParameter.VOL_DB_OFFSET,
                RuleParameter.REVERB_DB_OFFSET,
            ):
                key = (p.channel_id, p.parameter)
                per_param_sum[key] = per_param_sum.get(key, 0.0) + p.value

        cards: list[PendingRuleCard] = []
        now_ms = int(time.time() * 1000)

        for idx, patch in enumerate(patches):
            rule_id = f"rule_{uuid.uuid4().hex[:10]}"
            auto_ok = self._can_auto_apply(patch, is_multi_channel)
            if auto_ok:
                total_for_param = per_param_sum.get((patch.channel_id, patch.parameter), patch.value)
                vol_exceeded = patch.parameter == RuleParameter.VOL_DB_OFFSET and not (
                    AUTO_APPLY_VOL_MIN_DB <= total_for_param <= AUTO_APPLY_VOL_MAX_DB
                )
                rev_exceeded = patch.parameter == RuleParameter.REVERB_DB_OFFSET and not (
                    AUTO_APPLY_REVERB_MIN_DB <= total_for_param <= AUTO_APPLY_REVERB_MAX_DB
                )
                if vol_exceeded or rev_exceeded:
                    auto_ok = False

            expires_on = patch.expires_on if patch.expires_on is not None else RuleExpiry.SEGMENT_CHANGE
            created_ms = now_ms + idx

            if auto_ok:
                card = PendingRuleCard(
                    rule_id=rule_id,
                    description=patch.description,
                    phase=patch.phase,
                    channel_id=patch.channel_id,
                    parameter=patch.parameter,
                    value=patch.value,
                    condition=patch.condition,
                    scope=RuleScope.EPHEMERAL_SESSION,
                    expires_on=expires_on,
                    requires_preview=False,
                    applied_in_session=True,
                    created_at_ms=created_ms,
                )
                ephemeral_rule = MixingRule(
                    id=rule_id,
                    description=patch.description,
                    phase=patch.phase,
                    channel_id=patch.channel_id,
                    parameter=patch.parameter,
                    value=patch.value,
                    condition=patch.condition,
                    enabled=True,
                    source="llm_auto_ephemeral",
                    scope=RuleScope.EPHEMERAL_SESSION,
                    expires_on=expires_on,
                    created_at_ms=created_ms,
                )
                self.ephemeral_rules.append(ephemeral_rule)
                self._assign_order(rule_id)
            else:
                card = PendingRuleCard(
                    rule_id=rule_id,
                    description=patch.description,
                    phase=patch.phase,
                    channel_id=patch.channel_id,
                    parameter=patch.parameter,
                    value=patch.value,
                    condition=patch.condition,
                    scope=patch.scope,
                    expires_on=expires_on,
                    requires_preview=True,
                    applied_in_session=False,
                    created_at_ms=created_ms,
                )

            self.pending_rules[rule_id] = card
            cards.append(card)

        return cards

    def resolve_pending_rule(
        self,
        rule_id: str,
        action: Literal["apply_session", "promote", "dismiss"],
    ) -> MixingRule | None:
        """Resolve a pending rule card via apply_session, promote, or dismiss."""
        card = self.pending_rules.get(rule_id)
        if card is None:
            raise KeyError(f"Unknown pending rule_id: {rule_id}")

        if action not in ("apply_session", "promote", "dismiss"):
            raise ValueError(f"Unsupported pending rule action: {action}")

        if action == "dismiss":
            self.pending_rules.pop(rule_id, None)
            self.ephemeral_rules = [r for r in self.ephemeral_rules if r.id != rule_id]
            return None

        self._validate_target(card.channel_id, card.parameter, card.condition)
        now_ms = int(time.time() * 1000)

        if action == "apply_session":
            existing = next((r for r in self.ephemeral_rules if r.id == rule_id), None)
            if existing is None:
                rule = MixingRule(
                    id=card.rule_id,
                    description=card.description,
                    phase=card.phase,
                    channel_id=card.channel_id,
                    parameter=card.parameter,
                    value=card.value,
                    condition=card.condition,
                    enabled=True,
                    source="operator_session",
                    scope=RuleScope.EPHEMERAL_SESSION,
                    expires_on=card.expires_on,
                    created_at_ms=now_ms,
                )
                self.ephemeral_rules.append(rule)
                self._assign_order(rule.id)
            else:
                existing.enabled = True
                rule = existing

            card.applied_in_session = True
            card.requires_preview = False
            card.scope = RuleScope.EPHEMERAL_SESSION
            return rule

        # action == "promote"
        self.ephemeral_rules = [r for r in self.ephemeral_rules if r.id != rule_id]
        self.rules.learned_rules = [r for r in self.rules.learned_rules if r.id != rule_id]
        permanent_rule = MixingRule(
            id=card.rule_id,
            description=card.description,
            phase=card.phase,
            channel_id=card.channel_id,
            parameter=card.parameter,
            value=card.value,
            condition=card.condition,
            enabled=True,
            source="operator_confirmed",
            scope=RuleScope.CANDIDATE_PERMANENT,
            expires_on=None,
            created_at_ms=now_ms,
        )
        self.rules.learned_rules.append(permanent_rule)
        self._assign_order(permanent_rule.id)
        self.pending_rules.pop(rule_id, None)
        if self.rules_path is not None:
            self.save_atomic()
        return permanent_rule

    def add_permanent_rule(self, rule: MixingRule) -> MixingRule:
        """Directly add a validated permanent rule and persist atomically if configured."""
        self._validate_target(rule.channel_id, rule.parameter, rule.condition)
        self.rules.learned_rules = [r for r in self.rules.learned_rules if r.id != rule.id]
        self.rules.learned_rules.append(rule)
        self._assign_order(rule.id)
        if self.rules_path is not None:
            self.save_atomic()
        return rule

    def toggle_rule(self, rule_id: str, enabled: bool) -> MixingRule:
        """Enable or disable a permanent or ephemeral rule by ID."""
        for rule in self.rules.learned_rules:
            if rule.id == rule_id:
                rule.enabled = bool(enabled)
                if self.rules_path is not None:
                    self.save_atomic()
                return rule
        for rule in self.ephemeral_rules:
            if rule.id == rule_id:
                rule.enabled = bool(enabled)
                return rule
        raise KeyError(f"Unknown rule_id: {rule_id}")

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule from permanent, ephemeral, and pending stores."""
        deleted_perm = False
        before_perm = len(self.rules.learned_rules)
        self.rules.learned_rules = [r for r in self.rules.learned_rules if r.id != rule_id]
        if len(self.rules.learned_rules) < before_perm:
            deleted_perm = True

        before_eph = len(self.ephemeral_rules)
        self.ephemeral_rules = [r for r in self.ephemeral_rules if r.id != rule_id]
        deleted_eph = len(self.ephemeral_rules) < before_eph

        deleted_pend = self.pending_rules.pop(rule_id, None) is not None

        if deleted_perm and self.rules_path is not None:
            self.save_atomic()
        return deleted_perm or deleted_eph or deleted_pend

    def expire_ephemeral(self, event: RuleExpiry | str) -> list[str]:
        """Expire ephemeral session rules matching the lifecycle event."""
        evt = event if isinstance(event, RuleExpiry) else RuleExpiry(event)
        if evt == RuleExpiry.PHASE_CHANGE:
            expire_set = {RuleExpiry.PHASE_CHANGE}
        elif evt == RuleExpiry.SEGMENT_CHANGE:
            expire_set = {RuleExpiry.PHASE_CHANGE, RuleExpiry.SEGMENT_CHANGE}
        else:  # END_OF_SERVICE
            expire_set = {
                RuleExpiry.PHASE_CHANGE,
                RuleExpiry.SEGMENT_CHANGE,
                RuleExpiry.END_OF_SERVICE,
            }

        expired_ids: list[str] = []
        remaining: list[MixingRule] = []
        for rule in self.ephemeral_rules:
            rule_exp = rule.expires_on or RuleExpiry.SEGMENT_CHANGE
            if rule_exp in expire_set:
                expired_ids.append(rule.id)
                self.pending_rules.pop(rule.id, None)
            else:
                remaining.append(rule)
        self.ephemeral_rules = remaining
        return expired_ids

    def list_active_rules(self) -> list[MixingRule]:
        """Return all permanent and active ephemeral rules."""
        return [*self.rules.learned_rules, *self.ephemeral_rules]

    def list_pending_rules(self) -> list[PendingRuleCard]:
        """Return all pending rule cards in insertion order."""
        return list(self.pending_rules.values())

    def _rule_matches_phase(
        self,
        rule_phase: RulePhase,
        active_phase: ServicePhase,
        fallback_macro_phase: ServicePhase | None = None,
    ) -> bool:
        if rule_phase == RulePhase.ANY:
            return True
        if active_phase == ServicePhase.TRANSITION:
            if fallback_macro_phase is not None and fallback_macro_phase != ServicePhase.TRANSITION:
                return rule_phase.value == fallback_macro_phase.value
            return False
        return rule_phase.value == active_phase.value

    def _condition_is_true(
        self,
        condition: RuleCondition | None,
        speech_open_map: Mapping[str, bool],
        active_map: Mapping[str, bool],
    ) -> bool:
        if condition is None:
            return True
        if condition.predicate == ConditionPredicate.SPEECH_OPEN:
            return bool(speech_open_map.get(condition.when_channel, False))
        if condition.predicate == ConditionPredicate.ACTIVE:
            return bool(active_map.get(condition.when_channel, False))
        return False

    def _resolve_conflicts(
        self,
        phase: ServicePhase,
        channel_id: str,
        fallback_macro_phase: ServicePhase | None = None,
    ) -> list[MixingRule]:
        """Return enabled matching rules with most-recent rule winning identical conflict keys."""
        winners: dict[
            tuple[RulePhase, str, RuleParameter, tuple[str, ConditionPredicate] | None],
            tuple[tuple[int, int], MixingRule],
        ] = {}

        for idx, rule in enumerate(self.list_active_rules()):
            if not rule.enabled:
                continue
            if rule.channel_id != channel_id:
                continue
            if not self._rule_matches_phase(rule.phase, phase, fallback_macro_phase):
                continue
            cond_key = (
                (rule.condition.when_channel, rule.condition.predicate)
                if rule.condition is not None
                else None
            )
            conflict_key = (rule.phase, rule.channel_id, rule.parameter, cond_key)
            recency = (rule.created_at_ms, self._rule_order.get(rule.id, idx))
            prev = winners.get(conflict_key)
            if prev is None or recency >= prev[0]:
                winners[conflict_key] = (recency, rule)

        return [item[1] for item in winners.values()]

    def evaluate_channel(
        self,
        phase: ServicePhase | str,
        channel_id: str,
        control: str,
        speech_open_map: Mapping[str, bool] | None = None,
        active_map: Mapping[str, bool] | None = None,
        fallback_macro_phase: ServicePhase | None = None,
    ) -> tuple[float, float, bool]:
        """Evaluate rules for (phase, channel_id, control).

        Returns:
            (continuous_sum_db, conditional_delta_db, conditional_mute)
        """
        if channel_id not in self._channel_map:
            raise KeyError(f"Unknown channel_id: {channel_id}")

        active_phase = phase if isinstance(phase, ServicePhase) else ServicePhase(phase)
        ch = self._channel_map[channel_id]
        s_map = speech_open_map or {}
        a_map = active_map or {}

        matching = self._resolve_conflicts(active_phase, channel_id, fallback_macro_phase)

        continuous_sum = 0.0
        conditional_sum = 0.0
        conditional_mute = False

        if control == "reverb":
            for rule in matching:
                if rule.parameter == RuleParameter.REVERB_DB_OFFSET and self._condition_is_true(
                    rule.condition, s_map, a_map
                ):
                    continuous_sum += rule.value
            continuous_clamped = max(CONTINUOUS_STACK_MIN_DB, min(CONTINUOUS_STACK_MAX_DB, continuous_sum))
            return float(continuous_clamped), 0.0, False

        if control != "vol":
            return 0.0, 0.0, False

        for rule in matching:
            if not self._condition_is_true(rule.condition, s_map, a_map):
                continue
            if rule.parameter == RuleParameter.VOL_DB_OFFSET:
                continuous_sum += rule.value
            elif rule.parameter == RuleParameter.CONDITIONAL_ATTENUATE:
                conditional_sum += rule.value
            elif (
                rule.parameter == RuleParameter.CONDITIONAL_MUTE
                and ch.role != ChannelRole.SPOKEN_PRIMARY
                and rule.value >= 0.5
            ):
                conditional_mute = True

        continuous_clamped = max(CONTINUOUS_STACK_MIN_DB, min(CONTINUOUS_STACK_MAX_DB, continuous_sum))
        conditional_clamped = max(
            CONDITIONAL_ATTENUATE_MIN_DB, min(CONDITIONAL_ATTENUATE_MAX_DB, conditional_sum)
        )
        return float(continuous_clamped), float(conditional_clamped), conditional_mute

    def evaluate_duck_depth_db(
        self,
        phase: ServicePhase | str,
        channel_id: str,
        default_duck_depth_db: float,
        fallback_macro_phase: ServicePhase | None = None,
    ) -> float:
        """Return overridden duck_depth_db if a matching DUCK_DEPTH_DB rule exists, else default."""
        active_phase = phase if isinstance(phase, ServicePhase) else ServicePhase(phase)
        matching = self._resolve_conflicts(active_phase, channel_id, fallback_macro_phase)
        duck_rules = [r for r in matching if r.parameter == RuleParameter.DUCK_DEPTH_DB]
        if not duck_rules:
            return float(default_duck_depth_db)
        best = max(duck_rules, key=lambda r: (r.created_at_ms, self._rule_order.get(r.id, 0)))
        return float(max(-12.0, min(0.0, best.value)))

    def save_atomic(self) -> Path | None:
        """Persist permanent MixingRules atomically (tmp + os.replace) with timestamped backup."""
        if self.rules_path is None:
            return None

        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        if self.rules_path.exists() and self.backup_dir is not None:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            backup_path = self.backup_dir / f"{self.rules_path.stem}_{stamp}{self.rules_path.suffix}"
            shutil.copy2(self.rules_path, backup_path)

        serialized = json.dumps(self.rules.model_dump(mode="json"), indent=2) + "\n"
        tmp_path = self.rules_path.with_name(f".{self.rules_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(serialized)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.rules_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return self.rules_path
