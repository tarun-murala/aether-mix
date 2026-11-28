"""Canonical AetherMix v5.1 protocol schemas, taper math, and cross-validators.

All engine modules, bridge IPC clients, tests, and the SwiftUI ProtocolModels mirror
conform strictly to these models.
"""

from __future__ import annotations

import itertools
import math
import os
import secrets
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL_VERSION = "5.1"
SCHEMA_VERSION = 5

DEFAULT_AETHERMIX_DIR = Path.home() / ".aethermix"
DEFAULT_TOKEN_PATH = DEFAULT_AETHERMIX_DIR / "session.token"
DEFAULT_BRIDGE_SOCKET_PATH = DEFAULT_AETHERMIX_DIR / "bridge.sock"
DEFAULT_BRIDGE_STATE_PATH = DEFAULT_AETHERMIX_DIR / "bridge_state.json"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ServicePhase(str, Enum):
    SPOKEN_WORD = "SPOKEN_WORD"
    PRAYER_UNDERSCORE = "PRAYER_UNDERSCORE"
    WORSHIP_LOW = "WORSHIP_LOW"
    WORSHIP_HIGH = "WORSHIP_HIGH"
    TRANSITION = "TRANSITION"


MACRO_PHASES: tuple[ServicePhase, ...] = (
    ServicePhase.SPOKEN_WORD,
    ServicePhase.PRAYER_UNDERSCORE,
    ServicePhase.WORSHIP_LOW,
    ServicePhase.WORSHIP_HIGH,
)


class RulePhase(str, Enum):
    ANY = "ANY"
    SPOKEN_WORD = "SPOKEN_WORD"
    PRAYER_UNDERSCORE = "PRAYER_UNDERSCORE"
    WORSHIP_LOW = "WORSHIP_LOW"
    WORSHIP_HIGH = "WORSHIP_HIGH"


class AutonomyMode(str, Enum):
    SHADOW_OBSERVE = "SHADOW_OBSERVE"
    ACTIVE = "ACTIVE"
    EMERGENCY_BYPASS = "EMERGENCY_BYPASS"


class SyncStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SYNCED = "SYNCED"


class ChannelRole(str, Enum):
    SPOKEN_PRIMARY = "spoken_primary"
    VOCAL_LEAD = "vocal_lead"
    VOCAL_BG = "vocal_bg"
    INSTRUMENT_PAD = "instrument_pad"
    INSTRUMENT_GUITAR = "instrument_guitar"
    INSTRUMENT_BAND = "instrument_band"
    ROOM_AMBIENCE = "room_ambience"
    MASTER_METER = "master_meter"


BAND_ROLES: frozenset[ChannelRole] = frozenset(
    {
        ChannelRole.INSTRUMENT_PAD,
        ChannelRole.INSTRUMENT_GUITAR,
        ChannelRole.INSTRUMENT_BAND,
    }
)


class ControlType(str, Enum):
    VOL = "vol"
    REVERB = "reverb"
    MUTE = "mute"


class TakeoverMode(str, Enum):
    PICKUP_CATCHUP = "pickup_catchup"
    IMMEDIATE = "immediate"
    RELATIVE_ENCODER = "relative_encoder"


class MainStageOutMode(str, Enum):
    VIRTUAL_PORT = "virtual_port"
    IAC_BUS = "iac_bus"


class RuleParameter(str, Enum):
    VOL_DB_OFFSET = "vol_db_offset"
    REVERB_DB_OFFSET = "reverb_db_offset"
    DUCK_DEPTH_DB = "duck_depth_db"
    CONDITIONAL_ATTENUATE = "conditional_attenuate"
    CONDITIONAL_MUTE = "conditional_mute"


class RuleScope(str, Enum):
    EPHEMERAL_SESSION = "ephemeral_session"
    CANDIDATE_PERMANENT = "candidate_permanent"


class RuleExpiry(str, Enum):
    PHASE_CHANGE = "phase_change"
    SEGMENT_CHANGE = "segment_change"
    END_OF_SERVICE = "end_of_service"


class ConditionPredicate(str, Enum):
    SPEECH_OPEN = "speech_open"
    ACTIVE = "active"


class BrainCategory(str, Enum):
    PERCEPTION = "PERCEPTION"
    ACTION = "ACTION"
    SAFETY = "SAFETY"
    LLM = "LLM"
    RLHF = "RLHF"


# Per-parameter Pydantic bounds for LLM & operator rule patches
RULE_PARAMETER_BOUNDS: dict[RuleParameter, tuple[float, float]] = {
    RuleParameter.VOL_DB_OFFSET: (-12.0, 6.0),
    RuleParameter.REVERB_DB_OFFSET: (-18.0, 12.0),
    RuleParameter.DUCK_DEPTH_DB: (-12.0, 0.0),
    RuleParameter.CONDITIONAL_ATTENUATE: (-36.0, 0.0),
    RuleParameter.CONDITIONAL_MUTE: (0.0, 1.0),
}

DEFAULT_VOL_TAPER: list[tuple[int, float | None]] = [
    # Measured against MainStage's native channel-strip fader law (volume mapped Absolute, -inf..+6 dB).
    (0, None),
    (1, -78.2),
    (4, -54.1),
    (8, -42.1),
    (16, -30.1),
    (24, -23.0),
    (32, -18.0),
    (48, -11.0),
    (64, -6.0),
    (80, -2.1),
    (96, 1.1),
    (112, 3.7),
    (127, 6.0),
]

DEFAULT_REVERB_TAPER: list[tuple[int, float | None]] = [
    (0, None),
    (1, -60.0),
    (32, -24.0),
    (64, -12.0),
    (96, 0.0),
    (114, 6.0),
    (127, 6.0),
]


# ---------------------------------------------------------------------------
# Taper math (CC 0 = -inf/None, strictly monotonic CC 1..plateau, 0..127)
# ---------------------------------------------------------------------------


def validate_taper(points: list[tuple[int, float | None]]) -> list[tuple[int, float | None]]:
    if len(points) < 2:
        raise ValueError("CC taper must contain at least 2 points")
    if points[0][0] != 0 or points[-1][0] != 127:
        raise ValueError("CC taper must span full MIDI range 0..127")
    prev_cc = -1
    prev_db: float | None = None
    seen_non_null = False
    for cc, db in points:
        if not (0 <= cc <= 127):
            raise ValueError(f"CC {cc} out of 0..127 range")
        if cc <= prev_cc:
            raise ValueError("CC taper x-values must be strictly increasing")
        if db is None:
            if cc != 0:
                raise ValueError("Only CC 0 may map to null (-inf)")
        else:
            if seen_non_null and prev_db is not None and db < prev_db:
                raise ValueError("CC taper dB values must be monotonically non-decreasing")
            seen_non_null = True
            prev_db = db
        prev_cc = cc
    if not seen_non_null:
        raise ValueError("CC taper must contain at least one numeric dB point")
    return points


def cc_to_db(cc: int, taper: list[tuple[int, float | None]]) -> float:
    """Convert 7-bit MIDI CC (0..127) to float dB (-inf for CC 0 -> -120.0 floor)."""
    cc_clamped = max(0, min(127, int(cc)))
    if cc_clamped == 0 and taper[0][1] is None:
        return -math.inf
    numeric_pts = [(c, d) for c, d in taper if d is not None]
    if cc_clamped <= numeric_pts[0][0]:
        return float(numeric_pts[0][1])
    if cc_clamped >= numeric_pts[-1][0]:
        return float(numeric_pts[-1][1])
    for (c0, d0), (c1, d1) in itertools.pairwise(numeric_pts):
        if c0 <= cc_clamped <= c1:
            if c1 == c0:
                return float(d0)
            frac = (cc_clamped - c0) / (c1 - c0)
            return float(d0 + frac * (d1 - d0))
    return float(numeric_pts[-1][1])


def db_to_cc(db: float, taper: list[tuple[int, float | None]]) -> int:
    """Inverse taper: convert float dB to 7-bit MIDI CC (picks lowest CC on top plateau)."""
    if math.isinf(db) and db < 0:
        return 0
    numeric_pts = [(c, d) for c, d in taper if d is not None]
    min_cc, min_db = numeric_pts[0]
    if db < min_db - 1e-6 and taper[0][1] is None:
        return 0
    if db <= min_db:
        return min_cc
    # Pick lowest CC on any flat plateau (e.g., CC 114 for +6.0 dB)
    for (c0, d0), (c1, d1) in itertools.pairwise(numeric_pts):
        if db <= d0 + 1e-9:
            return c0
        if d0 <= db <= d1:
            if abs(d1 - d0) < 1e-9:
                return c0
            frac = (db - d0) / (d1 - d0)
            return round(c0 + frac * (c1 - c0))
    # Above max: return lowest CC that reaches the maximum dB
    max_db = numeric_pts[-1][1]
    for c, d in numeric_pts:
        if abs(d - max_db) < 1e-9:
            return c
    return numeric_pts[-1][0]


# ---------------------------------------------------------------------------
# Service Plan Schema
# ---------------------------------------------------------------------------


class FaderMapEntry(BaseModel):
    hw_channel: int = Field(ge=1, le=16)
    hw_cc: int = Field(ge=0, le=127)
    channel_id: str
    control: Literal["vol", "reverb"] = "vol"


class ActionBinding(BaseModel):
    hw_channel: int = Field(ge=1, le=16)
    hw_cc: int = Field(ge=0, le=127)


class ActionBindings(BaseModel):
    thumbs_up: ActionBinding | None = None
    thumbs_down: ActionBinding | None = None
    emergency_bypass: ActionBinding | None = None


class MidiConfig(BaseModel):
    hardware_in_port: str | None = None
    takeover_mode: TakeoverMode = TakeoverMode.PICKUP_CATCHUP
    mainstage_out_mode: MainStageOutMode = MainStageOutMode.VIRTUAL_PORT
    mainstage_out_port: str = "AetherMix To MainStage"
    mainstage_feedback_in_port: str | None = None
    fader_map: list[FaderMapEntry] = Field(default_factory=list)
    action_bindings: ActionBindings = Field(default_factory=ActionBindings)


class ChannelConfig(BaseModel):
    id: str
    label: str
    tap_channels: list[int] = Field(min_length=1)
    role: ChannelRole
    vad_enabled: bool = False
    feedback_guard: bool = False
    nominal_rms_db: float = -18.0
    nominal_vband_rms_db: float = -20.0
    mic_noise_floor_norm_db: float = -36.0
    bleed_offset_db: float = -12.0
    midi_channel: int = Field(default=1, ge=1, le=16)
    vol_cc: int = Field(ge=0, le=127)
    reverb_cc: int | None = Field(default=None, ge=0, le=127)
    mute_cc: int | None = Field(default=None, ge=0, le=127)
    mute_active_high: bool = True
    hard_min_db: float = -60.0
    hard_max_db: float = 0.0
    cc_taper: list[tuple[int, float | None]] = Field(default_factory=lambda: list(DEFAULT_VOL_TAPER))
    reverb_cc_taper: list[tuple[int, float | None]] = Field(
        default_factory=lambda: list(DEFAULT_REVERB_TAPER)
    )

    @field_validator("cc_taper", "reverb_cc_taper")
    @classmethod
    def _check_taper(cls, v: list[tuple[int, float | None]]) -> list[tuple[int, float | None]]:
        return validate_taper(v)

    @model_validator(mode="after")
    def _check_bounds_and_spoken_mute(self) -> ChannelConfig:
        if self.hard_min_db >= self.hard_max_db:
            raise ValueError(f"{self.id}: hard_min_db must be < hard_max_db")
        if self.role == ChannelRole.SPOKEN_PRIMARY and self.mute_cc is not None:
            raise ValueError(f"{self.id}: automatic mute_cc is forbidden on spoken_primary channels")
        return self


class ServiceSegment(BaseModel):
    id: str
    title: str
    expected_phase: ServicePhase


class ServicePlan(BaseModel):
    schema_version: int = SCHEMA_VERSION
    sample_rate: int = 48000
    audio_device_hint: str = "BlackHole 16ch"
    ollama_model: str = "qwen3:14b"
    ollama_think: bool = False
    midi: MidiConfig = Field(default_factory=MidiConfig)
    room_tap_channels: list[int] = Field(default_factory=lambda: [13, 14])
    master_tap_channels: list[int] = Field(default_factory=lambda: [15, 16])
    channels: list[ChannelConfig] = Field(min_length=1)
    use_service_plan: bool = True
    order_of_service: list[ServiceSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> ServicePlan:
        ids = [ch.id for ch in self.channels]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate channel IDs in service_plan.json")
        id_set = set(ids)
        for entry in self.midi.fader_map:
            if entry.channel_id not in id_set:
                raise ValueError(f"fader_map references unknown channel_id: {entry.channel_id}")
        return self

    def channel_by_id(self, channel_id: str) -> ChannelConfig:
        for ch in self.channels:
            if ch.id == channel_id:
                return ch
        raise KeyError(f"Unknown channel_id: {channel_id}")

    @property
    def channel_ids(self) -> list[str]:
        return [ch.id for ch in self.channels]


# ---------------------------------------------------------------------------
# Mixing Rules & Firewall Schema
# ---------------------------------------------------------------------------


class FirewallConfig(BaseModel):
    touch_lock_seconds: float = Field(default=15.0, ge=1.0, le=120.0)
    echo_suppress_ms: int = Field(default=200, ge=20, le=1000)
    transition_instrument_trim_db: float = Field(default=-4.0, ge=-12.0, le=0.0)
    speech_open_vad_min: float = Field(default=0.70, ge=0.1, le=0.99)
    speech_open_snr_db: float = Field(default=6.0, ge=0.0, le=30.0)
    speech_open_hold_spoken_ms: int = Field(default=150, ge=20, le=2000)
    speech_open_hold_worship_ms: int = Field(default=400, ge=50, le=2000)
    speech_open_release_silence_s: float = Field(default=3.0, ge=0.5, le=15.0)
    instrument_open_silent_norm_db: float = Field(default=-30.0, ge=-60.0, le=-10.0)
    instrument_open_silent_min_s: float = Field(default=3.0, ge=0.5, le=30.0)
    instrument_open_active_norm_db: float = Field(default=-12.0, ge=-30.0, le=0.0)
    instrument_open_hold_ms: int = Field(default=200, ge=40, le=2000)
    instrument_open_ensemble_window_s: float = Field(default=1.0, ge=0.2, le=5.0)
    instrument_open_max_hold_s: float = Field(default=8.0, ge=2.0, le=30.0)
    duck_depth_db: float = Field(default=-6.0, ge=-18.0, le=0.0)
    duck_snr_spoken_db: float = Field(default=3.0, ge=0.0, le=24.0)
    duck_targets_by_phase: dict[ServicePhase, list[ChannelRole]] = Field(
        default_factory=lambda: {
            ServicePhase.SPOKEN_WORD: [ChannelRole.INSTRUMENT_PAD],
            ServicePhase.PRAYER_UNDERSCORE: [ChannelRole.INSTRUMENT_PAD],
            ServicePhase.TRANSITION: [ChannelRole.INSTRUMENT_PAD],
            ServicePhase.WORSHIP_LOW: [
                ChannelRole.INSTRUMENT_PAD,
                ChannelRole.INSTRUMENT_GUITAR,
                ChannelRole.INSTRUMENT_BAND,
            ],
            ServicePhase.WORSHIP_HIGH: [
                ChannelRole.INSTRUMENT_PAD,
                ChannelRole.INSTRUMENT_GUITAR,
                ChannelRole.INSTRUMENT_BAND,
            ],
        }
    )
    slew_protect_down_db_per_sec: float = Field(default=48.0, ge=10.0, le=120.0)
    slew_speech_open_up_db_per_sec: float = Field(default=48.0, ge=10.0, le=120.0)
    slew_instrument_open_up_db_per_sec: float = Field(default=24.0, ge=6.0, le=96.0)
    slew_duck_release_up_db_per_sec: float = Field(default=8.0, ge=1.0, le=48.0)
    slew_musical_db_per_sec: float = Field(default=4.0, ge=0.5, le=24.0)
    slew_handover_db_per_sec: float = Field(default=2.0, ge=0.5, le=12.0)
    feedback_peak_above_floor_db: float = Field(default=14.0, ge=6.0, le=30.0)
    feedback_growth_db: float = Field(default=2.0, ge=0.0, le=12.0)
    feedback_hold_ms: int = Field(default=250, ge=60, le=2000)
    feedback_cut_step_db: float = Field(default=-4.0, ge=-12.0, le=-1.0)
    feedback_max_cut_db: float = Field(default=-16.0, ge=-30.0, le=-4.0)
    feedback_release_delay_s: float = Field(default=5.0, ge=1.0, le=60.0)
    feedback_release_rate_db_per_sec: float = Field(default=1.0, ge=0.1, le=6.0)
    watchdog_max_tick_ms: float = Field(default=50.0, ge=20.0, le=500.0)
    watchdog_max_consecutive_late: int = Field(default=5, ge=2, le=50)


class RuleCondition(BaseModel):
    when_channel: str
    predicate: ConditionPredicate


class MixingRule(BaseModel):
    id: str
    description: str = Field(max_length=120)
    phase: RulePhase
    channel_id: str
    parameter: RuleParameter
    value: float
    condition: RuleCondition | None = None
    enabled: bool = True
    source: str = "operator_confirmed"
    scope: RuleScope = RuleScope.CANDIDATE_PERMANENT
    expires_on: RuleExpiry | None = None
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    @model_validator(mode="after")
    def _validate_bounds(self) -> MixingRule:
        lo, hi = RULE_PARAMETER_BOUNDS[self.parameter]
        if not (lo <= self.value <= hi):
            raise ValueError(
                f"Rule {self.id}: parameter {self.parameter.value} value {self.value} "
                f"outside allowed bounds [{lo}, {hi}]"
            )
        if self.scope == RuleScope.EPHEMERAL_SESSION and self.expires_on is None:
            self.expires_on = RuleExpiry.SEGMENT_CHANGE
        return self


class MixingRules(BaseModel):
    schema_version: int = SCHEMA_VERSION
    calibrated_at: str | None = None
    firewall: FirewallConfig = Field(default_factory=FirewallConfig)
    phase_target_lufs: dict[ServicePhase, float]
    phase_baselines_db: dict[ServicePhase, dict[str, float]]
    phase_reverb_baselines_db: dict[ServicePhase, dict[str, float]] = Field(default_factory=dict)
    learned_rules: list[MixingRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_transition_baselines(self) -> MixingRules:
        if ServicePhase.TRANSITION in self.phase_baselines_db:
            raise ValueError("TRANSITION must not have fixed baselines (it is hold + settle)")
        if ServicePhase.TRANSITION in self.phase_target_lufs:
            raise ValueError("TRANSITION must not have a fixed LUFS target")
        for phase in MACRO_PHASES:
            if phase not in self.phase_baselines_db:
                raise ValueError(f"Missing phase_baselines_db entry for {phase.value}")
            if phase not in self.phase_target_lufs:
                raise ValueError(f"Missing phase_target_lufs entry for {phase.value}")
        return self


def cross_validate_plan_and_rules(plan: ServicePlan, rules: MixingRules) -> None:
    """Enforce load-time invariants between service_plan.json and mixing_rules.json."""
    channel_ids = set(plan.channel_ids)
    for phase in MACRO_PHASES:
        baselines = rules.phase_baselines_db[phase]
        missing = channel_ids - set(baselines.keys())
        if missing:
            raise ValueError(f"Phase {phase.value} missing baselines for channels: {sorted(missing)}")
        rev_baselines = rules.phase_reverb_baselines_db.get(phase, {})
        for ch in plan.channels:
            if ch.reverb_cc is not None and ch.id not in rev_baselines:
                raise ValueError(
                    f"Phase {phase.value} missing reverb baseline for reverb-enabled channel {ch.id}"
                )
    for rule in rules.learned_rules:
        if rule.channel_id not in channel_ids:
            raise ValueError(f"Rule {rule.id} references unknown channel_id: {rule.channel_id}")
        ch = plan.channel_by_id(rule.channel_id)
        if rule.parameter == RuleParameter.CONDITIONAL_MUTE and ch.role == ChannelRole.SPOKEN_PRIMARY:
            raise ValueError(f"Rule {rule.id}: conditional_mute forbidden on spoken_primary {ch.id}")
        if rule.condition is not None and rule.condition.when_channel not in channel_ids:
            raise ValueError(
                f"Rule {rule.id} condition references unknown channel_id: {rule.condition.when_channel}"
            )


# ---------------------------------------------------------------------------
# Ollama LLM Output Schema (Section 6.3)
# ---------------------------------------------------------------------------


class ProposedPatch(BaseModel):
    description: str = Field(max_length=120)
    scope: RuleScope
    expires_on: RuleExpiry = RuleExpiry.SEGMENT_CHANGE
    phase: RulePhase
    channel_id: str
    parameter: RuleParameter
    value: float = Field(ge=-36.0, le=12.0)
    condition: RuleCondition | None = None

    @model_validator(mode="after")
    def _validate_param_bounds(self) -> ProposedPatch:
        lo, hi = RULE_PARAMETER_BOUNDS[self.parameter]
        if not (lo <= self.value <= hi):
            raise ValueError(
                f"Patch value {self.value} out of bounds [{lo}, {hi}] for {self.parameter.value}"
            )
        return self


class LLMSupervisorResponse(BaseModel):
    thought_summary: str = Field(max_length=140)
    reasoning: str = Field(max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_patches: list[ProposedPatch] = Field(default_factory=list, max_length=4)


OLLAMA_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["thought_summary", "reasoning", "confidence", "proposed_patches"],
    "properties": {
        "thought_summary": {"type": "string", "maxLength": 140},
        "reasoning": {"type": "string", "maxLength": 500},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "proposed_patches": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["description", "scope", "phase", "channel_id", "parameter", "value"],
                "properties": {
                    "description": {"type": "string", "maxLength": 120},
                    "scope": {"type": "string", "enum": ["ephemeral_session", "candidate_permanent"]},
                    "expires_on": {
                        "type": "string",
                        "enum": ["phase_change", "segment_change", "end_of_service"],
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["ANY", "SPOKEN_WORD", "PRAYER_UNDERSCORE", "WORSHIP_LOW", "WORSHIP_HIGH"],
                    },
                    "channel_id": {"type": "string"},
                    "parameter": {
                        "type": "string",
                        "enum": [
                            "vol_db_offset",
                            "reverb_db_offset",
                            "duck_depth_db",
                            "conditional_attenuate",
                            "conditional_mute",
                        ],
                    },
                    "value": {"type": "number", "minimum": -36.0, "maximum": 12.0},
                    "condition": {
                        "type": ["object", "null"],
                        "properties": {
                            "when_channel": {"type": "string"},
                            "predicate": {"type": "string", "enum": ["speech_open", "active"]},
                        },
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# WebSocket v5.1 Messages (Section 6.4)
# ---------------------------------------------------------------------------


class WSEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    seq: int = 0
    req_id: str | None = None
    ts_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    payload: dict[str, Any] = Field(default_factory=dict)


class ChannelRuntimeState(BaseModel):
    armed: bool = False
    sync_status: SyncStatus = SyncStatus.UNVERIFIED
    vol_current_db: float = -60.0
    vol_target_db: float = -60.0
    vol_actual_cc: int = 0
    vol_target_cc: int = 0
    vol_touch_lock_s: float = 0.0
    vol_human_offset_db: float = 0.0
    vol_pickup_dir: int = 0
    reverb_current_db: float | None = None
    reverb_target_db: float | None = None
    reverb_actual_cc: int | None = None
    reverb_target_cc: int | None = None
    reverb_touch_lock_s: float = 0.0
    reverb_human_offset_db: float = 0.0
    reverb_pickup_dir: int = 0
    feedback_ceiling_db: float | None = None
    speech_open: bool = False
    instrument_open: bool = False


class PendingRuleCard(BaseModel):
    rule_id: str
    description: str
    phase: RulePhase
    channel_id: str
    parameter: RuleParameter
    value: float
    condition: RuleCondition | None = None
    scope: RuleScope
    expires_on: RuleExpiry = RuleExpiry.SEGMENT_CHANGE
    requires_preview: bool = False
    applied_in_session: bool = False
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class StateSnapshotPayload(BaseModel):
    channel_ids: list[str]
    autonomy_mode: AutonomyMode = AutonomyMode.SHADOW_OBSERVE
    bypass_latched: bool = False
    freeze_learning: bool = False
    active_phase: ServicePhase = ServicePhase.SPOKEN_WORD
    phase_confidence: float = 0.0
    hysteresis_progress: float = 0.0
    service_step_idx: int = 0
    channels: dict[str, ChannelRuntimeState] = Field(default_factory=dict)
    active_rules: list[MixingRule] = Field(default_factory=list)
    pending_rules: list[PendingRuleCard] = Field(default_factory=list)
    queued_review_count: int = 0
    hardware_warnings: list[str] = Field(default_factory=list)


class MeterFramePayload(BaseModel):
    rms_db: list[float]
    peak_db: list[float]
    clip: list[bool]
    vad: list[float]
    master_lufs: float = -60.0
    vol_target_cc: list[int]
    vol_actual_cc: list[int]
    reverb_target_cc: list[int | None]
    reverb_actual_cc: list[int | None]


class BrainEventPayload(BaseModel):
    id: str
    category: BrainCategory
    summary: str
    detail: str = ""
    confidence: float = 1.0
    ts_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class HeartbeatPayload(BaseModel):
    uptime_s: float
    loop_p95_ms: float
    loop_p99_ms: float
    audio_ok: bool
    bridge_ok: bool
    clap_ok: bool
    ollama_ok: bool


# ---------------------------------------------------------------------------
# Bridge NDJSON Protocol Helpers (Section 6.5)
# ---------------------------------------------------------------------------


class BridgeStatePayload(BaseModel):
    last_sent_cc: dict[str, int] = Field(default_factory=dict)
    human_seq: dict[str, int] = Field(default_factory=dict)
    bypass_latched: bool = False
    controller_connected: bool = False
    engine_connected: bool = False
    pickup_pending: dict[str, int] = Field(default_factory=dict)
    restored_from_disk_at: int | None = None


def ensure_session_token(token_path: Path = DEFAULT_TOKEN_PATH) -> str:
    """Create (0600) or load the shared local session token for WS and Bridge auth."""
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if token_path.exists():
        tok = token_path.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    tmp_path = token_path.with_suffix(".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (tok + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp_path, token_path)
    os.chmod(token_path, 0o600)
    return tok
