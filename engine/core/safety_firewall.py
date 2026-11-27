"""Deterministic Real-Time Tier-1 Safety Firewall (SPEC v5.1 §5.2 & §5.3.3).

Enforces all live mixing safety invariants at 50 Hz (20 ms tick):
1. Boots into SHADOW_OBSERVE with all channels disarmed; bridge disk-restore safety drop.
2. Watchdog (>50 ms x 5 consecutive late ticks or >500 ms audio stall) drops to SHADOW_OBSERVE
   emitting zero agent writes (never sends CC 0 / mute).
3. Per-(channel_id, control) 15 s touch lock (including pre-pickup on_human_touch),
   frozen_learned_db (zero drift), suspended conditional rules while human override is active,
   human-above-hard_max_db respect, and smooth 4 dB/s clearing on phase/segment change or release.
4. Tier-1 speech_open reflex on spoken_primary (48 dB/s, 150 ms spoken / 400 ms worship, 3 s release).
5. Tier-1 ensemble-gated instrument_open reflex (never in PRAYER_UNDERSCORE, silent>=3s -> active>=200ms,
   drums alone or >=2 instrument roles within 1.0 s, 24 dB/s up-slew, speech-over-intro duck cap,
   8.0 s auto-release, plus on_segment_advance pre-staging).
6. Bleed-safe, role-configurable auto-ducking inside unified 5-path slew limiter.
7. Bleed-rejected, escalating anti-feedback detector overriding touch lock (-4 dB -> -16 dB,
   +1 dB/s lift after 5 s clear).
8. Smart thumbs_down rollback (T-5s first press, T-15s/pre-transition second press within 5 s,
   skipping touch-locked controls, holding 10 s) and thumbs_up reinforcement.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.ndimage import median_filter

from engine.core.offset_learner import OffsetLearner, build_context_vector
from engine.core.protocol import (
    BAND_ROLES,
    MACRO_PHASES,
    AutonomyMode,
    BrainCategory,
    BrainEventPayload,
    BridgeStatePayload,
    ChannelConfig,
    ChannelRole,
    ChannelRuntimeState,
    MixingRules,
    ServicePhase,
    ServicePlan,
    SyncStatus,
    cc_to_db,
    cross_validate_plan_and_rules,
    db_to_cc,
)
from engine.core.rules_manager import RulesManager

if TYPE_CHECKING:
    from engine.core.audio_tap import AudioFrameMetrics

# Anti-feedback growth tracking.
# A musical note reaches its sustain level within a few ticks of its onset, whereas acoustic
# feedback rings up continuously. The growth baseline is therefore the loudest level seen during
# the first FEEDBACK_ONSET_SETTLE_S of a candidate, so a note's attack never counts as "growth".
FEEDBACK_ONSET_SETTLE_S = 0.06
# On a singing channel's fundamental bins, growth must also be a sustained ring-up: the magnitude
# must rise by feedback_growth_db across the trailing feedback_hold_ms window (>= 8 dB/s with the
# default 2 dB / 250 ms rules) without dipping more than this below the window's starting level.
# Held notes, crescendos and expression swells wander up and down; a howl only climbs, and it
# climbs to new highs: growth is measured from the higher of the window start and the loudest
# level this candidate already held earlier (so recovering from an expression dip is not growth).
FEEDBACK_RINGUP_MAX_DIP_DB = 1.0
FEEDBACK_RINGUP_EDGE_TICKS = 3  # average this many ticks at each window edge (jitter guard)


@dataclass
class AgentCCWrite:
    """Single quantized MIDI CC command emitted by the firewall to midi_bridge_daemon."""

    channel_id: str
    control: Literal["vol", "reverb", "mute"]
    cc: int
    target_db: float | None = None
    ack_human_seq: int = 0


@dataclass
class FirewallOutput:
    """Per-tick deterministic output of SafetyFirewall.tick()."""

    agent_writes: list[AgentCCWrite]
    channels_state: dict[str, ChannelRuntimeState]
    events: list[BrainEventPayload]
    watchdog_tripped: bool
    autonomy_mode: AutonomyMode


@dataclass
class _ControlTrack:
    current_db: float
    target_db: float
    actual_cc: int
    target_cc: int
    touch_lock_until_s: float = 0.0
    human_override_active: bool = False
    human_anchor_db: float | None = None
    human_session_offset_db: float = 0.0
    frozen_learned_db: float = 0.0
    clear_offset_after_touch_lock: bool = False
    pickup_dir: int = 0
    in_duck_release: bool = False
    was_ducked: bool = False
    handover_gliding: bool = False
    rollback_hold_until_s: float = 0.0
    rollback_target_db: float | None = None
    last_phi: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64))


@dataclass
class _ChannelInternal:
    config: ChannelConfig
    armed: bool = False
    sync_status: SyncStatus = SyncStatus.UNVERIFIED
    vol: _ControlTrack = field(default_factory=lambda: _ControlTrack(-60.0, -60.0, 0, 0))
    reverb: _ControlTrack | None = None
    muted: bool = False

    # Speech-open state (spoken_primary)
    speech_hold_s: float = 0.0
    speech_silence_s: float = 0.0
    speech_open: bool = False
    snr_env_db: float = -60.0

    # Duck trigger hold state (spoken_primary / vocal_lead)
    duck_trigger_hold_s: float = 0.0
    duck_trigger_active: bool = False

    # Instrument-open onset tracking (BAND_ROLES)
    silent_duration_s: float = 0.0
    was_sustained_silent: bool = False
    mid_level_s: float = 0.0
    active_onset_hold_s: float = 0.0
    last_onset_s: float | None = None
    instrument_open: bool = False

    # Anti-feedback state (feedback_guard == True)
    fb_candidate_bin: int | None = None
    fb_hold_s: float = 0.0
    fb_initial_mag_db: float = 0.0
    fb_initial_prom_db: float = 0.0
    fb_mag_hist: deque[tuple[float, float]] = field(default_factory=deque)  # (hold_s, mag_db)
    fb_prior_peak_db: float = -120.0  # loudest level held before the trailing ring-up window
    fb_total_cut_db: float = 0.0
    fb_last_cut_s: float | None = None
    fb_clear_s: float = 0.0
    feedback_ceiling_db: float | None = None


@dataclass
class _HistorySnapshot:
    ts_s: float
    active_phase: ServicePhase
    last_macro_phase: ServicePhase
    controls: dict[tuple[str, str], tuple[float, float, np.ndarray]]
    # (current_db, effective_offset_db, phi)


class SafetyFirewall:
    """Deterministic Tier-1 safety firewall, reflex engine, and 5-path slew limiter."""

    def __init__(
        self,
        plan: ServicePlan,
        rules: MixingRules,
        rules_manager: RulesManager,
        offset_learner: OffsetLearner,
    ) -> None:
        cross_validate_plan_and_rules(plan, rules)
        self.plan = plan
        self.rules = rules
        self.rules_manager = rules_manager
        self.offset_learner = offset_learner

        # Boot invariants (SPEC §1, §5.2): SHADOW_OBSERVE, all channels disarmed
        self.autonomy_mode: AutonomyMode = AutonomyMode.SHADOW_OBSERVE
        self.bypass_latched: bool = False
        self.active_phase: ServicePhase = ServicePhase.SPOKEN_WORD
        self.last_macro_phase: ServicePhase = ServicePhase.SPOKEN_WORD
        self.service_step_idx: int = 0

        self._clock_s: float = 0.0
        self._consecutive_late_ticks: int = 0
        self._watchdog_tripped: bool = False
        self._prestage_worship: bool = False

        # Ensemble instrument-open state
        self._instrument_open_active: bool = False
        self._instrument_open_started_at_s: float = 0.0

        # Bridge deduplication, hydration & human_seq tracking
        self._last_sent_cc: dict[tuple[str, str], int] = {}
        self._human_seq: dict[tuple[str, str], int] = {}
        self._last_handled_restored_at: int | None = None

        # Pending events emitted outside tick() (e.g. hydrate_from_bridge)
        self._pending_events: list[BrainEventPayload] = []

        # History for pre-move phi_bar, thumbs_up trailing phi, and smart thumbs_down rollback
        self._history: deque[_HistorySnapshot] = deque(maxlen=1200)  # ~24s at 50 Hz
        self._pre_transition_snapshot: _HistorySnapshot | None = None
        self._last_thumbs_down_s: float | None = None

        # Per-channel internal state
        self._channels: dict[str, _ChannelInternal] = {}
        spoken_baselines = rules.phase_baselines_db[ServicePhase.SPOKEN_WORD]
        spoken_rev_baselines = rules.phase_reverb_baselines_db.get(ServicePhase.SPOKEN_WORD, {})

        for ch in plan.channels:
            init_vol_db = float(spoken_baselines[ch.id])
            init_vol_cc = db_to_cc(init_vol_db, ch.cc_taper)
            vol_track = _ControlTrack(
                current_db=init_vol_db,
                target_db=init_vol_db,
                actual_cc=init_vol_cc,
                target_cc=init_vol_cc,
            )
            self._last_sent_cc[(ch.id, "vol")] = init_vol_cc

            rev_track: _ControlTrack | None = None
            if ch.reverb_cc is not None:
                init_rev_db = float(spoken_rev_baselines.get(ch.id, -24.0))
                init_rev_cc = db_to_cc(init_rev_db, ch.reverb_cc_taper)
                rev_track = _ControlTrack(
                    current_db=init_rev_db,
                    target_db=init_rev_db,
                    actual_cc=init_rev_cc,
                    target_cc=init_rev_cc,
                )
                self._last_sent_cc[(ch.id, "reverb")] = init_rev_cc

            self._channels[ch.id] = _ChannelInternal(
                config=ch,
                armed=False,
                sync_status=SyncStatus.UNVERIFIED,
                vol=vol_track,
                reverb=rev_track,
                muted=False,
            )

    # ------------------------------------------------------------------
    # Event helper & clock
    # ------------------------------------------------------------------

    def _make_event(
        self,
        category: BrainCategory,
        summary: str,
        detail: str = "",
        confidence: float = 1.0,
    ) -> BrainEventPayload:
        return BrainEventPayload(
            id=f"evt_{uuid.uuid4().hex[:10]}",
            category=category,
            summary=summary,
            detail=detail,
            confidence=confidence,
            ts_ms=int(time.time() * 1000),
        )

    def _resolve_now(self, now_s: float | None) -> float:
        if now_s is not None:
            self._clock_s = float(now_s)
        return self._clock_s

    def _get_track(self, channel_id: str, control: str) -> tuple[_ChannelInternal, _ControlTrack]:
        if channel_id not in self._channels:
            raise KeyError(f"Unknown channel_id: {channel_id}")
        ch_int = self._channels[channel_id]
        if control == "vol":
            return ch_int, ch_int.vol
        if control == "reverb":
            if ch_int.reverb is None:
                raise ValueError(f"Channel {channel_id} does not have reverb control")
            return ch_int, ch_int.reverb
        raise ValueError(f"Unsupported control: {control}")

    def _taper_for(self, ch: ChannelConfig, control: str) -> list[tuple[int, float | None]]:
        return ch.cc_taper if control == "vol" else ch.reverb_cc_taper

    def _phase_baseline_db(
        self,
        channel_id: str,
        control: str,
        phase: ServicePhase | None = None,
        last_macro: ServicePhase | None = None,
    ) -> float:
        p = self.active_phase if phase is None else phase
        macro = self.last_macro_phase if last_macro is None else last_macro
        if macro == ServicePhase.TRANSITION:
            macro = ServicePhase.SPOKEN_WORD
        ch = self._channels[channel_id].config

        if control == "reverb":
            return float(self.rules.phase_reverb_baselines_db.get(macro, {}).get(channel_id, -24.0))

        if p == ServicePhase.TRANSITION:
            base = float(self.rules.phase_baselines_db[macro][channel_id])
            if ch.role in BAND_ROLES:
                base += float(self.rules.firewall.transition_instrument_trim_db)
            return base
        return float(self.rules.phase_baselines_db[p][channel_id])

    def _next_worship_phase(self) -> ServicePhase:
        """Select upcoming worship phase from Order of Service, defaulting to WORSHIP_LOW."""
        segments = self.plan.order_of_service
        if segments and 0 <= self.service_step_idx < len(segments):
            # Look from current step onward for a worship segment
            for seg in segments[self.service_step_idx :]:
                if seg.expected_phase in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH):
                    return seg.expected_phase
        return ServicePhase.WORSHIP_LOW

    # ------------------------------------------------------------------
    # Bridge hydration, mode, arming, sync & operator controls
    # ------------------------------------------------------------------

    def hydrate_from_bridge(
        self,
        bridge_state: BridgeStatePayload | Mapping[str, Any],
        was_active: bool = False,
        is_full_sync: bool = True,
    ) -> list[BrainEventPayload]:
        """Hydrate fader/reverb positions and sync status from midi_bridge_daemon.

        - Tracks `_last_handled_restored_at` so a bridge disk-restore triggers the
          `ACTIVE -> SHADOW_OBSERVE` + `UNVERIFIED` safety drop exactly once per restart.
        - When `is_full_sync=False` (incremental `bridge_state` updates such as
          `pickup_pending` or `bypass_latched`), only updates `bypass_latched`,
          `human_seq`, and `pickup_pending` without clobbering mid-glide faders or
          `SYNCED` status (unless a brand-new `restored_from_disk_at` is seen).
        - When `bypass_latched` transitions `True -> False`, resets `autonomy_mode`
          from `EMERGENCY_BYPASS` to `SHADOW_OBSERVE`.
        """
        state = (
            bridge_state
            if isinstance(bridge_state, BridgeStatePayload)
            else BridgeStatePayload.model_validate(bridge_state)
        )
        events: list[BrainEventPayload] = []

        was_bypass = self.bypass_latched
        self.bypass_latched = bool(state.bypass_latched)
        if self.bypass_latched:
            self.autonomy_mode = AutonomyMode.EMERGENCY_BYPASS
        elif was_bypass and self.autonomy_mode == AutonomyMode.EMERGENCY_BYPASS:
            # Unlatching bridge bypass drops safely to SHADOW_OBSERVE (never auto-ACTIVE)
            self.autonomy_mode = AutonomyMode.SHADOW_OBSERVE

        for key_str, seq in state.human_seq.items():
            if ":" in key_str:
                ch_id, ctrl = key_str.split(":", 1)
                prev = self._human_seq.get((ch_id, ctrl), 0)
                self._human_seq[(ch_id, ctrl)] = max(prev, int(seq))

        # Refresh pickup_dir for all controls (clearing resolved pickup entries back to 0)
        for ch_id, ch_int in self._channels.items():
            ch_int.vol.pickup_dir = int(state.pickup_pending.get(f"{ch_id}:vol", 0))
            if ch_int.reverb is not None:
                ch_int.reverb.pickup_dir = int(state.pickup_pending.get(f"{ch_id}:reverb", 0))

        new_disk_restore = (state.restored_from_disk_at is not None) and (
            state.restored_from_disk_at != self._last_handled_restored_at
        )
        if new_disk_restore:
            self._last_handled_restored_at = state.restored_from_disk_at

        if is_full_sync or new_disk_restore:
            for ch_id, ch_int in self._channels.items():
                vol_key = f"{ch_id}:vol"
                if vol_key in state.last_sent_cc:
                    cc = int(state.last_sent_cc[vol_key])
                    db = cc_to_db(cc, ch_int.config.cc_taper)
                    if math.isinf(db):
                        db = ch_int.config.hard_min_db
                    ch_int.vol.current_db = db
                    ch_int.vol.target_db = db
                    ch_int.vol.actual_cc = cc
                    ch_int.vol.target_cc = cc
                    self._last_sent_cc[(ch_id, "vol")] = cc
                    ch_int.sync_status = (
                        SyncStatus.UNVERIFIED if state.restored_from_disk_at is not None else SyncStatus.SYNCED
                    )
                else:
                    ch_int.sync_status = SyncStatus.UNVERIFIED

                if ch_int.reverb is not None:
                    rev_key = f"{ch_id}:reverb"
                    if rev_key in state.last_sent_cc:
                        cc = int(state.last_sent_cc[rev_key])
                        db = cc_to_db(cc, ch_int.config.reverb_cc_taper)
                        if math.isinf(db):
                            db = ch_int.config.hard_min_db
                        ch_int.reverb.current_db = db
                        ch_int.reverb.target_db = db
                        ch_int.reverb.actual_cc = cc
                        ch_int.reverb.target_cc = cc
                        self._last_sent_cc[(ch_id, "reverb")] = cc

        if new_disk_restore:
            for ch_int in self._channels.values():
                ch_int.sync_status = SyncStatus.UNVERIFIED
            if was_active or self.autonomy_mode == AutonomyMode.ACTIVE:
                self.autonomy_mode = AutonomyMode.SHADOW_OBSERVE
                evt = self._make_event(
                    BrainCategory.SAFETY,
                    "Bridge restored from disk — dropped to SHADOW_OBSERVE",
                    "All controls marked UNVERIFIED; confirm sync before re-enabling ACTIVE.",
                )
                events.append(evt)
                self._pending_events.append(evt)

        return events

    def set_mode(self, mode: AutonomyMode | str) -> None:
        m = mode if isinstance(mode, AutonomyMode) else AutonomyMode(mode)
        if m == AutonomyMode.EMERGENCY_BYPASS:
            self.bypass_latched = True
            self.autonomy_mode = AutonomyMode.EMERGENCY_BYPASS
        elif m == AutonomyMode.SHADOW_OBSERVE:
            self.bypass_latched = False
            self.autonomy_mode = AutonomyMode.SHADOW_OBSERVE
        elif m == AutonomyMode.ACTIVE:
            if self.bypass_latched:
                raise RuntimeError("Cannot enter ACTIVE while emergency bypass is latched")
            self._watchdog_tripped = False
            self._consecutive_late_ticks = 0
            self.autonomy_mode = AutonomyMode.ACTIVE

    def set_channel_armed(self, channel_id: str, armed: bool) -> None:
        if channel_id not in self._channels:
            raise KeyError(f"Unknown channel_id: {channel_id}")
        self._channels[channel_id].armed = bool(armed)

    def set_armed_channels(self, channel_ids: Iterable[str]) -> None:
        armed_set = set(channel_ids)
        for ch_id, ch_int in self._channels.items():
            ch_int.armed = ch_id in armed_set

    def get_armed_channel_ids(self) -> list[str]:
        return [ch_id for ch_id, ch_int in self._channels.items() if ch_int.armed]

    def confirm_sync(self, channel_id: str) -> None:
        """Confirm channel sync (enables smooth 2.0 dB/s handover glide to target)."""
        if channel_id not in self._channels:
            raise KeyError(f"Unknown channel_id: {channel_id}")
        ch_int = self._channels[channel_id]
        ch_int.sync_status = SyncStatus.SYNCED
        ch_int.vol.handover_gliding = True
        if ch_int.reverb is not None:
            ch_int.reverb.handover_gliding = True

    def on_human_touch(
        self,
        channel_id: str,
        control: str = "vol",
        human_seq: int | None = None,
        now_s: float | None = None,
    ) -> None:
        """Pre-pickup touch event: immediately engages 15s touch lock & freezes agent glides."""
        t = self._resolve_now(now_s)
        _, track = self._get_track(channel_id, control)
        if human_seq is not None:
            self._human_seq[(channel_id, control)] = max(
                self._human_seq.get((channel_id, control), 0), int(human_seq)
            )
        track.touch_lock_until_s = t + float(self.rules.firewall.touch_lock_seconds)
        track.rollback_hold_until_s = 0.0
        track.rollback_target_db = None
        if not track.human_override_active:
            track.frozen_learned_db = self.offset_learner.predict(
                self.active_phase,
                channel_id,
                control,
                track.last_phi,
                fallback_macro_phase=self.last_macro_phase,
            )
        track.target_db = track.current_db
        track.target_cc = track.actual_cc

    def _mean_phi_window(self, channel_id: str, control: str, window_s: float, now_s: float) -> np.ndarray:
        cutoff = now_s - window_s
        phis: list[np.ndarray] = []
        key = (channel_id, control)
        for snap in reversed(self._history):
            if snap.ts_s < cutoff:
                break
            entry = snap.controls.get(key)
            if entry is not None:
                phis.append(entry[2])
        if phis:
            return np.mean(np.stack(phis, axis=0), axis=0)
        _, track = self._get_track(channel_id, control)
        return track.last_phi.copy()

    def on_human_cc(
        self,
        channel_id: str,
        control: str,
        cc: int,
        human_seq: int | None = None,
        now_s: float | None = None,
        freeze_learning: bool = False,
        source: str = "hardware",
    ) -> dict[str, Any]:
        """Handle a human hardware or UI CC move: touch lock, zero-drift offset, 1-sample update."""
        t = self._resolve_now(now_s)
        ch_int, track = self._get_track(channel_id, control)
        taper = self._taper_for(ch_int.config, control)

        cc_clamped = max(0, min(127, int(cc)))
        human_db = cc_to_db(cc_clamped, taper)
        if math.isinf(human_db):
            human_db = ch_int.config.hard_min_db

        if human_seq is not None:
            self._human_seq[(channel_id, control)] = max(
                self._human_seq.get((channel_id, control), 0), int(human_seq)
            )

        ch_int.sync_status = SyncStatus.SYNCED
        track.pickup_dir = 0
        track.handover_gliding = False
        track.rollback_hold_until_s = 0.0
        track.rollback_target_db = None
        track.clear_offset_after_touch_lock = False
        track.in_duck_release = False
        track.was_ducked = False

        # Capture frozen_learned_db BEFORE updating offset_learner weights (anti-double-count invariant!)
        if not track.human_override_active:
            track.frozen_learned_db = self.offset_learner.predict(
                self.active_phase,
                channel_id,
                control,
                track.last_phi,
                fallback_macro_phase=self.last_macro_phase,
            )

        phase_base_db = self._phase_baseline_db(channel_id, control)
        speech_map = {cid: c.speech_open for cid, c in self._channels.items()}
        continuous_rule_sum_db, _, _ = self.rules_manager.evaluate_channel(
            self.active_phase,
            channel_id,
            control,
            speech_open_map=speech_map,
            active_map={},
            fallback_macro_phase=self.last_macro_phase,
        )

        track.current_db = float(human_db)
        track.target_db = float(human_db)
        track.actual_cc = cc_clamped
        track.target_cc = cc_clamped
        self._last_sent_cc[(channel_id, control)] = cc_clamped

        track.touch_lock_until_s = t + float(self.rules.firewall.touch_lock_seconds)
        track.human_override_active = True
        track.human_anchor_db = float(human_db)
        track.human_session_offset_db = float(
            human_db - (phase_base_db + continuous_rule_sum_db + track.frozen_learned_db)
        )

        # Feed exact 1-sample correction into OffsetLearner using 2-second pre-move mean phi
        phi_bar = self._mean_phi_window(channel_id, control, 2.0, t)
        target_offset_db = float(human_db - (phase_base_db + continuous_rule_sum_db))
        sample = self.offset_learner.record_human_correction(
            phase=self.active_phase,
            channel_id=channel_id,
            control=control,
            phi=phi_bar,
            target_offset_db=target_offset_db,
            freeze=freeze_learning,
            fallback_macro_phase=self.last_macro_phase,
        )
        return sample

    def release_human_offset(
        self,
        channel_id: str,
        control: str = "vol",
        now_s: float | None = None,
    ) -> None:
        """Manually release human offset & touch lock; channel glides smoothly at 4 dB/s."""
        self._resolve_now(now_s)
        _, track = self._get_track(channel_id, control)
        track.touch_lock_until_s = 0.0
        track.human_override_active = False
        track.human_anchor_db = None
        track.human_session_offset_db = 0.0
        track.frozen_learned_db = 0.0
        track.clear_offset_after_touch_lock = False

    def _clear_human_offsets(self, now_s: float) -> None:
        """Clear human overrides immediately if touch lock expired, else schedule upon expiry."""
        for ch_int in self._channels.values():
            for track in (ch_int.vol, ch_int.reverb):
                if track is None or not track.human_override_active:
                    continue
                if now_s >= track.touch_lock_until_s:
                    track.human_override_active = False
                    track.human_anchor_db = None
                    track.human_session_offset_db = 0.0
                    track.frozen_learned_db = 0.0
                    track.clear_offset_after_touch_lock = False
                else:
                    track.clear_offset_after_touch_lock = True

    def on_segment_advance(self, step_idx: int, now_s: float | None = None) -> None:
        """Advance Order-of-Service segment: clear human offsets & pre-stage worship instruments."""
        t = self._resolve_now(now_s)
        self.service_step_idx = max(0, int(step_idx))
        self._clear_human_offsets(t)

        segments = self.plan.order_of_service
        if segments and 0 <= self.service_step_idx < len(segments):
            expected = segments[self.service_step_idx].expected_phase
            if expected in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH) and self.active_phase not in (
                ServicePhase.WORSHIP_LOW,
                ServicePhase.WORSHIP_HIGH,
            ):
                self._prestage_worship = True
            else:
                self._prestage_worship = False
        else:
            self._prestage_worship = False

    def _handle_phase_change(
        self,
        new_phase: ServicePhase,
        new_last_macro: ServicePhase | None,
        now_s: float,
    ) -> None:
        old_phase = self.active_phase
        old_macro = self.last_macro_phase

        if new_phase == old_phase and (new_last_macro is None or new_last_macro == old_macro):
            return

        # Save pre-transition snapshot for 2nd-press thumbs_down rollback
        if self._history:
            self._pre_transition_snapshot = self._history[-1]

        self.active_phase = new_phase
        if new_phase in MACRO_PHASES:
            self.last_macro_phase = new_phase
        elif new_last_macro is not None and new_last_macro in MACRO_PHASES:
            self.last_macro_phase = new_last_macro

        # Macro phase transition clears human offsets smoothly (at 4 dB/s)
        if new_phase in MACRO_PHASES and new_phase != old_macro:
            self._clear_human_offsets(now_s)

        # Entering worship completes any pre-staging or instrument_open reflex cleanly
        if new_phase in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH):
            self._prestage_worship = False
            self._instrument_open_active = False
            for ch_int in self._channels.values():
                ch_int.instrument_open = False
        elif new_phase == ServicePhase.PRAYER_UNDERSCORE:
            self._instrument_open_active = False
            for ch_int in self._channels.values():
                ch_int.instrument_open = False

    # ------------------------------------------------------------------
    # Thumbs Up / Smart Thumbs Down Rollback
    # ------------------------------------------------------------------

    def _find_snapshot_at_or_before(self, target_ts_s: float) -> _HistorySnapshot | None:
        if not self._history:
            return None
        best = self._history[0]
        for snap in self._history:
            if snap.ts_s <= target_ts_s:
                best = snap
            else:
                break
        return best

    def handle_thumbs_up(
        self,
        now_s: float | None = None,
        freeze_learning: bool = False,
    ) -> tuple[list[dict[str, Any]], list[BrainEventPayload]]:
        """Record 1-sample positive reinforcement (alpha=1.0) per armed control."""
        t = self._resolve_now(now_s)
        speech_map = {cid: c.speech_open for cid, c in self._channels.items()}
        armed_targets: list[dict[str, Any]] = []

        for ch_id, ch_int in self._channels.items():
            if not ch_int.armed:
                continue
            for ctrl_name, track in (("vol", ch_int.vol), ("reverb", ch_int.reverb)):
                if track is None:
                    continue
                phase_base_db = self._phase_baseline_db(ch_id, ctrl_name)
                cont_sum, _, _ = self.rules_manager.evaluate_channel(
                    self.active_phase,
                    ch_id,
                    ctrl_name,
                    speech_open_map=speech_map,
                    active_map={},
                    fallback_macro_phase=self.last_macro_phase,
                )
                eff_offset = track.current_db - (phase_base_db + cont_sum)
                phi_bar = self._mean_phi_window(ch_id, ctrl_name, 5.0, t)
                armed_targets.append(
                    {
                        "channel_id": ch_id,
                        "control": ctrl_name,
                        "phi": phi_bar,
                        "target_offset_db": eff_offset,
                    }
                )

        samples = self.offset_learner.record_thumbs_up(
            self.active_phase,
            armed_targets,
            freeze=freeze_learning,
            fallback_macro_phase=self.last_macro_phase,
        )
        evt = self._make_event(
            BrainCategory.RLHF,
            f"Thumbs Up reinforced {len(samples)} armed control(s)",
            "Queued for post-service review (Freeze Learning active)."
            if freeze_learning
            else "Applied 1-sample Ridge reinforcement (alpha=1.0).",
        )
        return samples, [evt]

    def handle_thumbs_down(
        self,
        now_s: float | None = None,
        freeze_learning: bool = False,
    ) -> tuple[list[dict[str, Any]], list[BrainEventPayload]]:
        """Smart rollback: T-5s on first press, T-15s/pre-transition on second press within 5s.

        Skips touch-locked controls, holds rolled-back levels for 10s, and records 1-sample
        updates (alpha=1.5) into OffsetLearner (or queues if freeze_learning=True).
        """
        t = self._resolve_now(now_s)
        is_second_press = self._last_thumbs_down_s is not None and (t - self._last_thumbs_down_s) <= 5.0

        if is_second_press:
            snap_15 = self._find_snapshot_at_or_before(t - 15.0)
            if self._pre_transition_snapshot is not None and (
                snap_15 is None or self._pre_transition_snapshot.ts_s >= snap_15.ts_s
            ):
                target_snap = self._pre_transition_snapshot
                lookback_label = "pre-transition"
            else:
                target_snap = snap_15
                lookback_label = "T-15s"
            self._last_thumbs_down_s = None
        else:
            target_snap = self._find_snapshot_at_or_before(t - 5.0)
            lookback_label = "T-5s"
            self._last_thumbs_down_s = t

        rolled_back_targets: list[dict[str, Any]] = []
        skipped_locked = 0

        if target_snap is not None:
            for ch_id, ch_int in self._channels.items():
                if not ch_int.armed:
                    continue
                for ctrl_name, track in (("vol", ch_int.vol), ("reverb", ch_int.reverb)):
                    if track is None:
                        continue
                    # Skip touch-locked controls!
                    if t < track.touch_lock_until_s:
                        skipped_locked += 1
                        continue
                    snap_entry = target_snap.controls.get((ch_id, ctrl_name))
                    if snap_entry is None:
                        continue
                    snap_db, snap_offset_db, snap_phi = snap_entry
                    track.rollback_hold_until_s = t + 10.0
                    track.rollback_target_db = float(snap_db)
                    rolled_back_targets.append(
                        {
                            "channel_id": ch_id,
                            "control": ctrl_name,
                            "phi": snap_phi,
                            "target_offset_db": snap_offset_db,
                        }
                    )

        samples = self.offset_learner.record_thumbs_down(
            self.active_phase,
            rolled_back_targets,
            freeze=freeze_learning,
            fallback_macro_phase=self.last_macro_phase,
        )
        evt = self._make_event(
            BrainCategory.RLHF,
            f"Thumbs Down rollback ({lookback_label}) on {len(samples)} control(s)",
            f"Holding 10s; skipped {skipped_locked} touch-locked control(s).",
        )
        return samples, [evt]

    # ------------------------------------------------------------------
    # Bleed-Rejected Escalating Anti-Feedback Detector
    # ------------------------------------------------------------------

    def _compute_fft_prominence(self, spec: np.ndarray) -> np.ndarray:
        arr = np.asarray(spec, dtype=np.float64)
        if arr.size < 25:
            return np.zeros_like(arr)
        floor = median_filter(arr, size=21, mode="nearest")
        return arr - floor

    def _update_feedback_guards(
        self,
        fft_mag_db: Mapping[str, np.ndarray],
        dt_s: float,
        now_s: float,
        events: list[BrainEventPayload],
        sung_vocals: Mapping[str, bool] | None = None,
    ) -> None:
        fw = self.rules.firewall
        inst_prom_cache: dict[str, np.ndarray] = {}

        def _is_instrument_bleed(peak_bin: int, mic_prom: float) -> bool:
            for b_id, b_int in self._channels.items():
                if b_int.config.role not in BAND_ROLES:
                    continue
                b_spec = fft_mag_db.get(b_id)
                if b_spec is None or len(b_spec) == 0:
                    continue
                if b_id not in inst_prom_cache:
                    inst_prom_cache[b_id] = self._compute_fft_prominence(b_spec)
                b_prom = inst_prom_cache[b_id]
                if peak_bin >= b_prom.size:
                    continue
                lo = max(0, peak_bin - 2)
                hi = min(b_prom.size, peak_bin + 3)
                b_near = float(np.max(b_prom[lo:hi]))
                # Reject if instrument tap has matching bin within 6 dB prominence
                if b_near >= mic_prom - 6.0 and b_near > 8.0:
                    return True
            return False

        for ch_id, ch_int in self._channels.items():
            if not ch_int.config.feedback_guard:
                continue

            spec_raw = fft_mag_db.get(ch_id)
            detected_bin: int | None = None
            detected_mag: float = -120.0
            detected_prom: float = 0.0
            is_harmonic_vocal = False

            if spec_raw is not None:
                spec = np.asarray(spec_raw, dtype=np.float64)
                if spec.size >= 25:
                    prom = self._compute_fft_prominence(spec)
                    # Ignore DC / sub-140Hz bins (<3) and top edge
                    search_prom = prom[3 : spec.size - 3]
                    cand_indices = np.flatnonzero(search_prom > fw.feedback_peak_above_floor_db)
                    if cand_indices.size > 0:
                        # Check strongest candidates first so synth bleed at one frequency
                        # cannot mask a genuine feedback howl at another frequency
                        order = np.argsort(search_prom[cand_indices])[::-1]
                        for idx in order[:8]:
                            k = int(cand_indices[idx]) + 3
                            p_val = float(prom[k])
                            mag_val = float(spec[k])
                            # Ignore sub-floor ambience/noise peaks; real PA feedback is audible energy
                            min_fb_mag_db = ch_int.config.nominal_rms_db - 24.0
                            if mag_val < min_fb_mag_db:
                                continue
                            # Narrowband high-Q check: real feedback drops steeply by ±3 bins (~140 Hz)
                            skirt_avg = 0.5 * (float(spec[k - 3]) + float(spec[k + 3]))
                            if (mag_val - skirt_avg) < 12.0:
                                continue
                            if not _is_instrument_bleed(k, p_val):
                                detected_bin = k
                                detected_mag = mag_val
                                detected_prom = p_val
                                # Check for multi-harmonic sung vocal / instrument overtone structure
                                harmonic_hits = 0
                                has_octave_overtone = False
                                for mult_num, mult_den in (
                                    (1, 4),
                                    (1, 3),
                                    (1, 2),
                                    (2, 3),
                                    (3, 4),
                                    (4, 3),
                                    (3, 2),
                                    (2, 1),
                                    (3, 1),
                                ):
                                    hk = round(k * mult_num / mult_den)
                                    if 3 <= hk < spec.size - 3 and abs(hk - k) >= 3:
                                        lo_h = max(0, hk - 1)
                                        hi_h = min(spec.size, hk + 2)
                                        h_prom = float(np.max(prom[lo_h:hi_h]))
                                        if h_prom > 8.5:
                                            harmonic_hits += 1
                                            if (mult_num, mult_den) in ((2, 1), (1, 2), (3, 1), (1, 3)):
                                                has_octave_overtone = True
                                # Pure PA feedback is a single sinusoid (0 harmonics); sustained
                                # vocal fundamentals (<=1.05 kHz, bin <=22) with a clear octave/fifth
                                # harmonic or any peak with >=2 harmonics are musical vocals.
                                is_harmonic_vocal = (harmonic_hits >= 2) or (
                                    k <= 22 and has_octave_overtone and harmonic_hits >= 1
                                )
                                break

            if detected_bin is not None:
                ch_int.fb_clear_s = 0.0
                if ch_int.fb_candidate_bin is not None and abs(detected_bin - ch_int.fb_candidate_bin) <= 1:
                    ch_int.fb_hold_s += dt_s
                    if ch_int.fb_hold_s <= FEEDBACK_ONSET_SETTLE_S + 1e-9:
                        # Still inside the onset: track the attack up to its sustain level
                        ch_int.fb_initial_mag_db = max(ch_int.fb_initial_mag_db, detected_mag)
                        ch_int.fb_initial_prom_db = max(ch_int.fb_initial_prom_db, detected_prom)
                else:
                    ch_int.fb_candidate_bin = detected_bin
                    ch_int.fb_hold_s = dt_s
                    ch_int.fb_initial_mag_db = detected_mag
                    ch_int.fb_initial_prom_db = detected_prom
                    ch_int.fb_mag_hist.clear()
                    ch_int.fb_prior_peak_db = -120.0

                hold_target_s = fw.feedback_hold_ms / 1000.0
                ch_int.fb_mag_hist.append((ch_int.fb_hold_s, detected_mag))
                window_start_s = max(FEEDBACK_ONSET_SETTLE_S, ch_int.fb_hold_s - hold_target_s)
                while ch_int.fb_mag_hist and ch_int.fb_mag_hist[0][0] < window_start_s - 1e-9:
                    popped_hold_s, popped_mag = ch_int.fb_mag_hist.popleft()
                    if popped_hold_s > FEEDBACK_ONSET_SETTLE_S + 1e-9:
                        ch_int.fb_prior_peak_db = max(ch_int.fb_prior_peak_db, popped_mag)

                # Steady held vocal fundamentals (<=1.12 kHz) on an actively singing channel
                # require actual acoustic growth to trigger; pure high-frequency howls (>1.12 kHz)
                # or open non-singing mics lock on immediately.
                is_steady_sung_fundamental = bool((sung_vocals or {}).get(ch_id, False)) and (
                    detected_bin <= 24
                )
                if is_steady_sung_fundamental:
                    # Prominence follows the band floor under a sung note, so only the mic's own
                    # magnitude counts, and it must be ringing up across the trailing window.
                    window_mags = np.fromiter((m for _, m in ch_int.fb_mag_hist), dtype=np.float64)
                    edge = FEEDBACK_RINGUP_EDGE_TICKS
                    if window_mags.size >= 2 * edge:
                        start_db = float(np.mean(window_mags[:edge]))
                        end_db = float(np.mean(window_mags[-edge:]))
                        ringing_up = float(np.min(window_mags)) >= start_db - FEEDBACK_RINGUP_MAX_DIP_DB
                        ref_db = max(start_db, ch_int.fb_prior_peak_db, ch_int.fb_initial_mag_db)
                        meets_growth = ringing_up and (end_db - ref_db) >= fw.feedback_growth_db
                    else:
                        meets_growth = False
                    meets_growth = meets_growth and (not is_harmonic_vocal)
                else:
                    growth_mag = detected_mag - ch_int.fb_initial_mag_db
                    growth_prom = detected_prom - ch_int.fb_initial_prom_db
                    meets_growth = (
                        (growth_mag >= fw.feedback_growth_db) or (growth_prom >= fw.feedback_growth_db)
                    ) and (not is_harmonic_vocal)
                meets_lockon = (
                    (detected_prom > 18.0)
                    and (not is_harmonic_vocal)
                    and (not is_steady_sung_fundamental)
                )

                if ch_int.fb_hold_s >= hold_target_s - 1e-9 and (meets_growth or meets_lockon):
                    touch_override_note = (
                        " (OVERRIDING TOUCH LOCK)" if now_s < ch_int.vol.touch_lock_until_s else ""
                    )
                    if ch_int.feedback_ceiling_db is None or ch_int.fb_last_cut_s is None:
                        ch_int.fb_total_cut_db = float(fw.feedback_cut_step_db)
                        ch_int.feedback_ceiling_db = max(
                            ch_int.config.hard_min_db,
                            ch_int.vol.current_db + float(fw.feedback_cut_step_db),
                        )
                        ch_int.fb_last_cut_s = now_s
                        events.append(
                            self._make_event(
                                BrainCategory.SAFETY,
                                f"ANTI-FEEDBACK CUT ({ch_int.fb_total_cut_db:.1f} dB) on {ch_id}{touch_override_note}",
                                f"Bin {detected_bin}, prominence +{detected_prom:.1f} dB; ceiling={ch_int.feedback_ceiling_db:.1f} dB.",
                            )
                        )
                    elif (now_s - ch_int.fb_last_cut_s) >= 0.400 - 1e-9 and (
                        ch_int.fb_total_cut_db > fw.feedback_max_cut_db + 1e-9
                    ):
                        ch_int.fb_total_cut_db = max(
                            float(fw.feedback_max_cut_db),
                            ch_int.fb_total_cut_db + float(fw.feedback_cut_step_db),
                        )
                        ch_int.feedback_ceiling_db = max(
                            ch_int.config.hard_min_db,
                            ch_int.feedback_ceiling_db + float(fw.feedback_cut_step_db),
                        )
                        ch_int.fb_last_cut_s = now_s
                        events.append(
                            self._make_event(
                                BrainCategory.SAFETY,
                                f"ANTI-FEEDBACK ESCALATED ({ch_int.fb_total_cut_db:.1f} dB) on {ch_id}{touch_override_note}",
                                f"Sustained howl bin {detected_bin}; ceiling={ch_int.feedback_ceiling_db:.1f} dB.",
                            )
                        )
            else:
                ch_int.fb_candidate_bin = None
                ch_int.fb_hold_s = 0.0
                ch_int.fb_mag_hist.clear()
                ch_int.fb_prior_peak_db = -120.0
                if ch_int.feedback_ceiling_db is not None:
                    ch_int.fb_clear_s += dt_s
                    if ch_int.fb_clear_s >= fw.feedback_release_delay_s - 1e-9:
                        ch_int.feedback_ceiling_db += float(fw.feedback_release_rate_db_per_sec) * dt_s
                        upper_ref = ch_int.config.hard_max_db
                        if ch_int.vol.human_override_active and ch_int.vol.human_anchor_db is not None:
                            upper_ref = max(upper_ref, ch_int.vol.human_anchor_db)
                        if ch_int.feedback_ceiling_db >= upper_ref:
                            ch_int.feedback_ceiling_db = None
                            ch_int.fb_total_cut_db = 0.0
                            ch_int.fb_last_cut_s = None
                            ch_int.fb_clear_s = 0.0

    # ------------------------------------------------------------------
    # Tier-1 Speech-Open, Instrument-Open, and Ducking Detectors
    # ------------------------------------------------------------------

    def _update_speech_and_duck(
        self,
        vad_probs: Mapping[str, float],
        snr_map: Mapping[str, float],
        dt_s: float,
        events: list[BrainEventPayload],
    ) -> bool:
        """Update speech_open reflex and return whether auto-ducking is currently triggered."""
        fw = self.rules.firewall
        in_worship = self.active_phase in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH)
        speech_hold_req_s = (
            fw.speech_open_hold_worship_ms / 1000.0
            if in_worship
            else fw.speech_open_hold_spoken_ms / 1000.0
        )

        any_spoken_primary_open = False
        any_spoken_duck_trigger = False

        for ch_id, ch_int in self._channels.items():
            vad = float(vad_probs.get(ch_id, 0.0))
            snr_raw = float(snr_map.get(ch_id, -60.0))
            # Fast-attack, syllable-length (~150ms) release envelope on SNR so unvoiced
            # consonants (t/k/p/s) don't reset speech_open while steady band bleed stays ~0 dB.
            if snr_raw >= ch_int.snr_env_db:
                ch_int.snr_env_db = snr_raw
            else:
                decay_alpha = math.exp(-dt_s / 0.15)
                ch_int.snr_env_db = decay_alpha * ch_int.snr_env_db + (1.0 - decay_alpha) * snr_raw
            snr = ch_int.snr_env_db

            if ch_int.config.role == ChannelRole.SPOKEN_PRIMARY:
                if vad >= fw.speech_open_vad_min and snr >= fw.speech_open_snr_db:
                    ch_int.speech_hold_s += dt_s
                    ch_int.speech_silence_s = 0.0
                    if not ch_int.speech_open and ch_int.speech_hold_s >= speech_hold_req_s - 1e-9:
                        ch_int.speech_open = True
                        events.append(
                            self._make_event(
                                BrainCategory.ACTION,
                                f"Speech-open reflex engaged on {ch_id}",
                                f"VAD={vad:.2f}, SNR=+{snr:.1f} dB in {self.active_phase.value}.",
                            )
                        )
                else:
                    # Consonant/micro-pause tolerant leaky decay: preserves real spoken phrases
                    # across 40-100ms unvoiced consonants while rejecting brief bleed/clicks.
                    if vad >= 0.45 and snr >= (fw.speech_open_snr_db - 4.0):
                        ch_int.speech_hold_s = max(0.0, ch_int.speech_hold_s - 0.5 * dt_s)
                    else:
                        ch_int.speech_hold_s = max(0.0, ch_int.speech_hold_s - 2.0 * dt_s)
                    if ch_int.speech_open:
                        ch_int.speech_silence_s += dt_s
                        if ch_int.speech_silence_s >= fw.speech_open_release_silence_s - 1e-9:
                            ch_int.speech_open = False
                            ch_int.speech_silence_s = 0.0

                if ch_int.speech_open:
                    any_spoken_primary_open = True

            # Ducking trigger in spoken / prayer / transition (Ch 1 or Ch 2 with vad_enabled)
            if ch_int.config.role in (ChannelRole.SPOKEN_PRIMARY, ChannelRole.VOCAL_LEAD) and (
                ch_int.config.vad_enabled or ch_int.config.role == ChannelRole.SPOKEN_PRIMARY
            ):
                if vad >= 0.65 and snr >= fw.duck_snr_spoken_db:
                    ch_int.duck_trigger_hold_s += dt_s
                    if ch_int.duck_trigger_hold_s >= 0.150 - 1e-9:
                        ch_int.duck_trigger_active = True
                else:
                    if vad >= 0.40 and snr >= (fw.duck_snr_spoken_db - 2.0):
                        ch_int.duck_trigger_hold_s = max(0.0, ch_int.duck_trigger_hold_s - 0.5 * dt_s)
                    else:
                        ch_int.duck_trigger_hold_s = 0.0
                        ch_int.duck_trigger_active = False

                if ch_int.duck_trigger_active or ch_int.speech_open:
                    any_spoken_duck_trigger = True

        if in_worship:
            return any_spoken_primary_open
        return any_spoken_duck_trigger

    def _update_instrument_open(
        self,
        norm_vband_db: Mapping[str, float],
        dt_s: float,
        now_s: float,
        events: list[BrainEventPayload],
    ) -> None:
        fw = self.rules.firewall
        in_worship = self.active_phase in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH)
        in_prayer = self.active_phase == ServicePhase.PRAYER_UNDERSCORE

        # Always track per-instrument silent -> active onsets
        for ch_id, ch_int in self._channels.items():
            if ch_int.config.role not in BAND_ROLES:
                continue
            level = float(norm_vband_db.get(ch_id, -60.0))

            if level <= fw.instrument_open_silent_norm_db:
                ch_int.silent_duration_s += dt_s
                ch_int.active_onset_hold_s = 0.0
                ch_int.mid_level_s = 0.0
                if ch_int.silent_duration_s >= fw.instrument_open_silent_min_s - 1e-9:
                    ch_int.was_sustained_silent = True
            elif level >= fw.instrument_open_active_norm_db:
                ch_int.silent_duration_s = 0.0
                ch_int.mid_level_s = 0.0
                if ch_int.was_sustained_silent:
                    ch_int.active_onset_hold_s += dt_s
                    if ch_int.active_onset_hold_s >= (fw.instrument_open_hold_ms / 1000.0) - 1e-9:
                        ch_int.last_onset_s = now_s
                        ch_int.was_sustained_silent = False
                        ch_int.active_onset_hold_s = 0.0
            else:
                # Intermediate zone (-30 < level < -12)
                ch_int.silent_duration_s = 0.0
                ch_int.active_onset_hold_s = 0.0
                ch_int.mid_level_s += dt_s
                if ch_int.mid_level_s > 1.0:
                    ch_int.was_sustained_silent = False

        # Never trigger in PRAYER_UNDERSCORE or when already in WORSHIP
        if in_prayer or in_worship:
            self._instrument_open_active = False
            for ch_int in self._channels.values():
                ch_int.instrument_open = False
            return

        # Check auto-release timeout (8.0 s) if currently active
        if self._instrument_open_active:
            if (now_s - self._instrument_open_started_at_s) >= fw.instrument_open_max_hold_s - 1e-9:
                self._instrument_open_active = False
                for ch_int in self._channels.values():
                    ch_int.instrument_open = False
                    ch_int.last_onset_s = None
                events.append(
                    self._make_event(
                        BrainCategory.ACTION,
                        "Instrument-open reflex auto-released after 8s guard",
                        "Macro phase did not transition to WORSHIP; gliding back at 4 dB/s.",
                    )
                )
                return

            # Release individual channels if silent >= 3s; deactivate all if none remain open
            any_open = False
            for ch_id, ch_int in self._channels.items():
                if ch_int.config.role not in BAND_ROLES:
                    continue
                level = float(norm_vband_db.get(ch_id, -60.0))
                if ch_int.silent_duration_s >= fw.instrument_open_silent_min_s - 1e-9:
                    ch_int.instrument_open = False
                elif level >= fw.instrument_open_active_norm_db:
                    ch_int.instrument_open = True
                if ch_int.instrument_open:
                    any_open = True
            if not any_open:
                self._instrument_open_active = False
            return

        # Evaluate Song-Start Ensemble Gate within 1.0 s window
        window_s = fw.instrument_open_ensemble_window_s
        recent_channels: list[_ChannelInternal] = []
        recent_roles: set[ChannelRole] = set()
        drums_onset = False

        for ch_int in self._channels.values():
            if ch_int.config.role not in BAND_ROLES:
                continue
            if ch_int.last_onset_s is not None and (now_s - ch_int.last_onset_s) <= window_s + 1e-9:
                recent_channels.append(ch_int)
                recent_roles.add(ch_int.config.role)
                if ch_int.config.role == ChannelRole.INSTRUMENT_BAND:
                    drums_onset = True

        if drums_onset or len(recent_roles) >= 2:
            self._instrument_open_active = True
            self._instrument_open_started_at_s = now_s
            for ch_id, ch_int in self._channels.items():
                if ch_int.config.role not in BAND_ROLES:
                    continue
                level = float(norm_vband_db.get(ch_id, -60.0))
                if ch_int in recent_channels or level >= fw.instrument_open_active_norm_db:
                    ch_int.instrument_open = True
                ch_int.last_onset_s = None
            reason = "drums onset" if drums_onset else f"{len(recent_roles)} instrument roles"
            events.append(
                self._make_event(
                    BrainCategory.ACTION,
                    f"Instrument-open reflex engaged ({reason})",
                    f"Opening active instruments at {fw.slew_instrument_open_up_db_per_sec:.0f} dB/s.",
                )
            )

    # ------------------------------------------------------------------
    # Main 50 Hz Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        metrics: AudioFrameMetrics | Any,
        vad_probs: Mapping[str, float],
        sung_vocals: Mapping[str, bool] | None = None,
        active_phase: ServicePhase | None = None,
        last_macro_phase: ServicePhase | None = None,
        dt_s: float = 0.02,
        now_s: float | None = None,
        tick_elapsed_ms: float | None = None,
        audio_stale_s: float = 0.0,
    ) -> FirewallOutput:
        """Execute one deterministic 50 Hz firewall tick."""
        t = self._resolve_now(now_s if now_s is not None else self._clock_s + dt_s)
        fw = self.rules.firewall
        events: list[BrainEventPayload] = []
        if self._pending_events:
            events.extend(self._pending_events)
            self._pending_events.clear()

        # Phase transition bookkeeping
        if active_phase is not None:
            self._handle_phase_change(active_phase, last_macro_phase, t)

        # 1. Watchdog check
        if tick_elapsed_ms is not None and tick_elapsed_ms > fw.watchdog_max_tick_ms:
            self._consecutive_late_ticks += 1
        else:
            self._consecutive_late_ticks = 0

        watchdog_now = (
            self._consecutive_late_ticks >= fw.watchdog_max_consecutive_late
            or audio_stale_s > 0.5
        )
        if watchdog_now:
            self._watchdog_tripped = True
            if self.autonomy_mode == AutonomyMode.ACTIVE:
                self.autonomy_mode = AutonomyMode.SHADOW_OBSERVE
            events.append(
                self._make_event(
                    BrainCategory.SAFETY,
                    "WATCHDOG_DROP_TO_SHADOW",
                    f"Late ticks={self._consecutive_late_ticks}, audio_stale_s={audio_stale_s:.3f}. "
                    "Agent writes frozen; manual bridge passthrough unaffected.",
                )
            )
            return FirewallOutput(
                agent_writes=[],
                channels_state=self.snapshot_channels_state(t),
                events=events,
                watchdog_tripped=True,
                autonomy_mode=self.autonomy_mode,
            )

        # Extract duck-typed audio telemetry
        norm_vband_db: Mapping[str, float] = getattr(metrics, "norm_vband_db", {}) or {}
        snr_map: Mapping[str, float] = getattr(metrics, "speech_to_band_snr_db", {}) or {}
        fft_mag_db: Mapping[str, np.ndarray] = getattr(metrics, "fft_mag_db", {}) or {}
        target_lufs = float(self.rules.phase_target_lufs[self.last_macro_phase])
        master_lufs_st = float(getattr(metrics, "master_lufs_short_term", target_lufs))
        lufs_err_norm = (
            0.0 if (math.isinf(master_lufs_st) or master_lufs_st < -70.0) else (master_lufs_st - target_lufs) / 12.0
        )

        # Expire deferred offset clears once touch lock expires
        for ch_int in self._channels.values():
            for track in (ch_int.vol, ch_int.reverb):
                if track is None:
                    continue
                if track.clear_offset_after_touch_lock and t >= track.touch_lock_until_s:
                    track.human_override_active = False
                    track.human_anchor_db = None
                    track.human_session_offset_db = 0.0
                    track.frozen_learned_db = 0.0
                    track.clear_offset_after_touch_lock = False
                if track.rollback_hold_until_s > 0.0 and t >= track.rollback_hold_until_s:
                    track.rollback_hold_until_s = 0.0
                    track.rollback_target_db = None

        # 2. Update reflexes & anti-feedback
        duck_active = self._update_speech_and_duck(vad_probs, snr_map, dt_s, events)
        self._update_instrument_open(norm_vband_db, dt_s, t, events)
        self._update_feedback_guards(fft_mag_db, dt_s, t, events, sung_vocals=sung_vocals)

        any_spoken_open = any(
            c.speech_open for c in self._channels.values() if c.config.role == ChannelRole.SPOKEN_PRIMARY
        )
        speech_open_map = {cid: c.speech_open for cid, c in self._channels.items()}
        active_map = {
            cid: (
                float(norm_vband_db.get(cid, -60.0)) >= -18.0
                or bool((sung_vocals or {}).get(cid, False))
                or float(vad_probs.get(cid, 0.0)) >= 0.5
            )
            for cid in self._channels
        }

        # Precompute vocal_to_band_ratio_norm for 5D context vectors
        max_vox_norm = max(
            (
                float(norm_vband_db.get(c.config.id, -60.0))
                for c in self._channels.values()
                if c.config.role in (ChannelRole.SPOKEN_PRIMARY, ChannelRole.VOCAL_LEAD, ChannelRole.VOCAL_BG)
            ),
            default=-60.0,
        )
        max_band_norm = max(
            (
                float(norm_vband_db.get(c.config.id, -60.0))
                for c in self._channels.values()
                if c.config.role in BAND_ROLES
            ),
            default=-60.0,
        )
        vocal_to_band_ratio_norm = (max_vox_norm - max_band_norm) / 24.0

        duck_target_roles = set(fw.duck_targets_by_phase.get(self.active_phase, []))
        can_emit_writes = (self.autonomy_mode == AutonomyMode.ACTIVE) and (not self.bypass_latched)
        agent_writes: list[AgentCCWrite] = []
        history_controls: dict[tuple[str, str], tuple[float, float, np.ndarray]] = {}

        # 3. Target assembly, 5-path slew limiting, clamps & MIDI write emission
        for ch_id, ch_int in self._channels.items():
            ch = ch_int.config
            nv_db = float(norm_vband_db.get(ch_id, -60.0))
            is_ch_active = active_map[ch_id]
            phi = build_context_vector(nv_db, vocal_to_band_ratio_norm, is_ch_active, lufs_err_norm)

            # --- Volume Control ---
            vol_track = ch_int.vol
            vol_track.last_phi = phi
            vol_touch_locked = t < vol_track.touch_lock_until_s
            vol_base_phase_db = self._phase_baseline_db(ch_id, "vol")

            cont_vol_db, cond_vol_db, cond_mute = self.rules_manager.evaluate_channel(
                self.active_phase,
                ch_id,
                "vol",
                speech_open_map=speech_open_map,
                active_map=active_map,
                fallback_macro_phase=self.last_macro_phase,
            )

            # Suspend conditional rules while human override is active!
            if vol_track.human_override_active:
                cond_vol_db = 0.0
                cond_mute = False
                eff_learned_vol = vol_track.frozen_learned_db
            else:
                eff_learned_vol = self.offset_learner.predict(
                    self.active_phase,
                    ch_id,
                    "vol",
                    phi,
                    fallback_macro_phase=self.last_macro_phase,
                )

            base_target_vol = (
                vol_base_phase_db
                + cont_vol_db
                + eff_learned_vol
                + vol_track.human_session_offset_db
                + cond_vol_db
            )

            # Determine active reflex / rollback / prestage modifiers
            speech_open_driving = False
            inst_open_driving = False
            is_ducked = False
            desired_vol = base_target_vol

            if vol_track.rollback_target_db is not None and t < vol_track.rollback_hold_until_s:
                desired_vol = vol_track.rollback_target_db
            elif ch.role == ChannelRole.SPOKEN_PRIMARY and ch_int.speech_open:
                spoken_target = (
                    float(self.rules.phase_baselines_db[ServicePhase.SPOKEN_WORD][ch_id])
                    + cont_vol_db
                    + eff_learned_vol
                    + vol_track.human_session_offset_db
                )
                desired_vol = max(base_target_vol, spoken_target)
                speech_open_driving = desired_vol > vol_track.current_db + 1e-6
            elif ch.role in BAND_ROLES and ch_int.instrument_open:
                worship_phase = self._next_worship_phase()
                worship_target = float(self.rules.phase_baselines_db[worship_phase][ch_id])
                if any_spoken_open:
                    duck_d = self.rules_manager.evaluate_duck_depth_db(
                        self.active_phase, ch_id, fw.duck_depth_db, self.last_macro_phase
                    )
                    worship_target = worship_target + duck_d
                desired_vol = worship_target
                inst_open_driving = True
            elif ch.role in BAND_ROLES and self._prestage_worship:
                desired_vol = float(self.rules.phase_baselines_db[ServicePhase.WORSHIP_LOW][ch_id])
            elif duck_active and ch.role in duck_target_roles:
                duck_d = self.rules_manager.evaluate_duck_depth_db(
                    self.active_phase, ch_id, fw.duck_depth_db, self.last_macro_phase
                )
                desired_vol = base_target_vol + duck_d
                is_ducked = True

            # Track duck release transition for 8 dB/s up-slew
            if is_ducked:
                vol_track.was_ducked = True
                vol_track.in_duck_release = False
            elif vol_track.was_ducked:
                vol_track.was_ducked = False
                vol_track.in_duck_release = True

            # Human-respecting clamps (hard_min_db .. max(hard_max_db, human_anchor_db))
            vol_upper = ch.hard_max_db
            if vol_track.human_override_active and vol_track.human_anchor_db is not None:
                vol_upper = max(vol_upper, vol_track.human_anchor_db)
            desired_vol = float(np.clip(desired_vol, ch.hard_min_db, vol_upper))

            # Anti-feedback ceiling ALWAYS wins (even over touch lock & human_anchor_db!)
            fb_cutting = False
            if ch_int.feedback_ceiling_db is not None:
                desired_vol = min(desired_vol, ch_int.feedback_ceiling_db)
                if vol_track.current_db > ch_int.feedback_ceiling_db + 1e-6:
                    fb_cutting = True

            # Touch lock blocks ALL reflexes/ducking/glides EXCEPT anti-feedback cuts
            if vol_touch_locked:
                if fb_cutting and ch_int.feedback_ceiling_db is not None:
                    desired_vol = ch_int.feedback_ceiling_db
                else:
                    desired_vol = vol_track.current_db

            vol_track.target_db = desired_vol
            vol_track.target_cc = db_to_cc(desired_vol, ch.cc_taper)

            if can_emit_writes and ch_int.armed:
                # Choose unified 5-path slew rate
                delta = desired_vol - vol_track.current_db
                if delta < -1e-9:
                    if fb_cutting or is_ducked:
                        rate = fw.slew_protect_down_db_per_sec
                    elif vol_track.handover_gliding:
                        rate = fw.slew_handover_db_per_sec
                    else:
                        rate = fw.slew_musical_db_per_sec
                elif delta > 1e-9:
                    if speech_open_driving:
                        rate = fw.slew_speech_open_up_db_per_sec
                    elif inst_open_driving:
                        rate = fw.slew_instrument_open_up_db_per_sec
                    elif vol_track.in_duck_release:
                        rate = fw.slew_duck_release_up_db_per_sec
                    elif vol_track.handover_gliding:
                        rate = fw.slew_handover_db_per_sec
                    else:
                        rate = fw.slew_musical_db_per_sec
                else:
                    rate = fw.slew_musical_db_per_sec

                max_step = rate * dt_s
                next_vol = vol_track.current_db + float(np.clip(delta, -max_step, max_step))
                vol_track.current_db = next_vol

                if abs(desired_vol - next_vol) <= 0.05:
                    vol_track.in_duck_release = False
                    vol_track.handover_gliding = False

                next_cc = db_to_cc(next_vol, ch.cc_taper)
                vol_track.actual_cc = next_cc
                if self._last_sent_cc.get((ch_id, "vol")) != next_cc:
                    self._last_sent_cc[(ch_id, "vol")] = next_cc
                    agent_writes.append(
                        AgentCCWrite(
                            channel_id=ch_id,
                            control="vol",
                            cc=next_cc,
                            target_db=next_vol,
                            ack_human_seq=self._human_seq.get((ch_id, "vol"), 0),
                        )
                    )

            eff_vol_offset = vol_track.current_db - (vol_base_phase_db + cont_vol_db)
            history_controls[(ch_id, "vol")] = (vol_track.current_db, eff_vol_offset, phi)

            # --- Reverb Control (if channel has reverb_cc) ---
            if ch_int.reverb is not None:
                rev_track = ch_int.reverb
                rev_track.last_phi = phi
                rev_touch_locked = t < rev_track.touch_lock_until_s
                rev_base_phase_db = self._phase_baseline_db(ch_id, "reverb")

                cont_rev_db, _, _ = self.rules_manager.evaluate_channel(
                    self.active_phase,
                    ch_id,
                    "reverb",
                    speech_open_map=speech_open_map,
                    active_map=active_map,
                    fallback_macro_phase=self.last_macro_phase,
                )

                if rev_track.human_override_active:
                    eff_learned_rev = rev_track.frozen_learned_db
                else:
                    eff_learned_rev = self.offset_learner.predict(
                        self.active_phase,
                        ch_id,
                        "reverb",
                        phi,
                        fallback_macro_phase=self.last_macro_phase,
                    )

                desired_rev = (
                    rev_base_phase_db + cont_rev_db + eff_learned_rev + rev_track.human_session_offset_db
                )
                if rev_track.rollback_target_db is not None and t < rev_track.rollback_hold_until_s:
                    desired_rev = rev_track.rollback_target_db

                rev_upper = ch.hard_max_db
                if rev_track.human_override_active and rev_track.human_anchor_db is not None:
                    rev_upper = max(rev_upper, rev_track.human_anchor_db)
                desired_rev = float(np.clip(desired_rev, ch.hard_min_db, rev_upper))

                if rev_touch_locked:
                    desired_rev = rev_track.current_db

                rev_track.target_db = desired_rev
                rev_track.target_cc = db_to_cc(desired_rev, ch.reverb_cc_taper)

                if can_emit_writes and ch_int.armed:
                    delta_r = desired_rev - rev_track.current_db
                    rate_r = (
                        fw.slew_handover_db_per_sec
                        if rev_track.handover_gliding
                        else fw.slew_musical_db_per_sec
                    )
                    max_step_r = rate_r * dt_s
                    next_rev = rev_track.current_db + float(np.clip(delta_r, -max_step_r, max_step_r))
                    rev_track.current_db = next_rev
                    if abs(desired_rev - next_rev) <= 0.05:
                        rev_track.handover_gliding = False
                    next_rev_cc = db_to_cc(next_rev, ch.reverb_cc_taper)
                    rev_track.actual_cc = next_rev_cc
                    if self._last_sent_cc.get((ch_id, "reverb")) != next_rev_cc:
                        self._last_sent_cc[(ch_id, "reverb")] = next_rev_cc
                        agent_writes.append(
                            AgentCCWrite(
                                channel_id=ch_id,
                                control="reverb",
                                cc=next_rev_cc,
                                target_db=next_rev,
                                ack_human_seq=self._human_seq.get((ch_id, "reverb"), 0),
                            )
                        )

                eff_rev_offset = rev_track.current_db - (rev_base_phase_db + cont_rev_db)
                history_controls[(ch_id, "reverb")] = (rev_track.current_db, eff_rev_offset, phi)

            # --- Conditional Mute Control (STRICTLY forbidden on spoken_primary!) ---
            if ch.mute_cc is not None and ch.role != ChannelRole.SPOKEN_PRIMARY:
                desired_mute = bool(cond_mute) and (not vol_touch_locked) and (not vol_track.human_override_active)
                if can_emit_writes and ch_int.armed and desired_mute != ch_int.muted:
                    ch_int.muted = desired_mute
                    mute_cc_val = (127 if desired_mute else 0) if ch.mute_active_high else (0 if desired_mute else 127)
                    self._last_sent_cc[(ch_id, "mute")] = mute_cc_val
                    agent_writes.append(
                        AgentCCWrite(
                            channel_id=ch_id,
                            control="mute",
                            cc=mute_cc_val,
                            target_db=None,
                            ack_human_seq=self._human_seq.get((ch_id, "mute"), 0),
                        )
                    )

        # Record compact history snapshot for rollback & pre-move phi_bar
        self._history.append(
            _HistorySnapshot(
                ts_s=t,
                active_phase=self.active_phase,
                last_macro_phase=self.last_macro_phase,
                controls=history_controls,
            )
        )

        return FirewallOutput(
            agent_writes=agent_writes,
            channels_state=self.snapshot_channels_state(t),
            events=events,
            watchdog_tripped=False,
            autonomy_mode=self.autonomy_mode,
        )

    def snapshot_channels_state(self, now_s: float | None = None) -> dict[str, ChannelRuntimeState]:
        """Build ChannelRuntimeState snapshots for all configured channels."""
        t = self._clock_s if now_s is None else float(now_s)
        out: dict[str, ChannelRuntimeState] = {}
        for ch_id, ch_int in self._channels.items():
            v = ch_int.vol
            r = ch_int.reverb
            out[ch_id] = ChannelRuntimeState(
                armed=ch_int.armed,
                sync_status=ch_int.sync_status,
                vol_current_db=round(v.current_db, 4),
                vol_target_db=round(v.target_db, 4),
                vol_actual_cc=v.actual_cc,
                vol_target_cc=v.target_cc,
                vol_touch_lock_s=round(max(0.0, v.touch_lock_until_s - t), 3),
                vol_human_offset_db=round(v.human_session_offset_db, 4),
                vol_pickup_dir=v.pickup_dir,
                reverb_current_db=round(r.current_db, 4) if r is not None else None,
                reverb_target_db=round(r.target_db, 4) if r is not None else None,
                reverb_actual_cc=r.actual_cc if r is not None else None,
                reverb_target_cc=r.target_cc if r is not None else None,
                reverb_touch_lock_s=round(max(0.0, r.touch_lock_until_s - t), 3) if r is not None else 0.0,
                reverb_human_offset_db=round(r.human_session_offset_db, 4) if r is not None else 0.0,
                reverb_pickup_dir=r.pickup_dir if r is not None else 0,
                feedback_ceiling_db=round(ch_int.feedback_ceiling_db, 4)
                if ch_int.feedback_ceiling_db is not None
                else None,
                speech_open=ch_int.speech_open,
                instrument_open=ch_int.instrument_open,
            )
        return out
