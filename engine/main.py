"""AetherMix v5.1 Engine Orchestrator & Authenticated FastAPI WebSocket Server.

Integrates BridgeClient, AudioTap, VADWorker, CLAPWorker, SceneClassifier,
OffsetLearner, RulesManager, SafetyFirewall, LLMSupervisor, and TelemetryStore.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import math
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core.audio_tap import AudioFrameMetrics, AudioTap
from engine.core.bridge_client import BridgeClient
from engine.core.clap_worker import CLAPWorker
from engine.core.llm_supervisor import LLMSupervisor
from engine.core.offset_learner import OffsetLearner
from engine.core.protocol import (
    DEFAULT_BRIDGE_SOCKET_PATH,
    DEFAULT_TOKEN_PATH,
    MACRO_PHASES,
    PROTOCOL_VERSION,
    AutonomyMode,
    BrainCategory,
    BrainEventPayload,
    BridgeStatePayload,
    ChannelRole,
    ChannelRuntimeState,
    HeartbeatPayload,
    MeterFramePayload,
    MixingRules,
    RuleExpiry,
    ServicePhase,
    ServicePlan,
    ServiceSegment,
    WSEnvelope,
    cc_to_db,
    cross_validate_plan_and_rules,
    ensure_session_token,
)
from engine.core.rules_manager import RulesManager
from engine.core.safety_firewall import FirewallOutput, SafetyFirewall
from engine.core.scene_classifier import SceneClassifier, SceneDecision
from engine.core.telemetry_store import DEFAULT_DB_PATH, TelemetryStore
from engine.core.vad_worker import VADWorker

logger = logging.getLogger("aethermix.engine")

DEFAULT_SERVICE_PLAN_PATH = REPO_ROOT / "engine" / "schemas" / "service_plan.json"
DEFAULT_MIXING_RULES_PATH = REPO_ROOT / "engine" / "schemas" / "mixing_rules.json"
_IS_APP_BUNDLE = ".app/" in str(REPO_ROOT)
DEFAULT_BACKUP_DIR = (
    Path.home() / ".aethermix" / "storage" / "rule_backups"
    if _IS_APP_BUNDLE
    else REPO_ROOT / "engine" / "storage" / "rule_backups"
)
DEFAULT_ENGINE_DB_PATH = (
    Path.home() / ".aethermix" / "storage" / "telemetry.db"
    if _IS_APP_BUNDLE
    else DEFAULT_DB_PATH
)


class AetherMixEngine:
    """Core orchestrator driving the 50 Hz Tier-1 loop, 0.5 Hz Tier-2 fusion, and WS RPCs."""

    def __init__(
        self,
        plan: ServicePlan,
        rules: MixingRules,
        *,
        plan_path: Path | None = None,
        rules_path: Path | None = None,
        backup_dir: Path | None = None,
        db_path: Path | str = DEFAULT_DB_PATH,
        bridge_socket_path: Path | str = DEFAULT_BRIDGE_SOCKET_PATH,
        token_path: Path = DEFAULT_TOKEN_PATH,
        token: str | None = None,
        headless: bool = False,
        enable_clap: bool | None = None,
        use_service_plan: bool | None = None,
        bridge_client: Any | None = None,
        llm_supervisor: Any | None = None,
        audio_tap: AudioTap | None = None,
        vad_worker: VADWorker | None = None,
        clap_worker: CLAPWorker | None = None,
        telemetry_store: TelemetryStore | None = None,
    ) -> None:
        cross_validate_plan_and_rules(plan, rules)
        self.plan = plan
        self.rules = rules
        self.plan_path = Path(plan_path) if plan_path is not None else None
        self.rules_path = Path(rules_path) if rules_path is not None else None
        self.backup_dir = Path(backup_dir) if backup_dir is not None else DEFAULT_BACKUP_DIR
        self.headless = bool(headless)
        self.use_service_plan = (
            bool(plan.use_service_plan) if use_service_plan is None else bool(use_service_plan)
        )
        self.token = token if token is not None else ensure_session_token(token_path)

        # Core subsystems
        self.telemetry_store = telemetry_store or TelemetryStore(db_path=db_path)
        self.offset_learner = OffsetLearner(plan)
        self.rules_manager = RulesManager(
            plan=plan,
            rules=rules,
            rules_path=self.rules_path,
            backup_dir=self.backup_dir,
        )
        self.safety_firewall = SafetyFirewall(
            plan=plan,
            rules=rules,
            rules_manager=self.rules_manager,
            offset_learner=self.offset_learner,
        )
        self.audio_tap = audio_tap or AudioTap(plan, headless=self.headless)
        self.vad_worker = vad_worker or VADWorker(plan)
        clap_enabled = (not self.headless) if enable_clap is None else bool(enable_clap)
        self.clap_worker = clap_worker or CLAPWorker(enabled=clap_enabled)
        self.scene_classifier = SceneClassifier(plan)
        self.llm_supervisor = llm_supervisor or LLMSupervisor(plan)

        if bridge_client is not None:
            self.bridge_client = bridge_client
            self._wire_bridge_callbacks(self.bridge_client)
        else:
            self.bridge_client = BridgeClient(
                socket_path=bridge_socket_path,
                token=self.token,
                on_human_touch=self._on_bridge_human_touch,
                on_human_cc=self._on_bridge_human_cc,
                on_action_button=self._on_bridge_action_button,
                on_state_change=self._on_bridge_state_change,
                on_reconnect=self._on_bridge_reconnect,
            )

        # Runtime state
        self.freeze_learning: bool = False
        self.service_step_idx: int = 0
        self.phase_confidence: float = 0.0
        self.hysteresis_progress: float = 0.0
        self.hardware_warnings: set[str] = set()
        self._queued_review_count: int = 0

        self._started: bool = False
        self._start_mono_s: float = time.monotonic()
        self._sim_clock_s: float = 0.0
        self._last_scene_eval_s: float = -10.0
        self._last_1hz_s: float = -10.0
        self._last_tick_ms: float = 0.0
        self._last_seen_audio_ts_ms: int = 0
        self._last_audio_ts_s: float | None = None
        self._tick_latencies_ms: deque[float] = deque(maxlen=250)
        self._recent_audio_blocks: deque[np.ndarray] = deque(maxlen=500)
        self._ws_seq: int = 0
        self._ws_clients: set[WebSocket] = set()
        self._loop_task: asyncio.Task[None] | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._last_metrics: AudioFrameMetrics | None = None
        self._last_vad_probs: dict[str, float] = {ch.id: 0.0 for ch in plan.channels}
        self._last_sung_vocals: dict[str, bool] = {}
        self._last_channels_state: dict[str, ChannelRuntimeState] = (
            self.safety_firewall.snapshot_channels_state(0.0)
        )
        self._last_decision: SceneDecision | None = None
        self._event_log: deque[BrainEventPayload] = deque(maxlen=200)
        self._last_snapshot_sig: str = ""

    def _wire_bridge_callbacks(self, client: Any) -> None:
        if hasattr(client, "_on_human_touch") and client._on_human_touch is None:
            client._on_human_touch = self._on_bridge_human_touch
        if hasattr(client, "_on_human_cc") and client._on_human_cc is None:
            client._on_human_cc = self._on_bridge_human_cc
        if hasattr(client, "_on_action_button") and client._on_action_button is None:
            client._on_action_button = self._on_bridge_action_button
        if hasattr(client, "_on_state_change") and client._on_state_change is None:
            client._on_state_change = self._on_bridge_state_change
        if hasattr(client, "_on_reconnect") and client._on_reconnect is None:
            client._on_reconnect = self._on_bridge_reconnect

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, start_loop: bool = True) -> None:
        if self._started:
            return
        self._started = True
        self._start_mono_s = time.monotonic()

        self.telemetry_store.start()
        self._queued_review_count = len(self.telemetry_store.list_queued_reviews())
        snap = self.telemetry_store.load_latest_policy_snapshot()
        if snap is not None:
            _ver, state_dict = snap
            try:
                self.offset_learner.load_state(state_dict)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Failed to load policy snapshot: %s", exc)

        self.audio_tap.start()
        self.vad_worker.start()
        self.clap_worker.start()

        # Connect to standalone MIDI bridge and hydrate authoritative state
        try:
            await self.bridge_client.connect()
            self._hydrate_bridge_state(self.bridge_client.state, initial=True, is_full_sync=True)
        except (OSError, ConnectionError, PermissionError, RuntimeError, TimeoutError) as exc:
            logger.warning("Bridge unavailable at startup (%s); continuing in SHADOW_OBSERVE", exc)
            self.hardware_warnings.add("BRIDGE_DISCONNECTED")
            if hasattr(self.bridge_client, "start_reconnect"):
                self.bridge_client.start_reconnect()

        if not self.headless and hasattr(self.llm_supervisor, "warm_up"):
            self._spawn_bg(self._warm_up_llm())

        if start_loop:
            self._loop_task = asyncio.create_task(self._run_50hz_loop(), name="aethermix-50hz-loop")

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()

        self._persist_last_known_state()
        self.clap_worker.stop()
        self.vad_worker.stop()
        self.audio_tap.stop()

        try:
            await self.bridge_client.close()
        except (OSError, ConnectionError, RuntimeError) as exc:
            logger.debug("Bridge close error: %s", exc)
        if hasattr(self.llm_supervisor, "close"):
            try:
                await self.llm_supervisor.close()
            except (OSError, RuntimeError) as exc:
                logger.debug("LLM supervisor close error: %s", exc)
        self.telemetry_store.close()

    async def _warm_up_llm(self) -> None:
        try:
            await self.llm_supervisor.warm_up()
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("LLM warm_up failed: %s", exc)

    def _spawn_bg(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ------------------------------------------------------------------
    # Bridge Callbacks & Persistence Helpers
    # ------------------------------------------------------------------

    def _hydrate_bridge_state(
        self,
        state: BridgeStatePayload,
        *,
        initial: bool = False,
        is_full_sync: bool = True,
    ) -> None:
        self.hardware_warnings.discard("BRIDGE_DISCONNECTED")
        if state.controller_connected:
            self.hardware_warnings.discard("CONTROLLER_DISCONNECTED")
        else:
            self.hardware_warnings.add("CONTROLLER_DISCONNECTED")

        was_active = (not initial) and (self.safety_firewall.autonomy_mode == AutonomyMode.ACTIVE)
        events = self.safety_firewall.hydrate_from_bridge(
            state,
            was_active=was_active,
            is_full_sync=is_full_sync,
        )
        self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
        for evt in events:
            self._record_and_schedule_event(evt)

    async def _on_bridge_reconnect(self, state: BridgeStatePayload) -> None:
        """Full re-hydration after BridgeClient reconnects to a restarted midi_bridge_daemon."""
        self._hydrate_bridge_state(state, initial=False, is_full_sync=True)
        await self.broadcast_state_snapshot()

    async def _on_bridge_state_change(self, state: BridgeStatePayload) -> None:
        """Incremental bridge_state update (bypass, pickup_pending, human_seq); never clobbers glides."""
        self._hydrate_bridge_state(state, initial=False, is_full_sync=False)
        await self.broadcast_state_snapshot()

    async def _on_bridge_human_touch(self, evt: dict[str, Any]) -> None:
        ch_id = str(evt.get("channel_id", ""))
        ctrl = str(evt.get("control", "vol"))
        seq = evt.get("human_seq")
        if ch_id in {c.id for c in self.plan.channels}:
            self.safety_firewall.on_human_touch(ch_id, ctrl, human_seq=seq, now_s=self._sim_clock_s)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()

    async def _on_bridge_human_cc(self, evt: dict[str, Any]) -> None:
        ch_id = str(evt.get("channel_id", ""))
        ctrl = str(evt.get("control", "vol"))
        cc = int(evt.get("cc", 0))
        seq = evt.get("human_seq")
        source = str(evt.get("source", "hardware"))
        if ch_id not in {c.id for c in self.plan.channels}:
            return
        sample = self.safety_firewall.on_human_cc(
            ch_id,
            ctrl,
            cc,
            human_seq=seq,
            now_s=self._sim_clock_s,
            freeze_learning=self.freeze_learning,
            source=source,
        )
        if self.freeze_learning:
            sample_tagged = dict(sample)
            sample_tagged.update(self._current_segment_meta())
            self.telemetry_store.queue_review_sample(sample_tagged)
            self._queued_review_count += 1
            await self.broadcast_queued_reviews()
        else:
            self.telemetry_store.save_policy_snapshot(
                self.offset_learner.version, self.offset_learner.export_state()
            )
        self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
        self._persist_last_known_state()
        await self.broadcast_state_snapshot()

    async def _on_bridge_action_button(self, evt: dict[str, Any]) -> None:
        action = str(evt.get("action", ""))
        if action == "thumbs_up":
            await self._handle_feedback_thumb("up")
        elif action == "thumbs_down":
            await self._handle_feedback_thumb("down")
        elif action == "emergency_bypass":
            if self.bridge_client.state.bypass_latched:
                self.safety_firewall.set_mode(AutonomyMode.EMERGENCY_BYPASS)
            else:
                self.safety_firewall.set_mode(AutonomyMode.SHADOW_OBSERVE)
            await self.broadcast_state_snapshot()

    def _persist_last_known_state(self) -> None:
        armed_ids = self.safety_firewall.get_armed_channel_ids()
        channels_cc: dict[str, int] = {}
        for ch_id, ch_state in self._last_channels_state.items():
            channels_cc[f"{ch_id}:vol"] = int(ch_state.vol_actual_cc)
            if ch_state.reverb_actual_cc is not None:
                channels_cc[f"{ch_id}:reverb"] = int(ch_state.reverb_actual_cc)
        self.telemetry_store.save_last_known_state(armed_ids, channels_cc)

    # ------------------------------------------------------------------
    # 50 Hz Loop & Single-Step Execution
    # ------------------------------------------------------------------

    async def _run_50hz_loop(self) -> None:
        period_s = 0.02
        next_tick = time.monotonic()
        while self._started:
            try:
                await self.step_once(dt_s=period_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error inside 50 Hz engine loop")
            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
                await asyncio.sleep(0)

    def push_audio_block(self, block: np.ndarray) -> None:
        """Push a (frames, 16) float32 block into AudioTap (used in headless mode/tests)."""
        self.audio_tap.push_block(block)
        self._recent_audio_blocks.append(block.copy())
        self._last_audio_ts_s = time.monotonic()

    async def step_once(
        self,
        dt_s: float = 0.02,
        now_s: float | None = None,
        audio_block: np.ndarray | None = None,
        broadcast: bool = True,
    ) -> FirewallOutput:
        """Execute one complete 50 Hz tick (deterministic when audio_block & now_s are supplied)."""
        t_start = time.perf_counter()

        if audio_block is not None:
            self.push_audio_block(audio_block)

        if now_s is not None:
            self._sim_clock_s = float(now_s)
        else:
            self._sim_clock_s += float(dt_s)
        t_now = self._sim_clock_s

        # 1. Audio DSP + VAD
        metrics = self.audio_tap.poll_metrics()
        self._last_metrics = metrics
        vad_probs, sung_vocals = self.vad_worker.process_frame(metrics)
        self._last_vad_probs = vad_probs
        self._last_sung_vocals = sung_vocals
        self.scene_classifier.observe_tick(metrics, vad_probs, sung_vocals)

        if metrics.tcc_silence_warning:
            self.hardware_warnings.add("HARDWARE_SILENCE_WARNING")
        else:
            self.hardware_warnings.discard("HARDWARE_SILENCE_WARNING")

        # 2. Tier-2 Scene Fusion at 0.5 Hz (every 2.0 s)
        if (t_now - self._last_scene_eval_s) >= 2.0 - 1e-9:
            self._last_scene_eval_s = t_now
            self.clap_worker.submit_audio(metrics.pre_fader_sum_3s, self.plan.sample_rate)
            clap_probs, _clap_ok = self.clap_worker.latest_probs()
            decision = self.scene_classifier.update(
                metrics=metrics,
                vad_probs=vad_probs,
                sung_vocals=sung_vocals,
                clap_probs=clap_probs,
                service_step_idx=self.service_step_idx,
                use_service_plan=self.use_service_plan,
                now_s=t_now,
            )
            self._last_decision = decision
            self.phase_confidence = decision.confidence
            self.hysteresis_progress = decision.hysteresis_progress
            if decision.transitioned:
                self.rules_manager.expire_ephemeral(RuleExpiry.PHASE_CHANGE)
                evt = BrainEventPayload(
                    id=f"evt_{uuid.uuid4().hex[:10]}",
                    category=BrainCategory.PERCEPTION,
                    summary=f"Phase -> {decision.active_phase.value}",
                    detail=decision.reason,
                    confidence=decision.confidence,
                )
                self._record_and_schedule_event(evt)
                if not self.headless:
                    self._spawn_bg(self._run_advisory_llm("phase_transition"))

        # 3. Tier-1 Deterministic Safety Firewall
        # Track live audio freshness strictly via advancing block timestamps (Defect #6)
        tap_ts_ms = int(getattr(self.audio_tap, "last_block_ts_ms", metrics.ts_ms))
        if self._last_audio_ts_s is None:
            self._last_audio_ts_s = t_now
        if tap_ts_ms > self._last_seen_audio_ts_ms:
            self._last_seen_audio_ts_ms = tap_ts_ms
            self._last_audio_ts_s = t_now
        audio_stale_s = 0.0 if self.headless else max(0.0, t_now - self._last_audio_ts_s)
        self._last_audio_stale_s = audio_stale_s
        if not self.headless and audio_stale_s > 0.5:
            self.hardware_warnings.add("AUDIO_STREAM_STALLED")
        else:
            self.hardware_warnings.discard("AUDIO_STREAM_STALLED")

        fw_out = self.safety_firewall.tick(
            metrics=metrics,
            vad_probs=vad_probs,
            sung_vocals=sung_vocals,
            active_phase=self.scene_classifier.active_phase,
            last_macro_phase=self.scene_classifier.last_macro_phase,
            dt_s=dt_s,
            now_s=t_now,
            tick_elapsed_ms=self._last_tick_ms,
            audio_stale_s=audio_stale_s,
        )
        self._last_channels_state = fw_out.channels_state

        # 4. Emit bridge heartbeat & agent CC writes
        if self.bridge_client.connected:
            try:
                await self.bridge_client.send_heartbeat()
                for write in fw_out.agent_writes:
                    await self.bridge_client.send_agent_cc(
                        channel_id=write.channel_id,
                        control=write.control,
                        cc=write.cc,
                        ack_human_seq=write.ack_human_seq,
                    )
            except (OSError, ConnectionError, RuntimeError) as exc:
                logger.warning("Bridge write/heartbeat failed: %s", exc)
                self.hardware_warnings.add("BRIDGE_DISCONNECTED")

        # 5. Record & stream brain events
        for evt in fw_out.events:
            self._record_and_schedule_event(evt)
            if evt.category == BrainCategory.SAFETY and not self.headless:
                self._spawn_bg(self._run_advisory_llm("safety_event", extra_note=evt.summary))

        # 6. Measure tick latency
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._last_tick_ms = elapsed_ms
        self._tick_latencies_ms.append(elapsed_ms)

        # 7. Stream WS meter_frame, delta state_snapshot, and 1Hz heartbeat + SQLite summary
        if broadcast and self._ws_clients:
            await self._broadcast_meter_frame(metrics, fw_out.channels_state, vad_probs)
            sig = self._snapshot_signature()
            if sig != self._last_snapshot_sig:
                self._last_snapshot_sig = sig
                await self.broadcast_state_snapshot()

        if (t_now - self._last_1hz_s) >= 1.0 - 1e-9:
            self._last_1hz_s = t_now
            hb = self.build_heartbeat_payload()
            self.telemetry_store.log_1hz_summary(
                active_phase=self.scene_classifier.active_phase.value,
                autonomy_mode=self.safety_firewall.autonomy_mode.value,
                master_lufs=metrics.master_lufs_short_term,
                loop_p95_ms=hb.loop_p95_ms,
                loop_p99_ms=hb.loop_p99_ms,
                summary={
                    "phase_confidence": round(self.phase_confidence, 3),
                    "armed_count": len(self.safety_firewall.get_armed_channel_ids()),
                },
            )
            if broadcast and self._ws_clients:
                await self._broadcast_envelope("heartbeat", hb.model_dump(mode="json"))

        return fw_out

    def _record_and_schedule_event(self, evt: BrainEventPayload) -> None:
        self._event_log.append(evt)
        self.telemetry_store.log_event(evt.category.value, evt.model_dump(mode="json"), ts_ms=evt.ts_ms)
        if self._ws_clients:
            try:
                asyncio.get_running_loop()
                self._spawn_bg(self._broadcast_envelope("brain_event", evt.model_dump(mode="json")))
            except RuntimeError:
                pass

    async def _run_advisory_llm(self, trigger_type: str, extra_note: str | None = None) -> None:
        summary = self._build_telemetry_summary()
        if extra_note:
            summary["event_note"] = extra_note
        try:
            resp = await self.llm_supervisor.analyze_event(trigger_type, summary)
            evt = BrainEventPayload(
                id=f"evt_{uuid.uuid4().hex[:10]}",
                category=BrainCategory.LLM,
                summary=resp.thought_summary,
                detail=resp.reasoning,
                confidence=resp.confidence,
            )
            self._record_and_schedule_event(evt)
        except (OSError, RuntimeError, TimeoutError, ValidationError, ValueError) as exc:
            logger.debug("Advisory LLM call failed: %s", exc)

    def _build_telemetry_summary(self) -> dict[str, Any]:
        return {
            "active_phase": self.scene_classifier.active_phase.value,
            "last_macro_phase": self.scene_classifier.last_macro_phase.value,
            "autonomy_mode": self.safety_firewall.autonomy_mode.value,
            "service_step_idx": self.service_step_idx,
            "master_lufs": (
                round(self._last_metrics.master_lufs_short_term, 2)
                if self._last_metrics is not None
                else -60.0
            ),
            "channels": {
                cid: {
                    "armed": st.armed,
                    "vol_db": st.vol_current_db,
                    "speech_open": st.speech_open,
                    "instrument_open": st.instrument_open,
                }
                for cid, st in self._last_channels_state.items()
            },
        }

    # ------------------------------------------------------------------
    # Payload Builders & WebSocket Broadcasting
    # ------------------------------------------------------------------

    def _current_segment_meta(self) -> dict[str, Any]:
        segs = self.plan.order_of_service
        if segs and 0 <= self.service_step_idx < len(segs):
            seg = segs[self.service_step_idx]
            return {"segment_id": seg.id, "segment_title": seg.title}
        return {}

    def build_state_snapshot(self) -> dict[str, Any]:
        payload = {
            "channel_ids": list(self.plan.channel_ids),
            "autonomy_mode": self.safety_firewall.autonomy_mode.value,
            "bypass_latched": self.safety_firewall.bypass_latched,
            "freeze_learning": self.freeze_learning,
            "use_service_plan": self.use_service_plan,
            "order_of_service": [s.model_dump(mode="json") for s in self.plan.order_of_service],
            "active_phase": self.scene_classifier.active_phase.value,
            "phase_confidence": round(self.phase_confidence, 4),
            "hysteresis_progress": round(self.hysteresis_progress, 4),
            "service_step_idx": self.service_step_idx,
            "channels": {
                cid: st.model_dump(mode="json") for cid, st in self._last_channels_state.items()
            },
            "active_rules": [r.model_dump(mode="json") for r in self.rules_manager.list_active_rules()],
            "pending_rules": [c.model_dump(mode="json") for c in self.rules_manager.list_pending_rules()],
            "queued_review_count": self._queued_review_count,
            "hardware_warnings": sorted(self.hardware_warnings),
        }
        return payload

    def build_queued_reviews_payload(self) -> dict[str, Any]:
        samples = self.telemetry_store.list_queued_reviews() if self._started else []
        self._queued_review_count = len(samples)
        ui_samples = [
            {
                "id": str(s.get("id", "")),
                "phase": str(s.get("phase", "SPOKEN_WORD")),
                "channel_id": str(s.get("channel_id", "")),
                "control": str(s.get("control", "vol")),
                "target_offset_db": float(s.get("target_offset_db", 0.0)),
                "source": str(s.get("source", "human_correction")),
                "ts_ms": int(s.get("ts_ms", 0)),
            }
            for s in samples
        ]
        return {"samples": ui_samples}

    async def broadcast_queued_reviews(self) -> None:
        if not self._ws_clients:
            return
        await self._broadcast_envelope("queued_reviews", self.build_queued_reviews_payload())

    def _snapshot_signature(self) -> str:
        return (
            f"{self.safety_firewall.autonomy_mode.value}|{self.safety_firewall.bypass_latched}|"
            f"{self.freeze_learning}|{self.use_service_plan}|"
            f"{[(s.id, s.title, s.expected_phase.value) for s in self.plan.order_of_service]}|"
            f"{self.scene_classifier.active_phase.value}|"
            f"{round(self.phase_confidence, 2)}|{round(self.hysteresis_progress, 2)}|"
            f"{self.service_step_idx}|{self._queued_review_count}|"
            f"{len(self.rules_manager.list_active_rules())}|{len(self.rules_manager.list_pending_rules())}|"
            f"{sorted(self.hardware_warnings)}|"
            + ",".join(
                f"{cid}:{st.armed}:{st.sync_status.value}:{st.speech_open}:{st.instrument_open}:"
                f"{st.feedback_ceiling_db}:{round(st.vol_touch_lock_s, 1)}:{round(st.reverb_touch_lock_s, 1)}:"
                f"{round(st.vol_human_offset_db, 2)}:{round(st.reverb_human_offset_db, 2)}:"
                f"{st.vol_pickup_dir}:{st.reverb_pickup_dir}"
                for cid, st in self._last_channels_state.items()
            )
        )

    def build_meter_frame(
        self,
        metrics: AudioFrameMetrics,
        channels_state: dict[str, ChannelRuntimeState],
        vad_probs: dict[str, float],
    ) -> MeterFramePayload:
        cids = self.plan.channel_ids
        return MeterFramePayload(
            rms_db=[round(float(metrics.rms_db.get(c, -120.0)), 2) for c in cids],
            peak_db=[round(float(metrics.peak_db.get(c, -120.0)), 2) for c in cids],
            clip=[bool(metrics.clip.get(c, False)) for c in cids],
            vad=[round(float(vad_probs.get(c, 0.0)), 3) for c in cids],
            master_lufs=round(float(metrics.master_lufs_momentary), 2),
            vol_target_cc=[int(channels_state[c].vol_target_cc) for c in cids],
            vol_actual_cc=[int(channels_state[c].vol_actual_cc) for c in cids],
            reverb_target_cc=[channels_state[c].reverb_target_cc for c in cids],
            reverb_actual_cc=[channels_state[c].reverb_actual_cc for c in cids],
        )

    def build_heartbeat_payload(self) -> HeartbeatPayload:
        samples = sorted(self._tick_latencies_ms)
        if samples:
            p95 = samples[int(0.95 * (len(samples) - 1))]
            p99 = samples[int(0.99 * (len(samples) - 1))]
        else:
            p95 = 0.0
            p99 = 0.0
        _, clap_ok = self.clap_worker.latest_probs()
        ollama_ok = bool(getattr(self.llm_supervisor, "ollama_ok", getattr(self.llm_supervisor, "ok", False)))
        return HeartbeatPayload(
            uptime_s=round(max(0.0, time.monotonic() - self._start_mono_s), 2),
            loop_p95_ms=round(p95, 3),
            loop_p99_ms=round(p99, 3),
            audio_ok=(
                "HARDWARE_SILENCE_WARNING" not in self.hardware_warnings
                and "AUDIO_STREAM_STALLED" not in self.hardware_warnings
            ),
            bridge_ok=bool(self.bridge_client.connected),
            clap_ok=bool(clap_ok),
            ollama_ok=ollama_ok,
        )

    def _next_envelope(self, msg_type: str, payload: dict[str, Any], req_id: str | None = None) -> dict[str, Any]:
        self._ws_seq += 1
        env = WSEnvelope(
            type=msg_type,
            seq=self._ws_seq,
            req_id=req_id,
            ts_ms=int(time.time() * 1000),
            payload=payload,
        )
        return env.model_dump(mode="json")

    async def _broadcast_envelope(self, msg_type: str, payload: dict[str, Any]) -> None:
        if not self._ws_clients:
            return
        frame = self._next_envelope(msg_type, payload)
        dead: list[WebSocket] = []
        for ws in list(self._ws_clients):
            try:
                await ws.send_json(frame)
            except (WebSocketDisconnect, RuntimeError, OSError):
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    async def _broadcast_meter_frame(
        self,
        metrics: AudioFrameMetrics,
        channels_state: dict[str, ChannelRuntimeState],
        vad_probs: dict[str, float],
    ) -> None:
        mf = self.build_meter_frame(metrics, channels_state, vad_probs)
        await self._broadcast_envelope("meter_frame", mf.model_dump(mode="json"))

    async def broadcast_state_snapshot(self) -> None:
        await self._broadcast_envelope("state_snapshot", self.build_state_snapshot())

    # ------------------------------------------------------------------
    # Feedback Helper
    # ------------------------------------------------------------------

    async def _handle_feedback_thumb(self, direction: str) -> dict[str, Any]:
        if direction == "up":
            samples, events = self.safety_firewall.handle_thumbs_up(
                now_s=self._sim_clock_s, freeze_learning=self.freeze_learning
            )
        elif direction == "down":
            samples, events = self.safety_firewall.handle_thumbs_down(
                now_s=self._sim_clock_s, freeze_learning=self.freeze_learning
            )
        else:
            raise ValueError(f"Unsupported feedback_thumb direction: {direction!r}")

        if self.freeze_learning:
            seg_meta = self._current_segment_meta()
            for sample in samples:
                tagged = dict(sample)
                tagged.update(seg_meta)
                self.telemetry_store.queue_review_sample(tagged)
            self._queued_review_count += len(samples)
            await self.broadcast_queued_reviews()
        elif samples:
            self.telemetry_store.save_policy_snapshot(
                self.offset_learner.version, self.offset_learner.export_state()
            )

        for evt in events:
            self._record_and_schedule_event(evt)
        self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
        await self.broadcast_state_snapshot()
        return {"direction": direction, "sample_count": len(samples), "queued": self.freeze_learning}

    # ------------------------------------------------------------------
    # All v5.1 Client RPC Handlers
    # ------------------------------------------------------------------

    async def handle_rpc(
        self,
        msg_type: str,
        payload: dict[str, Any],
        req_id: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch a client RPC and return a correlated ack or error WSEnvelope."""
        try:
            result = await self._dispatch_rpc(msg_type, payload)
            return self._next_envelope(
                "ack",
                {"req_id": req_id, "ok": True, "code": "OK", "message": "ok", "data": result},
                req_id=req_id,
            )
        except PermissionError as exc:
            return self._next_envelope(
                "error",
                {"req_id": req_id, "ok": False, "code": "SAFE_UPDATE_BLOCKED", "message": str(exc)},
                req_id=req_id,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return self._next_envelope(
                "error",
                {"req_id": req_id, "ok": False, "code": "RPC_ERROR", "message": str(exc)},
                req_id=req_id,
            )
        except Exception as exc:
            logger.exception("Unhandled error in RPC %s", msg_type)
            return self._next_envelope(
                "error",
                {"req_id": req_id, "ok": False, "code": "INTERNAL_ERROR", "message": str(exc)},
                req_id=req_id,
            )

    async def _dispatch_rpc(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if msg_type == "set_mode":
            mode = AutonomyMode(str(payload["mode"]))
            self.safety_firewall.set_mode(mode)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()
            return {"autonomy_mode": mode.value}

        if msg_type == "set_channel_armed":
            ch_id = str(payload["channel_id"])
            armed = bool(payload["armed"])
            self.safety_firewall.set_channel_armed(ch_id, armed)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            self._persist_last_known_state()
            await self.broadcast_state_snapshot()
            return {"channel_id": ch_id, "armed": armed}

        if msg_type == "restore_previous_arming":
            saved = self.telemetry_store.load_last_known_state()
            armed_ids = [cid for cid in saved.get("armed_ids", []) if cid in set(self.plan.channel_ids)]
            self.safety_firewall.set_armed_channels(armed_ids)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()
            return {"restored_armed_ids": armed_ids}

        if msg_type == "sync_channel":
            ch_id = str(payload["channel_id"])
            self.safety_firewall.confirm_sync(ch_id)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()
            return {"channel_id": ch_id, "sync_status": "SYNCED"}

        if msg_type == "release_human_offset":
            ch_id = str(payload["channel_id"])
            ctrl = str(payload.get("control", "vol"))
            self.safety_firewall.release_human_offset(ch_id, ctrl, now_s=self._sim_clock_s)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()
            return {"channel_id": ch_id, "control": ctrl}

        if msg_type == "set_freeze_learning":
            flag = payload.get("freeze", payload.get("enabled", False))
            self.freeze_learning = bool(flag)
            await self.broadcast_state_snapshot()
            return {"freeze_learning": self.freeze_learning}

        if msg_type == "override_phase":
            prev_phase = self.scene_classifier.active_phase
            phase = ServicePhase(str(payload["phase"]))
            hold_s = float(payload.get("hold_seconds", 60.0))
            self.scene_classifier.override_phase(phase, hold_seconds=hold_s, now_s=self._sim_clock_s)
            self.safety_firewall._handle_phase_change(
                self.scene_classifier.active_phase,
                self.scene_classifier.last_macro_phase,
                self._sim_clock_s,
            )
            if phase != prev_phase:
                self.rules_manager.expire_ephemeral(RuleExpiry.PHASE_CHANGE)
            self.phase_confidence = 1.0
            self.hysteresis_progress = 1.0
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()
            return {"active_phase": phase.value, "hold_seconds": hold_s}

        if msg_type == "set_use_service_plan":
            if "ignore" in payload:
                enabled = not bool(payload["ignore"])
            else:
                enabled = bool(payload.get("enabled", payload.get("use_service_plan", True)))
            self.use_service_plan = enabled
            self.plan.use_service_plan = enabled
            if not enabled:
                self.safety_firewall._prestage_worship = False
            self._save_service_plan_atomic()
            evt = BrainEventPayload(
                id=f"evt_{uuid.uuid4().hex[:10]}",
                category=BrainCategory.ACTION,
                summary=(
                    "Order of Service ACTIVE (20% soft prior)"
                    if enabled
                    else "Order of Service IGNORED (100% live audio)"
                ),
                detail=(
                    "Classifier blends 20% Order-of-Service prior with live audio; segments pre-stage worship."
                    if enabled
                    else "Classifier runs purely on live audio; you can still mark segments for learning & telemetry."
                ),
            )
            self._record_and_schedule_event(evt)
            await self.broadcast_state_snapshot()
            return {"use_service_plan": self.use_service_plan, "ignore_order_of_service": not self.use_service_plan}

        if msg_type == "update_order_of_service":
            raw_segments = payload.get("segments")
            if not isinstance(raw_segments, list):
                raise ValueError("update_order_of_service requires a 'segments' list")
            validated: list[ServiceSegment] = []
            seen_ids: set[str] = set()
            for idx, item in enumerate(raw_segments):
                if not isinstance(item, dict):
                    raise TypeError(f"Segment at index {idx} must be an object")
                seg_id = str(item.get("id") or f"seg_{idx + 1}").strip()
                if not seg_id or seg_id in seen_ids:
                    seg_id = f"seg_{idx + 1}_{uuid.uuid4().hex[:4]}"
                seen_ids.add(seg_id)
                title = str(item.get("title", "")).strip()
                if not title:
                    raise ValueError(f"Segment at index {idx} must have a non-empty title")
                phase = ServicePhase(str(item["expected_phase"]))
                validated.append(ServiceSegment(id=seg_id, title=title, expected_phase=phase))

            self.plan.order_of_service = validated
            if "use_service_plan" in payload:
                self.use_service_plan = bool(payload["use_service_plan"])
                self.plan.use_service_plan = self.use_service_plan
            elif "ignore" in payload:
                self.use_service_plan = not bool(payload["ignore"])
                self.plan.use_service_plan = self.use_service_plan

            if validated:
                self.service_step_idx = min(self.service_step_idx, len(validated) - 1)
            else:
                self.service_step_idx = 0
            self.safety_firewall.service_step_idx = self.service_step_idx
            self._save_service_plan_atomic()
            evt = BrainEventPayload(
                id=f"evt_{uuid.uuid4().hex[:10]}",
                category=BrainCategory.ACTION,
                summary=f"Updated Order of Service ({len(validated)} segment(s))",
                detail=(
                    "Plan active (20% soft prior)."
                    if self.use_service_plan
                    else "Plan ignored for live audio (marking enabled for learning)."
                ),
            )
            self._record_and_schedule_event(evt)
            await self.broadcast_state_snapshot()
            return {
                "order_of_service": [s.model_dump(mode="json") for s in validated],
                "use_service_plan": self.use_service_plan,
                "service_step_idx": self.service_step_idx,
            }

        if msg_type == "advance_service_step":
            total_steps = len(self.plan.order_of_service)
            if "step_idx" in payload and payload["step_idx"] is not None:
                next_idx = int(payload["step_idx"])
            elif total_steps > 0:
                next_idx = min(total_steps - 1, self.service_step_idx + 1)
            else:
                next_idx = 0
            if total_steps > 0 and not (0 <= next_idx < total_steps):
                raise ValueError(f"step_idx {next_idx} out of bounds [0, {total_steps - 1}]")
            self.service_step_idx = next_idx
            self.safety_firewall.on_segment_advance(
                next_idx, now_s=self._sim_clock_s, prestage=self.use_service_plan
            )
            self.rules_manager.expire_ephemeral(RuleExpiry.SEGMENT_CHANGE)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            if total_steps > 0:
                seg = self.plan.order_of_service[next_idx]
                evt = BrainEventPayload(
                    id=f"evt_{uuid.uuid4().hex[:10]}",
                    category=BrainCategory.RLHF if not self.use_service_plan else BrainCategory.ACTION,
                    summary=f"Segment marked: {seg.title}",
                    detail=(
                        f"Expected {seg.expected_phase.value} (logged for learning; plan ignored by live classifier)."
                        if not self.use_service_plan
                        else f"Expected {seg.expected_phase.value} (20% soft prior active)."
                    ),
                )
                self._record_and_schedule_event(evt)
            await self.broadcast_state_snapshot()
            return {"service_step_idx": self.service_step_idx, "use_service_plan": self.use_service_plan}

        if msg_type == "feedback_thumb":
            direction = str(payload.get("direction", payload.get("thumb", ""))).lower()
            return await self._handle_feedback_thumb(direction)

        if msg_type == "submit_critique":
            text = str(payload.get("text", payload.get("critique", ""))).strip()
            if not text:
                raise ValueError("Critique text must not be empty")
            summary = self._build_telemetry_summary()
            resp = await self.llm_supervisor.analyze_event(
                "operator_critique", summary, critique_text=text
            )
            cards = self.rules_manager.ingest_proposed_patches(resp.proposed_patches)
            evt = BrainEventPayload(
                id=f"evt_{uuid.uuid4().hex[:10]}",
                category=BrainCategory.LLM,
                summary=resp.thought_summary,
                detail=resp.reasoning,
                confidence=resp.confidence,
            )
            self._record_and_schedule_event(evt)
            await self.broadcast_state_snapshot()
            return {
                "thought_summary": resp.thought_summary,
                "pending_rules": [c.model_dump(mode="json") for c in cards],
            }

        if msg_type == "resolve_pending_rule":
            rule_id = str(payload["rule_id"])
            action = str(payload["action"])
            resolved = self.rules_manager.resolve_pending_rule(rule_id, action)  # type: ignore[arg-type]
            evt = BrainEventPayload(
                id=f"evt_{uuid.uuid4().hex[:10]}",
                category=BrainCategory.RLHF,
                summary=f"Rule {rule_id}: {action}",
                detail=resolved.description if resolved is not None else "Dismissed",
            )
            self._record_and_schedule_event(evt)
            await self.broadcast_state_snapshot()
            return {
                "rule_id": rule_id,
                "action": action,
                "rule": resolved.model_dump(mode="json") if resolved is not None else None,
            }

        if msg_type == "toggle_rule":
            rule_id = str(payload["rule_id"])
            enabled = bool(payload["enabled"])
            rule = self.rules_manager.toggle_rule(rule_id, enabled)
            await self.broadcast_state_snapshot()
            return {"rule_id": rule_id, "enabled": rule.enabled}

        if msg_type == "delete_rule":
            rule_id = str(payload["rule_id"])
            deleted = self.rules_manager.delete_rule(rule_id)
            if not deleted:
                raise KeyError(f"Unknown rule_id: {rule_id}")
            await self.broadcast_state_snapshot()
            return {"rule_id": rule_id, "deleted": True}

        if msg_type == "reset_learning":
            phase = payload.get("phase")
            ch_id = payload.get("channel_id")
            ctrl = payload.get("control")
            count = self.offset_learner.reset_learning(phase=phase, channel_id=ch_id, control=ctrl)
            self.telemetry_store.save_policy_snapshot(
                self.offset_learner.version, self.offset_learner.export_state()
            )
            await self.broadcast_state_snapshot()
            return {"reset_models": count}

        if msg_type == "commit_queued_reviews":
            action = str(payload.get("action", "commit")).lower()
            if action not in ("commit", "discard"):
                raise ValueError(f"Unsupported commit_queued_reviews action: {action}")
            raw_ids = payload.get("sample_ids")
            selected_ids: list[str] | None = (
                [str(sid) for sid in raw_ids] if isinstance(raw_ids, list) and len(raw_ids) > 0 else None
            )
            all_samples = self.telemetry_store.list_queued_reviews()
            if selected_ids is not None:
                id_set = set(selected_ids)
                target_samples = [s for s in all_samples if str(s.get("id", "")) in id_set]
                delete_ids: list[str] | None = [str(s["id"]) for s in target_samples]
            else:
                target_samples = all_samples
                delete_ids = None

            committed = 0
            if action == "commit" and target_samples:
                committed = self.offset_learner.commit_queued_samples(target_samples)
                self.telemetry_store.save_policy_snapshot(
                    self.offset_learner.version, self.offset_learner.export_state()
                )
            if selected_ids is None:
                self.telemetry_store.clear_queued_reviews()
            elif delete_ids:
                self.telemetry_store.delete_queued_reviews(delete_ids)
            self.telemetry_store.flush()
            remaining = self.telemetry_store.list_queued_reviews()
            self._queued_review_count = len(remaining)
            await self.broadcast_queued_reviews()
            await self.broadcast_state_snapshot()
            return {
                "action": action,
                "committed": committed,
                "cleared": len(target_samples),
                "remaining": len(remaining),
            }

        if msg_type == "capture_room_silence":
            duration_s = float(payload.get("duration_s", 5.0))
            blocks = await self._collect_calibration_blocks(duration_s)
            floors = self.audio_tap.capture_room_silence(blocks)
            self._save_service_plan_atomic()
            return {"mic_noise_floor_vband_db": floors}

        if msg_type == "capture_input_nominals":
            group = str(payload.get("group", "spoken"))
            duration_s = float(payload.get("duration_s", 10.0))
            blocks = await self._collect_calibration_blocks(duration_s)
            nominals = self.audio_tap.capture_input_nominals(group, blocks)  # type: ignore[arg-type]
            self._save_service_plan_atomic()
            return {"group": group, "nominals": {k: list(v) for k, v in nominals.items()}}

        if msg_type == "capture_bleed_floor":
            duration_s = float(payload.get("duration_s", 10.0))
            blocks = await self._collect_calibration_blocks(duration_s)
            offsets = self.audio_tap.capture_bleed_floor(blocks)
            self._save_service_plan_atomic()
            return {"bleed_offset_db": offsets}

        if msg_type == "capture_baseline":
            phase = ServicePhase(str(payload["phase"]))
            if phase not in MACRO_PHASES:
                raise ValueError("capture_baseline requires a macro phase (not TRANSITION)")
            quick_derive = bool(payload.get("quick_derive_others", False))
            bridge_cc = dict(self.bridge_client.state.last_sent_cc)
            vol_map: dict[str, float] = {}
            rev_map: dict[str, float] = {}
            for ch in self.plan.channels:
                v_cc = bridge_cc.get(f"{ch.id}:vol", self._last_channels_state[ch.id].vol_actual_cc)
                v_db = cc_to_db(v_cc, ch.cc_taper)
                if math.isinf(v_db):
                    v_db = ch.hard_min_db
                vol_map[ch.id] = round(float(v_db), 2)
                if ch.reverb_cc is not None:
                    r_cc = bridge_cc.get(
                        f"{ch.id}:reverb",
                        self._last_channels_state[ch.id].reverb_actual_cc or 32,
                    )
                    r_db = cc_to_db(r_cc, ch.reverb_cc_taper)
                    if math.isinf(r_db):
                        r_db = ch.hard_min_db
                    rev_map[ch.id] = round(float(r_db), 2)

            self.rules.phase_baselines_db[phase] = vol_map
            if rev_map:
                self.rules.phase_reverb_baselines_db[phase] = rev_map
            if quick_derive:
                self._derive_other_phase_baselines(phase, vol_map)
            self.rules.calibrated_at = datetime.now(UTC).isoformat()
            if self.rules_path is not None:
                self.rules_manager.save_atomic()
            for ch in self.plan.channels:
                self.safety_firewall.confirm_sync(ch.id)
            self._last_channels_state = self.safety_firewall.snapshot_channels_state(self._sim_clock_s)
            await self.broadcast_state_snapshot()
            return {"phase": phase.value, "calibrated_at": self.rules.calibrated_at, "vol_db": vol_map}

        if msg_type == "midi_learn_pulse":
            ch_id = str(payload["channel_id"])
            ctrl = str(payload.get("control", "vol"))
            self.plan.channel_by_id(ch_id)  # validate channel exists
            if self.bridge_client.connected:
                await self.bridge_client.send_midi_learn_pulse(ch_id, ctrl)
            return {"channel_id": ch_id, "control": ctrl, "pulsed": True}

        if msg_type == "verify_mute_polarity":
            ch_id = str(payload["channel_id"])
            ch = self.plan.channel_by_id(ch_id)
            if ch.role == ChannelRole.SPOKEN_PRIMARY or ch.mute_cc is None:
                raise ValueError(f"Mute polarity verification not permitted on {ch_id}")
            raw_polarity = payload.get("mute_active_high")
            if raw_polarity is None:
                raw_polarity = payload.get("active_high")
            if raw_polarity is not None:
                ch.mute_active_high = bool(raw_polarity)
                self._save_service_plan_atomic()
            test_mute = bool(payload.get("test_mute", True))
            cc_val = (127 if test_mute else 0) if ch.mute_active_high else (0 if test_mute else 127)
            if self.bridge_client.connected:
                await self.bridge_client.send_agent_cc(ch_id, "mute", cc_val)
            return {
                "channel_id": ch_id,
                "mute_active_high": ch.mute_active_high,
                "active_high": ch.mute_active_high,
                "test_mute": test_mute,
                "sent_cc": cc_val,
            }

        if msg_type in ("safe_update", "apply_update", "check_for_updates"):
            if self.safety_firewall.autonomy_mode == AutonomyMode.ACTIVE:
                raise PermissionError(
                    "Safe-Update Lock: updates are blocked while autonomy_mode == ACTIVE. "
                    "Switch to SHADOW_OBSERVE or EMERGENCY_BYPASS first."
                )
            return {"allowed": True, "autonomy_mode": self.safety_firewall.autonomy_mode.value}

        raise ValueError(f"Unsupported RPC type: {msg_type}")

    def _derive_other_phase_baselines(self, source_phase: ServicePhase, vol_map: dict[str, float]) -> None:
        """Derive consistent companion macro-phase baselines using musical relative trims."""

        def _clamp_ch(ch_id: str, val: float) -> float:
            ch = self.plan.channel_by_id(ch_id)
            return round(max(ch.hard_min_db, min(ch.hard_max_db, val)), 2)

        if source_phase == ServicePhase.WORSHIP_HIGH:
            low_map: dict[str, float] = {}
            prayer_map = dict(self.rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE])
            for ch in self.plan.channels:
                base = vol_map[ch.id]
                if ch.role == ChannelRole.INSTRUMENT_BAND:
                    low_map[ch.id] = _clamp_ch(ch.id, base - 10.0)
                    prayer_map[ch.id] = _clamp_ch(ch.id, -16.0)
                elif ch.role == ChannelRole.INSTRUMENT_GUITAR:
                    low_map[ch.id] = _clamp_ch(ch.id, base - 2.0)
                    prayer_map[ch.id] = _clamp_ch(ch.id, -16.0)
                elif ch.role == ChannelRole.INSTRUMENT_PAD:
                    low_map[ch.id] = _clamp_ch(ch.id, base - 1.0)
                    prayer_map[ch.id] = _clamp_ch(ch.id, base - 7.0)
                elif ch.role == ChannelRole.VOCAL_BG:
                    low_map[ch.id] = _clamp_ch(ch.id, base - 3.0)
                elif ch.role == ChannelRole.SPOKEN_PRIMARY:
                    low_map[ch.id] = _clamp_ch(ch.id, base + 2.0)
                else:
                    low_map[ch.id] = _clamp_ch(ch.id, base)
            self.rules.phase_baselines_db[ServicePhase.WORSHIP_LOW] = low_map
            self.rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE] = prayer_map
        elif source_phase == ServicePhase.WORSHIP_LOW:
            high_map: dict[str, float] = {}
            prayer_map = dict(self.rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE])
            for ch in self.plan.channels:
                base = vol_map[ch.id]
                if ch.role == ChannelRole.INSTRUMENT_BAND:
                    high_map[ch.id] = _clamp_ch(ch.id, base + 10.0)
                    prayer_map[ch.id] = _clamp_ch(ch.id, -16.0)
                elif ch.role == ChannelRole.INSTRUMENT_GUITAR:
                    high_map[ch.id] = _clamp_ch(ch.id, base + 2.0)
                    prayer_map[ch.id] = _clamp_ch(ch.id, -16.0)
                elif ch.role == ChannelRole.INSTRUMENT_PAD:
                    high_map[ch.id] = _clamp_ch(ch.id, base + 1.0)
                    prayer_map[ch.id] = _clamp_ch(ch.id, base - 6.0)
                elif ch.role == ChannelRole.VOCAL_BG:
                    high_map[ch.id] = _clamp_ch(ch.id, base + 3.0)
                elif ch.role == ChannelRole.SPOKEN_PRIMARY:
                    high_map[ch.id] = _clamp_ch(ch.id, base - 2.0)
                else:
                    high_map[ch.id] = _clamp_ch(ch.id, base)
            self.rules.phase_baselines_db[ServicePhase.WORSHIP_HIGH] = high_map
            self.rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE] = prayer_map
        elif source_phase == ServicePhase.SPOKEN_WORD:
            for ch in self.plan.channels:
                if ch.role == ChannelRole.SPOKEN_PRIMARY:
                    base = vol_map[ch.id]
                    self.rules.phase_baselines_db[ServicePhase.PRAYER_UNDERSCORE][ch.id] = _clamp_ch(
                        ch.id, base
                    )
                    self.rules.phase_baselines_db[ServicePhase.WORSHIP_LOW][ch.id] = _clamp_ch(
                        ch.id, base - 18.0
                    )
                    self.rules.phase_baselines_db[ServicePhase.WORSHIP_HIGH][ch.id] = _clamp_ch(
                        ch.id, base - 20.0
                    )
        elif source_phase == ServicePhase.PRAYER_UNDERSCORE:
            for ch in self.plan.channels:
                if ch.role == ChannelRole.SPOKEN_PRIMARY:
                    self.rules.phase_baselines_db[ServicePhase.SPOKEN_WORD][ch.id] = _clamp_ch(
                        ch.id, vol_map[ch.id]
                    )

    def _save_service_plan_atomic(self) -> Path | None:
        if self.plan_path is None:
            return None
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.plan.model_dump(mode="json"), indent=2) + "\n"
        tmp_path = self.plan_path.with_name(f".{self.plan_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(serialized)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.plan_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return self.plan_path

    async def _collect_calibration_blocks(self, duration_s: float = 5.0) -> list[np.ndarray]:
        clamped_dur = max(0.1, min(30.0, float(duration_s)))
        if not self.headless:
            # Allow live PortAudio blocks to accumulate over the requested capture window
            deadline = time.monotonic() + clamped_dur
            while time.monotonic() < deadline:
                self.audio_tap.poll_metrics()
                await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if hasattr(self.audio_tap, "recent_raw_blocks"):
            blocks = self.audio_tap.recent_raw_blocks(duration_s=clamped_dur)
            if blocks:
                return blocks
        if self._recent_audio_blocks:
            return list(self._recent_audio_blocks)[-50:]
        return [np.full((960, 16), 1e-4, dtype=np.float32)]


# ---------------------------------------------------------------------------
# FastAPI WebSocket Application Factory
# ---------------------------------------------------------------------------


def _extract_bearer_token(websocket: WebSocket) -> str | None:
    auth = websocket.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    query_tok = websocket.query_params.get("token")
    if query_tok:
        return query_tok.strip()
    return None


def create_app(engine: AetherMixEngine, *, auto_start_loop: bool = True) -> FastAPI:
    """Create the FastAPI app hosting ws://127.0.0.1:8765/ws with Bearer token auth."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await engine.start(start_loop=auto_start_loop)
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="AetherMix Engine v5.1", version="0.5.1", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    async def health() -> dict[str, Any]:
        hb = engine.build_heartbeat_payload()
        return {
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "autonomy_mode": engine.safety_firewall.autonomy_mode.value,
            "active_phase": engine.scene_classifier.active_phase.value,
            "heartbeat": hb.model_dump(mode="json"),
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        provided = _extract_bearer_token(websocket)
        if not provided or not hmac.compare_digest(provided, engine.token):
            await websocket.close(code=4401, reason="Unauthorized")
            return

        await websocket.accept()
        try:
            # Expect protocol v5.1 hello handshake
            raw_hello = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            msg_type = raw_hello.get("type")
            req_id = raw_hello.get("req_id")
            payload = raw_hello.get("payload") or {}
            proto_ver = str(payload.get("protocol_version") or raw_hello.get("protocol_version") or "")

            if msg_type != "hello" or proto_ver not in (PROTOCOL_VERSION, "5.0"):
                err = engine._next_envelope(
                    "error",
                    {
                        "req_id": req_id,
                        "ok": False,
                        "code": "HANDSHAKE_FAILED",
                        "message": f"Expected hello with protocol_version '{PROTOCOL_VERSION}', got {msg_type!r}/{proto_ver!r}",
                    },
                    req_id=req_id,
                )
                await websocket.send_json(err)
                await websocket.close(code=1008, reason="Handshake failed")
                return

            engine._ws_clients.add(websocket)
            hello_ack = engine._next_envelope(
                "ack",
                {
                    "req_id": req_id,
                    "ok": True,
                    "code": "HELLO_OK",
                    "message": "connected",
                    "protocol_version": PROTOCOL_VERSION,
                },
                req_id=req_id,
            )
            await websocket.send_json(hello_ack)
            await websocket.send_json(
                engine._next_envelope("state_snapshot", engine.build_state_snapshot())
            )
            await websocket.send_json(
                engine._next_envelope("queued_reviews", engine.build_queued_reviews_payload())
            )

            while True:
                data = await websocket.receive_json()
                rpc_type = str(data.get("type", ""))
                rpc_req_id = data.get("req_id")
                rpc_payload = data.get("payload") or {}
                if not isinstance(rpc_payload, dict):
                    rpc_payload = {}
                resp = await engine.handle_rpc(rpc_type, rpc_payload, req_id=rpc_req_id)
                await websocket.send_json(resp)
        except WebSocketDisconnect:
            pass
        except TimeoutError:
            await websocket.close(code=1008, reason="Hello timeout")
        finally:
            engine._ws_clients.discard(websocket)

    return app


def build_engine_from_files(
    service_plan_path: Path = DEFAULT_SERVICE_PLAN_PATH,
    mixing_rules_path: Path = DEFAULT_MIXING_RULES_PATH,
    *,
    headless: bool = False,
    bridge_socket_path: Path = DEFAULT_BRIDGE_SOCKET_PATH,
    token_path: Path = DEFAULT_TOKEN_PATH,
    db_path: Path = DEFAULT_ENGINE_DB_PATH,
) -> AetherMixEngine:
    plan = ServicePlan.model_validate_json(service_plan_path.read_text(encoding="utf-8"))
    rules = MixingRules.model_validate_json(mixing_rules_path.read_text(encoding="utf-8"))
    return AetherMixEngine(
        plan=plan,
        rules=rules,
        plan_path=service_plan_path,
        rules_path=mixing_rules_path,
        backup_dir=DEFAULT_BACKUP_DIR,
        db_path=db_path,
        bridge_socket_path=bridge_socket_path,
        token_path=token_path,
        headless=headless,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherMix v5.1 AI Live Sound Engine")
    parser.add_argument("--service-plan", type=Path, default=DEFAULT_SERVICE_PLAN_PATH)
    parser.add_argument("--mixing-rules", type=Path, default=DEFAULT_MIXING_RULES_PATH)
    parser.add_argument("--bridge-socket", type=Path, default=DEFAULT_BRIDGE_SOCKET_PATH)
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_ENGINE_DB_PATH)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--headless", action="store_true", help="Run without PortAudio/CLAP hardware")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = build_engine_from_files(
        service_plan_path=args.service_plan,
        mixing_rules_path=args.mixing_rules,
        headless=args.headless,
        bridge_socket_path=args.bridge_socket,
        token_path=args.token_path,
        db_path=args.db_path,
    )
    app = create_app(engine, auto_start_loop=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
