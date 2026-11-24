"""Deterministic headless Audio & Bridge IPC simulators for AetherMix tests and service simulation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from engine.core.protocol import (
    BridgeStatePayload,
    ConditionPredicate,
    LLMSupervisorResponse,
    ProposedPatch,
    RuleCondition,
    RuleExpiry,
    RuleParameter,
    RulePhase,
    RuleScope,
    ServicePhase,
    ServicePlan,
)


@dataclass
class SceneAudioConfig:
    """Controls synthetic 16-channel audio generation for a given block/window."""

    phase: ServicePhase = ServicePhase.SPOKEN_WORD
    pastor_speaking: bool = True
    lead_singing: bool = False
    bv_singing: bool = False
    keys_playing: bool = False
    guitars_playing: bool = False
    drums_playing: bool = False
    room_applause: bool = False
    stage_bleed: bool = True
    feedback_channel_idx: int | None = None  # 0-based channel index (e.g. 0 for ch1_pastor)
    feedback_freq_hz: float = 2500.0
    feedback_amp: float = 0.0  # linear amplitude of narrowband howl
    tcc_zero_silence: bool = False

    @classmethod
    def for_phase(cls, phase: ServicePhase) -> SceneAudioConfig:
        if phase == ServicePhase.SPOKEN_WORD:
            return cls(
                phase=phase,
                pastor_speaking=True,
                lead_singing=False,
                bv_singing=False,
                keys_playing=False,
                guitars_playing=False,
                drums_playing=False,
            )
        if phase == ServicePhase.PRAYER_UNDERSCORE:
            return cls(
                phase=phase,
                pastor_speaking=True,
                lead_singing=False,
                bv_singing=False,
                keys_playing=True,
                guitars_playing=False,
                drums_playing=False,
            )
        if phase == ServicePhase.WORSHIP_LOW:
            return cls(
                phase=phase,
                pastor_speaking=False,
                lead_singing=True,
                bv_singing=False,
                keys_playing=True,
                guitars_playing=True,
                drums_playing=False,
            )
        if phase == ServicePhase.WORSHIP_HIGH:
            return cls(
                phase=phase,
                pastor_speaking=False,
                lead_singing=True,
                bv_singing=True,
                keys_playing=True,
                guitars_playing=True,
                drums_playing=True,
            )
        # TRANSITION
        return cls(
            phase=ServicePhase.TRANSITION,
            pastor_speaking=False,
            lead_singing=False,
            bv_singing=False,
            keys_playing=False,
            guitars_playing=False,
            drums_playing=False,
            room_applause=True,
        )


class SyntheticServiceAudioGenerator:
    """Deterministic 48 kHz 16-channel multi-stem church service audio generator."""

    def __init__(self, sample_rate: int = 48000, seed: int = 1337) -> None:
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng(seed)
        self._sample_cursor = 0

    def reset(self, seed: int = 1337) -> None:
        self._rng = np.random.default_rng(seed)
        self._sample_cursor = 0

    def generate_block(
        self,
        config: SceneAudioConfig,
        frames: int = 960,
        master_gain_db: float = 0.0,
    ) -> np.ndarray:
        """Generate one (frames, 16) float32 block at 48 kHz."""
        if config.tcc_zero_silence:
            self._sample_cursor += frames
            return np.zeros((frames, 16), dtype=np.float32)

        idx = np.arange(self._sample_cursor, self._sample_cursor + frames, dtype=np.float64)
        t = idx / float(self.sample_rate)
        self._sample_cursor += frames

        block = np.zeros((frames, 16), dtype=np.float64)

        # Ambient microphone noise floor (~ -58 dBFS); keep room mic (12, 13) tonal HVAC hum when quiet
        noise_amp = 10.0 ** (-58.0 / 20.0)
        block[:, :12] = self._rng.standard_normal((frames, 12)) * noise_amp
        hvac_hum = (10.0 ** (-52.0 / 20.0)) * (
            0.7 * np.sin(2.0 * np.pi * 120.0 * t) + 0.3 * np.sin(2.0 * np.pi * 240.0 * t)
        )
        block[:, 12] = hvac_hum
        block[:, 13] = hvac_hum

        # Keys / Pad (Ch 5-6 -> indices 4, 5) — rich chord at ~ -17 dBFS
        keys_l = np.zeros(frames, dtype=np.float64)
        keys_r = np.zeros(frames, dtype=np.float64)
        if config.keys_playing:
            amp = 0.14 if config.phase in (ServicePhase.WORSHIP_LOW, ServicePhase.WORSHIP_HIGH) else 0.09
            keys_l = amp * (
                0.45 * np.sin(2.0 * np.pi * 261.63 * t)
                + 0.35 * np.sin(2.0 * np.pi * 329.63 * t)
                + 0.25 * np.sin(2.0 * np.pi * 392.00 * t)
                + 0.15 * np.sin(2.0 * np.pi * 523.25 * t)
            )
            keys_r = amp * (
                0.45 * np.sin(2.0 * np.pi * 261.63 * t + 0.4)
                + 0.35 * np.sin(2.0 * np.pi * 329.63 * t + 0.7)
                + 0.25 * np.sin(2.0 * np.pi * 392.00 * t + 1.1)
                + 0.15 * np.sin(2.0 * np.pi * 523.25 * t + 1.4)
            )
            block[:, 4] += keys_l
            block[:, 5] += keys_r

        # Guitars (Ch 7-8 -> indices 6, 7) — strummed harmonics at ~ -16 dBFS
        gtr = np.zeros(frames, dtype=np.float64)
        if config.guitars_playing:
            strum_env = 0.65 + 0.35 * np.abs(np.sin(2.0 * np.pi * 2.5 * t))
            gtr = (
                0.15
                * strum_env
                * (
                    0.45 * np.sin(2.0 * np.pi * 330.0 * t)
                    + 0.35 * np.sin(2.0 * np.pi * 660.0 * t)
                    + 0.25 * np.sin(2.0 * np.pi * 990.0 * t)
                    + 0.15 * np.sin(2.0 * np.pi * 1320.0 * t)
                )
            )
            block[:, 6] += gtr
            block[:, 7] += gtr * 0.95

        # Drums / Bass / Tracks (Ch 9-12 -> indices 8..11) at ~ -14 dBFS wideband / strong voice-band presence
        drums = np.zeros(frames, dtype=np.float64)
        if config.drums_playing:
            beat = np.mod(t, 0.5)
            kick = np.exp(-beat * 18.0) * np.sin(2.0 * np.pi * 65.0 * t)
            snare_t = np.mod(t + 0.25, 0.5)
            snare = np.exp(-snare_t * 22.0) * (
                0.6 * np.sin(2.0 * np.pi * 420.0 * t) + 0.6 * self._rng.standard_normal(frames)
            )
            bass = 0.25 * np.sin(2.0 * np.pi * 110.0 * t) + 0.35 * np.sin(2.0 * np.pi * 550.0 * t)
            hi_hat = 0.25 * self._rng.standard_normal(frames) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * t))
            drums = 0.36 * (kick + snare + bass + hi_hat)
            block[:, 8] += drums
            block[:, 9] += drums * 0.92
            block[:, 10] += 0.30 * (bass + snare * 0.6)
            block[:, 11] += 0.28 * (bass + hi_hat)

        # Stage bleed into vocal/spoken mics (attenuated safely below bleed_offset_db priors)
        if config.stage_bleed:
            band_sum = 0.33 * (0.5 * (keys_l + keys_r) + gtr + drums)
            bleed_pastor = 10.0 ** (-22.0 / 20.0)
            bleed_vox = 10.0 ** (-22.0 / 20.0)
            block[:, 0] += band_sum * bleed_pastor
            block[:, 1] += band_sum * bleed_vox
            block[:, 2] += band_sum * bleed_vox
            block[:, 3] += band_sum * bleed_vox

        # Pastor speech (Ch 1 -> index 0): realistic broadband formant-shaped voiced speech
        # (sweeping pitch + dense harmonics + breathy vocal tract bed -> high VAD, zero single-bin feedback spike)
        if config.pastor_speaking:
            syllable_env = 0.65 + 0.35 * np.sin(2.0 * np.pi * 3.8 * t)
            # Natural intonational pitch contour sweeping across multiple FFT bins (115..185 Hz)
            f0 = 150.0 + 35.0 * np.sin(2.0 * np.pi * 2.1 * t) + 12.0 * np.sin(2.0 * np.pi * 5.3 * t)
            ph = 2.0 * np.pi * np.cumsum(f0) / float(self.sample_rate)
            # Rich harmonic glottal train (k=2..14) + shaped vocal noise bed
            glottal = np.zeros(frames, dtype=np.float64)
            for k in range(2, 15):
                glottal += (1.0 / (k ** 0.75)) * np.sin(k * ph + 0.3 * k)
            # Add smooth broadband voice-band energy so local 21-bin median floor stays realistic
            raw_noise = self._rng.standard_normal(frames)
            # Simple 1-pole smoothed broadband noise (~300-3000 Hz body)
            breath = np.convolve(raw_noise, np.array([0.2, 0.3, 0.3, 0.2]), mode="same")
            speech = 0.13 * syllable_env * (glottal + 1.4 * breath)
            block[:, 0] += speech

        # Lead Vocal singing (Ch 2 -> index 1): rich multi-harmonic sung tone with musical vibrato
        # (high autocorrelation harmonicity >0.65, broadband harmonic spacing -> zero false feedback cuts)
        if config.lead_singing:
            vib = 240.0 + 22.0 * np.sin(2.0 * np.pi * 5.5 * t) + 18.0 * np.sin(2.0 * np.pi * 1.1 * t)
            ph = 2.0 * np.pi * np.cumsum(vib) / float(self.sample_rate)
            lead_harm = np.zeros(frames, dtype=np.float64)
            for k in range(1, 13):
                lead_harm += (1.0 / (k ** 0.7)) * np.sin(k * ph)
            lead_bed = np.convolve(self._rng.standard_normal(frames), np.array([0.25, 0.5, 0.25]), mode="same")
            block[:, 1] += 0.075 * (lead_harm + 0.45 * lead_bed)

        # Backup Vocals (Ch 3, 4 -> indices 2, 3)
        if config.bv_singing:
            vib1 = 295.0 + 20.0 * np.sin(2.0 * np.pi * 5.2 * t + 0.5)
            vib2 = 365.0 + 24.0 * np.sin(2.0 * np.pi * 4.8 * t + 1.1)
            ph1 = 2.0 * np.pi * np.cumsum(vib1) / float(self.sample_rate)
            ph2 = 2.0 * np.pi * np.cumsum(vib2) / float(self.sample_rate)
            bv1 = sum((1.0 / (k ** 0.7)) * np.sin(k * ph1) for k in range(1, 10))
            bv2 = sum((1.0 / (k ** 0.7)) * np.sin(k * ph2) for k in range(1, 10))
            bed1 = np.convolve(self._rng.standard_normal(frames), np.array([0.25, 0.5, 0.25]), mode="same")
            bed2 = np.convolve(self._rng.standard_normal(frames), np.array([0.25, 0.5, 0.25]), mode="same")
            block[:, 2] += 0.055 * (bv1 + 0.45 * bed1)
            block[:, 3] += 0.055 * (bv2 + 0.45 * bed2)

        # Room Ambience (Ch 13-14 -> indices 12, 13)
        if config.room_applause:
            applause_amp = 0.12
            block[:, 12] += self._rng.standard_normal(frames) * applause_amp
            block[:, 13] += self._rng.standard_normal(frames) * applause_amp

        # Optional narrowband feedback howl (strictly on target mic, absent from instruments)
        if config.feedback_channel_idx is not None and config.feedback_amp > 0.0:
            howl = config.feedback_amp * np.sin(2.0 * np.pi * config.feedback_freq_hz * t)
            block[:, config.feedback_channel_idx] += howl

        # Post-fader Master stereo mix (Ch 15-16 -> indices 14, 15)
        gain_lin = 10.0 ** (master_gain_db / 20.0)
        mix_l = (
            block[:, 0]
            + block[:, 1]
            + 0.7 * block[:, 2]
            + 0.7 * block[:, 3]
            + block[:, 4]
            + block[:, 6]
            + block[:, 8]
        ) * 0.35 * gain_lin
        mix_r = (
            block[:, 0]
            + block[:, 1]
            + 0.7 * block[:, 2]
            + 0.7 * block[:, 3]
            + block[:, 5]
            + block[:, 7]
            + block[:, 9]
        ) * 0.35 * gain_lin
        block[:, 14] = np.clip(mix_l, -0.99, 0.99)
        block[:, 15] = np.clip(mix_r, -0.99, 0.99)

        return np.clip(block, -1.0, 1.0).astype(np.float32)


class MockBridgeClient:
    """In-memory deterministic mock matching engine.core.bridge_client.BridgeClient.

    Enforces bridge-level bypass latch, human_seq fence, and 250ms in-flight drop window.
    """

    def __init__(
        self,
        plan: ServicePlan,
        on_human_touch: Callable[[dict[str, Any]], Any] | None = None,
        on_human_cc: Callable[[dict[str, Any]], Any] | None = None,
        on_action_button: Callable[[dict[str, Any]], Any] | None = None,
        on_state_change: Callable[[BridgeStatePayload], Any] | None = None,
        on_reconnect: Callable[[BridgeStatePayload], Any] | None = None,
        initial_cc: dict[str, int] | None = None,
    ) -> None:
        self.plan = plan
        self._on_human_touch = on_human_touch
        self._on_human_cc = on_human_cc
        self._on_action_button = on_action_button
        self._on_state_change = on_state_change
        self._on_reconnect = on_reconnect

        default_cc: dict[str, int] = {}
        for ch in plan.channels:
            default_cc[f"{ch.id}:vol"] = 64
            if ch.reverb_cc is not None:
                default_cc[f"{ch.id}:reverb"] = 32
        if initial_cc:
            default_cc.update(initial_cc)

        self._state = BridgeStatePayload(
            last_sent_cc=default_cc,
            human_seq={k: 0 for k in default_cc},
            bypass_latched=False,
            controller_connected=True,
            engine_connected=False,
            pickup_pending={},
            restored_from_disk_at=None,
        )
        self._connected = False
        self._human_guard_until: dict[str, float] = {}
        self.sent_agent_writes: list[dict[str, Any]] = []
        self.dropped_agent_writes: list[dict[str, Any]] = []
        self.midi_learn_pulses: list[tuple[str, str]] = []
        self.heartbeat_count = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> BridgeStatePayload:
        return self._state

    @property
    def human_seq(self) -> dict[str, int]:
        return dict(self._state.human_seq)

    async def connect(self) -> None:
        self._connected = True
        self._state.engine_connected = True
        if self._on_state_change is not None:
            res = self._on_state_change(self._state)
            if hasattr(res, "__await__"):
                await res

    async def close(self) -> None:
        self._connected = False
        self._state.engine_connected = False

    async def get_state(self) -> BridgeStatePayload:
        return self._state

    async def send_heartbeat(self) -> None:
        self.heartbeat_count += 1

    async def send_midi_learn_pulse(self, channel_id: str, control: str) -> None:
        self.midi_learn_pulses.append((channel_id, control))

    async def send_agent_cc(
        self,
        channel_id: str,
        control: str,
        cc: int,
        ack_human_seq: int = 0,
    ) -> bool:
        key = f"{channel_id}:{control}"
        cur_seq = self._state.human_seq.get(key, 0)
        now = time.monotonic()
        if (
            self._state.bypass_latched
            or ack_human_seq < cur_seq
            or now < self._human_guard_until.get(key, 0.0)
        ):
            self.dropped_agent_writes.append(
                {
                    "channel_id": channel_id,
                    "control": control,
                    "cc": int(cc),
                    "ack_human_seq": ack_human_seq,
                    "cur_human_seq": cur_seq,
                }
            )
            return False

        cc_clamped = max(0, min(127, int(cc)))
        self._state.last_sent_cc[key] = cc_clamped
        self.sent_agent_writes.append(
            {
                "channel_id": channel_id,
                "control": control,
                "cc": cc_clamped,
                "ack_human_seq": ack_human_seq,
                "ts_ms": int(time.time() * 1000),
            }
        )
        return True

    async def simulate_human_touch(
        self,
        channel_id: str,
        control: str = "vol",
        picked_up: bool = False,
        source: str = "hardware",
    ) -> int:
        key = f"{channel_id}:{control}"
        seq = self._state.human_seq.get(key, 0) + 1
        self._state.human_seq[key] = seq
        self._human_guard_until[key] = time.monotonic() + 0.25
        evt = {
            "channel_id": channel_id,
            "control": control,
            "human_seq": seq,
            "picked_up": picked_up,
            "source": source,
            "ts_ms": int(time.time() * 1000),
        }
        if self._on_human_touch is not None:
            res = self._on_human_touch(evt)
            if hasattr(res, "__await__"):
                await res
        return seq

    async def simulate_human_cc(
        self,
        channel_id: str,
        control: str,
        cc: int,
        source: str = "hardware",
    ) -> int:
        key = f"{channel_id}:{control}"
        seq = self._state.human_seq.get(key, 0) + 1
        self._state.human_seq[key] = seq
        self._human_guard_until[key] = time.monotonic() + 0.25
        cc_clamped = max(0, min(127, int(cc)))
        self._state.last_sent_cc[key] = cc_clamped
        evt = {
            "channel_id": channel_id,
            "control": control,
            "cc": cc_clamped,
            "human_seq": seq,
            "source": source,
            "ts_ms": int(time.time() * 1000),
        }
        if self._on_human_cc is not None:
            res = self._on_human_cc(evt)
            if hasattr(res, "__await__"):
                await res
        return seq

    async def simulate_reconnect(self, restored_from_disk_at: int | None = None) -> None:
        self._connected = True
        self._state.engine_connected = True
        self._state.restored_from_disk_at = restored_from_disk_at
        if self._on_reconnect is not None:
            res = self._on_reconnect(self._state)
            if hasattr(res, "__await__"):
                await res

    async def simulate_state_broadcast(self) -> None:
        if self._on_state_change is not None:
            res = self._on_state_change(self._state)
            if hasattr(res, "__await__"):
                await res

    async def simulate_action_button(self, action: str) -> None:
        if action == "emergency_bypass":
            self._state.bypass_latched = not self._state.bypass_latched
            if self._on_state_change is not None:
                res = self._on_state_change(self._state)
                if hasattr(res, "__await__"):
                    await res
        evt = {"action": action, "ts_ms": int(time.time() * 1000)}
        if self._on_action_button is not None:
            res = self._on_action_button(evt)
            if hasattr(res, "__await__"):
                await res


class MockLLMSupervisor:
    """Deterministic local mock for LLMSupervisor used in E2E and headless simulation."""

    def __init__(self, plan: ServicePlan) -> None:
        self.plan = plan
        self.calls: list[dict[str, Any]] = []
        self.ok = True

    async def warm_up(self) -> bool:
        return self.ok

    async def analyze_event(
        self,
        trigger_type: str,
        telemetry_summary: dict[str, Any],
        critique_text: str | None = None,
    ) -> LLMSupervisorResponse:
        self.calls.append(
            {
                "trigger_type": trigger_type,
                "telemetry_summary": telemetry_summary,
                "critique_text": critique_text,
            }
        )
        if trigger_type != "operator_critique":
            return LLMSupervisorResponse(
                thought_summary=f"Advisory observation for {trigger_type}",
                reasoning="Automatic triggers are strictly advisory; no rule patches emitted.",
                confidence=0.92,
                proposed_patches=[],
            )

        text = (critique_text or "").lower()
        phase_str = str(telemetry_summary.get("active_phase", "ANY"))
        rule_phase = RulePhase.ANY if phase_str == "TRANSITION" else RulePhase(phase_str)

        # Parse simple deterministic test critiques
        if "mute" in text and ("bv" in text or "backup" in text):
            return LLMSupervisorResponse(
                thought_summary="Propose conditional attenuate/mute on backup vocals under speech",
                reasoning="Operator requested muting backup vocals when pastor speaks; requires preview.",
                confidence=0.94,
                proposed_patches=[
                    ProposedPatch(
                        description="Mute BV1 when pastor speaks",
                        scope=RuleScope.CANDIDATE_PERMANENT,
                        expires_on=RuleExpiry.END_OF_SERVICE,
                        phase=RulePhase.ANY,
                        channel_id="ch3_bv1",
                        parameter=RuleParameter.CONDITIONAL_MUTE,
                        value=1.0,
                        condition=RuleCondition(
                            when_channel="ch1_pastor",
                            predicate=ConditionPredicate.SPEECH_OPEN,
                        ),
                    )
                ],
            )
        if "5 db" in text or "+5" in text or "-5" in text or "much louder" in text or "much softer" in text:
            val = -5.0 if ("softer" in text or "-5" in text or "down" in text) else 5.0
            ch = "ch5_keys" if "key" in text else "ch7_guitars"
            return LLMSupervisorResponse(
                thought_summary=f"Propose {val:+.1f} dB adjustment on {ch} (preview required)",
                reasoning="Exceeds +/-3.0 dB auto-apply threshold so operator preview is required.",
                confidence=0.90,
                proposed_patches=[
                    ProposedPatch(
                        description=f"Adjust {ch} by {val:+.1f} dB",
                        scope=RuleScope.CANDIDATE_PERMANENT,
                        expires_on=RuleExpiry.SEGMENT_CHANGE,
                        phase=rule_phase,
                        channel_id=ch,
                        parameter=RuleParameter.VOL_DB_OFFSET,
                        value=val,
                    )
                ],
            )
        # Default small single-channel tweak (auto-applied ephemeral session rule)
        ch = "ch5_keys" if "key" in text else ("ch7_guitars" if "guitar" in text else "ch2_lead_vox")
        val = -2.0 if ("down" in text or "softer" in text or "less" in text or "too loud" in text) else 2.0
        return LLMSupervisorResponse(
            thought_summary=f"Apply gentle {val:+.1f} dB tweak on {ch}",
            reasoning="Single-channel offset within +/-3.0 dB auto-apply gate.",
            confidence=0.91,
            proposed_patches=[
                ProposedPatch(
                    description=f"Nudge {ch} {val:+.1f} dB",
                    scope=RuleScope.EPHEMERAL_SESSION,
                    expires_on=RuleExpiry.SEGMENT_CHANGE,
                    phase=rule_phase,
                    channel_id=ch,
                    parameter=RuleParameter.VOL_DB_OFFSET,
                    value=val,
                )
            ],
        )
