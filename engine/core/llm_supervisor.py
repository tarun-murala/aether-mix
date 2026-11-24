"""Async Ollama Supervisory LLM Client (SPEC v5.1 §5.3 Tier 2 & §6.3).

Features:
- Warm-pinning via keep_alive="60m"
- Strict Ollama JSON Schema `format=OLLAMA_RESPONSE_JSON_SCHEMA`
- Hard code-level enforcement: `proposed_patches=[]` whenever
  `trigger_type != "operator_critique"` (advisory-only on phase_transition / safety_event)
- Sanitizes any hallucinated unknown channel_ids or forbidden spoken_primary mutes
- Graceful fallback LLMSupervisorResponse on timeout, network failure, or malformed JSON
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from engine.core.protocol import (
    OLLAMA_RESPONSE_JSON_SCHEMA,
    ChannelRole,
    LLMSupervisorResponse,
    ProposedPatch,
    RuleParameter,
    ServicePlan,
)

TriggerType = Literal["phase_transition", "safety_event", "operator_critique"]


class LLMSupervisor:
    """Async Ollama client for supervisory reasoning and critique-to-rule translation."""

    def __init__(
        self,
        plan: ServicePlan,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.plan = plan
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.model = plan.ollama_model
        self.think = plan.ollama_think
        self.ollama_ok: bool = False
        self.last_latency_ms: float | None = None

        self._channel_map = {ch.id: ch for ch in plan.channels}
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs: dict[str, Any] = {
                "base_url": self.base_url,
                "timeout": httpx.Timeout(self.timeout_s),
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx AsyncClient."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def warm_up(self) -> bool:
        """Warm-load the configured Ollama model and pin it with keep_alive='60m'."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
            "keep_alive": "60m",
        }
        if not self.think:
            payload["think"] = False

        t0 = time.perf_counter()
        try:
            client = self._get_client()
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.ollama_ok = True
            return True
        except (httpx.HTTPError, ValueError, RuntimeError):
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.ollama_ok = False
            return False

    def _build_system_prompt(self, trigger_type: TriggerType) -> str:
        channel_desc = ", ".join(
            f"{ch.id} ({ch.label}, role={ch.role.value})" for ch in self.plan.channels
        )
        advisory_clause = (
            "You MAY propose up to 4 typed rule patches matching the operator's critique."
            if trigger_type == "operator_critique"
            else "This is an AUTOMATIC event (advisory only). You MUST return proposed_patches as an empty list []."
        )
        return (
            "You are AetherMix's Supervisory Live Sound Engineer for Apple MainStage.\n"
            f"Valid channels: {channel_desc}.\n"
            "Rules:\n"
            "1. Never propose conditional_mute on any spoken_primary channel.\n"
            "2. Parameter bounds: vol_db_offset [-12, +6], reverb_db_offset [-18, +12], "
            "duck_depth_db [-12, 0], conditional_attenuate [-36, 0], conditional_mute [0, 1].\n"
            "3. Default expires_on to 'segment_change' for ephemeral_session patches.\n"
            f"4. {advisory_clause}\n"
            "Respond strictly with JSON conforming to the provided schema."
        )

    def _build_user_prompt(
        self,
        trigger_type: TriggerType,
        telemetry_summary: dict[str, Any],
        critique_text: str | None,
    ) -> str:
        body: dict[str, Any] = {
            "trigger_type": trigger_type,
            "telemetry": telemetry_summary,
        }
        if critique_text is not None:
            body["operator_critique"] = critique_text
        return json.dumps(body, default=str)

    def _sanitize_patches(self, patches: list[ProposedPatch]) -> list[ProposedPatch]:
        valid: list[ProposedPatch] = []
        for patch in patches:
            ch = self._channel_map.get(patch.channel_id)
            if ch is None:
                continue
            if patch.parameter == RuleParameter.CONDITIONAL_MUTE and ch.role == ChannelRole.SPOKEN_PRIMARY:
                continue
            if patch.condition is not None and patch.condition.when_channel not in self._channel_map:
                continue
            valid.append(patch)
        return valid

    def _fallback_response(self, trigger_type: TriggerType, reason: str) -> LLMSupervisorResponse:
        summary = f"LLM unavailable ({trigger_type}); Tier-1 deterministic mix active"
        return LLMSupervisorResponse(
            thought_summary=summary[:140],
            reasoning=reason[:500],
            confidence=0.0,
            proposed_patches=[],
        )

    async def analyze_event(
        self,
        trigger_type: TriggerType,
        telemetry_summary: dict[str, Any],
        critique_text: str | None = None,
    ) -> LLMSupervisorResponse:
        """Analyze a phase transition, safety event, or operator critique via Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(trigger_type)},
                {
                    "role": "user",
                    "content": self._build_user_prompt(trigger_type, telemetry_summary, critique_text),
                },
            ],
            "stream": False,
            "keep_alive": "60m",
            "format": OLLAMA_RESPONSE_JSON_SCHEMA,
            "options": {"temperature": 0.1},
        }
        if not self.think:
            payload["think"] = False

        t0 = time.perf_counter()
        try:
            client = self._get_client()
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        except httpx.TimeoutException:
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.ollama_ok = False
            return self._fallback_response(
                trigger_type,
                f"Ollama request timed out after {self.timeout_s:.1f}s.",
            )
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.ollama_ok = False
            return self._fallback_response(
                trigger_type,
                f"Ollama request failed ({type(exc).__name__}): {exc}",
            )

        raw_content: Any = None
        if isinstance(data, dict):
            msg = data.get("message")
            if isinstance(msg, dict) and "content" in msg:
                raw_content = msg["content"]
            elif "response" in data:
                raw_content = data["response"]

        if raw_content is None:
            self.ollama_ok = False
            return self._fallback_response(trigger_type, "Ollama returned empty message content.")

        try:
            if isinstance(raw_content, str):
                parsed = LLMSupervisorResponse.model_validate_json(raw_content)
            elif isinstance(raw_content, dict):
                parsed = LLMSupervisorResponse.model_validate(raw_content)
            else:
                raise TypeError(f"Unexpected content type: {type(raw_content).__name__}")
        except (ValidationError, ValueError, TypeError) as exc:
            self.ollama_ok = False
            return self._fallback_response(
                trigger_type,
                f"Ollama returned invalid JSON schema payload: {exc}",
            )

        self.ollama_ok = True

        # Hard code-level safety invariant: non-critique triggers are strictly advisory-only
        if trigger_type != "operator_critique":
            parsed.proposed_patches = []
        else:
            parsed.proposed_patches = self._sanitize_patches(parsed.proposed_patches)

        return parsed
