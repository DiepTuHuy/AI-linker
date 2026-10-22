"""Anthropic Claude connector implementation."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ai_linker.connectors.base import (
    AIConnectorError,
    AuthenticationError,
    BaseAIConnector,
    RateLimitError,
)
from ai_linker.types import Message, ModelResponse, Role, Usage


class AnthropicConnector(BaseAIConnector):
    """Connector for Anthropic API (Claude 3.5 Sonnet, Claude 3 Opus, Haiku)."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        anthropic_version: str = "2023-06-01",
        **kwargs: Any,
    ):
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.anthropic_version = anthropic_version

    @property
    def provider(self) -> str:
        return "anthropic"

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise AuthenticationError("No Anthropic API key provided. Set ANTHROPIC_API_KEY.")
        return key

    async def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{self.base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": self._get_api_key(),
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

        system_parts = []
        user_assistant_msgs = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_parts.append(m.content)
            else:
                user_assistant_msgs.append({
                    "role": "user" if m.role == Role.USER else "assistant",
                    "content": m.content,
                })

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": user_assistant_msgs,
            "max_tokens": max_tokens or self.max_tokens or 4096,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature is not None or self.temperature is not None:
            payload["temperature"] = temperature if temperature is not None else self.temperature

        payload.update(self.extra_kwargs)
        payload.update(kwargs)

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as e:
                raise AIConnectorError(f"Network error connecting to Anthropic: {e}")

        latency = (time.time() - t0) * 1000

        if resp.status_code == 401:
            raise AuthenticationError(f"Anthropic 401 Unauthorized: {resp.text}")
        elif resp.status_code == 429:
            raise RateLimitError(f"Anthropic Rate Limit Exceeded: {resp.text}")
        elif resp.status_code != 200:
            raise AIConnectorError(f"Anthropic Error ({resp.status_code}): {resp.text}")

        data = resp.json()
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return ModelResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.provider,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=latency,
            raw_response=data,
            finish_reason=data.get("stop_reason"),
        )
