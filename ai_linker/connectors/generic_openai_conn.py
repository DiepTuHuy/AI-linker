"""Universal OpenAI-compatible API connector (OpenAI, DeepSeek, Groq, Together, vLLM, OpenRouter)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ai_linker.connectors.base import (
    AIConnectorError,
    AuthenticationError,
    BaseAIConnector,
    RateLimitError,
)
from ai_linker.types import Message, ModelResponse, StreamChunk, Usage


class GenericOpenAIConnector(BaseAIConnector):
    """Connects to any endpoint that adheres to the OpenAI `/v1/chat/completions` spec."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.default_headers = default_headers or {}

    @property
    def provider(self) -> str:
        return "openai_compatible"

    def _get_base_url(self) -> str:
        return (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise AuthenticationError(f"No API key provided for {self.provider} connector.")
        return key

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        formatted = []
        for m in messages:
            item = {"role": m.role.value, "content": m.content}
            if m.name:
                item["name"] = m.name
            formatted.append(item)
        return formatted

    async def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{self._get_base_url()}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
            **self.default_headers,
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens or self.max_tokens:
            payload["max_tokens"] = max_tokens or self.max_tokens

        payload.update(self.extra_kwargs)
        payload.update(kwargs)

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as e:
                raise AIConnectorError(f"Network error connecting to {url}: {e}")

        latency = (time.time() - t0) * 1000

        if resp.status_code == 401:
            raise AuthenticationError(f"Invalid API key for {self.provider}: {resp.text}")
        elif resp.status_code == 429:
            raise RateLimitError(f"Rate limit exceeded on {self.provider}: {resp.text}")
        elif resp.status_code != 200:
            raise AIConnectorError(f"API Error ({resp.status_code}) from {url}: {resp.text}")

        data = resp.json()
        choice = data["choices"][0]
        usage_data = data.get("usage", {})

        return ModelResponse(
            content=choice["message"]["content"] or "",
            model=data.get("model", self.model),
            provider=self.provider,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            latency_ms=latency,
            raw_response=data,
            finish_reason=choice.get("finish_reason"),
        )
