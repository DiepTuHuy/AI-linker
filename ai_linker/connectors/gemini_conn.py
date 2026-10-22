"""Google Gemini connector implementation."""
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


class GeminiConnector(BaseAIConnector):
    """Connector for Google Gemini API (gemini-1.5-pro, gemini-1.5-flash, gemini-2.0, etc.)."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        base_url = base_url or os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @property
    def provider(self) -> str:
        return "gemini"

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise AuthenticationError("No Gemini API key provided. Set GEMINI_API_KEY.")
        return key

    async def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        key = self._get_api_key()
        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent?key={key}"

        contents = []
        system_instruction = None

        for m in messages:
            if m.role == Role.SYSTEM:
                system_instruction = {"parts": [{"text": m.content}]}
            else:
                gemini_role = "user" if m.role == Role.USER else "model"
                contents.append({"role": gemini_role, "parts": [{"text": m.content}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        generation_config: Dict[str, Any] = {}
        if temperature is not None or self.temperature is not None:
            generation_config["temperature"] = temperature if temperature is not None else self.temperature
        if max_tokens or self.max_tokens:
            generation_config["maxOutputTokens"] = max_tokens or self.max_tokens

        if generation_config:
            payload["generationConfig"] = generation_config

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
            except httpx.RequestError as e:
                raise AIConnectorError(f"Network error connecting to Gemini API: {e}")

        latency = (time.time() - t0) * 1000

        if resp.status_code == 400 and "API_KEY_INVALID" in resp.text:
            raise AuthenticationError(f"Gemini API key is invalid: {resp.text}")
        elif resp.status_code == 429:
            raise RateLimitError(f"Gemini Rate Limit Exceeded: {resp.text}")
        elif resp.status_code != 200:
            raise AIConnectorError(f"Gemini Error ({resp.status_code}): {resp.text}")

        data = resp.json()
        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        candidates_tokens = usage_meta.get("candidatesTokenCount", 0)

        return ModelResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=candidates_tokens,
                total_tokens=prompt_tokens + candidates_tokens,
            ),
            latency_ms=latency,
            raw_response=data,
            finish_reason=candidates[0].get("finishReason") if candidates else None,
        )
