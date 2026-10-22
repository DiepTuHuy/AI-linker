"""Ollama local model connector implementation."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ai_linker.connectors.base import AIConnectorError, BaseAIConnector
from ai_linker.types import Message, ModelResponse, Usage


class OllamaConnector(BaseAIConnector):
    """Connector for local Ollama instances (Llama 3, Qwen 2.5, Mistral, DeepSeek-R1-distill)."""

    def __init__(
        self,
        model: str = "llama3:latest",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        super().__init__(model=model, base_url=base_url, **kwargs)

    @property
    def provider(self) -> str:
        return "ollama"

    async def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        url = f"{self.base_url.rstrip('/')}/api/chat"

        formatted = [{"role": m.role.value, "content": m.content} for m in messages]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "stream": False,
        }

        options: Dict[str, Any] = {}
        if temperature is not None or self.temperature is not None:
            options["temperature"] = temperature if temperature is not None else self.temperature
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens or self.max_tokens

        if options:
            payload["options"] = options

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
            except httpx.RequestError as e:
                raise AIConnectorError(
                    f"Failed to connect to Ollama at {url}. Is Ollama running? Error: {e}"
                )

        latency = (time.time() - t0) * 1000

        if resp.status_code != 200:
            raise AIConnectorError(f"Ollama API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return ModelResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.provider,
            usage=Usage(
                prompt_tokens=prompt_eval,
                completion_tokens=eval_count,
                total_tokens=prompt_eval + eval_count,
            ),
            latency_ms=latency,
            raw_response=data,
            finish_reason=data.get("done_reason"),
        )
