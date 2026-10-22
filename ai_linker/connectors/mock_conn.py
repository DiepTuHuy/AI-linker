"""Mock AI Connector for testing, CI/CD, and offline demonstration."""
from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

from ai_linker.connectors.base import BaseAIConnector
from ai_linker.types import Message, ModelResponse, StreamChunk, Usage


class MockConnector(BaseAIConnector):
    """Predictable mock connector without needing any API key."""

    def __init__(
        self,
        model: str = "mock-model",
        custom_response: Optional[str] = None,
        response_map: Optional[Dict[str, str]] = None,
        fail_with_error: Optional[Exception] = None,
        latency_seconds: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.custom_response = custom_response
        self.response_map = response_map or {}
        self.fail_with_error = fail_with_error
        self.latency_seconds = latency_seconds
        self.call_history: List[List[Message]] = []

    @property
    def provider(self) -> str:
        return "mock"

    async def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.call_history.append(messages)

        if self.fail_with_error:
            raise self.fail_with_error

        last_prompt = messages[-1].content if messages else ""

        # Check response map or fallback
        if last_prompt in self.response_map:
            content = self.response_map[last_prompt]
        elif self.custom_response is not None:
            content = self.custom_response
        else:
            content = f"[{self.model}] Mock response to: {last_prompt[:120]}"

        prompt_len = sum(len(m.content) for m in messages) // 4
        comp_len = len(content) // 4

        return ModelResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            usage=Usage(
                prompt_tokens=max(1, prompt_len),
                completion_tokens=max(1, comp_len),
                total_tokens=max(2, prompt_len + comp_len),
            ),
            latency_ms=self.latency_seconds * 1000,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        words = resp.content.split(" ")
        accum = ""
        for i, word in enumerate(words):
            delta = word + (" " if i < len(words) - 1 else "")
            accum += delta
            is_final = (i == len(words) - 1)
            yield StreamChunk(delta=delta, accumulated=accum, is_final=is_final)
