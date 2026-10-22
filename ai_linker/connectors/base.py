"""Abstract base class for all AI connectors in AI-Linker."""
from __future__ import annotations

import abc
import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from ai_linker.types import Message, ModelResponse, StreamChunk


class AIConnectorError(Exception):
    """Base exception for AI connector errors."""
    pass


class AuthenticationError(AIConnectorError):
    """Raised when API key or authentication fails."""
    pass


class RateLimitError(AIConnectorError):
    """Raised when provider rate limit is exceeded."""
    pass


class BaseAIConnector(abc.ABC):
    """Universal interface for any LLM provider."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        fallback_connector: Optional["BaseAIConnector"] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_connector = fallback_connector
        self.extra_kwargs = kwargs

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Name of the provider (e.g. 'openai', 'anthropic', 'gemini')."""
        pass

    @abc.abstractmethod
    async def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send chat messages and return the full ModelResponse."""
        pass

    def chat_sync(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Synchronous wrapper around chat()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if loop and loop.is_running():
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)

    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Default stream implementation falls back to full chat if not overridden."""
        resp = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        yield StreamChunk(delta=resp.content, accumulated=resp.content, is_final=True)

    def with_fallback(self, fallback: "BaseAIConnector") -> "BaseAIConnector":
        """Chain a fallback connector if this connector fails."""
        self.fallback_connector = fallback
        return self

    async def chat_with_fallback(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> ModelResponse:
        """Attempt primary chat; if failed and fallback configured, route to fallback."""
        try:
            return await self.chat(messages, **kwargs)
        except Exception as err:
            if self.fallback_connector is not None:
                return await self.fallback_connector.chat_with_fallback(messages, **kwargs)
            raise err

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider='{self.provider}' model='{self.model}'>"
