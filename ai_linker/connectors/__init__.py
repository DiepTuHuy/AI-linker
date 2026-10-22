"""Universal Connector Registry and Factory."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from ai_linker.connectors.base import BaseAIConnector
from ai_linker.connectors.generic_openai_conn import GenericOpenAIConnector
from ai_linker.connectors.openai_conn import OpenAIConnector
from ai_linker.connectors.anthropic_conn import AnthropicConnector
from ai_linker.connectors.gemini_conn import GeminiConnector
from ai_linker.connectors.deepseek_conn import DeepSeekConnector
from ai_linker.connectors.ollama_conn import OllamaConnector
from ai_linker.connectors.mock_conn import MockConnector

PROVIDER_REGISTRY: Dict[str, Type[BaseAIConnector]] = {
    "openai": OpenAIConnector,
    "anthropic": AnthropicConnector,
    "claude": AnthropicConnector,
    "gemini": GeminiConnector,
    "google": GeminiConnector,
    "deepseek": DeepSeekConnector,
    "ollama": OllamaConnector,
    "generic": GenericOpenAIConnector,
    "vllm": GenericOpenAIConnector,
    "groq": GenericOpenAIConnector,
    "openrouter": GenericOpenAIConnector,
    "mock": MockConnector,
}


class AIConnector:
    """Factory to instantiate and link any AI model with a single line of code."""

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback: Optional[BaseAIConnector] = None,
        **kwargs: Any,
    ) -> BaseAIConnector:
        """Create a connector instance from provider name and model name."""
        provider_key = provider.lower().strip()
        connector_cls = PROVIDER_REGISTRY.get(provider_key)
        if connector_cls is None:
            # Fallback to Generic OpenAI compatible
            connector_cls = GenericOpenAIConnector

        instance = connector_cls(model=model, api_key=api_key, base_url=base_url, **kwargs)
        if fallback:
            instance.with_fallback(fallback)
        return instance

    @classmethod
    def from_string(cls, spec: str, fallback: Optional[BaseAIConnector] = None, **kwargs: Any) -> BaseAIConnector:
        """
        Parse spec like 'openai:gpt-4o', 'claude:claude-3-5-sonnet', 'mock:test'
        or bare model names with auto-detection.
        """
        spec = spec.strip()
        if ":" in spec:
            provider, model = spec.split(":", 1)
        else:
            # Auto-detect provider from model name
            model = spec
            lower = model.lower()
            if "claude" in lower:
                provider = "anthropic"
            elif "gpt" in lower or "o1" in lower or "o3" in lower:
                provider = "openai"
            elif "gemini" in lower:
                provider = "gemini"
            elif "deepseek" in lower:
                provider = "deepseek"
            elif "mock" in lower:
                provider = "mock"
            else:
                provider = "ollama"

        return cls.create(provider=provider, model=model, fallback=fallback, **kwargs)


__all__ = [
    "BaseAIConnector",
    "OpenAIConnector",
    "AnthropicConnector",
    "GeminiConnector",
    "DeepSeekConnector",
    "OllamaConnector",
    "GenericOpenAIConnector",
    "MockConnector",
    "AIConnector",
    "PROVIDER_REGISTRY",
]
