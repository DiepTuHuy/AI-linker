"""DeepSeek connector implementation."""
from __future__ import annotations

import os
from typing import Any, Optional

from ai_linker.connectors.generic_openai_conn import GenericOpenAIConnector


class DeepSeekConnector(GenericOpenAIConnector):
    """Connector for DeepSeek API (deepseek-chat / V3, deepseek-reasoner / R1)."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @property
    def provider(self) -> str:
        return "deepseek"
