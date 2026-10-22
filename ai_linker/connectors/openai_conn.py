"""OpenAI connector implementation."""
from __future__ import annotations

import os
from typing import Any, Optional

from ai_linker.connectors.generic_openai_conn import GenericOpenAIConnector


class OpenAIConnector(GenericOpenAIConnector):
    """Connector for official OpenAI API (GPT-4o, GPT-4o-mini, o1, o3, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @property
    def provider(self) -> str:
        return "openai"
