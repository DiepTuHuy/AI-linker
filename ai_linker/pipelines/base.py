"""Abstract base class and data containers for AI Pipelines."""
from __future__ import annotations

import abc
import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PipelineStep(BaseModel):
    step_index: int
    agent_name: str
    agent_role: str
    model: str
    provider: str
    input_text: str
    output_text: str
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    pipeline_name: str
    initial_input: str
    final_output: str
    steps: List[PipelineStep] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None

    def __str__(self) -> str:
        return self.final_output


class BasePipeline(abc.ABC):
    """Base class for all multi-agent orchestration pipelines."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abc.abstractmethod
    async def run(self, initial_input: str, **kwargs: Any) -> PipelineResult:
        """Execute the pipeline asynchronously."""
        pass

    def run_sync(self, initial_input: str, **kwargs: Any) -> PipelineResult:
        """Execute the pipeline synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.run(initial_input, **kwargs)
        if loop and loop.is_running():
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)
