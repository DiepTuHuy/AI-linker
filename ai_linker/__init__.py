"""AI-Linker: Universal Multi-Agent Orchestrator & Pluggable Model Connector Hub."""
from __future__ import annotations

from ai_linker.agents import Agent, BaseMemory, BufferMemory, WindowMemory
from ai_linker.connectors import (
    AIConnector,
    AnthropicConnector,
    BaseAIConnector,
    DeepSeekConnector,
    GeminiConnector,
    GenericOpenAIConnector,
    MockConnector,
    OllamaConnector,
    OpenAIConnector,
)
from ai_linker.pipelines import (
    BasePipeline,
    DebateLoop,
    ParallelConsensus,
    PipelineResult,
    PipelineStep,
    SequentialPipeline,
    SupervisorPipeline,
)
from ai_linker.types import Message, ModelResponse, Role, StreamChunk, Usage

__version__ = "0.1.0"


def link(spec: str, **kwargs) -> BaseAIConnector:
    """Quick helper to create a connector: e.g. `ai_linker.link('claude:claude-3-5-sonnet')`."""
    return AIConnector.from_string(spec, **kwargs)


def create_agent(name: str, role: str, model: str = "mock:default", **kwargs) -> Agent:
    """Quick helper to create an Agent with a bound model."""
    connector = link(model)
    return Agent(name=name, role=role, connector=connector, **kwargs)


def chain(*agents: Agent, name: str = "ChainedRelay") -> SequentialPipeline:
    """Quick helper to chain multiple agents into a sequential pipeline."""
    return SequentialPipeline(list(agents), name=name)


__all__ = [
    "__version__",
    # Agents
    "Agent",
    "BaseMemory",
    "BufferMemory",
    "WindowMemory",
    "create_agent",
    # Connectors
    "AIConnector",
    "BaseAIConnector",
    "OpenAIConnector",
    "AnthropicConnector",
    "GeminiConnector",
    "DeepSeekConnector",
    "OllamaConnector",
    "GenericOpenAIConnector",
    "MockConnector",
    "link",
    # Pipelines
    "BasePipeline",
    "SequentialPipeline",
    "ParallelConsensus",
    "DebateLoop",
    "SupervisorPipeline",
    "PipelineResult",
    "PipelineStep",
    "chain",
    # Types
    "Message",
    "Role",
    "ModelResponse",
    "StreamChunk",
    "Usage",
]
