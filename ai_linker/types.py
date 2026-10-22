"""Data structures and types for AI-Linker."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role = Role.USER
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def system(cls, content: str, name: Optional[str] = None) -> "Message":
        return cls(role=Role.SYSTEM, content=content, name=name)

    @classmethod
    def user(cls, content: str, name: Optional[str] = None) -> "Message":
        return cls(role=Role.USER, content=content, name=name)

    @classmethod
    def assistant(cls, content: str, name: Optional[str] = None) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, name=name)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    def __str__(self) -> str:
        return self.content


class StreamChunk(BaseModel):
    delta: str
    accumulated: str = ""
    is_final: bool = False
