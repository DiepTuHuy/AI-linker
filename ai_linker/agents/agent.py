"""Agent class with pluggable AI Connector and persistent role/memory."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ai_linker.agents.memory import BaseMemory, BufferMemory
from ai_linker.connectors.base import BaseAIConnector
from ai_linker.connectors import AIConnector
from ai_linker.types import Message, ModelResponse, Role, StreamChunk


class Agent:
    """An autonomous AI Agent with a defined role, memory, and hot-swappable AI model connector."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: Optional[str] = None,
        connector: Optional[Union[BaseAIConnector, str]] = None,
        memory: Optional[BaseMemory] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or f"You are {name}, a helpful AI acting as: {role}."
        self.memory = memory if memory is not None else BufferMemory()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if connector is None:
            # Default to mock if nothing provided
            self.connector: BaseAIConnector = AIConnector.from_string("mock:agent-default")
        elif isinstance(connector, str):
            self.connector = AIConnector.from_string(connector)
        else:
            self.connector = connector

    def set_connector(self, connector: Union[BaseAIConnector, str]) -> "Agent":
        """
        Hot-swap the underlying AI model/provider in a single line.
        Supports spec string (e.g. 'anthropic:claude-3-5-sonnet') or connector instance.
        """
        if isinstance(connector, str):
            self.connector = AIConnector.from_string(connector)
        else:
            self.connector = connector
        return self

    def _prepare_messages(self, user_input: Optional[str] = None) -> List[Message]:
        messages: List[Message] = []
        # Prepend system prompt with agent identity
        sys_msg = Message.system(
            f"[Agent Identity: {self.name} | Role: {self.role}]\n{self.system_prompt}"
        )
        messages.append(sys_msg)

        # Append memory history
        messages.extend(self.memory.get_messages())

        # Append current user turn
        if user_input:
            messages.append(Message.user(user_input))

        return messages

    async def chat(self, user_input: str, **kwargs: Any) -> ModelResponse:
        """Asynchronously send message to agent and record in memory."""
        self.memory.add_user_message(user_input)
        messages = self._prepare_messages()

        response = await self.connector.chat_with_fallback(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        self.memory.add_assistant_message(response.content)
        return response

    def chat_sync(self, user_input: str, **kwargs: Any) -> ModelResponse:
        """Synchronously chat with agent."""
        self.memory.add_user_message(user_input)
        messages = self._prepare_messages()

        response = self.connector.chat_sync(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        self.memory.add_assistant_message(response.content)
        return response

    async def stream(self, user_input: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream response tokens from agent."""
        self.memory.add_user_message(user_input)
        messages = self._prepare_messages()

        final_content = ""
        async for chunk in self.connector.stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        ):
            final_content = chunk.accumulated
            yield chunk

        self.memory.add_assistant_message(final_content)

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()

    def __rshift__(self, other: Union["Agent", Any]) -> Any:
        """Support sequential chaining syntax: agent1 >> agent2."""
        from ai_linker.pipelines.sequential import SequentialPipeline
        if isinstance(other, Agent):
            return SequentialPipeline([self, other])
        elif hasattr(other, "agents"):
            return SequentialPipeline([self] + list(other.agents))
        raise TypeError(f"Cannot chain Agent with type {type(other)}")

    def __repr__(self) -> str:
        return f"<Agent name='{self.name}' role='{self.role}' connector={self.connector}>"
