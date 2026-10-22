"""Conversation Memory management for Agents."""
from __future__ import annotations

import abc
from typing import List

from ai_linker.types import Message, Role


class BaseMemory(abc.ABC):
    """Abstract base class for agent memory."""

    @abc.abstractmethod
    def add_message(self, message: Message) -> None:
        pass

    @abc.abstractmethod
    def get_messages(self) -> List[Message]:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass

    def add_user_message(self, content: str) -> None:
        self.add_message(Message.user(content))

    def add_assistant_message(self, content: str) -> None:
        self.add_message(Message.assistant(content))

    def add_system_message(self, content: str) -> None:
        self.add_message(Message.system(content))


class BufferMemory(BaseMemory):
    """Stores full chronological conversation history."""

    def __init__(self) -> None:
        self.messages: List[Message] = []

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def get_messages(self) -> List[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()


class WindowMemory(BaseMemory):
    """Keeps only the most recent N message pairs (turns)."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self.system_message: Message | None = None
        self.history: List[Message] = []

    def add_message(self, message: Message) -> None:
        if message.role == Role.SYSTEM:
            self.system_message = message
        else:
            self.history.append(message)
            max_messages = self.max_turns * 2
            if len(self.history) > max_messages:
                self.history = self.history[-max_messages:]

    def get_messages(self) -> List[Message]:
        msgs = []
        if self.system_message:
            msgs.append(self.system_message)
        msgs.extend(self.history)
        return msgs

    def clear(self) -> None:
        self.system_message = None
        self.history.clear()
