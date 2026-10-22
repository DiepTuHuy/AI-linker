"""Tests for Agents and Memory."""
import pytest
from ai_linker import Agent, BufferMemory, WindowMemory
from ai_linker.connectors import MockConnector
from ai_linker.types import Message


def test_agent_creation_and_memory():
    mock = MockConnector(model="agent-m", custom_response="I understand your request.")
    agent = Agent(name="Alice", role="Researcher", connector=mock)

    resp = agent.chat_sync("Can you summarize quantum computing?")
    assert resp.content == "I understand your request."

    # Check memory recorded both turns
    msgs = agent.memory.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "Can you summarize quantum computing?"
    assert msgs[1].content == "I understand your request."


def test_agent_hot_swap_connector():
    agent = Agent(name="Bob", role="Engineer", connector="mock:v1")
    assert agent.connector.model == "v1"

    # Hot-swap to another model in 1 line
    agent.set_connector("mock:v2")
    assert agent.connector.model == "v2"

    agent.set_connector(MockConnector(model="v3-instance"))
    assert agent.connector.model == "v3-instance"


def test_window_memory():
    mem = WindowMemory(max_turns=2)
    mem.add_user_message("msg 1")
    mem.add_assistant_message("rep 1")
    mem.add_user_message("msg 2")
    mem.add_assistant_message("rep 2")
    mem.add_user_message("msg 3")
    mem.add_assistant_message("rep 3")

    msgs = mem.get_messages()
    # 2 turns * 2 = 4 messages max
    assert len(msgs) == 4
    assert msgs[0].content == "msg 2"
    assert msgs[-1].content == "rep 3"


def test_agent_chaining_syntax():
    a1 = Agent("A1", "Planner", connector="mock:m1")
    a2 = Agent("A2", "Coder", connector="mock:m2")
    a3 = Agent("A3", "Reviewer", connector="mock:m3")

    pipeline = a1 >> a2 >> a3
    assert len(pipeline.agents) == 3
    assert pipeline.agents[0].name == "A1"
    assert pipeline.agents[2].name == "A3"
