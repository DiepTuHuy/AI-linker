"""Tests for AI-Linker Connectors."""
import asyncio
from ai_linker.connectors import AIConnector, MockConnector, OpenAIConnector, AnthropicConnector, GeminiConnector
from ai_linker.types import Message


def test_mock_connector_chat_sync():
    conn = MockConnector(model="test-ai", custom_response="Fixed output")
    resp = conn.chat_sync([Message.user("Hello")])
    assert resp.content == "Fixed output"
    assert resp.model == "test-ai"
    assert resp.provider == "mock"
    assert resp.usage.total_tokens > 0


def test_mock_connector_stream():
    async def _test():
        conn = MockConnector(model="test-ai", custom_response="Alpha Beta Gamma")
        chunks = []
        async for chunk in conn.stream([Message.user("Hello")]):
            chunks.append(chunk.delta)
        assert "".join(chunks) == "Alpha Beta Gamma"

    asyncio.run(_test())


def test_factory_string_parsing():
    c1 = AIConnector.from_string("openai:gpt-4o")
    assert isinstance(c1, OpenAIConnector)
    assert c1.model == "gpt-4o"

    c2 = AIConnector.from_string("claude:claude-3-5-sonnet")
    assert isinstance(c2, AnthropicConnector)
    assert c2.model == "claude-3-5-sonnet"

    c3 = AIConnector.from_string("gemini:gemini-1.5-pro")
    assert isinstance(c3, GeminiConnector)
    assert c3.model == "gemini-1.5-pro"

    # Auto-detection
    c4 = AIConnector.from_string("gpt-4o-mini")
    assert isinstance(c4, OpenAIConnector)

    c5 = AIConnector.from_string("claude-3-opus")
    assert isinstance(c5, AnthropicConnector)


def test_connector_fallback():
    async def _test():
        primary = MockConnector(model="primary", fail_with_error=RuntimeError("Quota exceeded"))
        fallback = MockConnector(model="fallback", custom_response="Saved by fallback!")
        primary.with_fallback(fallback)

        resp = await primary.chat_with_fallback([Message.user("Urgent task")])
        assert resp.content == "Saved by fallback!"
        assert resp.model == "fallback"

    asyncio.run(_test())
