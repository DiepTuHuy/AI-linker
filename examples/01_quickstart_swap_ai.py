"""
Example 01: Quickstart & Hot-Swapping Any Model in 1 Line.
Demonstrates how AI-Linker allows you to switch between Claude, GPT, Gemini,
DeepSeek, Ollama, or Mock models seamlessly.
"""
from ai_linker import Agent, link

def main():
    # 1. Create an agent with default model
    agent = Agent(
        name="Assistant",
        role="Product Consultant",
        connector="mock:mock-v1"
    )

    print(f"Initial model: {agent.connector.model} ({agent.connector.provider})")
    resp = agent.chat_sync("Hello! Give me a quick tip on building SaaS.")
    print("Agent Response:\n", resp.content)
    print("-" * 50)

    # 2. Hot-swap to Anthropic Claude in ONE line!
    # (If you have ANTHROPIC_API_KEY set, uncomment: agent.set_connector("claude:claude-3-5-sonnet-latest"))
    agent.set_connector("mock:claude-3-5-sonnet")
    print(f"\nHot-swapped to: {agent.connector.model} ({agent.connector.provider})")
    resp = agent.chat_sync("How should I price the MVP?")
    print("Agent Response:\n", resp.content)
    print("-" * 50)

    # 3. Hot-swap to DeepSeek R1 in ONE line!
    agent.set_connector("mock:deepseek-r1")
    print(f"\nHot-swapped to: {agent.connector.model} ({agent.connector.provider})")
    resp = agent.chat_sync("What tech stack do you recommend?")
    print("Agent Response:\n", resp.content)


if __name__ == "__main__":
    main()
