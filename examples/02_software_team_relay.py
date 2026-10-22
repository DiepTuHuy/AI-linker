"""
Example 02: Software Team Relay Pipeline.
Demonstrates chaining specialized agents:
  Planner (Claude) >> Coder (DeepSeek) >> Reviewer (GPT-4o)
"""
from ai_linker import Agent
from ai_linker.utils.display import print_step, print_pipeline_summary

def main():
    # Define team members with their distinct roles and preferred models
    planner = Agent(
        name="Astra",
        role="Software Architect & Planner",
        connector="mock:claude-3-5-sonnet",
        system_prompt="You create structured, bulleted engineering architecture blueprints."
    )

    coder = Agent(
        name="Sol",
        role="Senior Python Engineer",
        connector="mock:deepseek-chat",
        system_prompt="You write robust, PEP8 compliant Python code with type annotations."
    )

    reviewer = Agent(
        name="Sentinel",
        role="Security & Quality QA Auditor",
        connector="mock:gpt-4o",
        system_prompt="You review code for security vulnerabilities, edge cases, and test coverage."
    )

    # Chain agents using the >> operator
    dev_team = planner >> coder >> reviewer

    task = "Build an in-memory key-value cache with TTL expiration and thread safety."
    print(f"🚀 Running Software Team Relay on Task:\n'{task}'\n")

    result = dev_team.run_sync(task)

    for step in result.steps:
        print_step(step)

    print_pipeline_summary(result)


if __name__ == "__main__":
    main()
