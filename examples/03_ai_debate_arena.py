"""
Example 03: Multi-AI Debate Arena.
Two different AI models debate a technical dilemma over 2 rounds,
and a Judge delivers the final synthesized verdict.
"""
from ai_linker import Agent
from ai_linker.pipelines import DebateLoop
from ai_linker.utils.display import print_step, print_pipeline_summary

def main():
    agent_sql = Agent(
        name="SQL_Advocate",
        role="Database Engineer (Pro-Relational / ACID)",
        connector="mock:claude-3-5-sonnet"
    )

    agent_nosql = Agent(
        name="NoSQL_Advocate",
        role="Cloud Architect (Pro-Distributed / Flexible Schema)",
        connector="mock:deepseek-r1"
    )

    judge = Agent(
        name="Chief_Judge",
        role="Principal Systems Architect",
        connector="mock:gpt-4o"
    )

    arena = DebateLoop(
        debaters=[agent_sql, agent_nosql],
        judge=judge,
        rounds=2,
        name="SQL vs NoSQL Architecture Debate"
    )

    topic = "Should an early-stage fintech startup choose Postgres or MongoDB for financial ledger transactions?"
    print(f"⚔️ Starting AI Debate Arena on topic:\n'{topic}'\n")

    result = arena.run_sync(topic)

    for step in result.steps:
        print_step(step)

    print_pipeline_summary(result)


if __name__ == "__main__":
    main()
