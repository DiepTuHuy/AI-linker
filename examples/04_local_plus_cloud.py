"""
Example 04: Combining Local Ollama Models with Cloud Models.
Demonstrates privacy/cost optimization:
Local Ollama model filters and redacts data first, then passes to Cloud model.
"""
from ai_linker import Agent
from ai_linker.utils.display import print_step, print_pipeline_summary

def main():
    # Local model handles private extraction on local machine
    local_filter = Agent(
        name="Local_Filter",
        role="Privacy Guard & Extractor (Runs on local Ollama)",
        connector="mock:ollama-llama3"
    )

    # Cloud model handles deep creative synthesis
    cloud_synthesizer = Agent(
        name="Cloud_Reasoning",
        role="Senior Strategist (Cloud LLM)",
        connector="mock:claude-3-5-sonnet"
    )

    hybrid_pipeline = local_filter >> cloud_synthesizer

    task = "Customer support call transcript: extract anonymized sentiment and key complaints."
    print(f"🔒 Running Hybrid Local + Cloud Pipeline on:\n'{task}'\n")

    result = hybrid_pipeline.run_sync(task)

    for step in result.steps:
        print_step(step)

    print_pipeline_summary(result)


if __name__ == "__main__":
    main()
