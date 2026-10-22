"""Parallel Ensemble & Consensus Pipeline."""
from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional

from ai_linker.agents.agent import Agent
from ai_linker.pipelines.base import BasePipeline, PipelineResult, PipelineStep


class ParallelConsensus(BasePipeline):
    """
    Sends the same prompt to multiple agents concurrently,
    then uses an optional Judge/Synthesizer agent to reach a final consensus.
    """

    def __init__(
        self,
        agents: List[Agent],
        judge: Optional[Agent] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "ParallelConsensus")
        if not agents:
            raise ValueError("ParallelConsensus requires at least one Agent.")
        self.agents = list(agents)
        self.judge = judge

    async def run(self, initial_input: str, **kwargs: Any) -> PipelineResult:
        t0 = time.time()
        steps: List[PipelineStep] = []
        total_tokens = 0

        # Execute all agents in parallel
        async def _run_agent(idx: int, ag: Agent):
            resp = await ag.chat(initial_input, **kwargs)
            return idx, ag, resp

        tasks = [_run_agent(i, ag) for i, ag in enumerate(self.agents)]
        results = await asyncio.gather(*tasks)

        candidate_reports = []
        for idx, ag, resp in results:
            total_tokens += resp.usage.total_tokens
            steps.append(
                PipelineStep(
                    step_index=idx + 1,
                    agent_name=ag.name,
                    agent_role=ag.role,
                    model=ag.connector.model,
                    provider=ag.connector.provider,
                    input_text=initial_input,
                    output_text=resp.content,
                    latency_ms=resp.latency_ms,
                )
            )
            candidate_reports.append(f"### [Agent: {ag.name} ({ag.role} | Model: {ag.connector.model})]:\n{resp.content}")

        aggregated_candidates = "\n\n".join(candidate_reports)

        # If judge is provided, synthesize or select
        if self.judge:
            judge_prompt = (
                f"Task: {initial_input}\n\n"
                f"Below are the perspectives/answers from {len(self.agents)} different AI agents:\n\n"
                f"{aggregated_candidates}\n\n"
                f"As the Judge/Synthesizer, critique these responses, identify consensus, "
                f"resolve any contradictions, and produce the definitive best final answer."
            )
            judge_resp = await self.judge.chat(judge_prompt, **kwargs)
            total_tokens += judge_resp.usage.total_tokens
            steps.append(
                PipelineStep(
                    step_index=len(self.agents) + 1,
                    agent_name=self.judge.name,
                    agent_role="Synthesizer / Judge",
                    model=self.judge.connector.model,
                    provider=self.judge.connector.provider,
                    input_text=judge_prompt,
                    output_text=judge_resp.content,
                    latency_ms=judge_resp.latency_ms,
                )
            )
            final_output = judge_resp.content
        else:
            final_output = aggregated_candidates

        return PipelineResult(
            pipeline_name=self.name,
            initial_input=initial_input,
            final_output=final_output,
            steps=steps,
            total_latency_ms=(time.time() - t0) * 1000,
            total_tokens=total_tokens,
            success=True,
        )
