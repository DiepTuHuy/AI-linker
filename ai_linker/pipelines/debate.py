"""Adversarial Debate and Consensus Loop."""
from __future__ import annotations

import time
from typing import Any, List, Optional

from ai_linker.agents.agent import Agent
from ai_linker.pipelines.base import BasePipeline, PipelineResult, PipelineStep


class DebateLoop(BasePipeline):
    """
    Two or more agents debate an issue over N rounds,
    refuting and refining their stances before a Judge delivers the verdict.
    """

    def __init__(
        self,
        debaters: List[Agent],
        judge: Agent,
        rounds: int = 2,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "DebateArena")
        if len(debaters) < 2:
            raise ValueError("DebateLoop requires at least 2 debater agents.")
        self.debaters = debaters
        self.judge = judge
        self.rounds = rounds

    async def run(self, initial_input: str, **kwargs: Any) -> PipelineResult:
        t0 = time.time()
        steps: List[PipelineStep] = []
        total_tokens = 0
        step_idx = 1
        debate_transcript: List[str] = []

        for r in range(1, self.rounds + 1):
            for debater in self.debaters:
                if r == 1 and not debate_transcript:
                    prompt = (
                        f"Topic/Problem: {initial_input}\n"
                        f"Round 1: State your initial thesis and well-reasoned argument."
                    )
                else:
                    context = "\n\n".join(debate_transcript[-len(self.debaters):])
                    prompt = (
                        f"Topic: {initial_input}\n"
                        f"Round {r} of {self.rounds}.\n"
                        f"Recent statements from other debaters:\n{context}\n\n"
                        f"Refute counterarguments, defend your reasoning, or concede valid points."
                    )

                resp = await debater.chat(prompt, **kwargs)
                total_tokens += resp.usage.total_tokens

                entry = f"[Round {r} - {debater.name} ({debater.role})]:\n{resp.content}"
                debate_transcript.append(entry)

                steps.append(
                    PipelineStep(
                        step_index=step_idx,
                        agent_name=debater.name,
                        agent_role=f"Debater (Round {r})",
                        model=debater.connector.model,
                        provider=debater.connector.provider,
                        input_text=prompt,
                        output_text=resp.content,
                        latency_ms=resp.latency_ms,
                    )
                )
                step_idx += 1

        # Judge delivers consensus
        full_transcript = "\n\n".join(debate_transcript)
        judge_prompt = (
            f"Topic: {initial_input}\n\n"
            f"Here is the full debate transcript between the debaters:\n\n"
            f"{full_transcript}\n\n"
            f"As the Chief Judge, evaluate the logical coherence, evidence, and persuasiveness of both sides. "
            f"Declare a winner or synthesized compromise, and state the final reasoned conclusion."
        )

        judge_resp = await self.judge.chat(judge_prompt, **kwargs)
        total_tokens += judge_resp.usage.total_tokens

        steps.append(
            PipelineStep(
                step_index=step_idx,
                agent_name=self.judge.name,
                agent_role="Chief Judge",
                model=self.judge.connector.model,
                provider=self.judge.connector.provider,
                input_text=judge_prompt,
                output_text=judge_resp.content,
                latency_ms=judge_resp.latency_ms,
            )
        )

        return PipelineResult(
            pipeline_name=self.name,
            initial_input=initial_input,
            final_output=judge_resp.content,
            steps=steps,
            total_latency_ms=(time.time() - t0) * 1000,
            total_tokens=total_tokens,
            success=True,
        )
