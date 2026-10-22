"""Sequential Relay Pipeline (Agent A >> Agent B >> Agent C)."""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Union

from ai_linker.agents.agent import Agent
from ai_linker.pipelines.base import BasePipeline, PipelineResult, PipelineStep


class SequentialPipeline(BasePipeline):
    """
    Executes agents in a sequential relay chain:
    Each agent consumes the output produced by the previous agent.
    """

    def __init__(
        self,
        agents: List[Agent],
        name: Optional[str] = None,
        handoff_formatter: Optional[Callable[[Agent, str, str], str]] = None,
    ):
        super().__init__(name=name or "SequentialRelay")
        if not agents:
            raise ValueError("SequentialPipeline requires at least one Agent.")
        self.agents = list(agents)
        self.handoff_formatter = handoff_formatter or self._default_handoff

    @staticmethod
    def _default_handoff(agent: Agent, previous_output: str, initial_task: str) -> str:
        return (
            f"Original Task: {initial_task}\n\n"
            f"Previous Step Result:\n{previous_output}\n\n"
            f"Now perform your task as {agent.role}."
        )

    def add_agent(self, agent: Agent) -> "SequentialPipeline":
        self.agents.append(agent)
        return self

    def __rshift__(self, other: Union[Agent, "SequentialPipeline"]) -> "SequentialPipeline":
        if isinstance(other, Agent):
            self.agents.append(other)
            return self
        elif isinstance(other, SequentialPipeline):
            self.agents.extend(other.agents)
            return self
        raise TypeError(f"Cannot chain SequentialPipeline with {type(other)}")

    async def run(self, initial_input: str, **kwargs: Any) -> PipelineResult:
        t0 = time.time()
        steps: List[PipelineStep] = []
        current_input = initial_input
        total_tokens = 0

        for i, agent in enumerate(self.agents):
            step_prompt = (
                current_input
                if i == 0
                else self.handoff_formatter(agent, current_input, initial_input)
            )

            resp = await agent.chat(step_prompt, **kwargs)
            total_tokens += resp.usage.total_tokens

            step = PipelineStep(
                step_index=i + 1,
                agent_name=agent.name,
                agent_role=agent.role,
                model=agent.connector.model,
                provider=agent.connector.provider,
                input_text=step_prompt,
                output_text=resp.content,
                latency_ms=resp.latency_ms,
            )
            steps.append(step)
            current_input = resp.content

        return PipelineResult(
            pipeline_name=self.name,
            initial_input=initial_input,
            final_output=current_input,
            steps=steps,
            total_latency_ms=(time.time() - t0) * 1000,
            total_tokens=total_tokens,
            success=True,
        )
