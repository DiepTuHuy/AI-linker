"""Supervisor / Brain-Worker Orchestration Pipeline."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ai_linker.agents.agent import Agent
from ai_linker.pipelines.base import BasePipeline, PipelineResult, PipelineStep


class SupervisorPipeline(BasePipeline):
    """
    A Brain/Supervisor agent creates a plan, delegates specific subtasks
    to specialized Worker agents, and reviews the consolidated result.
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: Dict[str, Agent],
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "SupervisorBrain")
        self.supervisor = supervisor
        self.workers = workers

    async def run(self, initial_input: str, **kwargs: Any) -> PipelineResult:
        t0 = time.time()
        steps: List[PipelineStep] = []
        total_tokens = 0
        step_idx = 1

        worker_descriptions = "\n".join(
            [f"- {key}: {worker.name} ({worker.role})" for key, worker in self.workers.items()]
        )

        plan_prompt = (
            f"Task: {initial_input}\n\n"
            f"Available specialist workers:\n{worker_descriptions}\n\n"
            f"Create a breakdown of subtasks for each applicable worker to execute."
        )

        sup_plan = await self.supervisor.chat(plan_prompt, **kwargs)
        total_tokens += sup_plan.usage.total_tokens
        steps.append(
            PipelineStep(
                step_index=step_idx,
                agent_name=self.supervisor.name,
                agent_role="Supervisor (Planning)",
                model=self.supervisor.connector.model,
                provider=self.supervisor.connector.provider,
                input_text=plan_prompt,
                output_text=sup_plan.content,
                latency_ms=sup_plan.latency_ms,
            )
        )
        step_idx += 1

        # Execute each worker
        worker_outputs: Dict[str, str] = {}
        for key, worker in self.workers.items():
            worker_prompt = (
                f"Master Task: {initial_input}\n"
                f"Supervisor Plan:\n{sup_plan.content}\n\n"
                f"You are the {key} specialist ({worker.role}). Execute your portion of the work."
            )
            w_resp = await worker.chat(worker_prompt, **kwargs)
            total_tokens += w_resp.usage.total_tokens
            worker_outputs[key] = w_resp.content

            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    agent_name=worker.name,
                    agent_role=f"Worker ({key})",
                    model=worker.connector.model,
                    provider=worker.connector.provider,
                    input_text=worker_prompt,
                    output_text=w_resp.content,
                    latency_ms=w_resp.latency_ms,
                )
            )
            step_idx += 1

        # Supervisor consolidates and audits
        worker_summary = "\n\n".join(
            [f"=== Worker [{k}] ({self.workers[k].name}) ===\n{out}" for k, out in worker_outputs.items()]
        )
        audit_prompt = (
            f"Master Task: {initial_input}\n\n"
            f"Workers completed deliverables:\n{worker_summary}\n\n"
            f"Audit and integrate the workers' outputs into the complete, final polished delivery."
        )

        final_resp = await self.supervisor.chat(audit_prompt, **kwargs)
        total_tokens += final_resp.usage.total_tokens
        steps.append(
            PipelineStep(
                step_index=step_idx,
                agent_name=self.supervisor.name,
                agent_role="Supervisor (Consolidation & Audit)",
                model=self.supervisor.connector.model,
                provider=self.supervisor.connector.provider,
                input_text=audit_prompt,
                output_text=final_resp.content,
                latency_ms=final_resp.latency_ms,
            )
        )

        return PipelineResult(
            pipeline_name=self.name,
            initial_input=initial_input,
            final_output=final_resp.content,
            steps=steps,
            total_latency_ms=(time.time() - t0) * 1000,
            total_tokens=total_tokens,
            success=True,
        )
