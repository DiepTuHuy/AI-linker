"""Rich UI display utilities for AI-Linker."""
from __future__ import annotations

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ai_linker.pipelines.base import PipelineResult, PipelineStep

console = Console()


def print_banner() -> None:
    text = Text()
    text.append("⚡ AI-LINKER ", style="bold cyan")
    text.append("v0.1.0 — Universal Multi-Agent Orchestrator\n", style="bold yellow")
    text.append("Connect & Swap Any AI Model (Claude, GPT, Gemini, DeepSeek, Ollama) in 1 Line\n", style="dim italic")
    console.print(Panel(text, border_style="cyan"))


def print_step(step: PipelineStep) -> None:
    header = f"Step {step.step_index}: [{step.agent_name}] — {step.agent_role} ({step.provider}:{step.model})"
    content = step.output_text
    meta = f"Latency: {step.latency_ms:.1f}ms"

    panel = Panel(
        content,
        title=f"[bold green]{header}[/bold green]",
        subtitle=f"[dim]{meta}[/dim]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def print_pipeline_summary(result: PipelineResult) -> None:
    table = Table(title=f"🚀 Pipeline Execution Summary: {result.pipeline_name}", show_header=True, header_style="bold magenta")
    table.add_column("Step", style="dim", width=6)
    table.add_column("Agent", style="bold cyan")
    table.add_column("Role", style="yellow")
    table.add_column("Provider / Model", style="green")
    table.add_column("Latency (ms)", justify="right")

    for s in result.steps:
        table.add_row(
            str(s.step_index),
            s.agent_name,
            s.agent_role,
            f"{s.provider}:{s.model}",
            f"{s.latency_ms:.1f}",
        )

    console.print(table)
    console.print(
        f"\n[bold green]✔ Complete![/bold green] Total latency: [cyan]{result.total_latency_ms:.1f}ms[/cyan] | "
        f"Total tokens: [cyan]{result.total_tokens}[/cyan]\n"
    )
