"""CLI entrypoint for AI-Linker."""
from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

from ai_linker import Agent, AIConnector, chain, link
from ai_linker.pipelines import DebateLoop, ParallelConsensus, SupervisorPipeline
from ai_linker.utils.display import console, print_banner, print_pipeline_summary, print_step


def cmd_providers(args: argparse.Namespace) -> None:
    print_banner()
    from rich.table import Table

    table = Table(title="Supported AI Providers & Swap Syntax", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="bold yellow")
    table.add_column("Prefix Syntax", style="green")
    table.add_column("Popular Models", style="white")
    table.add_column("Environment Variable", style="dim")

    table.add_row("OpenAI", "openai:model", "gpt-4o, gpt-4o-mini, o1-preview, o3-mini", "OPENAI_API_KEY")
    table.add_row("Anthropic", "claude:model", "claude-3-5-sonnet-latest, claude-3-opus", "ANTHROPIC_API_KEY")
    table.add_row("Google", "gemini:model", "gemini-1.5-pro, gemini-1.5-flash, gemini-2.0", "GEMINI_API_KEY")
    table.add_row("DeepSeek", "deepseek:model", "deepseek-chat, deepseek-reasoner", "DEEPSEEK_API_KEY")
    table.add_row("Ollama", "ollama:model", "llama3, qwen2.5, mistral, deepseek-r1", "OLLAMA_BASE_URL (Local)")
    table.add_row("Generic/vLLM", "generic:model", "Custom endpoint / vLLM / Groq", "OPENAI_BASE_URL")
    table.add_row("Mock", "mock:model", "Offline testing / dry-run", "None (Zero API keys needed)")

    console.print(table)


def cmd_chat(args: argparse.Namespace) -> None:
    print_banner()
    model_spec = args.model or "mock:chat-bot"
    console.print(f"[bold green]Starting interactive session with:[/] [cyan]{model_spec}[/]")
    console.print("[dim]Type 'exit' or 'quit' to end session.[/dim]\n")

    connector = link(model_spec)
    agent = Agent(name="Assistant", role="Helpful AI Partner", connector=connector)

    while True:
        try:
            user_input = console.input("[bold blue]You ❯ [/bold blue]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            resp = agent.chat_sync(user_input)
            console.print(f"[bold green]Agent ❯ [/bold green]{resp.content}\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break


def cmd_run(args: argparse.Namespace) -> None:
    print_banner()
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[bold red]Error: Config file not found at {config_path}[/bold red]")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pipeline_type = data.get("type", "sequential").lower()
    task = args.task or data.get("task", "Analyze this project requirements.")

    agents_dict = {}
    for ag_conf in data.get("agents", []):
        agent = Agent(
            name=ag_conf["name"],
            role=ag_conf["role"],
            connector=ag_conf.get("model", "mock:default"),
            system_prompt=ag_conf.get("system_prompt"),
        )
        agents_dict[ag_conf["name"]] = agent

    console.print(f"[bold yellow]Task:[/] {task}\n")

    if pipeline_type == "sequential":
        ordered_agents = [agents_dict[name] for name in data.get("order", list(agents_dict.keys()))]
        p = chain(*ordered_agents, name=data.get("name", "CustomRelay"))
    elif pipeline_type == "parallel":
        judge_name = data.get("judge")
        judge_agent = agents_dict.get(judge_name) if judge_name else None
        p = ParallelConsensus(list(agents_dict.values()), judge=judge_agent, name=data.get("name", "CustomParallel"))
    else:
        console.print(f"[bold red]Unknown pipeline type: {pipeline_type}[/bold red]")
        sys.exit(1)

    result = p.run_sync(task)
    for step in result.steps:
        print_step(step)
    print_pipeline_summary(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Linker CLI — Multi-Agent Orchestrator")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # chat command
    p_chat = subparsers.add_parser("chat", help="Start an interactive chat with any model")
    p_chat.add_argument("-m", "--model", default="mock:assistant", help="Model spec (e.g. claude:claude-3-5-sonnet, openai:gpt-4o, mock:test)")

    # run command
    p_run = subparsers.add_parser("run", help="Run a multi-agent pipeline from a YAML config")
    p_run.add_argument("config", help="Path to pipeline YAML configuration")
    p_run.add_argument("-t", "--task", default=None, help="Override task prompt")

    # providers command
    subparsers.add_parser("providers", help="List all supported providers and syntax")

    args = parser.parse_args()
    if args.subcommand == "chat":
        cmd_chat(args)
    elif args.subcommand == "run":
        cmd_run(args)
    elif args.subcommand == "providers":
        cmd_providers(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
