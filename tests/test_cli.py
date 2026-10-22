"""Tests for CLI."""
import subprocess
import sys


def test_cli_providers_help():
    proc = subprocess.run(
        [sys.executable, "-m", "ai_linker.cli.main", "providers"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "/Users/dieptuhuy/Documents/AI-linker"},
    )
    assert proc.returncode == 0
    assert "OpenAI" in proc.stdout
    assert "Anthropic" in proc.stdout
    assert "DeepSeek" in proc.stdout
    assert "Ollama" in proc.stdout


def test_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "ai_linker.cli.main", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "/Users/dieptuhuy/Documents/AI-linker"},
    )
    assert proc.returncode == 0
    assert "chat" in proc.stdout
    assert "run" in proc.stdout
