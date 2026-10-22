"""Tests for Multi-Agent Orchestration Pipelines."""
import pytest
from ai_linker import Agent, MockConnector
from ai_linker.pipelines import SequentialPipeline, ParallelConsensus, DebateLoop, SupervisorPipeline


def test_sequential_pipeline():
    c1 = MockConnector(custom_response="Architecture Plan: REST API")
    c2 = MockConnector(custom_response="def api(): return 200")
    c3 = MockConnector(custom_response="Code Approved: 100% test pass")

    planner = Agent("Planner", "Architect", connector=c1)
    coder = Agent("Coder", "Developer", connector=c2)
    reviewer = Agent("Reviewer", "QA Lead", connector=c3)

    pipeline = planner >> coder >> reviewer
    result = pipeline.run_sync("Build user auth")

    assert result.success is True
    assert len(result.steps) == 3
    assert result.final_output == "Code Approved: 100% test pass"
    assert result.steps[0].agent_name == "Planner"
    assert result.steps[1].agent_name == "Coder"
    assert result.steps[2].agent_name == "Reviewer"


def test_parallel_consensus():
    c1 = MockConnector(custom_response="Use PostgreSQL")
    c2 = MockConnector(custom_response="Use MongoDB")
    c_judge = MockConnector(custom_response="PostgreSQL is superior for relational integrity.")

    ag1 = Agent("DB_A", "Relational Advocate", connector=c1)
    ag2 = Agent("DB_B", "Document Advocate", connector=c2)
    judge = Agent("LeadArchitect", "Decision Maker", connector=c_judge)

    consensus = ParallelConsensus([ag1, ag2], judge=judge)
    result = consensus.run_sync("Which database for e-commerce?")

    assert result.success is True
    assert len(result.steps) == 3  # 2 candidates + 1 judge
    assert "PostgreSQL" in result.final_output


def test_debate_loop():
    d1 = MockConnector(custom_response="Monolith is faster to iterate initially.")
    d2 = MockConnector(custom_response="Microservices prevent tech debt scale bottlenecks.")
    j = MockConnector(custom_response="Verdict: Start with modular monolith, split later.")

    ag1 = Agent("MonolithFan", "Pro-Monolith", connector=d1)
    ag2 = Agent("MicroserviceFan", "Pro-Microservices", connector=d2)
    judge = Agent("Judge", "Impartial Evaluator", connector=j)

    debate = DebateLoop(debaters=[ag1, ag2], judge=judge, rounds=2)
    result = debate.run_sync("Monolith vs Microservices for a 5-person startup?")

    # 2 rounds * 2 debaters + 1 judge = 5 steps
    assert len(result.steps) == 5
    assert "modular monolith" in result.final_output


def test_supervisor_pipeline():
    sup_conn = MockConnector(custom_response="Plan: 1. Research backend, 2. Design UI.")
    w1_conn = MockConnector(custom_response="Backend designed with FastAPI.")
    w2_conn = MockConnector(custom_response="UI mockup designed with Tailwind.")

    supervisor = Agent("Astra", "Lead Conductor", connector=sup_conn)
    worker1 = Agent("Sol_BE", "Backend Specialist", connector=w1_conn)
    worker2 = Agent("Sol_FE", "Frontend Specialist", connector=w2_conn)

    sup_pipe = SupervisorPipeline(
        supervisor=supervisor,
        workers={"backend": worker1, "frontend": worker2},
    )

    result = sup_pipe.run_sync("Create fullstack prototype")
    assert result.success is True
    # Steps: Supervisor plan (1) + 2 workers (2) + Supervisor consolidation (1) = 4 steps
    assert len(result.steps) == 4
