"""LlamaIndex Workflow wiring planner -> researcher -> synthesizer.

Every LLM call in the workflow goes through ``Gateway``, so failover is inherited rather
than reimplemented per agent, and every tool call goes through ``ToolRunner``, so scope
validation cannot be bypassed by a step that forgets to ask.

The ``RunTrace`` produced here is the substrate for everything downstream: Phase 6 checks
citations against the URLs it records, and Phase 7 identifies silent regressions from the
providers it records.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from gateway.agents.base import Evidence, ResearchResult
from gateway.agents.planner import plan
from gateway.agents.researcher import research
from gateway.agents.synthesizer import synthesize
from gateway.llm.events import RunTrace, trace_run
from gateway.llm.model_list import PRIMARY_ALIAS
from gateway.llm.router import Gateway
from gateway.security.redaction import redact_obj
from gateway.security.scope import RunBudget, ScopePolicy
from gateway.tools.registry import ToolRunner, ToolSpec


class PlannedEvent(Event):
    sub_questions: list[str]


class EvidenceEvent(Event):
    sub_questions: list[str]
    evidence: list[Any]


class ResearchWorkflow(Workflow):
    def __init__(self, gateway: Gateway, runner: ToolRunner, *, model: str = PRIMARY_ALIAS, **kw):
        super().__init__(**kw)
        self.gateway = gateway
        self.runner = runner
        self.model = model

    @step
    async def plan_step(self, ev: StartEvent) -> PlannedEvent:
        question = str(ev.question)
        subs = await asyncio.to_thread(plan, question, self.gateway, model=self.model)
        return PlannedEvent(sub_questions=subs)

    @step
    async def research_step(self, ev: PlannedEvent) -> EvidenceEvent:
        evidence = await asyncio.to_thread(research, ev.sub_questions, self.runner)
        return EvidenceEvent(sub_questions=ev.sub_questions, evidence=evidence)

    @step
    async def synthesize_step(self, ev: EvidenceEvent) -> StopEvent:
        report = await asyncio.to_thread(
            synthesize, self._question, list(ev.evidence), self.gateway, model=self.model
        )
        return StopEvent(
            result={
                "report": report,
                "evidence": list(ev.evidence),
                "sub_questions": ev.sub_questions,
            }
        )

    async def run_question(self, question: str) -> dict:
        self._question = question
        return await self.run(question=question)


def build_runner(*, resolve_dns: bool = True, budget: RunBudget | None = None,
                 search_fn=None, fetch_fn=None, tavily_key: str | None = None,
                 parallel_key: str | None = None) -> ToolRunner:
    """Wire the researcher's two tools. Injecting the callables keeps tests offline."""
    from gateway.tools import fetch_url as fetch_mod
    from gateway.tools import web_search as search_mod

    runner = ToolRunner(
        "researcher", policy=ScopePolicy(resolve_dns=resolve_dns), budget=budget or RunBudget()
    )
    runner.register(
        ToolSpec(
            "web_search",
            search_fn or (lambda query: search_mod.search(
                query, api_key=tavily_key, parallel_api_key=parallel_key
            )),
            "Search the public web.",
        )
    )
    runner.register(
        ToolSpec(
            "fetch_url",
            fetch_fn or (lambda url: fetch_mod.fetch(url, resolve_dns=resolve_dns)),
            "Fetch and extract the text of an HTTPS page.",
        )
    )
    return runner


def run_research(
    question: str,
    gateway: Gateway,
    runner: ToolRunner,
    *,
    model: str = PRIMARY_ALIAS,
    gate=None,
    trace: RunTrace | None = None,
    save_dir: Path | None = None,
) -> tuple[ResearchResult, RunTrace]:
    """Synchronous entry point: run the workflow, then apply the quality gate."""
    with trace_run(trace, question=question) as active:
        workflow = ResearchWorkflow(gateway, runner, model=model, timeout=None)
        out = asyncio.run(workflow.run_question(question))

        result = ResearchResult(
            question=question,
            report=out["report"],
            evidence=list(out["evidence"]),
            sub_questions=list(out["sub_questions"]),
            served_by=sorted(p for p in active.providers_used() if p),
        )

        if gate is not None:
            result = gate.apply(result, active, gateway=gateway, runner=runner, model=model)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{active.run_id}.json").write_text(
            json.dumps(redact_obj(active.to_dict()), indent=2, default=str)
        )
    return result, active


__all__ = [
    "Evidence",
    "ResearchResult",
    "ResearchWorkflow",
    "build_runner",
    "run_research",
]
