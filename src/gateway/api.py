"""HTTP API.

``/research`` is also the surface garak probes (see `scripts/run_garak.sh`), which is why
it accepts a bare question and returns the report as a top-level field: the probes then
traverse the untrusted-content envelope, the scope gate and the eval gate rather than a
bare model.
"""

from __future__ import annotations

from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gateway.agents.orchestrator import build_runner, run_research
from gateway.eval.gate import QualityGate
from gateway.eval.judge import Judge, JudgeUnavailable
from gateway.llm.router import AllProvidersExhausted, build_gateway
from gateway.security.redaction import register_settings
from gateway.settings import get_settings

app = FastAPI(title="multi-agent-gateway", version="0.1.0")

_settings = get_settings()
register_settings(_settings)
_gateway = build_gateway(settings=_settings)
try:
    _judge = Judge(_settings)
except JudgeUnavailable:
    _judge = None

METRICS: Counter = Counter()
DEPTHS: list[int] = []
SCORES: list[float] = []


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    threshold: float | None = None


class ResearchResponse(BaseModel):
    run_id: str
    report: str
    sources: list[dict]
    scores: dict | None
    gate_passed: bool | None
    warning: str | None
    served_by: list[str]
    fallback_depth: int


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    runner = build_runner(tavily_key=_settings.tavily_api_key)
    gate = QualityGate(request.threshold or _settings.gate_threshold, _judge)
    METRICS["requests"] += 1
    try:
        result, trace = run_research(request.question, _gateway, runner, gate=gate)
    except AllProvidersExhausted as exc:
        METRICS["exhausted"] += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    METRICS["served"] += 1
    for provider in result.served_by:
        METRICS[f"provider:{provider}"] += 1
    DEPTHS.append(trace.max_fallback_depth)
    if result.scores:
        SCORES.append(result.scores["composite"])
        METRICS["gate_passed" if result.gate_passed else "gate_failed"] += 1

    return ResearchResponse(
        run_id=trace.run_id,
        report=result.report,
        sources=[
            {"id": e.source_id, "url": e.url, "title": e.title, "quote_only": e.quote_only}
            for e in result.evidence
        ],
        scores=result.scores,
        gate_passed=result.gate_passed,
        warning=result.warning,
        served_by=result.served_by,
        fallback_depth=trace.max_fallback_depth,
    )


@app.get("/health")
def health() -> dict:
    cooling = []
    try:
        cooling = list(_gateway.router.get_model_ids() or [])
    except Exception:
        pass
    return {
        "status": "ok",
        "providers": list(_gateway.chain),
        "n_providers": len(_gateway.chain),
        "judge_configured": _judge is not None,
        "deployments": cooling,
    }


@app.get("/metrics")
def metrics() -> dict:
    depth_histogram = Counter(DEPTHS)
    return {
        "counters": dict(METRICS),
        "fallback_depth_histogram": {str(k): v for k, v in sorted(depth_histogram.items())},
        "mean_fallback_depth": round(sum(DEPTHS) / len(DEPTHS), 3) if DEPTHS else None,
        "mean_composite_score": round(sum(SCORES) / len(SCORES), 3) if SCORES else None,
        "n_scored": len(SCORES),
    }
