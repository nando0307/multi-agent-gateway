"""HTTP API.

``/research`` is also the surface garak probes (see `scripts/run_garak.sh`), which is why
it accepts a bare question and returns the report as a top-level field: the probes then
traverse the untrusted-content envelope, the scope gate and the eval gate rather than a
bare model.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from gateway.agents.orchestrator import build_runner, run_research
from gateway.eval.gate import QualityGate
from gateway.eval.judge import Judge, JudgeUnavailable
from gateway.llm.router import AllProvidersExhausted, build_gateway
from gateway.security.redaction import register_settings
from gateway.settings import get_settings

app = FastAPI(title="multi-agent-gateway", version="0.1.0")

# Single self-contained page, served same-origin so there is no CORS surface to configure.
# A FileResponse rather than a StaticFiles mount: there is one asset, and a mount would add a
# directory of things served verbatim next to an endpoint that handles untrusted content.
_UI = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_UI, media_type="text/html")

# Worst-case fallback/retry chains can sum past any one client's timeout (multiple LLM
# calls per request, each with its own retry ladder, plus several judge calls with their
# own backoff). Without a server-side deadline the caller just hangs until it gives up on
# its own terms -- which is exactly what turned a single slow request into three garak runs
# crashing outright (garak treats a raw connection timeout as fatal, unlike an HTTP status
# it can skip). Bounding it here means a slow request fails fast and cleanly with a 504.
REQUEST_TIMEOUT_S = 480.0
# ponytail: the abandoned thread keeps running to completion in the background rather than
# being cancelled -- Python has no clean way to kill a running thread. Fine for bounded
# scan traffic (garak, smoke tests); if this becomes user-facing at volume, swap to a
# cancellable async pipeline instead of papering over it with a bigger pool.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="research")

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
    runner = build_runner(tavily_key=_settings.tavily_api_key, parallel_key=_settings.parallel_api_key)
    gate = QualityGate(request.threshold or _settings.gate_threshold, _judge)
    METRICS["requests"] += 1
    future = _executor.submit(run_research, request.question, _gateway, runner, gate=gate)
    try:
        result, trace = future.result(timeout=REQUEST_TIMEOUT_S)
    except AllProvidersExhausted as exc:
        METRICS["exhausted"] += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except concurrent.futures.TimeoutError as exc:
        METRICS["timeout"] += 1
        raise HTTPException(
            status_code=504, detail=f"research exceeded {REQUEST_TIMEOUT_S}s budget"
        ) from exc

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
