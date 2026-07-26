"""Run tracing.

Without this module the system has failover but no *evidence* of failover, and the
Phase 3 report would be an assertion rather than a measurement. Every LLM call records
which provider served it and at what fallback depth; every failed attempt records its
error class.

The trace is also the ground truth for Phase 6's citation check: a URL the agent cited
but never actually fetched is a hallucination provable by string comparison, no judge
required.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMAttempt:
    """One deployment attempt inside a single logical completion."""

    attempt_idx: int
    provider: str | None
    served: bool
    latency_ms: float
    error_class: str | None = None
    error_message: str | None = None


@dataclass
class LLMCallRecord:
    """One logical completion, however many providers it took."""

    call_id: str
    alias: str
    served_by: str | None
    fallback_depth: int
    latency_ms: float
    attempts: list[LLMAttempt] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class ToolCallRecord:
    tool: str
    role: str
    allowed: bool
    reason: str | None
    args: dict[str, Any]
    latency_ms: float = 0.0
    url: str | None = None
    flagged: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    question: str | None = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # --- derived views used by the reports ---------------------------------------
    @property
    def fetched_urls(self) -> set[str]:
        """URLs actually retrieved. Phase 6 checks citations against this set."""
        return {t.url for t in self.tool_calls if t.allowed and t.url}

    @property
    def max_fallback_depth(self) -> int:
        return max((c.fallback_depth for c in self.llm_calls), default=0)

    @property
    def used_fallback(self) -> bool:
        return self.max_fallback_depth > 0

    def providers_used(self) -> set[str]:
        return {c.served_by for c in self.llm_calls if c.served_by}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_current_trace: ContextVar[RunTrace | None] = ContextVar("gateway_run_trace", default=None)


def get_trace() -> RunTrace | None:
    return _current_trace.get()


def set_trace(trace: RunTrace | None):
    return _current_trace.set(trace)


class trace_run:
    """Context manager binding a RunTrace to the current context."""

    def __init__(self, trace: RunTrace | None = None, question: str | None = None):
        self.trace = trace or RunTrace(question=question)
        self._token = None

    def __enter__(self) -> RunTrace:
        self._token = _current_trace.set(self.trace)
        return self.trace

    def __exit__(self, *exc) -> bool:
        if self._token is not None:
            _current_trace.reset(self._token)
        return False
