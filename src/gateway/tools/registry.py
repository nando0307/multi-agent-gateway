"""The single choke point through which every tool call passes.

Agents do not hold callables. They hold a ``ToolRunner`` bound to a role, and the runner
consults ``ScopePolicy`` before dispatching. There is no code path from an agent to a tool
that bypasses the check -- that property is the point, and `tests/test_scope.py` asserts it.

Note what is *absent*: no filesystem tool, no shell tool, no code execution. The strongest
control against tool abuse is not having the tool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from gateway.llm.events import ToolCallRecord, get_trace
from gateway.security.scope import Decision, RunBudget, ScopePolicy


class ToolDenied(PermissionError):
    def __init__(self, decision: Decision, tool: str):
        self.decision = decision
        super().__init__(f"{tool} denied [{decision.rule}]: {decision.reason}")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    description: str


class ToolRunner:
    def __init__(
        self,
        role: str,
        *,
        policy: ScopePolicy | None = None,
        budget: RunBudget | None = None,
        tools: dict[str, ToolSpec] | None = None,
    ):
        self.role = role
        self.policy = policy or ScopePolicy()
        self.budget = budget or RunBudget()
        self.tools = tools if tools is not None else {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def call(self, tool: str, **args: Any) -> Any:
        decision = self.policy.check(self.role, tool, args, self.budget)
        record = ToolCallRecord(
            tool=tool,
            role=self.role,
            allowed=decision.allowed,
            reason=decision.reason,
            args={k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()},
            url=args.get("url"),
        )

        if not decision.allowed:
            self.budget.denials.append(decision.rule)
            self._record(record)
            raise ToolDenied(decision, tool)

        spec = self.tools.get(tool)
        if spec is None:
            record.allowed = False
            record.reason = "tool not registered"
            self._record(record)
            raise ToolDenied(Decision(False, "not_registered", "tool not registered"), tool)

        started = time.perf_counter()
        try:
            result = spec.fn(**args)
        finally:
            record.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            self._record(record)

        if tool == "web_search":
            self.budget.record_search()
        elif tool == "fetch_url":
            self.budget.record_fetch(getattr(result, "n_bytes", 0))
        return result

    @staticmethod
    def _record(record: ToolCallRecord) -> None:
        trace = get_trace()
        if trace is not None:
            trace.tool_calls.append(record)
