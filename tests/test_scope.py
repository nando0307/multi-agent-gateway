"""Scope-validation tests -- PLAN.md Phase 4.

Table-driven, because the value of this layer is coverage of the boring cases. The
interesting assertion is the last one: there is no path from an agent to a tool that
skips the check.
"""

from __future__ import annotations

import pytest

from gateway.llm.events import trace_run
from gateway.security.scope import RunBudget, ScopePolicy, check_url
from gateway.tools.registry import ToolDenied, ToolRunner, ToolSpec


@pytest.fixture
def policy():
    return ScopePolicy(resolve_dns=False)


@pytest.fixture
def budget():
    return RunBudget()


BLOCKED_URLS = [
    ("http://example.com/page", "scheme_not_allowed"),
    ("file:///etc/passwd", "scheme_not_allowed"),
    ("ftp://example.com/x", "scheme_not_allowed"),
    ("data:text/html,<script>", "scheme_not_allowed"),
    ("https://169.254.169.254/latest/meta-data/", "host_denylisted"),
    ("https://127.0.0.1:8080/admin", "private_address"),
    ("https://10.0.0.5/internal", "private_address"),
    ("https://192.168.1.1/router", "private_address"),
    ("https://[::1]/x", "private_address"),
    ("https://localhost/admin", "host_denylisted"),
    ("https://metadata.google.internal/computeMetadata/v1/", "host_denylisted"),
    ("https://printer.local/status", "host_denylisted"),
]


@pytest.mark.parametrize("url,rule", BLOCKED_URLS)
def test_dangerous_urls_are_blocked(url, rule):
    decision = check_url(url, resolve_dns=False)
    assert not decision.allowed
    assert decision.rule == rule


@pytest.mark.parametrize(
    "url", ["https://example.com/article", "https://en.wikipedia.org/wiki/Battery", "https://arxiv.org/abs/2401.00001"]
)
def test_ordinary_urls_are_allowed(url):
    assert check_url(url, resolve_dns=False).allowed


@pytest.mark.parametrize(
    "role,tool,expected",
    [
        ("researcher", "web_search", True),
        ("researcher", "fetch_url", True),
        ("planner", "web_search", False),
        ("planner", "fetch_url", False),
        # The writer has no tools at all, so injected text reaching the synthesis step
        # has nothing to trigger.
        ("synthesizer", "fetch_url", False),
        ("synthesizer", "web_search", False),
    ],
)
def test_role_tool_matrix(policy, budget, role, tool, expected):
    args = {"query": "batteries"} if tool == "web_search" else {"url": "https://example.com"}
    assert policy.check(role, tool, args, budget).allowed is expected


def test_unknown_role_and_unknown_tool_are_denied(policy, budget):
    assert not policy.check("attacker", "web_search", {"query": "x"}, budget).allowed
    assert not policy.check("researcher", "run_shell", {"cmd": "ls"}, budget).allowed


def test_argument_validation(policy, budget):
    assert not policy.check("researcher", "web_search", {"query": ""}, budget).allowed
    assert not policy.check("researcher", "web_search", {"query": 42}, budget).allowed
    assert not policy.check("researcher", "fetch_url", {}, budget).allowed
    long = policy.check("researcher", "web_search", {"query": "x" * 5000}, budget)
    assert not long.allowed and long.rule == "query_too_long"


def test_budgets_stop_a_runaway_loop(policy):
    """An injected 'keep searching forever' cannot spend more than the budget."""
    budget = RunBudget(max_searches=3)
    for _ in range(3):
        assert policy.check("researcher", "web_search", {"query": "q"}, budget).allowed
        budget.record_search()
    denied = policy.check("researcher", "web_search", {"query": "q"}, budget)
    assert not denied.allowed and denied.rule == "budget_searches"


def test_byte_budget_blocks_further_fetches(policy):
    budget = RunBudget(max_bytes=1000)
    budget.record_fetch(1200)
    denied = policy.check("researcher", "fetch_url", {"url": "https://example.com"}, budget)
    assert denied.rule == "budget_bytes"


# --- the property that makes the layer worth having -------------------------------
def _runner(role="researcher"):
    runner = ToolRunner(role, policy=ScopePolicy(resolve_dns=False), budget=RunBudget())
    runner.register(ToolSpec("fetch_url", lambda url: f"content of {url}", "fetch"))
    runner.register(ToolSpec("web_search", lambda query: [f"hit for {query}"], "search"))
    return runner


def test_runner_denies_before_the_tool_executes():
    executed = []
    runner = ToolRunner("researcher", policy=ScopePolicy(resolve_dns=False), budget=RunBudget())
    runner.register(ToolSpec("fetch_url", lambda url: executed.append(url), "fetch"))
    with pytest.raises(ToolDenied):
        runner.call("fetch_url", url="https://169.254.169.254/latest/meta-data/")
    assert executed == [], "the tool ran despite being denied"


def test_synthesizer_cannot_call_any_tool():
    with pytest.raises(ToolDenied):
        _runner("synthesizer").call("fetch_url", url="https://example.com")


def test_allow_and_deny_are_both_traced():
    runner = _runner()
    with trace_run() as trace:
        runner.call("fetch_url", url="https://example.com/a")
        with pytest.raises(ToolDenied):
            runner.call("fetch_url", url="https://127.0.0.1/x")
    assert [t.allowed for t in trace.tool_calls] == [True, False]
    assert trace.fetched_urls == {"https://example.com/a"}
