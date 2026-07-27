"""Deterministic failover tests -- PLAN.md Phase 3.

No probability here: each case forces an exact provider configuration and asserts exactly
which provider served the request and at what fallback depth. The probabilistic study
lives in `scripts/chaos_run.py`; this file is what proves the mechanism works at all.
"""

from __future__ import annotations

import pytest

from gateway.llm.events import trace_run
from gateway.llm.router import AllProvidersExhausted
from tests.conftest import EMPTY, FAIL_429, FAIL_500, DEFAULT_CHAIN, make_gateway


def test_healthy_primary_serves_at_depth_zero(messages):
    result = make_gateway().complete(messages)
    assert result.served_by == "gemini"
    assert result.fallback_depth == 0
    assert "gemini" in result.text


@pytest.mark.parametrize("down", DEFAULT_CHAIN[:-1])
def test_single_provider_down_falls_through(down, messages):
    """Each provider individually forced down; the chain must still answer."""
    gw = make_gateway({down: FAIL_500})
    result = gw.complete(messages)
    assert result.served_by != down
    assert result.text


def test_first_two_down_serves_from_third(messages):
    gw = make_gateway({"gemini": FAIL_500, "nim": FAIL_429})
    result = gw.complete(messages)
    assert result.served_by == "openrouter"
    assert result.fallback_depth == 2


def test_all_providers_down_raises_cleanly_and_does_not_hang(messages):
    """The failure mode that matters: exhaustion must surface, not stall."""
    gw = make_gateway({name: FAIL_500 for name in DEFAULT_CHAIN})
    with pytest.raises(AllProvidersExhausted) as exc:
        gw.complete(messages)
    assert "research-primary" in str(exc.value)


def test_only_local_tier_survives(messages):
    """The terminal tier is local and cannot rate-limit -- this is why Y approaches 0."""
    gw = make_gateway({n: FAIL_429 for n in DEFAULT_CHAIN if n != "ollama"})
    result = gw.complete(messages)
    assert result.served_by == "ollama"
    assert result.fallback_depth == 3


def test_trace_records_fallback_depth_and_failed_attempts(messages):
    gw = make_gateway({"gemini": FAIL_500})
    with trace_run(question="q") as trace:
        gw.complete(messages)
    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.succeeded
    assert call.fallback_depth >= 1
    assert trace.used_fallback is True
    # The swallowed failure is still on the record -- otherwise Phase 3 has no evidence.
    assert any(a.error_class for a in call.attempts)
    assert any(a.served for a in call.attempts)


def test_exhaustion_is_recorded_on_the_trace(messages):
    gw = make_gateway({name: FAIL_500 for name in DEFAULT_CHAIN})
    with trace_run() as trace:
        with pytest.raises(AllProvidersExhausted):
            gw.complete(messages)
    assert len(trace.llm_calls) == 1
    assert trace.llm_calls[0].succeeded is False


def test_pinned_provider_bypasses_the_chain(messages):
    """Phase 7 pins one provider per matrix row; a pinned call must not fall back."""
    gw = make_gateway({"nim": FAIL_500})
    with pytest.raises(Exception) as exc:
        gw.complete(messages, model="nim")
    assert not isinstance(exc.value, AssertionError)

    ok = gw.complete(messages, model="openrouter")
    assert ok.served_by == "openrouter"


async def test_async_path_falls_back_too(messages):
    gw = make_gateway({"gemini": FAIL_500, "nim": FAIL_500})
    result = await gw.acomplete(messages)
    assert result.served_by == "openrouter"
    assert result.fallback_depth == 2


def test_retries_are_attempted_before_falling_back(messages):
    """num_retries>0 must not skip the fallback path when retries are also exhausted."""
    gw = make_gateway({"gemini": FAIL_500}, num_retries=2)
    result = gw.complete(messages)
    assert result.served_by == "nim"


def test_empty_completion_is_treated_as_failure_not_success(messages):
    """A 200 with no content is a failure HTTP status cannot see.

    Observed for real: a reasoning model overran its context window and spent the whole
    budget thinking, returning an empty message. Scoring that as a merely-poor report
    instead of a broken one is exactly the silent failure this project exists to catch.
    """
    gw = make_gateway({"gemini": EMPTY})
    result = gw.complete(messages)
    assert result.served_by != "gemini"
    assert result.text.strip()


def test_empty_completion_from_the_last_tier_raises(messages):
    from gateway.llm.router import EmptyCompletion

    gw = make_gateway({name: EMPTY for name in DEFAULT_CHAIN})
    with pytest.raises((EmptyCompletion, AllProvidersExhausted)):
        gw.complete(messages)


def test_empty_completion_is_recorded_on_the_trace(messages):
    gw = make_gateway({"gemini": EMPTY})
    with trace_run() as trace:
        gw.complete(messages)
    classes = [a.error_class for c in trace.llm_calls for a in c.attempts]
    assert "EmptyCompletion" in classes
