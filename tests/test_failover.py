"""Deterministic failover tests -- PLAN.md Phase 3.

No probability here: each case forces an exact provider configuration and asserts exactly
which provider served the request and at what fallback depth. The probabilistic study
lives in `scripts/chaos_run.py`; this file is what proves the mechanism works at all.
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import litellm
from litellm.integrations.custom_logger import CustomLogger

import pytest

from gateway.llm.events import trace_run
from gateway.llm.router import AllProvidersExhausted, Gateway
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
    # The CLI and /research only show str(exc); it must name what was tried.
    assert "gemini:InternalServerError" in str(exc.value)


def test_a_cooled_provider_advances_the_chain_instead_of_aborting_it(messages):
    """A sustained outage must fail over, not answer 400 invalid_request.

    After ``allowed_fails`` failures LiteLLM cools the deployment and starts raising
    ``RouterRateLimitError`` (a ValueError with no status_code) for it. That is a provider
    outage belonging to the same class as 429/5xx: the chain has to keep walking.
    """
    gw = make_gateway({"gemini": FAIL_500, "nim": FAIL_500})
    errors = []
    for _ in range(6):
        result = gw.complete(messages)
        assert result.served_by == "openrouter", result.attempts
        errors.extend(attempt.error_class for attempt in result.attempts)
    assert "RouterRateLimitError" in errors, errors


def test_every_provider_cooled_is_exhaustion_not_a_client_error(messages):
    gw = make_gateway({"gemini": FAIL_500, "openrouter": FAIL_500}, chain=("gemini", "openrouter"))
    error = None
    for _ in range(6):
        try:
            gw.complete(messages)
        except AllProvidersExhausted as exc:
            error = exc
        else:
            error = None
    assert error is not None
    assert [a.error_class for a in error.attempts] == ["RouterRateLimitError", "RouterRateLimitError"]


def test_the_shipped_model_list_lets_the_router_cool_a_dead_provider(messages):
    """allowed_fails/cooldown_time_s must bind on the shipped config, not just in fixtures.

    LiteLLM treats its own default ``allowed_fails`` (3) as "unset" and switches to a
    policy where a single-deployment group only cools after 1000 requests, and it treats a
    model name that equals a deployment id as that one specific deployment, skipping the
    cooldown filter entirely. Both would leave a dead primary re-attempted forever.
    """
    from gateway.llm.model_list import build_model_list
    from gateway.settings import Settings

    settings = Settings(_env_file=None, gemini_api_key="k", openrouter_api_key="k")
    model_list, chain = build_model_list(settings)
    assert chain == ("gemini", "openrouter")
    for entry in model_list:
        entry["litellm_params"] = {
            "model": "openai/gpt-4o-mini", "api_key": "sk-test-not-real",
            "mock_response": FAIL_500 if entry["model_name"] == "gemini"
            else f"answer from {entry['model_name']}",
        }
    gw = Gateway(settings=settings, model_list=model_list)
    errors = []
    for _ in range(6):
        result = gw.complete(messages)
        assert result.served_by == "openrouter", result.attempts
        errors.extend(attempt.error_class for attempt in result.attempts)
    assert "RouterRateLimitError" in errors, errors


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


def test_concurrent_sync_async_calls_deliver_their_success_callbacks():
    """Short-lived caller loops must not discard LiteLLM's queued callbacks."""
    delivered = Queue()

    class Recorder(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            delivered.put(kwargs["messages"][0]["content"])

    recorder = Recorder()
    litellm.callbacks.append(recorder)
    gateway = make_gateway()
    try:
        def call(label, asynchronous):
            messages = [{"role": "user", "content": label}]
            if asynchronous:
                return asyncio.run(gateway.acomplete(messages))
            return gateway.complete(messages)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = [pool.submit(call, f"callback-{index}", bool(index % 2)) for index in range(4)]
            assert all(result.result(timeout=5).served_by == "gemini" for result in results)
        expected = {f"callback-{index}" for index in range(4)}
        seen = set()
        while not expected <= seen:
            # Other tests' callbacks can arrive late; only these four are the claim.
            seen.add(delivered.get(timeout=5))
        assert expected <= seen
    finally:
        litellm.callbacks.remove(recorder)
