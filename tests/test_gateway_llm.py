"""PLAN.md D2: prove the LlamaIndex bridge actually inherits the Router's failover.

This is the test that catches the project's most expensive possible mistake. Using
``llama-index-llms-litellm`` would make every one of these assertions fail silently in
production and pass in a demo: it calls ``litellm.completion()``, which has no Router and
therefore no fallbacks. If this file goes red, the failover claim is void.
"""

from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from gateway.llm.events import trace_run
from gateway.llm.gateway_llm import GatewayLLM
from tests.conftest import FAIL_429, FAIL_500, make_gateway


def _llm(behaviour=None, **kw) -> GatewayLLM:
    return GatewayLLM(gateway=make_gateway(behaviour, **kw))


def test_complete_reports_serving_provider():
    llm = _llm()
    response = llm.complete("summarise solid-state battery progress")
    assert response.additional_kwargs["served_by"] == "gemini"
    assert response.additional_kwargs["fallback_depth"] == 0


def test_agent_still_answers_when_primary_is_down():
    """The headline behaviour: kill the primary, the LlamaIndex layer keeps working."""
    llm = _llm({"gemini": FAIL_500})
    response = llm.complete("what changed in EU AI regulation this year?")
    assert response.text
    assert llm.last_served_by == "nim"
    assert llm.last_fallback_depth == 1


def test_chat_path_falls_back_and_stamps_provider():
    llm = _llm({"gemini": FAIL_500, "nim": FAIL_429})
    response = llm.chat([ChatMessage(role=MessageRole.USER, content="hello")])
    assert response.message.role == MessageRole.ASSISTANT
    assert response.additional_kwargs["served_by"] == "openrouter"
    assert response.additional_kwargs["fallback_depth"] == 2


async def test_async_chat_falls_back():
    llm = _llm({"gemini": FAIL_500})
    response = await llm.achat([ChatMessage(role=MessageRole.USER, content="hello")])
    assert response.additional_kwargs["fallback_depth"] == 1


def test_metadata_is_provider_agnostic():
    """Agents must never be able to see, or pin, a provider through this interface."""
    md = _llm().metadata
    assert md.model_name == "gateway:research-primary"
    assert md.is_chat_model
    for provider in ("gemini", "nim", "openrouter", "ollama"):
        assert provider not in md.model_name


def test_calls_through_the_bridge_land_on_the_trace():
    llm = _llm({"gemini": FAIL_500})
    with trace_run(question="q") as trace:
        llm.complete("anything")
        llm.chat([ChatMessage(role=MessageRole.USER, content="more")])
    assert len(trace.llm_calls) == 2
    assert trace.used_fallback
    assert trace.providers_used() == {"nim"}
