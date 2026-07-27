"""Offline test fixtures.

Every test in this suite runs without an API key and without a network call. Provider
behaviour is driven by LiteLLM's ``mock_response``, which accepts either a canned string
(success) or a sentinel like ``"litellm.RateLimitError"`` (deterministic failure). That
means the failover assertions exercise the *real* Router fallback machinery rather than a
stand-in for it -- only the transport is faked.
"""

from __future__ import annotations

import pytest

from gateway.llm.model_list import PRIMARY_ALIAS
from gateway.llm.router import Gateway
from gateway.settings import Settings

#: Sentinels understood by litellm's mock layer.
FAIL_500 = "litellm.InternalServerError"
FAIL_429 = "litellm.RateLimitError"
FAIL_CONTEXT = "litellm.ContextWindowExceededError"
#: A "successful" response with no content. Whitespace rather than "" because litellm
#: treats a falsy mock_response as "not mocked" and would call the real API.
EMPTY = "   "

DEFAULT_CHAIN = ("gemini", "nim", "openrouter", "ollama")


def make_settings(**overrides) -> Settings:
    base = dict(
        gemini_api_key="test-gemini",
        nvidia_nim_api_key="test-nim",
        openrouter_api_key="test-openrouter",
        num_retries=0,
        request_timeout_s=5.0,
        cooldown_time_s=60,
        allowed_fails=1,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_gateway(
    behaviour: dict[str, object] | None = None,
    chain: tuple[str, ...] = DEFAULT_CHAIN,
    settings: Settings | None = None,
    **router_kwargs,
) -> Gateway:
    """Build a Gateway whose providers behave exactly as ``behaviour`` dictates.

    ``behaviour[name]`` is either a success string or a ``FAIL_*`` sentinel. Unlisted
    providers succeed.
    """
    behaviour = behaviour or {}
    settings = settings or make_settings()

    def params(name: str) -> dict:
        return {
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-test-not-real",
            "mock_response": behaviour.get(name, f"answer from {name}"),
        }

    entries = [
        {
            "model_name": name,
            "litellm_params": params(name),
            "model_info": {"id": name, "provider": name, "chain_index": tier},
        }
        for tier, name in enumerate(chain)
    ]
    entries.append(
        {
            "model_name": PRIMARY_ALIAS,
            "litellm_params": params(chain[0]),
            "model_info": {"id": f"primary::{chain[0]}", "provider": chain[0], "chain_index": 0},
        }
    )
    router_kwargs.setdefault("num_retries", 0)
    return Gateway(settings=settings, model_list=entries, **router_kwargs)


@pytest.fixture
def gateway_factory():
    return make_gateway


@pytest.fixture
def messages():
    return [{"role": "user", "content": "What is the current state of solid-state batteries?"}]
