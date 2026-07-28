"""Provider inventory and the fallback chain.

The chain order is a *measurement*, not a preference. Phase 1 (`scripts/bench_latency.py`)
writes `results/latency.md`; this list must match it. The ordering key is p95, not p50 --
failover is a tail-latency problem, and a provider with a good median and a bad tail makes
a bad primary.

Ollama is deliberately last: it is local, free, and cannot rate-limit, so the chain always
terminates in something that works. That is what drives the residual failure rate toward
zero in Phase 3 -- and, because it is measurably weaker, it is also what makes the silent
quality regressions in Phase 7 real rather than hypothetical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.settings import Settings

#: The alias every agent asks for. Agents never name a provider.
PRIMARY_ALIAS = "research-primary"

#: Fallback order, set from measurement -- see results/latency.md (2026-07-27, n=20 each).
#: Ordered by p95, not p50: failover is a tail-latency mechanism, so the tail is the
#: ordering key. OpenRouter (p95 4.6s) is the clear primary; Gemini (23.1s, and the only
#: provider to fail during the benchmark) sits second.
#:
#: Ollama is third, not last. The plan assumed the local tier would terminate the chain by
#: being slowest -- it is not. At p95 23.6s it beats NIM's 138.3s by ~6x, and its p50/p95
#: differ by under 400ms against NIM's 114s spread. Chain order does not affect the
#: residual failure rate (that is P(all fail), which is order-independent), so there was no
#: availability reason to override the measurement. Caveat: these are serial numbers, and
#: Ollama is a single local process that will queue under concurrent fan-out.
#: Ollama was the terminal tier until 2026-07-28. It was replaced by Groq: a synthesis
#: call took ~214s locally (generation-bound, ~23 tok/s), which made a 30-question x
#: 4-provider matrix a 6-8 hour run. See PLAN.md for what that cost -- the chain is now
#: all-cloud, so it no longer terminates in something that cannot rate-limit.
DEFAULT_CHAIN: tuple[str, ...] = ("openrouter", "gemini", "groq", "nim")

#: litellm model-string prefixes per provider, used by the judge-independence guard.
PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "gemini": ("gemini/", "gemini-", "vertex_ai/"),
    "nim": ("nvidia_nim/",),
    "openrouter": ("openrouter/",),
    "groq": ("groq/",),
    "ollama": ("ollama/", "ollama_chat/"),
}


def provider_params(name: str, settings: Settings) -> dict[str, Any] | None:
    """litellm params for ``name``, or None when its credentials are absent.

    Returning None (rather than raising) is what lets ``N`` be computed from reality:
    a provider you could not get a key for simply is not in the router.
    """
    if name == "gemini":
        if not settings.gemini_api_key:
            return None
        return {
            "model": settings.gemini_model,
            "api_key": settings.gemini_api_key,
            "timeout": settings.gemini_timeout_s,
        }
    if name == "nim":
        if not settings.nvidia_nim_api_key:
            return None
        return {
            "model": settings.nim_model,
            "api_key": settings.nvidia_nim_api_key,
            "timeout": settings.nim_timeout_s,
        }
    if name == "openrouter":
        if not settings.openrouter_api_key:
            return None
        return {
            "model": settings.openrouter_model,
            "api_key": settings.openrouter_api_key,
            "timeout": settings.openrouter_timeout_s,
        }
    if name == "groq":
        if not settings.groq_api_key:
            return None
        return {
            "model": settings.groq_model,
            "api_key": settings.groq_api_key,
            "timeout": settings.groq_timeout_s,
        }
    if name == "ollama":
        # No key required; availability is a liveness question, answered by the smoke test.
        # Carries its own timeout -- see Settings.ollama_timeout_s.
        return {
            "model": settings.ollama_model,
            "api_base": settings.ollama_api_base,
            "timeout": settings.ollama_timeout_s,
            "num_ctx": settings.ollama_num_ctx,
        }
    raise KeyError(f"unknown provider {name!r}")


def build_model_list(
    settings: Settings, chain: tuple[str, ...] | None = None
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Return ``(model_list, effective_chain)``.

    Each provider is registered under its own alias so Phase 7 can pin a single provider
    for the eval matrix. The head of the chain is *additionally* registered under
    ``PRIMARY_ALIAS``, which is the only name agents ever use.
    """
    chain = chain or DEFAULT_CHAIN
    entries: list[dict[str, Any]] = []
    effective: list[str] = []

    for tier, name in enumerate(chain):
        params = provider_params(name, settings)
        if params is None:
            continue
        effective.append(name)
        entries.append(
            {
                "model_name": name,
                # A per-provider timeout in params wins over the global default.
                "litellm_params": {"timeout": settings.request_timeout_s, **params},
                "model_info": {"id": name, "provider": name, "chain_index": tier},
            }
        )

    if not effective:
        raise RuntimeError(
            "No providers configured. Set at least one of GEMINI_API_KEY, "
            "NVIDIA_NIM_API_KEY, OPENROUTER_API_KEY, or run Ollama locally."
        )

    head = effective[0]
    head_params = provider_params(head, settings)
    if head_params is None:  # unreachable: `effective` only holds configured providers
        raise RuntimeError(f"provider {head!r} lost its credentials mid-build")
    entries.append(
        {
            "model_name": PRIMARY_ALIAS,
            "litellm_params": {"timeout": settings.request_timeout_s, **head_params},
            "model_info": {"id": f"primary::{head}", "provider": head, "chain_index": 0},
        }
    )

    return entries, tuple(effective)
