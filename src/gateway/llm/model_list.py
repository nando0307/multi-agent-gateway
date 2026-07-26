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

#: Fallback order. Placeholder until Phase 1 runs -- see results/latency.md.
DEFAULT_CHAIN: tuple[str, ...] = ("gemini", "nim", "openrouter", "ollama")

#: litellm model-string prefixes per provider, used by the judge-independence guard.
PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "gemini": ("gemini/", "gemini-", "vertex_ai/"),
    "nim": ("nvidia_nim/",),
    "openrouter": ("openrouter/",),
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
        return {"model": settings.gemini_model, "api_key": settings.gemini_api_key}
    if name == "nim":
        if not settings.nvidia_nim_api_key:
            return None
        return {"model": settings.nim_model, "api_key": settings.nvidia_nim_api_key}
    if name == "openrouter":
        if not settings.openrouter_api_key:
            return None
        return {"model": settings.openrouter_model, "api_key": settings.openrouter_api_key}
    if name == "ollama":
        # No key required; availability is a liveness question, answered by the smoke test.
        # Carries its own timeout -- see Settings.ollama_timeout_s.
        return {
            "model": settings.ollama_model,
            "api_base": settings.ollama_api_base,
            "timeout": settings.ollama_timeout_s,
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
