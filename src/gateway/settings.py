"""Configuration for the gateway.

Every secret enters the process here and nowhere else. Two invariants are enforced
at import time rather than left to convention:

1. The eval judge must never be one of the routed providers (PLAN.md D5). If it is,
   the provider matrix measures self-preference instead of quality.
2. A provider with no credentials is *absent* from the router, not silently broken.
   ``N`` in the resume bullet is ``len(available_providers())``, computed, not asserted.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeIndependenceError(RuntimeError):
    """Raised when the eval judge is one of the models it would be scoring (PLAN.md D5)."""


def model_identity(model_string: str) -> str:
    """Reduce a litellm model string to the underlying model name.

    The same model reached by two different routes is still the same model:
    ``gemini/gemini-3.1-flash-lite`` and ``openrouter/google/gemini-3.1-flash-lite`` must
    compare equal, or the independence guard is trivially bypassed by changing route.
    """
    tail = model_string.strip().lower().rstrip("/").split("/")[-1]
    for variant in (":free", ":nitro", ":beta", ":extended", ":floor"):
        if tail.endswith(variant):
            tail = tail[: -len(variant)]
    return tail


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- routed providers (N=4; Azure dropped, see PLAN.md D6) -------------------
    gemini_api_key: str | None = None
    # Pinned, not an alias. `gemini-flash-latest` also works but moves under you, which
    # would silently invalidate a benchmark taken days earlier. Verified working on this
    # key; `gemini-2.5-flash` is closed to new keys and `gemini-3.5-flash` returned 503.
    gemini_model: str = "gemini/gemini-3.1-flash-lite"

    nvidia_nim_api_key: str | None = None
    # llama-3.3-70b answers in ~87s on this key -- usable but far too slow for a routine
    # tier. Nemotron-super-49b returns in ~0.4s and is the stronger model of the two that
    # are actually fast.
    nim_model: str = "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1"

    openrouter_api_key: str | None = None
    # Pick an upstream vendor distinct from Gemini/NIM so the tiers fail
    # independently (PLAN.md Phase 1). Verify the slug against openrouter.ai/models.
    openrouter_model: str = "openrouter/mistralai/mistral-small-3.2-24b-instruct"

    ollama_api_base: str = "http://localhost:11434"
    ollama_model: str = "ollama_chat/qwen3.5:9b"
    # The local tier needs its own budget. A timeout tuned for a hosted flash model kills
    # local inference outright -- the terminal tier would then fail on every request, which
    # is the opposite of what it is there for. Measured: a cold 9B load plus a reasoning
    # preamble runs well past 45s on a laptop.
    ollama_timeout_s: float = 300.0

    # --- tools -------------------------------------------------------------------
    tavily_api_key: str | None = None

    # --- eval judge: deliberately NOT in the router's model list -----------------
    judge_api_key: str | None = None
    judge_model: str = "claude-sonnet-5"

    # --- behaviour ---------------------------------------------------------------
    request_timeout_s: float = 45.0
    num_retries: int = 2
    allowed_fails: int = 3
    cooldown_time_s: int = 60
    gate_threshold: float = 0.70

    def available_providers(self) -> tuple[str, ...]:
        """Providers that actually have credentials, in fallback order."""
        from gateway.llm.model_list import DEFAULT_CHAIN, provider_params

        return tuple(n for n in DEFAULT_CHAIN if provider_params(n, self) is not None)

    def resolve_judge_key(self) -> str | None:
        """The judge's key: explicit if given, else inferred from its route.

        Lets JUDGE_MODEL=openrouter/... reuse OPENROUTER_API_KEY rather than duplicating
        the same secret under a second name in .env.
        """
        if self.judge_api_key:
            return self.judge_api_key
        model = self.judge_model.lower()
        if model.startswith("openrouter/"):
            return self.openrouter_api_key
        if model.startswith(("gemini/", "gemini-")):
            return self.gemini_api_key
        if model.startswith("nvidia_nim/"):
            return self.nvidia_nim_api_key
        return None

    def routed_model_identities(self) -> dict[str, str]:
        from gateway.llm.model_list import DEFAULT_CHAIN, provider_params

        out = {}
        for name in DEFAULT_CHAIN:
            params = provider_params(name, self)
            if params is not None:
                out[name] = model_identity(params["model"])
        return out

    def assert_judge_is_independent(self) -> None:
        """Guard for PLAN.md D5.

        The check is at *model* level, not provider level. Self-preference bias is a model
        rating its own output higher; a different model reached through a shared account
        does not have that problem. Sharing an account is a weaker claim than a wholly
        separate vendor, so it is recorded in `judge_independence_caveat()` and printed
        into the matrix report rather than passed over in silence.
        """
        judge = model_identity(self.judge_model)
        for name, identity in self.routed_model_identities().items():
            if judge == identity:
                raise JudgeIndependenceError(
                    f"JUDGE_MODEL={self.judge_model!r} is the same model served by routed "
                    f"provider {name!r}. A model scoring its own output measures "
                    "self-preference, not quality (PLAN.md D5). Pick a different model."
                )

    def judge_independence_caveat(self) -> str | None:
        """Non-fatal note when the judge shares an account with a routed provider."""
        judge = self.judge_model.lower()
        shared = None
        if judge.startswith("openrouter/") and self.openrouter_api_key:
            shared = "openrouter"
        elif judge.startswith(("gemini/", "gemini-")) and self.gemini_api_key:
            shared = "gemini"
        elif judge.startswith("nvidia_nim/") and self.nvidia_nim_api_key:
            shared = "nim"
        if shared is None:
            return None
        return (
            f"The judge (`{self.judge_model}`) is a different model from every routed one, so "
            f"there is no self-preference bias, but it is reached through the same `{shared}` "
            "account that also serves requests. A wholly separate vendor would be a stronger "
            "guarantee."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
