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
    """Raised when the eval judge shares a provider with the router (PLAN.md D5)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- routed providers (N=4; Azure dropped, see PLAN.md D6) -------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini/gemini-2.5-flash"

    nvidia_nim_api_key: str | None = None
    nim_model: str = "nvidia_nim/meta/llama-3.3-70b-instruct"

    openrouter_api_key: str | None = None
    # Pick an upstream vendor distinct from Gemini/NIM so the tiers fail
    # independently (PLAN.md Phase 1). Verify the slug against openrouter.ai/models.
    openrouter_model: str = "openrouter/mistralai/mistral-small-3.2-24b-instruct"

    ollama_api_base: str = "http://localhost:11434"
    ollama_model: str = "ollama_chat/qwen3.5:9b"

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

    def assert_judge_is_independent(self) -> None:
        """Guard for PLAN.md D5 -- makes the bias structurally impossible, not just discouraged."""
        from gateway.llm.model_list import PROVIDER_PREFIXES

        judge = self.judge_model.lower()
        for name in self.available_providers():
            for prefix in PROVIDER_PREFIXES[name]:
                if judge.startswith(prefix):
                    raise JudgeIndependenceError(
                        f"JUDGE_MODEL={self.judge_model!r} belongs to routed provider {name!r}. "
                        "The judge must be independent of every provider it scores "
                        "(PLAN.md D5); use a separate Anthropic/OpenAI key."
                    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
