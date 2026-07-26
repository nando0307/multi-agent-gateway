"""LlamaIndex <-> LiteLLM Router bridge.

**Why this file exists (PLAN.md D2, the highest-risk integration point).**
The obvious choice, ``llama-index-llms-litellm``, calls ``litellm.completion()`` -- which
has no Router and therefore *no fallbacks*. Dropping it into an agent looks like it works
and silently voids the entire failover story. This class routes every agent call through
``Gateway.complete()`` instead, and stamps the serving provider onto the response so the
agent layer stays observable.
"""

from __future__ import annotations

from typing import Any, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from pydantic import Field, PrivateAttr

from gateway.llm.model_list import PRIMARY_ALIAS
from gateway.llm.router import Gateway, build_gateway


def _to_litellm(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role.value, "content": m.content or ""} for m in messages]


class GatewayLLM(CustomLLM):
    """A LlamaIndex LLM whose every call inherits the Router's failover."""

    model_alias: str = Field(default=PRIMARY_ALIAS)
    context_window: int = Field(default=128_000)
    num_output: int = Field(default=2048)
    temperature: float = Field(default=0.2)

    _gateway: Gateway = PrivateAttr()
    _last_served_by: str | None = PrivateAttr(default=None)
    _last_depth: int = PrivateAttr(default=0)

    def __init__(self, gateway: Gateway | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._gateway = gateway if gateway is not None else build_gateway()

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=f"gateway:{self.model_alias}",
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @property
    def last_served_by(self) -> str | None:
        """Which provider answered most recently. Used by tests and the CLI live view."""
        return self._last_served_by

    @property
    def last_fallback_depth(self) -> int:
        return self._last_depth

    def _stamp(self, result) -> dict[str, Any]:
        self._last_served_by = result.served_by
        self._last_depth = result.fallback_depth
        return {"served_by": result.served_by, "fallback_depth": result.fallback_depth}

    # -- chat ---------------------------------------------------------------------
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        result = self._gateway.complete(
            _to_litellm(messages),
            model=self.model_alias,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=result.text),
            raw=result.raw,
            additional_kwargs=self._stamp(result),
        )

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        result = await self._gateway.acomplete(
            _to_litellm(messages),
            model=self.model_alias,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=result.text),
            raw=result.raw,
            additional_kwargs=self._stamp(result),
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        response = self.chat(messages, **kwargs)

        def gen() -> ChatResponseGen:
            yield response

        return gen()

    # -- completion ---------------------------------------------------------------
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        result = self._gateway.complete(
            [{"role": "user", "content": prompt}],
            model=self.model_alias,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        return CompletionResponse(
            text=result.text, raw=result.raw, additional_kwargs=self._stamp(result)
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        result = await self._gateway.acomplete(
            [{"role": "user", "content": prompt}],
            model=self.model_alias,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        return CompletionResponse(
            text=result.text, raw=result.raw, additional_kwargs=self._stamp(result)
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> CompletionResponseGen:
            yield response

        return gen()
