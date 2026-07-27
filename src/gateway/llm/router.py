"""The gateway: one provider-agnostic, self-healing call site.

Design note (PLAN.md D1): this is an in-process ``litellm.Router``, not the LiteLLM proxy
server. The proxy is the easier path but makes fallback events invisible to the test
suite -- observable only through logs. In-process, `tests/test_failover.py` can assert
exactly which provider served each request and at what depth, which is the entire point
of the failover claim.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger

from gateway.llm.events import LLMAttempt, LLMCallRecord, get_trace
from gateway.llm.model_list import PRIMARY_ALIAS, build_model_list
from gateway.settings import Settings, get_settings

litellm.drop_params = True
litellm.suppress_debug_info = True
# LiteLLM logs a full traceback for every deployment failure. During failover those are
# expected events, not errors -- they belong on the RunTrace, not on stderr.
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)


class EmptyCompletion(RuntimeError):
    """A provider returned success with no content.

    This is a failure that HTTP status cannot see: 200 OK, no exception, zero words. It
    was observed for real -- a reasoning model overran its context window and spent the
    entire budget thinking. Treating it as success let blank reports through to scoring,
    where they registered as merely poor rather than broken.
    """


class AllProvidersExhausted(RuntimeError):
    """Every provider in the chain failed. Raised cleanly -- never hang."""

    def __init__(self, alias: str, attempts: list[LLMAttempt], cause: BaseException | None = None):
        self.alias = alias
        self.attempts = attempts
        self.__cause__ = cause
        tried = ", ".join(f"{a.provider}:{a.error_class}" for a in attempts) or "none recorded"
        super().__init__(f"all providers exhausted for {alias!r} (tried: {tried})")


class _AttemptRecorder(CustomLogger):
    """Captures per-deployment failures that the Router swallows while falling back.

    Registered once, globally. Writes into whichever ``CallResult`` collector is active
    on this context, so concurrent researcher tasks don't cross-contaminate.
    """

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, start_time, end_time)

    @staticmethod
    def _record(kwargs, start_time, end_time):
        sink = _ATTEMPT_SINK.get()
        if sink is None:
            return
        info = (kwargs.get("litellm_params") or {}).get("model_info") or {}
        exc = kwargs.get("exception")
        try:
            latency = (end_time - start_time).total_seconds() * 1000
        except (TypeError, AttributeError):
            latency = 0.0
        sink.append(
            LLMAttempt(
                attempt_idx=len(sink),
                provider=info.get("provider") or kwargs.get("model"),
                served=False,
                latency_ms=round(latency, 1),
                error_class=type(exc).__name__ if exc else "UnknownError",
                error_message=str(exc)[:300] if exc else None,
            )
        )


from contextvars import ContextVar  # noqa: E402  (kept next to its only consumer)

_ATTEMPT_SINK: ContextVar[list[LLMAttempt] | None] = ContextVar("gateway_attempts", default=None)

_recorder = _AttemptRecorder()
if not any(isinstance(cb, _AttemptRecorder) for cb in litellm.callbacks):
    litellm.callbacks.append(_recorder)


class CallResult:
    __slots__ = ("text", "served_by", "fallback_depth", "attempts", "latency_ms", "raw", "usage")

    def __init__(self, text, served_by, fallback_depth, attempts, latency_ms, raw, usage):
        self.text = text
        self.served_by = served_by
        self.fallback_depth = fallback_depth
        self.attempts = attempts
        self.latency_ms = latency_ms
        self.raw = raw
        self.usage = usage

    def __repr__(self) -> str:
        return (
            f"CallResult(served_by={self.served_by!r}, depth={self.fallback_depth}, "
            f"latency_ms={self.latency_ms:.0f})"
        )


class Gateway:
    """Provider-agnostic completion with measured failover."""

    def __init__(
        self,
        settings: Settings | None = None,
        chain: tuple[str, ...] | None = None,
        model_list: list[dict[str, Any]] | None = None,
        **router_kwargs: Any,
    ):
        self.settings = settings or get_settings()
        if model_list is not None:
            self.model_list = model_list
            self.chain = tuple(
                e["model_info"]["provider"]
                for e in model_list
                if e["model_name"] != PRIMARY_ALIAS
            )
        else:
            self.model_list, self.chain = build_model_list(self.settings, chain)

        defaults: dict[str, Any] = dict(
            model_list=self.model_list,
            fallbacks=[{PRIMARY_ALIAS: list(self.chain[1:])}] if len(self.chain) > 1 else [],
            num_retries=self.settings.num_retries,
            timeout=self.settings.request_timeout_s,
            allowed_fails=self.settings.allowed_fails,
            cooldown_time=self.settings.cooldown_time_s,
            routing_strategy="latency-based-routing",
            set_verbose=False,
        )
        defaults.update(router_kwargs)
        self.router = Router(**defaults)

    # -- provider resolution ------------------------------------------------------
    def _resolve_provider(self, response: Any) -> str | None:
        hidden = getattr(response, "_hidden_params", None) or {}
        model_id = hidden.get("model_id")
        if isinstance(model_id, str):
            if model_id.startswith("primary::"):
                return model_id.split("::", 1)[1]
            if model_id in self.chain:
                return model_id
        model = getattr(response, "model", None) or ""
        for entry in self.model_list:
            target = entry["litellm_params"]["model"]
            if model and (model in target or target.endswith(model)):
                return entry["model_info"]["provider"]
        return None

    def _depth(self, provider: str | None) -> int:
        if provider is None or provider not in self.chain:
            return 0
        return self.chain.index(provider)

    # -- public API ---------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = PRIMARY_ALIAS,
        **kwargs: Any,
    ) -> CallResult:
        attempts: list[LLMAttempt] = []
        token = _ATTEMPT_SINK.set(attempts)
        started = time.perf_counter()
        call_id = uuid.uuid4().hex[:8]
        try:
            response = self.router.completion(model=model, messages=messages, **kwargs)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            self._record_call(call_id, model, None, 0, elapsed, attempts, error=repr(exc))
            raise AllProvidersExhausted(model, attempts, cause=exc) from exc
        finally:
            _ATTEMPT_SINK.reset(token)

        elapsed = (time.perf_counter() - started) * 1000
        provider = self._resolve_provider(response)
        depth = self._depth(provider)

        text = response.choices[0].message.content or ""
        if not text.strip():
            attempts.append(
                LLMAttempt(len(attempts), provider, False, round(elapsed, 1),
                           error_class="EmptyCompletion",
                           error_message="provider returned 200 with no content")
            )
            self._record_call(call_id, model, provider, depth, elapsed, attempts,
                              error="EmptyCompletion")
            remaining = [p for p in self.chain[self._depth(provider) + 1:]]
            if not remaining:
                raise EmptyCompletion(f"{provider} returned an empty completion, chain exhausted")
            return self.complete(messages, model=remaining[0], **kwargs)

        attempts.append(
            LLMAttempt(
                attempt_idx=len(attempts),
                provider=provider,
                served=True,
                latency_ms=round(elapsed, 1),
            )
        )
        usage = getattr(response, "usage", None)
        self._record_call(
            call_id, model, provider, depth, elapsed, attempts, response=response, usage=usage
        )
        return CallResult(
            text=text,
            served_by=provider,
            fallback_depth=depth,
            attempts=attempts,
            latency_ms=elapsed,
            raw=response,
            usage=usage,
        )

    async def acomplete(
        self, messages: list[dict[str, str]], *, model: str = PRIMARY_ALIAS, **kwargs: Any
    ) -> CallResult:
        attempts: list[LLMAttempt] = []
        token = _ATTEMPT_SINK.set(attempts)
        started = time.perf_counter()
        call_id = uuid.uuid4().hex[:8]
        try:
            response = await self.router.acompletion(model=model, messages=messages, **kwargs)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            self._record_call(call_id, model, None, 0, elapsed, attempts, error=repr(exc))
            raise AllProvidersExhausted(model, attempts, cause=exc) from exc
        finally:
            _ATTEMPT_SINK.reset(token)

        elapsed = (time.perf_counter() - started) * 1000
        provider = self._resolve_provider(response)
        depth = self._depth(provider)
        attempts.append(
            LLMAttempt(len(attempts), provider, True, round(elapsed, 1))
        )
        usage = getattr(response, "usage", None)
        self._record_call(
            call_id, model, provider, depth, elapsed, attempts, response=response, usage=usage
        )
        return CallResult(
            text=response.choices[0].message.content or "",
            served_by=provider,
            fallback_depth=depth,
            attempts=attempts,
            latency_ms=elapsed,
            raw=response,
            usage=usage,
        )

    @staticmethod
    def _record_call(
        call_id, alias, provider, depth, elapsed, attempts, response=None, usage=None, error=None
    ) -> None:
        trace = get_trace()
        if trace is None:
            return
        trace.llm_calls.append(
            LLMCallRecord(
                call_id=call_id,
                alias=alias,
                served_by=provider,
                fallback_depth=depth,
                latency_ms=round(elapsed, 1),
                attempts=list(attempts),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                error=error,
            )
        )


def build_gateway(**kwargs: Any) -> Gateway:
    return Gateway(**kwargs)
