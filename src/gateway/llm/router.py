"""Bounded generation over LiteLLM's provider adapters and deployment Router."""
from __future__ import annotations

import atexit
import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import ExitStack, aclosing
from functools import partial
from typing import Any

import litellm
from anyio.from_thread import start_blocking_portal
from jsonschema import Draft202012Validator, ValidationError
from litellm import Router

from gateway.llm.events import LLMAttempt, LLMCallRecord, get_trace
from gateway.llm.model_list import BOSS_ALIASES, PRIMARY_ALIAS, build_model_list
from gateway.settings import Settings, get_settings

litellm.drop_params = False
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)

# LiteLLM owns a process-wide async logging worker and cached HTTP clients.
# Keep their loop alive across sync callers and ASGI requests, rather than losing
# queued callbacks whenever a per-call asyncio.run() closes its loop.
_portal_stack = ExitStack()
_portal_lock = threading.Lock()
_portal = None
atexit.register(_portal_stack.close)


def _completion_portal():
    global _portal
    with _portal_lock:
        if _portal is None:
            _portal = _portal_stack.enter_context(start_blocking_portal(name="gateway-generation"))
        return _portal


class EmptyCompletion(RuntimeError):
    """No usable content from an otherwise successful provider response."""


class IncompleteCompletion(RuntimeError):
    """Missing terminal marker or an unfinished answer."""


class InvalidCompletion(RuntimeError):
    """Provider output does not satisfy the requested schema."""


class OutputTruncated(RuntimeError):
    """A healthy provider answered until an output cap rather than failing."""

    def __init__(self, max_tokens=None):
        self.max_tokens = max_tokens
        detail = (f"Output truncated at the requested max_tokens={max_tokens}" if max_tokens
                  else "Output truncated at the provider's default output cap")
        super().__init__(detail)


class GenerationRefused(RuntimeError):
    """Safety refusal; never replay against another provider."""


class GenerationDeadline(TimeoutError):
    """The shared generation deadline expired."""


class AllProvidersExhausted(RuntimeError):
    def __init__(self, alias, attempts, cause=None):
        self.alias = alias
        self.attempts = attempts
        self.__cause__ = cause
        tried = ", ".join(f"{a.provider}:{a.error_class}" for a in attempts) or (
            type(cause).__name__ if cause is not None else "none recorded")
        super().__init__(f"all providers exhausted for {alias!r} (tried: {tried})")


class TruncatedExhausted(AllProvidersExhausted):
    """Every eligible tier stopped at its own default output cap."""


def retryable(exc: Exception) -> bool:
    # Normalized safety/input errors win over an ambiguous upstream HTTP status.
    if isinstance(exc, OutputTruncated):
        # A cap the caller named is the request's own budget; a provider's default cap is
        # worth retrying on a tier that may have a larger one.
        return exc.max_tokens is None
    if isinstance(exc, (litellm.BadRequestError, litellm.ContentPolicyViolationError,
                        GenerationRefused)):
        return False
    if isinstance(exc, (EmptyCompletion, IncompleteCompletion, TimeoutError,
                        litellm.APIConnectionError, litellm.Timeout, litellm.RateLimitError,
                        litellm.AuthenticationError, litellm.PermissionDeniedError,
                        litellm.RouterRateLimitError, litellm.RouterRateLimitErrorBasic,
                        InvalidCompletion)):
        # Router cooldown errors are ValueErrors carrying no status code: the pinned
        # deployment has no healthy entries left, which is a provider outage, not a
        # client error.
        return True
    # Upstream 401/403 reach some adapters as a generic APIError, so classify by status
    # too: a provider rejecting its own credentials is an outage, not a caller error.
    return getattr(exc, "status_code", None) in (401, 403, 429, 500, 502, 503, 504)


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not a JSON number")


def validate_content(text: str, response_format: dict | None) -> None:
    if response_format and response_format.get("type") == "json_schema":
        try:
            # json.loads accepts NaN/Infinity/-Infinity, which JSON does not have and
            # jsonschema's numeric bounds do not reject.
            value = json.loads(text, parse_constant=_reject_constant)
            Draft202012Validator(response_format["json_schema"]["schema"]).validate(value)
        except (ValueError, ValidationError) as exc:
            raise InvalidCompletion("Provider returned invalid structured output") from exc


class CallResult:
    __slots__ = ("text", "served_by", "fallback_depth", "attempts", "latency_ms", "raw", "usage")

    def __init__(self, text, served_by, fallback_depth, attempts, latency_ms, raw, usage):
        self.text, self.served_by, self.fallback_depth = text, served_by, fallback_depth
        self.attempts, self.latency_ms, self.raw, self.usage = attempts, latency_ms, raw, usage


class _GenerationRouter(Router):
    async def _acompletion_streaming_iterator(self, model_response, messages, initial_kwargs):
        # LiteLLM 1.93 wraps streams with continuation fallback even when
        # disable_fallbacks=True. Keep its adapters, but never splice model answers.
        return model_response


class Gateway:
    """One ordered attempt per approved provider, sharing one wall-clock deadline."""

    def __init__(self, settings: Settings | None = None, chain=None, model_list=None, **router_kwargs):
        self.settings = settings or get_settings()
        if model_list is None:
            self.model_list, self.chain = build_model_list(self.settings, chain)
        else:
            self.model_list = model_list
            self.chain = tuple(e["model_info"]["provider"] for e in model_list
                               if e["model_name"] != PRIMARY_ALIAS)
        defaults = dict(model_list=self.model_list, num_retries=0,
                        fallbacks=[{PRIMARY_ALIAS: list(self.chain[1:])}],
                        timeout=self.settings.request_timeout_s,
                        allowed_fails=self.settings.allowed_fails,
                        cooldown_time=self.settings.cooldown_time_s,
                        routing_strategy="simple-shuffle", set_verbose=False)
        defaults.update(router_kwargs)
        # The wrapper owns the budget and retry classification. No nested retry ladder.
        defaults["num_retries"] = 0
        self.router = _GenerationRouter(**defaults)

    def _providers(self, model, response_format=None):
        if model in (PRIMARY_ALIAS, *BOSS_ALIASES):
            providers = self.chain if self.router.fallbacks else self.chain[:1]
        elif model in self.chain:
            providers = (model,)
        else:
            raise ValueError("Unknown model alias")
        if not response_format or response_format.get("type") != "json_schema":
            return providers
        eligible = tuple(p for p in providers if self._satisfies_schema(p))
        if not eligible:
            raise ValueError("Configured model does not support required JSON output")
        return eligible

    def _satisfies_schema(self, provider) -> bool:
        target = next(e for e in self.model_list
                      if e["model_name"] == provider)["litellm_params"]["model"]
        if litellm.supports_response_schema(model=target):
            return True
        return "response_format" in (litellm.get_supported_openai_params(model=target) or [])

    def _budget(self, model):
        """Boss generation shares one deadline; research is bounded per attempt only."""
        if model in BOSS_ALIASES:
            return self.settings.generation_deadline_s, self.settings.generation_attempt_s
        return None, self.settings.request_timeout_s

    def _cool_down(self, provider, exc):
        """Cool the deployment for a provider attempt that produced nothing.

        Our attempt window cancels LiteLLM before its transport timeout, so its failure
        handlers never see the failure, and its per-minute failure counter expires between
        requests spaced by a full attempt window. The provider is therefore cooled here
        directly, in the Router's own cooldown cache, which is also what the Router consults
        when choosing a deployment.
        """
        deployment = next(e for e in self.model_list
                          if e["model_name"] == provider)["model_info"]["id"]
        self.router.cooldown_cache.add_deployment_to_cooldown(
            model_id=deployment,
            original_exception=exc,
            exception_status=500,
            cooldown_time=self.settings.cooldown_time_s,
        )

    def _options(self, provider, messages, kwargs, remaining, attempt_s=None):
        entry = next(e for e in self.model_list if e["model_name"] == provider)
        params = entry["litellm_params"]
        options = dict(kwargs)
        response_format = options.get("response_format")
        if response_format and response_format.get("type") == "json_schema":
            if not litellm.supports_response_schema(model=params["model"]):
                # JSON-mode providers still must pass the exact local schema validator.
                messages = [{"role": "system", "content": "Return only JSON matching this schema: "
                             + json.dumps(response_format["json_schema"]["schema"])}, *messages]
                options["response_format"] = {"type": "json_object"}
        attempt_s = self.settings.generation_attempt_s if attempt_s is None else attempt_s
        timeout = min(remaining, attempt_s,
                      params.get("timeout", self.settings.request_timeout_s))
        options.update(timeout=timeout, num_retries=0, max_retries=0,
                       disable_fallbacks=True, drop_params=False)
        return messages, options, timeout

    def complete(self, messages, *, model=PRIMARY_ALIAS, **kwargs):
        """Blocking facade; async callers must use acomplete/astream."""
        future = _completion_portal().start_task_soon(partial(self._acomplete, messages, model=model, **kwargs))
        try:
            return future.result()
        except (AllProvidersExhausted, OutputTruncated):
            raise
        except Exception as exc:
            # One normalised failure for every routed-provider failure, which is what the
            # research callers were written against.
            raise AllProvidersExhausted(model, [], exc) from exc
        finally:
            future.cancel()

    async def acomplete(self, messages, *, model=PRIMARY_ALIAS, **kwargs):
        future = _completion_portal().start_task_soon(partial(self._acomplete, messages, model=model, **kwargs))
        try:
            return await asyncio.wrap_future(future)
        finally:
            future.cancel()

    async def _acomplete(self, messages, *, model=PRIMARY_ALIAS, **kwargs):
        if kwargs.pop("stream", False):
            raise ValueError("Use astream for streaming generation")
        deadline_s, attempt_s = self._budget(model)
        started, call_id, attempts = time.monotonic(), uuid.uuid4().hex[:8], []
        cause, in_flight = None, None
        truncated = True
        try:
            async with asyncio.timeout(deadline_s):
                for provider in self._providers(model, kwargs.get("response_format")):
                    attempt_start = time.monotonic()
                    remaining = (float("inf") if deadline_s is None
                                 else deadline_s - (attempt_start - started))
                    if remaining <= 0:
                        raise GenerationDeadline("Generation deadline exceeded")
                    in_flight = provider
                    try:
                        prompt, options, timeout = self._options(provider, messages, kwargs,
                                                                remaining, attempt_s)
                        async with asyncio.timeout(timeout):
                            response = await self.router.acompletion(model=provider, messages=prompt, **options)
                        choice = response.choices[0]
                        if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
                            raise GenerationRefused("Provider refused generation")
                        if choice.finish_reason == "length":
                            raise OutputTruncated(kwargs.get("max_tokens"))
                        if choice.finish_reason not in ("stop", "tool_calls", "function_call"):
                            raise IncompleteCompletion("Provider did not complete the answer")
                        text = choice.message.content or ""
                        if not text.strip() and not getattr(choice.message, "tool_calls", None):
                            raise EmptyCompletion("Provider returned no content")
                        validate_content(text, kwargs.get("response_format"))
                    except Exception as exc:
                        cause = exc
                        truncated = truncated and isinstance(exc, OutputTruncated)
                        attempts.append(LLMAttempt(len(attempts), provider, False,
                                        (time.monotonic() - attempt_start) * 1000,
                                        error_class=type(exc).__name__))
                        if isinstance(exc, TimeoutError):
                            self._cool_down(provider, exc)
                        in_flight = None
                        if not retryable(exc):
                            raise
                        continue
                    elapsed = (time.monotonic() - started) * 1000
                    attempts.append(LLMAttempt(len(attempts), provider, True,
                                    (time.monotonic() - attempt_start) * 1000))
                    depth = self.chain.index(provider)
                    self._record_call(call_id, model, provider, depth, elapsed, attempts, usage=response.usage)
                    return CallResult(text, provider, depth, attempts, elapsed, response, response.usage)
                if isinstance(cause, InvalidCompletion):
                    raise cause
                if truncated and attempts:
                    raise TruncatedExhausted(model, attempts, cause) from cause
                raise AllProvidersExhausted(model, attempts, cause) from cause
        except TimeoutError as exc:
            # The shared budget expired while this provider was still answering: it consumed
            # what was left, so it is charged the failure rather than the caller.
            if in_flight is not None:
                self._cool_down(in_flight, exc)
            self._record_call(call_id, model, None, 0, (time.monotonic()-started)*1000,
                              attempts, error="GenerationDeadline")
            raise GenerationDeadline("Generation deadline exceeded") from exc
        except Exception as exc:
            self._record_call(call_id, model, None, 0, (time.monotonic()-started)*1000,
                              attempts, error=type(exc).__name__)
            raise

    async def astream(self, messages, *, model=PRIMARY_ALIAS, **kwargs):
        # One buffered chunk provides backpressure without moving provider clients
        # between loops. Cancelling the consumer cancels the producer's AnyIO scope.
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=1)

        async def send(item):
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(queue.put(item), loop))

        async def produce():
            try:
                async with asyncio.timeout(self.settings.generation_deadline_s):
                    async with aclosing(self._astream(messages, model=model, **kwargs)) as stream:
                        async for chunk in stream:
                            await send(chunk)
            except TimeoutError:
                await send(GenerationDeadline("Generation deadline exceeded"))
            except Exception as exc:
                await send(exc)
            else:
                await send(None)

        future = _completion_portal().start_task_soon(produce)
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            future.cancel()

    async def _astream(self, messages, *, model=PRIMARY_ALIAS, **kwargs):
        started = time.monotonic()
        attempts, cause = [], None
        committed = False
        truncated = True
        for provider in self._providers(model, kwargs.get("response_format")):
            stream = None
            attempt_start = time.monotonic()
            remaining = self.settings.generation_deadline_s - (attempt_start - started)
            if remaining <= 0:
                raise GenerationDeadline("Generation deadline exceeded")
            try:
                prompt, options, timeout = self._options(provider, messages, kwargs, remaining)
                # The asyncio window below bounds the pre-content phase; the transport may
                # only be cut by the shared deadline once a stream has started.
                options["timeout"] = remaining
                attempt_deadline = time.monotonic() + timeout
                async with asyncio.timeout(timeout):
                    stream = await self.router.acompletion(model=provider, messages=prompt,
                                                          stream=True, **options)
                terminal, content, emitted = None, [], False
                stream_deadline = started + self.settings.generation_deadline_s
                while True:
                    # No cancellation scope crosses yield: a slow consumer must not
                    # receive the producer's cancellation instead of a terminal error.
                    try:
                        async with asyncio.timeout(
                            (stream_deadline if emitted else attempt_deadline) - time.monotonic()
                        ):
                            chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if getattr(choice.delta, "refusal", None) or choice.finish_reason == "content_filter":
                        raise GenerationRefused("Provider refused generation")
                    if getattr(choice.delta, "tool_calls", None):
                        raise InvalidCompletion("Tool output is unsupported on this stream")
                    if choice.finish_reason:
                        # LiteLLM synthesizes stop on clean EOF. Require adapter evidence.
                        if not getattr(stream, "received_finish_reason", None):
                            raise IncompleteCompletion("Upstream stream ended without a terminal marker")
                        if choice.finish_reason == "length":
                            raise OutputTruncated(kwargs.get("max_tokens"))
                        if choice.finish_reason != "stop":
                            raise IncompleteCompletion("Provider did not complete the answer")
                        terminal = chunk.model_copy(deep=True)
                        terminal.choices[0].delta.content = None
                        choice.finish_reason = None
                    text = choice.delta.content or ""
                    if text:
                        content.append(text)
                        emitted = True
                    if kwargs.get("response_format"):
                        # Structured responses are atomic: validate before exposing JSON.
                        continue
                    if text:
                        committed = True
                        yield chunk
                if terminal is None:
                    raise IncompleteCompletion("Upstream stream ended without a terminal marker")
                text = "".join(content)
                if not text.strip():
                    raise EmptyCompletion("Provider returned no content")
                validate_content(text, kwargs.get("response_format"))
                if kwargs.get("response_format"):
                    chunk = terminal.model_copy(deep=True)
                    chunk.choices[0].delta.content = text
                    chunk.choices[0].finish_reason = None
                    committed = True
                    yield chunk
                yield terminal
                return
            except Exception as exc:
                cause = exc
                truncated = truncated and isinstance(exc, OutputTruncated)
                attempts.append(LLMAttempt(len(attempts), provider, False,
                                (time.monotonic()-attempt_start)*1000,
                                error_class=type(exc).__name__))
                if isinstance(exc, TimeoutError) and not committed:
                    self._cool_down(provider, exc)
                if committed and isinstance(exc, TimeoutError):
                    if time.monotonic() >= started + self.settings.generation_deadline_s:
                        raise GenerationDeadline("Generation deadline exceeded") from exc
                    raise IncompleteCompletion("Stream attempt timed out") from exc
                if committed or not retryable(exc):
                    raise
            except asyncio.CancelledError:
                # Only the shared budget can cancel here; a caller or consumer cancelling
                # the request is not the provider's failure.
                if time.monotonic() >= started + self.settings.generation_deadline_s:
                    self._cool_down(provider, GenerationDeadline("Generation deadline exceeded"))
                raise
            finally:
                if stream is not None:
                    await stream.aclose()
        if isinstance(cause, InvalidCompletion):
            raise cause
        if truncated and attempts:
            raise TruncatedExhausted(model, attempts, cause) from cause
        raise AllProvidersExhausted(model, attempts, cause) from cause

    @staticmethod
    def _record_call(call_id, alias, provider, depth, elapsed, attempts, response=None, usage=None, error=None):
        trace = get_trace()
        if trace is not None:
            trace.llm_calls.append(LLMCallRecord(
                call_id=call_id, alias=alias, served_by=provider, fallback_depth=depth,
                latency_ms=round(elapsed, 1), attempts=list(attempts),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None), error=error))


def build_gateway(**kwargs: Any) -> Gateway:
    return Gateway(**kwargs)
