"""Internal, generation-only OpenAI Chat Completions subset used by Boss."""
from __future__ import annotations

import asyncio
import hmac
import json
from typing import Literal

import litellm
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.requests import ClientDisconnect

from gateway.llm.model_list import BOSS_ALIASES
from gateway.llm.router import (
    AllProvidersExhausted, GenerationDeadline, GenerationRefused, IncompleteCompletion,
    InvalidCompletion,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class JsonSchema(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    schema_: dict = Field(alias="schema")
    strict: Literal[True] = True

    @model_validator(mode="after")
    def valid_schema(self):
        Draft202012Validator.check_schema(self.schema_)
        if self.schema_.get("type") != "object":
            raise ValueError("Structured output requires an object schema")
        if self.schema_.get("$schema", "https://json-schema.org/draft/2020-12/schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError("Only JSON Schema 2020-12 is supported")
        # No remote schema resolution, including refs hidden under nested definitions.
        def check(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ("$id", "$dynamicRef"):
                        raise ValueError("Schema identifiers and dynamic references are unsupported")
                    if key == "$ref":
                        if not isinstance(child, str) or not child.startswith("#/"):
                            raise ValueError("Only local JSON pointer references are supported")
                        target = self.schema_
                        try:
                            for part in child[2:].split("/"):
                                target = target[part.replace("~1", "/").replace("~0", "~")]
                        except (KeyError, TypeError) as exc:
                            raise ValueError("Unresolved schema reference") from exc
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(self.schema_)
        return self


class ResponseFormat(StrictModel):
    type: Literal["json_schema"]
    json_schema: JsonSchema


class ChatRequest(StrictModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    response_format: ResponseFormat | None = None

    @model_validator(mode="after")
    def supported_request(self):
        if self.model not in BOSS_ALIASES:
            raise ValueError("Unknown model alias")
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Specify only one token limit")
        return self


def error_response(exc, *, streaming=False):
    if isinstance(exc, GenerationDeadline):
        status, code = 504, "generation_deadline"
    elif isinstance(exc, GenerationRefused) or isinstance(exc, litellm.ContentPolicyViolationError):
        status, code = 400, "generation_refused"
    elif isinstance(exc, (ValidationError, SchemaError, ValueError, litellm.BadRequestError)):
        status, code = 400, "invalid_request"
    elif isinstance(exc, InvalidCompletion):
        status, code = 502, "invalid_completion"
    elif isinstance(exc, AllProvidersExhausted):
        status, code = 503, "providers_exhausted"
    elif isinstance(exc, IncompleteCompletion) or streaming:
        status, code = 502, "stream_interrupted"
    else:
        status, code = 502, "generation_failed"
    return status, {"error": {"message": code.replace("_", " "), "type": code, "code": code}}


async def cancel_on_disconnect(request, operation):
    """ASGI disconnects cancel upstream work, including before any response bytes."""
    task = asyncio.create_task(operation)

    async def disconnected():
        while not await request.is_disconnected():
            await asyncio.sleep(0.05)

    monitor = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((task, monitor), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        raise ClientDisconnect()
    finally:
        task.cancel()
        monitor.cancel()
        await asyncio.gather(task, monitor, return_exceptions=True)


def generation_router(gateway, settings):
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat(request: Request):
        token = settings.gateway_service_token
        if not token:
            return JSONResponse(status_code=503, content={"error": {
                "message": "Generation service is not configured", "type": "service_unavailable",
                "code": "service_unavailable"}})
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            return JSONResponse(status_code=401, headers={"WWW-Authenticate": "Bearer"},
                                content={"error": {"message": "Unauthorized", "type": "authentication_error",
                                                   "code": "authentication_error"}})
        try:
            payload = ChatRequest.model_validate(await request.json())
            options = payload.model_dump(exclude_none=True, exclude={"messages", "model", "stream"}, by_alias=True)
            # All approved backends accept max_tokens; never shrink the requested budget.
            if "max_completion_tokens" in options:
                options["max_tokens"] = options.pop("max_completion_tokens")
            messages = [message.model_dump() for message in payload.messages]
        except (ValidationError, ValueError, SchemaError) as exc:
            status, error = error_response(exc)
            return JSONResponse(status_code=status, content=error)

        if payload.stream:
            async def events():
                stream = gateway.astream(messages, model=payload.model, **options)
                try:
                    async for chunk in stream:
                        yield "data: " + chunk.model_dump_json(exclude_none=True) + "\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    _, error = error_response(exc, streaming=True)
                    yield "data: " + json.dumps(error) + "\n\n"
                finally:
                    await stream.aclose()
            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            result = await cancel_on_disconnect(request, gateway.acomplete(messages, model=payload.model, **options))
            return JSONResponse(result.raw.model_dump(exclude_none=True), headers={
                "X-Gateway-Provider": result.served_by,
                "X-Gateway-Fallback-Depth": str(result.fallback_depth)})
        except ClientDisconnect:
            return Response(status_code=499)
        except Exception as exc:
            status, error = error_response(exc)
            return JSONResponse(status_code=status, content=error)

    return router
