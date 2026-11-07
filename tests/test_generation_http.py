"""Real loopback HTTP, including the LiteLLM parser (not mock_response)."""
import asyncio
import json
import select
import socket
import threading
import time
from contextlib import aclosing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from gateway.generation import generation_router
from gateway.llm.model_list import DEFAULT_CHAIN
from gateway.llm.router import AllProvidersExhausted, Gateway, GenerationDeadline, OutputTruncated
from gateway.settings import Settings


@pytest.fixture
def wire(request):
    calls, modes, bodies = [], {}, []
    closed = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            provider = self.path.split("/")[1]
            mode = modes.get(provider, "ok")
            calls.append(provider)
            bodies.append(body)
            if mode == "drop":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if mode == "slow":
                readable, _, _ = select.select([self.connection], [], [], 1.5)
                if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                    closed.set()
                    return
            if isinstance(mode, int):
                self.send_response(mode)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": "synthetic rejection",
                    "type": "invalid_request_error" if mode == 400 else "api_error"}}).encode())
                return
            content = "   " if mode == "empty" else "answer from " + provider
            finish = {"refusal": "content_filter", "truncate": "length"}.get(mode, "stop")
            if body.get("response_format"):
                content = '{"department":"hr"}'
                if mode == "bad_json":
                    content = '{"department":7}'
                elif mode == "nan_json":
                    content = '{"score": NaN}'
            if mode == "refusal":
                content = ""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.end_headers()
            base = {"id": "synthetic", "model": "gpt-4o-mini", "created": 1}
            try:
                if body.get("stream"):
                    def send(delta, finish=None):
                        self.wfile.write(("data: " + json.dumps({**base, "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode())
                        self.wfile.flush()
                    send({"role": "assistant"})
                    if mode == "pre_eof":
                        return
                    if mode == "drip":
                        # Split a valid schema answer across a slow drip.
                        left, right = content[:9], content[9:]
                        send({"content": left})
                        time.sleep(1.0)
                        send({"content": right})
                    else:
                        send({"content": content})
                    if mode == "post_eof":
                        return
                    if mode == "hang":
                        for _ in range(100):
                            time.sleep(0.01)
                            send({"content": "."})
                    send({}, finish)
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(json.dumps({**base, "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                     "finish_reason": finish}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}).encode())
            except (BrokenPipeError, ConnectionResetError):
                closed.set()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    settings = Settings(_env_file=None, gateway_service_token="synthetic-service-token",
                        generation_deadline_s=4, generation_attempt_s=1,
                        allowed_fails=100)
    entries = [{"model_name": name,
                "litellm_params": {"model": getattr(request, "param", "openai/gpt-4o-mini"), "api_key": "synthetic",
                    "api_base": f"http://127.0.0.1:{upstream.server_port}/{name}/v1"},
                "model_info": {"id": f"deployment::{name}", "provider": name}}
               for name in DEFAULT_CHAIN]
    gateway = Gateway(settings=settings, model_list=entries)
    app = FastAPI()
    app.include_router(generation_router(gateway, settings))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    # Deadlines are asserted explicitly elsewhere; this cap only needs to be wide
    # enough that CPU contention cannot masquerade as a gateway hang.
    client = httpx.Client(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
                          headers={"Authorization": "Bearer synthetic-service-token"}, timeout=15)
    yield gateway, client, modes, calls, bodies, closed
    client.close()
    server.should_exit = True
    thread.join(timeout=5)
    upstream.shutdown()
    upstream.server_close()


MESSAGES = [{"role": "user", "content": "Synthetic public test only"}]
PAYLOAD = {"model": "boss-default", "messages": MESSAGES, "max_completion_tokens": 384}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "RouteQuery", "strict": True,
          "schema": {"type": "object", "properties": {"department": {"type": "string", "enum": ["hr", "finance"]}},
                     "required": ["department"], "additionalProperties": False}}}


def test_empty_then_disconnect_reaches_third_sync_and_async(wire):
    gateway, _, modes, calls, _, _ = wire
    modes.update(gemini="empty", openrouter="drop")
    assert gateway.complete(MESSAGES).served_by == "groq"
    assert calls == ["gemini", "openrouter", "groq"]
    calls.clear()
    assert asyncio.run(gateway.acomplete(MESSAGES)).served_by == "groq"
    assert calls == ["gemini", "openrouter", "groq"]
    modes.update({name: "empty" for name in DEFAULT_CHAIN})
    with pytest.raises(AllProvidersExhausted):
        gateway.complete(MESSAGES)


@pytest.mark.parametrize("mode", ["drop", "slow", 429, 500, 502, 503, 504])
def test_http_transient_fallback_preserves_budget(wire, mode):
    _, client, modes, calls, bodies, _ = wire
    modes["gemini"] = mode
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == "openrouter"
    assert calls == ["gemini", "openrouter"]
    assert all(body["max_tokens"] == 384 and body["messages"] == MESSAGES for body in bodies)


@pytest.mark.parametrize("mode,status,code", [(400, 400, "invalid_request"),
                                              ("refusal", 400, "generation_refused")])
def test_terminal_failures_do_not_fallback(wire, mode, status, code):
    _, client, modes, calls, _, _ = wire
    modes["gemini"] = mode
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert calls == ["gemini"]


@pytest.mark.parametrize("mode", [401, 403])
def test_provider_credentials_rejected_advances_to_a_healthy_tier(wire, mode):
    """A revoked key or a disabled API is a provider outage, not a caller error."""
    _, client, modes, calls, _, _ = wire
    modes["gemini"] = mode
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-provider"] == "openrouter"
    assert calls == ["gemini", "openrouter"]


def test_auth_and_required_features_rejected_before_provider(wire):
    _, client, _, calls, _, _ = wire
    assert client.post("/v1/chat/completions", json=PAYLOAD, headers={"Authorization": "Bearer wrong"}).status_code == 401
    for extra in ({"tools": []}, {"api_base": "http://bad"}, {"model": "unapproved"},
                  {"max_tokens": 20}, {"response_format": {"type": "json_object"}}):
        assert client.post("/v1/chat/completions", json={**PAYLOAD, **extra}).status_code == 400
    assert calls == []


@pytest.mark.parametrize("wire", ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo"], indirect=True)
def test_schema_output_is_validated(wire):
    _, client, modes, calls, _, _ = wire
    request = {**PAYLOAD, "response_format": SCHEMA}
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"department": "hr"}
    modes["gemini"] = "bad_json"
    calls.clear()
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"department": "hr"}
    assert response.headers["x-gateway-provider"] == "openrouter"
    assert calls == ["gemini", "openrouter"]


def test_malformed_structured_output_surfaces_when_no_schema_tier_succeeds(wire):
    """All-eligible tiers answering off-schema must report the schema failure, not prose."""
    _, client, modes, calls, _, _ = wire
    modes.update({name: "bad_json" for name in DEFAULT_CHAIN})
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "response_format": SCHEMA})
    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "invalid_completion"
    assert calls == list(DEFAULT_CHAIN)


def test_research_callers_keep_their_own_budget(wire):
    """Boss's 60 s deadline must not reach the pre-existing research callers."""
    gateway, client, modes, calls, _, _ = wire
    gateway.settings.generation_deadline_s = 0.1
    gateway.settings.generation_attempt_s = 5
    modes["gemini"] = "slow"
    assert asyncio.run(gateway.acomplete(MESSAGES)).served_by == "gemini"
    assert gateway.complete(MESSAGES).served_by == "gemini"
    with pytest.raises(GenerationDeadline):
        asyncio.run(gateway.acomplete(MESSAGES, model="boss-default"))


@pytest.mark.parametrize("stream", [False, True])
def test_hanging_providers_are_cooled_after_a_wrapper_timeout(wire, stream):
    """A provider that never answers must enter cooldown, not be re-tried forever."""
    gateway, client, modes, calls, _, _ = wire
    assert client.post("/v1/chat/completions",
                       json={**PAYLOAD, "stream": stream}).status_code == 200
    calls.clear()
    gateway.settings.generation_attempt_s = 0.2
    gateway.settings.generation_deadline_s = 8
    modes.update({"gemini": "slow", "openrouter": "slow"})
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    assert response.status_code == 200, response.text
    calls.clear()
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    assert response.status_code == 200, response.text
    assert calls == ["groq"]


@pytest.mark.parametrize("stream", [False, True])
def test_serial_requests_recover_under_shipped_budget_ratios(wire, stream):
    """Attempt = deadline/2, as shipped (30 s/60 s): two hangs must still cool both tiers.

    The second hanging tier is cut by the shared deadline rather than by its own window,
    and the failure counter would expire between requests spaced a full budget apart, so
    both accounting paths have to hold for the chain to reach the healthy backup.
    """
    gateway, client, modes, calls, _, _ = wire
    assert client.post("/v1/chat/completions",
                       json={**PAYLOAD, "stream": stream}).status_code == 200
    calls.clear()
    gateway.settings.generation_attempt_s = 0.2
    gateway.settings.generation_deadline_s = 0.4
    modes.update({"gemini": "slow", "openrouter": "slow"})
    client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    calls.clear()
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    assert response.status_code == 200, response.text
    assert calls == ["groq"]


def test_non_json_numeric_constants_are_rejected(wire):
    """NaN passes json.loads and jsonschema's bounds; it is not valid JSON."""
    score_schema = {"type": "json_schema", "json_schema": {
        "name": "Readiness", "strict": True,
        "schema": {"type": "object",
                   "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
                   "required": ["score"], "additionalProperties": False}}}
    _, client, modes, _, _, _ = wire
    modes.update({name: "nan_json" for name in DEFAULT_CHAIN})
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "response_format": score_schema})
    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "invalid_completion"


def test_truncation_is_terminal_only_when_the_request_named_a_cap(wire):
    """A provider's own default cap must not abort a research call that named no budget."""
    gateway, _, modes, calls, _, _ = wire
    modes["gemini"] = "truncate"
    assert asyncio.run(gateway.acomplete(MESSAGES)).served_by == "openrouter"
    assert calls == ["gemini", "openrouter"]
    modes.update({name: "truncate" for name in DEFAULT_CHAIN})
    with pytest.raises(AllProvidersExhausted):
        asyncio.run(gateway.acomplete(MESSAGES))
    calls.clear()
    with pytest.raises(OutputTruncated, match="max_tokens=384"):
        asyncio.run(gateway.acomplete(MESSAGES, max_tokens=384))
    assert calls == ["gemini"]


@pytest.mark.parametrize("stream", [False, True])
def test_output_truncation_is_a_request_budget_error_not_an_outage(wire, stream):
    """A healthy provider answering to the requested cap must not fail over or look down."""
    _, client, modes, calls, _, _ = wire
    modes["gemini"] = "truncate"
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    assert calls == ["gemini"], "truncation must not advance the chain"
    assert "providers_exhausted" not in response.text
    assert "stream_interrupted" not in response.text
    if stream:
        assert response.status_code == 200, response.text
        assert '"code": "output_truncated"' in response.text
        assert "max_tokens=384" in response.text
        assert "[DONE]" not in response.text
    else:
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "output_truncated"
        assert "max_tokens=384" in error["message"]


@pytest.mark.parametrize("mode,expected", [("ok", ["gemini"]), ("pre_eof", ["gemini", "openrouter"]),
                                           ("post_eof", ["gemini"])])
def test_stream_terminal_evidence_and_no_model_splicing(wire, mode, expected):
    _, client, modes, calls, _, _ = wire
    modes["gemini"] = mode
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})
    assert calls == expected
    if mode == "post_eof":
        assert '"code": "stream_interrupted"' in response.text
        assert "[DONE]" not in response.text
        assert "answer from openrouter" not in response.text
    else:
        assert "[DONE]" in response.text, response.text
        assert '"finish_reason":"stop"' in response.text


def test_stream_disconnect_cancels_upstream(wire):
    _, client, modes, calls, _, closed = wire
    modes["gemini"] = "hang"
    with client.stream("POST", "/v1/chat/completions", json={**PAYLOAD, "stream": True}) as response:
        for line in response.iter_lines():
            if '"content"' in line:
                break
    assert closed.wait(2), "Upstream connection was not closed after client disconnect"
    assert calls == ["gemini"]
    # A caller walking away is not a provider failure: gemini must stay selectable.
    calls.clear()
    modes["gemini"] = "ok"
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})
    assert "[DONE]" in response.text, response.text
    assert calls == ["gemini"]


def test_started_stream_is_not_cut_by_the_attempt_window(wire):
    """Once content is emitted, only the total budget may interrupt the attempt."""
    gateway, client, modes, calls, _, _ = wire
    gateway.settings.generation_attempt_s = 0.2   # "hang" drips content for ~1s
    gateway.settings.generation_deadline_s = 10
    modes["gemini"] = "hang"
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})
    assert "[DONE]" in response.text, response.text
    assert '"finish_reason":"stop"' in response.text
    assert calls == ["gemini"]


def test_total_deadline_does_not_restart_per_provider(wire):
    gateway, client, modes, calls, _, _ = wire
    assert client.post("/v1/chat/completions", json=PAYLOAD).status_code == 200
    calls.clear()
    gateway.settings.generation_deadline_s = 0.18
    gateway.settings.generation_attempt_s = 0.1
    modes.update({name: "slow" for name in DEFAULT_CHAIN})
    start = time.monotonic()
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.status_code == 504, response.text
    assert response.json()["error"]["code"] == "generation_deadline"
    # Bound is loose under CPU contention; the chain assertion is the real proof.
    assert time.monotonic() - start < 3
    assert calls == ["gemini", "openrouter"]


def test_structured_stream_never_exposes_invalid_json(wire):
    _, client, modes, calls, _, _ = wire
    request = {**PAYLOAD, "stream": True, "response_format": SCHEMA}
    response = client.post("/v1/chat/completions", json=request)
    assert "[DONE]" in response.text
    calls.clear()
    modes.update({name: "bad_json" for name in DEFAULT_CHAIN})
    response = client.post("/v1/chat/completions", json=request)
    assert "invalid_completion" in response.text
    assert "department" not in response.text
    assert "[DONE]" not in response.text
    assert calls == list(DEFAULT_CHAIN)


def test_a_slow_structured_answer_is_not_abandoned_mid_stream(wire):
    """Buffered schema output is still provider output: the attempt must not be cut."""
    gateway, client, modes, calls, _, _ = wire
    assert client.post("/v1/chat/completions", json=PAYLOAD).status_code == 200
    calls.clear()
    gateway.settings.generation_attempt_s = 0.4   # the drip pauses for ~1s mid-answer
    gateway.settings.generation_deadline_s = 10
    modes["gemini"] = "drip"
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True,
                                                        "response_format": SCHEMA})
    assert response.status_code == 200, response.text
    assert "[DONE]" in response.text
    assert calls == ["gemini"]


def test_nonstream_disconnect_stops_fallback(wire):
    gateway, client, modes, calls, _, closed = wire
    gateway.settings.generation_attempt_s = 2
    modes["gemini"] = "slow"

    async def disconnect():
        async with httpx.AsyncClient(base_url=str(client.base_url), headers=client.headers) as http:
            task = asyncio.create_task(http.post("/v1/chat/completions", json=PAYLOAD))
            async with asyncio.timeout(2):
                while not calls:
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(disconnect())
    assert closed.wait(3), "Cancelled nonstream provider connection was left open"
    assert calls == ["gemini"]
    # A caller walking away is not a provider failure: gemini must stay selectable.
    calls.clear()
    modes["gemini"] = "ok"
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.headers["x-gateway-provider"] == "gemini", response.text
    assert calls == ["gemini"]


@pytest.mark.parametrize("mode,code", [("hang", "generation_deadline"), ("drop", "providers_exhausted")])
def test_stream_errors_keep_their_failure_reason(wire, mode, code):
    gateway, client, modes, calls, _, _ = wire
    assert client.post("/v1/chat/completions", json=PAYLOAD).status_code == 200
    calls.clear()
    gateway.settings.generation_attempt_s = 0.2
    if mode == "hang":
        # Content starts, so the remaining total budget bounds the attempt, not the window.
        gateway.settings.generation_deadline_s = 0.5
    modes.update({name: mode for name in DEFAULT_CHAIN})
    response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})
    assert code in response.text
    assert "[DONE]" not in response.text
    assert calls == (["gemini"] if mode == "hang" else list(DEFAULT_CHAIN))


async def test_slow_stream_consumer_gets_deadline_error_not_hanging_iterator(wire):
    gateway, _, modes, calls, _, closed = wire
    await gateway.acomplete(MESSAGES)
    calls.clear()
    gateway.settings.generation_deadline_s = 0.2
    modes["gemini"] = "hang"
    async with aclosing(gateway.astream(MESSAGES)) as stream:
        await anext(stream)
        await asyncio.sleep(0.35)
        with pytest.raises(GenerationDeadline):
            async with asyncio.timeout(0.5):
                async for _ in stream:
                    pass
    assert closed.wait(1)
    assert calls == ["gemini"]
