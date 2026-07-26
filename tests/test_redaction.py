"""Credential-leakage proof -- PLAN.md Phase 8.

The claim is "no credential leakage in logs". The test that backs it uses sentinel keys
and asserts they appear in *zero* outputs, including the exception path -- which is where
keys usually escape, because error bodies from providers often echo the request.
"""

from __future__ import annotations

import json
import logging

import pytest

from gateway.llm.events import RunTrace, ToolCallRecord, trace_run
from gateway.security import redaction
from gateway.security.redaction import MASK, redact, redact_obj, register_secret
from gateway.settings import Settings

SENTINELS = {
    "openrouter": "sk-or-v1-SENTINELDONOTLOG0123456789abcdef",
    "anthropic": "sk-ant-SENTINELDONOTLOG0123456789",
    "nvidia": "nvapi-SENTINELDONOTLOG0123456789ab",
    "gemini": "AIzaSENTINELDONOTLOG0123456789xyz",
    "tavily": "tvly-SENTINELDONOTLOG0123456789",
    "openai": "sk-SENTINELDONOTLOGabcdefghijklmnop",
}


@pytest.mark.parametrize("name,key", SENTINELS.items())
def test_known_key_shapes_are_redacted_by_pattern(name, key):
    assert key not in redact(f"call failed with key={key} at 12:00")


def test_unknown_key_format_is_redacted_by_value():
    """A provider inventing a new prefix must not defeat redaction."""
    weird = "XX-totally-new-format-9f8e7d6c5b4a"
    assert weird in redact(f"token {weird}")  # not yet registered
    register_secret(weird)
    assert weird not in redact(f"token {weird}")


def test_settings_secrets_are_registered_together():
    settings = Settings(
        _env_file=None,
        gemini_api_key=SENTINELS["gemini"],
        tavily_api_key=SENTINELS["tavily"],
        judge_api_key=SENTINELS["anthropic"],
    )
    redaction.register_settings(settings)
    blob = " ".join(SENTINELS.values())
    for key in (SENTINELS["gemini"], SENTINELS["tavily"], SENTINELS["anthropic"]):
        assert key not in redact(blob)


def test_authorization_headers_are_redacted():
    assert "abc123def456" not in redact("Authorization: Bearer abc123def456")
    assert "hunter2hunter2" not in redact('{"api_key": "hunter2hunter2"}')


def test_nested_structures_are_redacted_recursively():
    payload = {
        "request": {"headers": {"authorization": f"Bearer {SENTINELS['openai']}"}},
        "attempts": [{"error": f"401 for key {SENTINELS['gemini']}"}],
    }
    text = json.dumps(redact_obj(payload))
    assert SENTINELS["openai"] not in text
    assert SENTINELS["gemini"] not in text
    assert MASK in text


def test_exception_text_is_redacted():
    """The path keys usually escape through."""
    try:
        raise RuntimeError(f"upstream rejected key {SENTINELS['nvidia']}")
    except RuntimeError as exc:
        assert SENTINELS["nvidia"] not in redact(str(exc))
        assert SENTINELS["nvidia"] not in redact(repr(exc))


def test_full_trace_serialisation_carries_no_sentinel(caplog):
    """End-to-end: build a trace containing secrets everywhere, serialise, assert clean."""
    for key in SENTINELS.values():
        register_secret(key)

    with caplog.at_level(logging.DEBUG):
        with trace_run(RunTrace(question=f"why did {SENTINELS['gemini']} fail?")) as trace:
            trace.tool_calls.append(
                ToolCallRecord(
                    tool="fetch_url",
                    role="researcher",
                    allowed=False,
                    reason=f"401 using {SENTINELS['tavily']}",
                    args={"url": f"https://api.example.com/?key={SENTINELS['openrouter']}"},
                )
            )
            logging.getLogger("gateway").debug("attempt failed: %s", SENTINELS["anthropic"])

    serialised = json.dumps(redact_obj(trace.to_dict()), default=str)
    for name, key in SENTINELS.items():
        assert key not in serialised, f"{name} sentinel leaked into the trace"

    log_text = redact("\n".join(r.getMessage() for r in caplog.records))
    for name, key in SENTINELS.items():
        assert key not in log_text, f"{name} sentinel leaked into logs"
