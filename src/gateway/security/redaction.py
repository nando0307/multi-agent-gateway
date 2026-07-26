"""Secret redaction for logs, traces and error paths -- PLAN.md Phase 8.

Two layers, because either alone is insufficient:

* **Pattern redaction** catches known key shapes (``sk-``, ``AIza``, ``nvapi-``) including
  keys this process has never seen -- e.g. one echoed back inside a provider error body.
* **Value redaction** catches every secret in ``Settings`` verbatim, regardless of format.
  This is the layer that survives a provider inventing a new key prefix.

Redaction is applied to log records *and* to exception text, because the place credentials
usually escape is a traceback, not an info log.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MASK = "***REDACTED***"

PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),          # OpenRouter
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),          # Anthropic
    re.compile(r"nvapi-[A-Za-z0-9_\-]{16,}"),           # NVIDIA NIM
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),             # Google / Gemini
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                 # OpenAI-style
    re.compile(r"tvly-[A-Za-z0-9_\-]{16,}"),            # Tavily
]

#: (pattern, replacement) where the label is kept and only the value is masked, so a
#: redacted log still shows *which* credential was involved.
LABELLED: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"""(?ix)
            ( ["']? \b (?: authorization | api[-_\ ]?key | apikey | bearer
                          | token | secret | password ) \b ["']? \s* [:=] \s* ["']? )
            (?: Bearer \s+ )?
            ( [A-Za-z0-9._\-]{6,} )
            """
        ),
        r"\1" + MASK,
    ),
]

_extra_values: set[str] = set()


def register_secret(value: str | None) -> None:
    """Add a literal secret value to the redaction set."""
    if value and len(value) >= 8:
        _extra_values.add(value)


def register_settings(settings: Any) -> None:
    for field_name in (
        "gemini_api_key",
        "nvidia_nim_api_key",
        "openrouter_api_key",
        "tavily_api_key",
        "judge_api_key",
    ):
        register_secret(getattr(settings, field_name, None))


def redact(text: str) -> str:
    if not text:
        return text
    for value in _extra_values:
        text = text.replace(value, MASK)
    for pattern in PATTERNS:
        text = pattern.sub(MASK, text)
    for pattern, replacement in LABELLED:
        text = pattern.sub(replacement, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings inside dicts/lists -- used on RunTrace before it is written."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v) for v in obj)
    return obj


def structlog_processor(logger, method_name, event_dict: dict) -> dict:
    return {k: redact_obj(v) for k, v in event_dict.items()}


def known_secrets() -> Iterable[str]:
    return frozenset(_extra_values)
