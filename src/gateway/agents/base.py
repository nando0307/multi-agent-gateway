"""Shared types and prompt plumbing for the agents."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class Evidence:
    """One retrieved, sanitised source. ``source_id`` is what appears as ``[n]``."""

    source_id: int
    url: str
    title: str
    text: str
    sub_question: str
    quote_only: bool = False
    flags: list[str] = field(default_factory=list)
    retrieved_at: float = field(default_factory=time.time)

    def as_prompt_block(self) -> str:
        from gateway.security.sanitize import wrap

        return wrap(self.source_id, self.url, self.text, quote_only=self.quote_only, flags=self.flags)


@dataclass
class ResearchResult:
    question: str
    report: str
    evidence: list[Evidence]
    sub_questions: list[str]
    scores: dict[str, Any] | None = None
    gate_passed: bool | None = None
    warning: str | None = None
    served_by: list[str] = field(default_factory=list)

    @property
    def sources_section(self) -> str:
        return "\n".join(f"[{e.source_id}] {e.title} - {e.url}" for e in self.evidence)


def parse_json(text: str, default: Any = None) -> Any:
    """Best-effort JSON extraction from a model response.

    Models wrap JSON in prose or fences unpredictably, and a parse failure here should
    degrade the run, not end it -- so the caller supplies a default.
    """
    candidates = []
    block = JSON_BLOCK.search(text)
    if block:
        candidates.append(block.group(1))
    candidates.append(text)
    for start, end in (("[", "]"), ("{", "}")):
        i, j = text.find(start), text.rfind(end)
        if i != -1 and j > i:
            candidates.append(text[i : j + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
    return default
