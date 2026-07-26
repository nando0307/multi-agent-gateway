"""LLM-as-judge -- PLAN.md Phase 6, decision D5.

This module deliberately does **not** use ``Gateway``. The judge must never be one of the
providers it scores, or the Phase 7 matrix measures self-preference instead of quality;
routing it through the failover chain would eventually do exactly that. It calls
``litellm.completion`` directly against ``JUDGE_MODEL`` and fails loudly if that key is
missing, rather than silently falling back to a provider under test.

Rubrics are anchored -- each score level has a description. Unanchored 1-5 judges drift
badly between runs and make score deltas meaningless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import litellm

from gateway.agents.base import parse_json
from gateway.settings import Settings, get_settings

DEPTH_RUBRIC = """Score research DEPTH from 1 to 5.

5 - Covers every major facet of the question. Dense with specifics: figures, dates, named
    organisations. Notes where evidence is thin or sources disagree.
4 - Covers the main facets with several concrete specifics; minor gaps.
3 - Answers the question but stays general; few specifics; obvious facets untouched.
2 - Superficial. Mostly restates the question or offers generic commentary.
1 - Says almost nothing a reader could not have guessed without research."""

COHERENCE_RUBRIC = """Score COHERENCE from 1 to 5.

5 - Clear structure, no contradictions, directly answers the question asked, no repetition.
4 - Well organised with a minor digression or repeated point.
3 - Readable but meanders, or partially answers a different question.
2 - Disorganised, repetitive, or contains a self-contradiction.
1 - Incoherent, or answers a question that was not asked."""

TEMPLATE = """{rubric}

Question the briefing was meant to answer:
{question}

Briefing:
---
{report}
---

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""

NLI_TEMPLATE = """Does the SOURCE support the CLAIM? Answer only whether the source
entails it -- not whether the claim is true in general.

CLAIM: {claim}

SOURCE:
{source}

Reply with JSON only: {{"supported": true|false}}"""


class JudgeUnavailable(RuntimeError):
    pass


@dataclass
class JudgeScore:
    score: float
    reason: str = ""


class Judge:
    def __init__(self, settings: Settings | None = None, *, samples: int = 1):
        self.settings = settings or get_settings()
        self.settings.assert_judge_is_independent()
        if not self.settings.judge_api_key:
            raise JudgeUnavailable(
                "JUDGE_API_KEY is not set. The eval harness will not silently fall back to a "
                "routed provider -- that would invalidate the provider matrix (PLAN.md D5)."
            )
        self.model = self.settings.judge_model
        self.samples = samples

    def _ask(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model,
            api_key=self.settings.judge_api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""

    def _scored(self, prompt: str) -> JudgeScore:
        """Median of ``samples`` draws. Temperature is 0, so >1 sample only guards against
        the occasional malformed response, not against sampling noise."""
        scores, reason = [], ""
        for _ in range(self.samples):
            parsed = parse_json(self._ask(prompt), default={})
            if isinstance(parsed, dict) and "score" in parsed:
                try:
                    scores.append(float(parsed["score"]))
                    reason = reason or str(parsed.get("reason", ""))
                except (TypeError, ValueError):
                    continue
        if not scores:
            raise JudgeUnavailable("judge returned no parseable score")
        scores.sort()
        return JudgeScore(score=scores[len(scores) // 2], reason=reason)

    def depth(self, question: str, report: str) -> JudgeScore:
        return self._scored(TEMPLATE.format(rubric=DEPTH_RUBRIC, question=question, report=report))

    def coherence(self, question: str, report: str) -> JudgeScore:
        return self._scored(
            TEMPLATE.format(rubric=COHERENCE_RUBRIC, question=question, report=report)
        )

    def entails(self, claim: str, source: str) -> bool:
        parsed = parse_json(
            self._ask(NLI_TEMPLATE.format(claim=claim, source=source[:6000])), default={}
        )
        return bool(isinstance(parsed, dict) and parsed.get("supported"))

    def as_nli(self):
        return self.entails
