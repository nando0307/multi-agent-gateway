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

import re
from dataclasses import dataclass

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

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

Reply with JSON only: {{"score": <1-5>, "reason": "<one short sentence, max 25 words>"}}"""

NLI_TEMPLATE = """Does the SOURCE support the CLAIM? Answer only whether the source
entails it -- not whether the claim is true in general.

CLAIM: {claim}

SOURCE:
{source}

Reply with JSON only: {{"supported": true|false}}"""


class JudgeUnavailable(RuntimeError):
    pass


#: Last-resort extraction when the JSON is malformed or truncated. The score is emitted
#: first, so it survives a cut that destroys the rest of the object.
SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*([1-5])(?:\.0)?\b')


@dataclass
class JudgeScore:
    score: float
    reason: str = ""


class Judge:
    def __init__(self, settings: Settings | None = None, *, samples: int = 1):
        self.settings = settings or get_settings()
        self.settings.assert_judge_is_independent()
        self.api_key = self.settings.resolve_judge_key()
        if not self.api_key:
            raise JudgeUnavailable(
                "No key resolves for JUDGE_MODEL. Set JUDGE_API_KEY, or use a JUDGE_MODEL whose "
                "route already has a key. The eval harness will not silently fall back to a "
                "model under test -- that would invalidate the provider matrix (PLAN.md D5)."
            )
        self.caveat = self.settings.judge_independence_caveat()
        self.model = self.settings.judge_model
        self.samples = samples

    # Free-tier judge models sit behind a shared, often-overloaded pool (unlike routed
    # providers, which retry via litellm.Router). RateLimitError (429), Timeout, and
    # InternalServerError (observed as a 529 "temporarily overloaded") are all normal
    # traffic for a free model, not an outage -- seen all three across different judge
    # models while chasing this.
    @retry(
        retry=retry_if_exception_type((
            litellm.exceptions.RateLimitError,
            litellm.exceptions.Timeout,
            litellm.exceptions.InternalServerError,
        )),
        wait=wait_random_exponential(multiplier=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _ask(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # Generous enough that a long `reason` cannot truncate the JSON mid-string.
            # A truncated response is not a judgement failure, it is a budget bug, and it
            # used to surface as "judge returned no parseable score".
            max_tokens=600,
        )
        return response.choices[0].message.content or ""

    def _scored(self, prompt: str) -> JudgeScore:
        """Median of ``samples`` draws. Temperature is 0, so >1 sample only guards against
        the occasional malformed response, not against sampling noise."""
        scores, reason, last = [], "", ""
        for _ in range(self.samples):
            last = self._ask(prompt)
            parsed = parse_json(last, default={})
            if isinstance(parsed, dict) and "score" in parsed:
                try:
                    scores.append(float(parsed["score"]))
                    reason = reason or str(parsed.get("reason", ""))
                    continue
                except (TypeError, ValueError):
                    pass
            match = SCORE_RE.search(last)
            if match:
                scores.append(float(match.group(1)))
        if not scores:
            raise JudgeUnavailable(
                f"judge returned no parseable score; raw response was {last[:200]!r}"
            )
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
