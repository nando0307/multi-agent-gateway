"""The quality gate -- PLAN.md Phase 7.

Nothing reaches a user unscored. A report below threshold gets exactly one retry on a
*different* provider; if it still fails, it is returned with an explicit warning and its
failing sub-scores attached.

It is never returned silently. That is the entire thesis of the eval bullet: a request
that succeeds at the transport layer and fails at the quality layer looks green on every
ops dashboard, and the only thing that catches it is a score.
"""

from __future__ import annotations

from dataclasses import dataclass

from gateway.agents.base import ResearchResult
from gateway.agents.synthesizer import synthesize
from gateway.eval.rubric import Scores, score
from gateway.llm.events import RunTrace


@dataclass
class GateOutcome:
    passed: bool
    scores: Scores
    retried_on: str | None = None
    warning: str | None = None


class QualityGate:
    def __init__(self, threshold: float = 0.70, judge=None, *, retry_on_fail: bool = True):
        self.threshold = threshold
        self.judge = judge
        self.retry_on_fail = retry_on_fail

    def evaluate(self, result: ResearchResult, trace: RunTrace) -> Scores:
        return score(
            result.question,
            result.report,
            result.evidence,
            result.sub_questions,
            trace.fetched_urls,
            judge=self.judge,
        )

    def apply(self, result: ResearchResult, trace: RunTrace, *, gateway, runner=None,
              model: str = "research-primary") -> ResearchResult:
        scores = self.evaluate(result, trace)
        result.scores = scores.to_dict()
        result.gate_passed = scores.composite >= self.threshold

        if result.gate_passed or not self.retry_on_fail:
            if not result.gate_passed:
                result.warning = self._warning(scores)
            return result

        alternative = self._alternative_provider(gateway, trace)
        if alternative is None:
            result.warning = self._warning(scores, note="no alternative provider available")
            return result

        retry_report = synthesize(result.question, result.evidence, gateway, model=alternative)
        retry_scores = score(
            result.question,
            retry_report,
            result.evidence,
            result.sub_questions,
            trace.fetched_urls,
            judge=self.judge,
        )

        if retry_scores.composite > scores.composite:
            result.report = retry_report
            scores = retry_scores
        result.scores = scores.to_dict()
        result.scores["retried_on"] = alternative
        result.gate_passed = scores.composite >= self.threshold
        if not result.gate_passed:
            result.warning = self._warning(scores, note=f"retried on {alternative}")
        return result

    @staticmethod
    def _alternative_provider(gateway, trace: RunTrace) -> str | None:
        used = trace.providers_used()
        return next((p for p in gateway.chain if p not in used), None)

    def _warning(self, scores: Scores, note: str | None = None) -> str:
        failing = [
            f"{name} {value:.2f}"
            for name, value in (
                ("citation", scores.citation),
                ("depth", scores.depth),
                ("coherence", scores.coherence),
            )
            if value < self.threshold
        ]
        detail = f" ({note})" if note else ""
        return (
            f"QUALITY GATE FAILED{detail}: composite {scores.composite:.2f} < "
            f"{self.threshold:.2f}. Weak dimensions: {', '.join(failing) or 'none individually'}. "
            "Treat this briefing as unverified."
        )
