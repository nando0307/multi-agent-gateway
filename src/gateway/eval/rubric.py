"""Composite scoring -- PLAN.md Phase 6.

Depth blends deterministic signals with a judge 50/50. The deterministic half is what
keeps the score anchored when the judge drifts: source count, domain diversity, and
specificity density (figures, dates, proper nouns per 100 words) are countable and cannot
be talked up by confident prose.

Coherence is judge-only -- there is no honest deterministic proxy for "does this hang
together" -- which is exactly why the Phase 6 agreement check against hand labels is a
hard gate before the matrix is allowed to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from gateway.eval.citations import CitationReport, check as check_citations

NUMBER = re.compile(r"\b\d[\d,.]*\b|\b(19|20)\d{2}\b|\b\d+\s?(%|percent|GW|MW|kWh|bn|billion|million)\b", re.I)
PROPER = re.compile(r"\b(?!The|This|These|In|A|An|It|We|Its)[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b")

WEIGHTS = {"citation": 0.4, "depth": 0.3, "coherence": 0.3}


@dataclass
class Scores:
    citation: float = 0.0
    depth: float = 0.0
    coherence: float = 0.0
    composite: float = 0.0
    citation_detail: CitationReport | None = None
    depth_signals: dict = field(default_factory=dict)
    judge_used: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "citation": round(self.citation, 4),
            "depth": round(self.depth, 4),
            "coherence": round(self.coherence, 4),
            "composite": round(self.composite, 4),
            "judge_used": self.judge_used,
            "depth_signals": self.depth_signals,
            "citation_detail": {
                "resolvable": round(self.citation_detail.resolvable, 3),
                "grounded": round(self.citation_detail.grounded, 3),
                "cited_rate": round(self.citation_detail.cited_rate, 3),
                "supported": round(self.citation_detail.supported, 3),
                "hallucinated_markers": self.citation_detail.hallucinated_markers,
                "ungrounded_urls": self.citation_detail.ungrounded_urls,
            }
            if self.citation_detail
            else None,
            "notes": self.notes,
        }


def _norm(value: float, target: float) -> float:
    return min(1.0, value / target) if target else 0.0


def depth_signals(report: str, evidence: list, sub_questions: list[str]) -> dict:
    words = max(1, len(report.split()))
    domains = {urlparse(e.url).netloc for e in evidence if e.url}
    covered = sum(
        1
        for q in sub_questions
        if any(e.sub_question == q for e in evidence)
    )
    return {
        "n_sources": len(evidence),
        "n_domains": len(domains),
        "sub_questions_covered": covered,
        "sub_questions_total": len(sub_questions),
        "specificity_per_100w": round(
            100 * (len(NUMBER.findall(report)) + len(PROPER.findall(report))) / words, 2
        ),
        "words": words,
        "acknowledges_limits": bool(
            re.search(r"\b(limitation|not (?:clear|available|found)|sources? (?:disagree|differ)|"
                      r"no (?:evidence|source)|unclear)\b", report, re.I)
        ),
    }


def deterministic_depth(signals: dict) -> float:
    covered = (
        signals["sub_questions_covered"] / signals["sub_questions_total"]
        if signals["sub_questions_total"]
        else 0.0
    )
    return round(
        0.25 * _norm(signals["n_sources"], 6)
        + 0.2 * _norm(signals["n_domains"], 4)
        + 0.25 * covered
        + 0.2 * _norm(signals["specificity_per_100w"], 8)
        + 0.1 * (1.0 if signals["acknowledges_limits"] else 0.0),
        4,
    )


def score(
    question: str,
    report: str,
    evidence: list,
    sub_questions: list[str],
    fetched_urls: set[str] | None = None,
    *,
    judge=None,
) -> Scores:
    nli = judge.as_nli() if judge is not None else None
    citation = check_citations(report, evidence, fetched_urls, nli_judge=nli)

    signals = depth_signals(report, evidence, sub_questions)
    det_depth = deterministic_depth(signals)

    result = Scores(
        citation=citation.score,
        depth=det_depth,
        coherence=0.0,
        citation_detail=citation,
        depth_signals=signals,
    )

    if judge is None:
        # Deterministic-only mode. Coherence is unmeasurable without a judge, so it is
        # excluded from the composite rather than assumed -- scoring an unmeasured
        # dimension as 0 or 1 would both be lies.
        result.judge_used = False
        result.notes.append("no judge configured: coherence excluded, depth deterministic only")
        w = {"citation": WEIGHTS["citation"], "depth": WEIGHTS["depth"]}
        total = sum(w.values())
        result.composite = round(
            (w["citation"] * result.citation + w["depth"] * result.depth) / total, 4
        )
        return result

    result.judge_used = True
    judged_depth = judge.depth(question, report).score
    result.depth = round(0.5 * det_depth + 0.5 * ((judged_depth - 1) / 4), 4)
    result.coherence = round((judge.coherence(question, report).score - 1) / 4, 4)
    result.composite = round(
        WEIGHTS["citation"] * result.citation
        + WEIGHTS["depth"] * result.depth
        + WEIGHTS["coherence"] * result.coherence,
        4,
    )
    return result
