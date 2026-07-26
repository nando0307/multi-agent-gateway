"""Deterministic citation checking -- PLAN.md Phase 6, decision D4.

An LLM asked "are these citations accurate?" produces an unfalsifiable answer. Three of
the four checks here need no model at all:

* **resolvable** -- does ``[n]`` correspond to a source the agent was actually given?
* **grounded**   -- was the cited URL genuinely fetched during this run? Checked against
                    the ``RunTrace``, so a fabricated URL is caught by string comparison.
* **cited_rate** -- what fraction of claim-bearing sentences carry any citation at all?
* **supported**  -- is the claim entailed by the cited text? Two-stage: a cheap lexical
                    screen, then an NLI judge only on what the screen cannot settle.

The lexical screen exists to keep judge cost proportional. It is a *screen*, not the
verdict: high overlap is accepted as support, low overlap is escalated rather than
condemned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MARKER = re.compile(r"\[(\d{1,3})\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
WORD = re.compile(r"[a-z0-9][a-z0-9\-']+")

STOPWORDS = frozenset(
    """the a an and or but of to in on for with by from as at is are was were be been being
    this that these those it its their there here which who whom whose what when where how
    not no than then also more most some any can could may might will would should have has
    had do does did about into over under between during""".split()
)

#: Sentences that make no factual claim need no citation.
NON_CLAIM = re.compile(
    r"^(#|\||\s*[-*]\s*$|```)|^\s*(however|in summary|in short|overall|this (briefing|report))\b",
    re.I,
)

LEXICAL_SUPPORT_THRESHOLD = 0.45


@dataclass
class CitationReport:
    resolvable: float = 0.0
    grounded: float = 0.0
    cited_rate: float = 0.0
    supported: float = 0.0
    n_markers: int = 0
    n_claim_sentences: int = 0
    hallucinated_markers: list[int] = field(default_factory=list)
    ungrounded_urls: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    escalated: int = 0
    judge_used: bool = False

    @property
    def score(self) -> float:
        return round(
            0.25 * self.resolvable + 0.25 * self.grounded + 0.2 * self.cited_rate
            + 0.3 * self.supported,
            4,
        )


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOPWORDS}


def sentences(report: str) -> list[str]:
    out: list[str] = []
    for line in report.splitlines():
        line = line.strip()
        if not line or NON_CLAIM.match(line):
            continue
        out.extend(s.strip() for s in SENTENCE_SPLIT.split(line) if s.strip())
    return out


def lexical_support(sentence: str, source_text: str) -> float:
    """Fraction of the sentence's content words that appear in the cited source."""
    claim = content_words(MARKER.sub("", sentence))
    if not claim:
        return 1.0
    return len(claim & content_words(source_text)) / len(claim)


def check(
    report: str,
    evidence: list,
    fetched_urls: set[str] | None = None,
    *,
    nli_judge=None,
) -> CitationReport:
    by_id = {e.source_id: e for e in evidence}
    result = CitationReport()

    markers = [int(m) for m in MARKER.findall(report)]
    result.n_markers = len(markers)

    # -- resolvable ---------------------------------------------------------------
    if markers:
        bad = [m for m in markers if m not in by_id]
        result.hallucinated_markers = sorted(set(bad))
        result.resolvable = 1.0 - len(bad) / len(markers)
    else:
        result.resolvable = 0.0

    # -- grounded: cited URLs must appear in the run's fetch trace ------------------
    cited_ids = {m for m in markers if m in by_id}
    if cited_ids:
        if fetched_urls is None:
            result.grounded = 1.0
        else:
            ungrounded = [by_id[i].url for i in cited_ids if by_id[i].url not in fetched_urls]
            result.ungrounded_urls = sorted(set(ungrounded))
            result.grounded = 1.0 - len(ungrounded) / len(cited_ids)
    else:
        result.grounded = 0.0

    # -- cited_rate and supported --------------------------------------------------
    claim_sentences = [s for s in sentences(report) if len(content_words(s)) >= 4]
    result.n_claim_sentences = len(claim_sentences)
    if not claim_sentences:
        return result

    cited_sentences = [s for s in claim_sentences if MARKER.search(s)]
    result.cited_rate = len(cited_sentences) / len(claim_sentences)

    if not cited_sentences:
        return result

    supported = 0
    for sentence in cited_sentences:
        ids = [int(m) for m in MARKER.findall(sentence) if int(m) in by_id]
        if not ids:
            result.unsupported_sentences.append(sentence)
            continue
        source_text = " ".join(by_id[i].text for i in ids)
        if lexical_support(sentence, source_text) >= LEXICAL_SUPPORT_THRESHOLD:
            supported += 1
            continue
        # The screen cannot settle it -- escalate rather than condemn.
        result.escalated += 1
        if nli_judge is not None:
            result.judge_used = True
            if nli_judge(sentence, source_text):
                supported += 1
                continue
        result.unsupported_sentences.append(sentence)

    result.supported = supported / len(cited_sentences)
    return result
