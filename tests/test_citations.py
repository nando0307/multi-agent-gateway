"""Citation-checking tests -- PLAN.md Phase 6.

The point of these is that a fabricated citation is caught *deterministically*. If any of
these needed a judge to pass, the citation score would be an opinion.
"""

from __future__ import annotations

from gateway.agents.base import Evidence
from gateway.eval.citations import check

SOLAR = (
    "Global solar photovoltaic capacity additions reached 447 GW in 2023, according to the "
    "International Energy Agency. China accounted for roughly 60 percent of new installations. "
    "Module prices fell by about 50 percent over the same year."
)
GRID = (
    "Grid-scale battery storage deployment doubled in 2023 to 42 GW of installed capacity. "
    "Lithium iron phosphate chemistry now dominates new grid projects."
)


def ev(source_id, url, text, sub_question="q"):
    return Evidence(source_id=source_id, url=url, title=f"src{source_id}", text=text,
                    sub_question=sub_question)


EVIDENCE = [ev(1, "https://iea.org/solar", SOLAR), ev(2, "https://example.org/grid", GRID)]
FETCHED = {"https://iea.org/solar", "https://example.org/grid"}


def test_well_cited_report_scores_high():
    report = (
        "Solar photovoltaic capacity additions reached 447 GW in 2023 [1]. "
        "China accounted for roughly 60 percent of new installations [1]. "
        "Grid-scale battery storage deployment doubled in 2023 to 42 GW [2]."
    )
    result = check(report, EVIDENCE, FETCHED)
    assert result.resolvable == 1.0
    assert result.grounded == 1.0
    assert result.cited_rate == 1.0
    assert result.supported == 1.0
    assert result.score > 0.9


def test_hallucinated_marker_is_caught_without_a_judge():
    """[7] was never provided. This must be provable by string comparison."""
    report = "Solar additions reached 447 GW in 2023 [1]. Wind additions reached 117 GW [7]."
    result = check(report, EVIDENCE, FETCHED)
    assert result.hallucinated_markers == [7]
    assert result.resolvable == 0.5
    assert result.score < 0.8


def test_url_never_fetched_is_flagged_as_ungrounded():
    """The agent cites a source it was given, but the run never actually retrieved it."""
    report = "Solar additions reached 447 GW in 2023 [1]. Storage doubled to 42 GW [2]."
    result = check(report, EVIDENCE, fetched_urls={"https://iea.org/solar"})
    assert result.ungrounded_urls == ["https://example.org/grid"]
    assert result.grounded == 0.5


def test_uncited_claims_lower_the_cited_rate():
    report = (
        "Solar additions reached 447 GW in 2023 [1]. "
        "Fusion power will supply most of the grid by the end of the decade. "
        "Hydrogen electrolyser costs collapsed last year."
    )
    result = check(report, EVIDENCE, FETCHED)
    assert result.cited_rate < 0.5


def test_claim_not_in_the_cited_source_is_escalated_and_unsupported():
    report = "Offshore wind turbine blade lengths exceeded ninety metres in Denmark last winter [1]."
    result = check(report, EVIDENCE, FETCHED)
    assert result.escalated == 1
    assert result.supported == 0.0
    assert result.judge_used is False


def test_escalated_claim_can_be_rescued_by_the_nli_judge():
    report = "Offshore wind blade lengths exceeded ninety metres in Denmark last winter [1]."
    result = check(report, EVIDENCE, FETCHED, nli_judge=lambda claim, source: True)
    assert result.judge_used is True
    assert result.supported == 1.0


def test_report_with_no_citations_at_all_scores_near_zero():
    report = "Solar grew a lot last year. Battery storage also grew. Prices came down."
    result = check(report, EVIDENCE, FETCHED)
    assert result.n_markers == 0
    assert result.score < 0.2
