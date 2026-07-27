"""Quality-gate tests -- PLAN.md Phase 7.

The behaviour being pinned down: a report that fails the gate is never returned silently.
That is the whole thesis -- a request can succeed at the transport layer (HTTP 200, no
error, green dashboard) and fail at the quality layer, and only a score catches it.
"""

from __future__ import annotations

from gateway.agents.base import Evidence, ResearchResult
from gateway.eval.gate import QualityGate
from gateway.eval.rubric import score
from gateway.llm.events import RunTrace, ToolCallRecord, trace_run
from tests.conftest import FAIL_500, make_gateway

SOURCE = (
    "Global solar photovoltaic capacity additions reached 447 GW in 2023 according to the "
    "International Energy Agency. China accounted for about 60 percent of new installations. "
    "Module prices fell roughly 50 percent over the year."
)

EVIDENCE = [
    Evidence(1, "https://iea.org/solar", "IEA solar", SOURCE, "how much solar was added?"),
    Evidence(2, "https://ember.org/solar", "Ember", SOURCE, "who installed the most?"),
]
FETCHED = {"https://iea.org/solar", "https://ember.org/solar"}
SUBS = ["how much solar was added?", "who installed the most?"]

GOOD = (
    "Solar photovoltaic additions reached 447 GW in 2023 [1]. China accounted for about "
    "60 percent of new installations [2]. Module prices fell roughly 50 percent over the "
    "same year [1]. Limitations: these figures cover utility-scale capacity only."
)
BAD = (
    "Solar energy grew substantially last year and the outlook is positive [9]. "
    "Industry observers expect continued momentum across most markets [12]. "
    "Costs are widely believed to be falling."
)


def _trace(fetched=FETCHED):
    trace = RunTrace(question="q")
    for url in fetched:
        trace.tool_calls.append(
            ToolCallRecord(tool="fetch_url", role="researcher", allowed=True, reason=None,
                           args={"url": url}, url=url)
        )
    return trace


def _result(report):
    return ResearchResult(question="How much solar was added in 2023?", report=report,
                          evidence=EVIDENCE, sub_questions=SUBS)


def test_fabricated_citations_score_far_below_a_grounded_report():
    good = score("q", GOOD, EVIDENCE, SUBS, FETCHED)
    bad = score("q", BAD, EVIDENCE, SUBS, FETCHED)
    assert good.composite > bad.composite
    assert bad.citation_detail.hallucinated_markers == [9, 12]


def test_good_report_passes_the_gate():
    gate = QualityGate(threshold=0.6)
    result = gate.apply(_result(GOOD), _trace(), gateway=make_gateway())
    assert result.gate_passed is True
    assert result.warning is None


def test_failing_report_is_never_returned_silently():
    gate = QualityGate(threshold=0.9, retry_on_fail=False)
    result = gate.apply(_result(BAD), _trace(), gateway=make_gateway())
    assert result.gate_passed is False
    assert result.warning and "QUALITY GATE FAILED" in result.warning
    assert result.scores["composite"] < 0.9


def test_gate_failure_triggers_exactly_one_retry_on_a_different_provider():
    calls: list[str] = []
    gateway = make_gateway()
    original = gateway.complete

    def counting(messages, *, model="research-primary", **kw):
        calls.append(model)
        return original(messages, model=model, **kw)

    gateway.complete = counting
    trace = _trace()
    trace.llm_calls.append(
        __import__("gateway.llm.events", fromlist=["LLMCallRecord"]).LLMCallRecord(
            call_id="a", alias="research-primary", served_by="gemini", fallback_depth=0,
            latency_ms=1.0,
        )
    )

    gate = QualityGate(threshold=0.99)
    gate.apply(_result(BAD), trace, gateway=gateway)

    assert len(calls) == 1, "gate must retry exactly once, not loop"
    assert calls[0] != "gemini", "retry must use a provider that has not already answered"


def test_retry_keeps_the_better_of_the_two_reports():
    """A retry that produces something worse must not replace a better original."""
    gate = QualityGate(threshold=0.99)
    gateway = make_gateway()
    gateway.complete = lambda *a, **k: type("R", (), {"text": "Nothing to report.", "raw": None})()
    result = gate.apply(_result(GOOD), _trace(), gateway=gateway)
    assert "447 GW" in result.report


def test_scores_are_attached_even_when_the_run_used_a_fallback_provider():
    """The silent-regression case: transport succeeded, quality did not."""
    gateway = make_gateway({"gemini": FAIL_500})
    with trace_run(_trace()) as trace:
        gateway.complete([{"role": "user", "content": "hi"}])
        result = QualityGate(threshold=0.9, retry_on_fail=False).apply(
            _result(BAD), trace, gateway=gateway
        )
    assert trace.used_fallback is True
    assert result.gate_passed is False
    assert result.scores["citation_detail"]["hallucinated_markers"] == [9, 12]


def test_deterministic_mode_excludes_coherence_rather_than_guessing_it():
    result = score("q", GOOD, EVIDENCE, SUBS, FETCHED, judge=None)
    assert result.judge_used is False
    assert result.coherence == 0.0
    assert "coherence excluded" in result.notes[0]
    assert result.composite > 0.0


# --- judge independence (PLAN.md D5) ---------------------------------------------
def test_judge_may_not_be_a_model_under_test():
    """The bias that matters is a model scoring its own output."""
    import pytest

    from gateway.settings import JudgeIndependenceError, Settings

    settings = Settings(
        _env_file=None, gemini_api_key="k", judge_model="gemini/gemini-3.1-flash-lite",
        gemini_model="gemini/gemini-3.1-flash-lite",
    )
    with pytest.raises(JudgeIndependenceError):
        settings.assert_judge_is_independent()


def test_same_model_via_a_different_route_is_still_the_same_model():
    """Changing route must not be a way around the guard."""
    import pytest

    from gateway.settings import JudgeIndependenceError, Settings

    settings = Settings(
        _env_file=None, gemini_api_key="k", gemini_model="gemini/gemini-3.1-flash-lite",
        judge_model="openrouter/google/gemini-3.1-flash-lite",
    )
    with pytest.raises(JudgeIndependenceError):
        settings.assert_judge_is_independent()


def test_different_model_on_a_shared_account_is_allowed_but_flagged():
    from gateway.settings import Settings

    settings = Settings(
        _env_file=None, openrouter_api_key="k",
        openrouter_model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
        judge_model="openrouter/openai/gpt-oss-120b",
    )
    settings.assert_judge_is_independent()          # no self-preference: different model
    caveat = settings.judge_independence_caveat()   # but the shared account is recorded
    assert caveat and "same `openrouter` account" in caveat


def test_openrouter_judge_reuses_the_openrouter_key():
    from gateway.settings import Settings

    settings = Settings(_env_file=None, openrouter_api_key="or-key",
                        judge_model="openrouter/openai/gpt-oss-120b")
    assert settings.resolve_judge_key() == "or-key"
