"""Phase 7: the provider matrix and the silent-regression count.

Each question is run once per provider, **pinned** -- no failover -- so each row measures
that provider and nothing else. All providers see identical evidence, because the search
cache is shared: otherwise this would measure search variance rather than model quality.

A *silent quality regression* is defined precisely, and the definition is printed into the
report so the number can be argued with:

    A run where the request SUCCEEDED -- no error, no exception, within the latency SLO,
    green on every ops dashboard -- but whose composite score fell below the gate
    threshold, on a question the primary provider passed.

That is the failure mode failover introduces and monitoring cannot see: the request works,
the answer is worse.

Usage:
    python scripts/run_eval.py --questions 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gateway.agents.orchestrator import build_runner, run_research  # noqa: E402
from gateway.eval.gate import QualityGate  # noqa: E402
from gateway.eval.judge import Judge, JudgeUnavailable  # noqa: E402
from gateway.llm.router import build_gateway  # noqa: E402
from gateway.security.redaction import register_settings  # noqa: E402
from gateway.settings import get_settings  # noqa: E402


def load_questions(limit: int) -> list[dict]:
    path = ROOT / "datasets" / "research_questions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-judge", action="store_true",
                    help="deterministic scoring only; coherence is excluded, not guessed")
    args = ap.parse_args()

    settings = get_settings()
    register_settings(settings)
    threshold = args.threshold if args.threshold is not None else settings.gate_threshold

    judge = None
    if not args.no_judge:
        try:
            judge = Judge(settings)
        except JudgeUnavailable as exc:
            print(f"refusing to run with a routed provider as judge: {exc}", file=sys.stderr)
            return 2

    gateway = build_gateway(settings=settings)
    providers = list(gateway.chain)
    questions = load_questions(args.questions)
    print(f"{len(questions)} questions x {len(providers)} providers = "
          f"{len(questions) * len(providers)} runs\n")

    rows: list[dict] = []
    for question in questions:
        for provider in providers:
            runner = build_runner(tavily_key=settings.tavily_api_key)
            started = time.perf_counter()
            try:
                result, trace = run_research(
                    question["question"], gateway, runner,
                    model=provider,                       # pinned: no failover
                    gate=QualityGate(threshold, judge, retry_on_fail=False),
                    save_dir=ROOT / "results" / "runs",
                )
                elapsed = time.perf_counter() - started
                rows.append({
                    "question_id": question["id"],
                    "category": question.get("category"),
                    "provider": provider,
                    "succeeded": True,
                    "latency_s": round(elapsed, 2),
                    "composite": result.scores["composite"],
                    "citation": result.scores["citation"],
                    "depth": result.scores["depth"],
                    "coherence": result.scores["coherence"],
                    "passed": result.gate_passed,
                    "n_sources": len(result.evidence),
                    "trace": trace.run_id,
                })
                print(f"  {question['id']:<8} {provider:<11} "
                      f"composite {result.scores['composite']:.2f} "
                      f"{'PASS' if result.gate_passed else 'FAIL'}")
            except Exception as exc:
                rows.append({
                    "question_id": question["id"], "provider": provider, "succeeded": False,
                    "error": f"{type(exc).__name__}: {exc}", "composite": None, "passed": False,
                })
                print(f"  {question['id']:<8} {provider:<11} ERROR {type(exc).__name__}")

    # -- silent regressions ---------------------------------------------------------
    primary = providers[0]
    by_question: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["provider"]] = row

    regressions = []
    for qid, per_provider in by_question.items():
        base = per_provider.get(primary)
        if not base or not base["succeeded"] or not base["passed"]:
            continue
        for provider, row in per_provider.items():
            if provider == primary or not row["succeeded"]:
                continue
            if not row["passed"]:
                regressions.append({
                    "question_id": qid,
                    "provider": provider,
                    "primary_composite": base["composite"],
                    "fallback_composite": row["composite"],
                    "delta": round(row["composite"] - base["composite"], 4),
                })

    def stats(provider: str) -> dict:
        got = [r for r in rows if r["provider"] == provider and r["succeeded"]]
        scored = [r["composite"] for r in got]
        return {
            "runs": len([r for r in rows if r["provider"] == provider]),
            "succeeded": len(got),
            "mean_composite": round(statistics.mean(scored), 3) if scored else None,
            "mean_citation": round(statistics.mean([r["citation"] for r in got]), 3) if got else None,
            "mean_depth": round(statistics.mean([r["depth"] for r in got]), 3) if got else None,
            "mean_coherence": round(statistics.mean([r["coherence"] for r in got]), 3) if got else None,
            "pass_rate": round(100 * sum(bool(r["passed"]) for r in got) / len(got), 1) if got else None,
        }

    summary = {p: stats(p) for p in providers}
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "provider_matrix.json").write_text(
        json.dumps({"rows": rows, "summary": summary, "regressions": regressions,
                    "threshold": threshold, "judge": None if judge is None else settings.judge_model},
                   indent=2)
    )

    lines = [
        "# Phase 7 - Provider quality matrix",
        "",
        f"{len(questions)} questions x {len(providers)} providers, each run **pinned** to one "
        "provider with failover disabled, so every row measures that provider alone. "
        f"Gate threshold **{threshold}**. "
        + (f"Judge: `{settings.judge_model}`, temperature 0, a different model from every "
           "routed one." if judge
           else "Deterministic scoring only -- coherence is excluded rather than guessed."),
        "",]
    if judge is not None and judge.caveat:
        lines += ["> **Independence caveat.** " + judge.caveat, ""]
    lines += [
        "All providers saw **identical evidence**: the search cache is shared across the matrix, "
        "so this compares models, not search luck.",
        "",
        "| provider | runs | succeeded | mean composite | citation | depth | coherence | pass rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for provider in providers:
        s = summary[provider]
        lines.append(
            f"| {provider} | {s['runs']} | {s['succeeded']} | {s['mean_composite']} | "
            f"{s['mean_citation']} | {s['mean_depth']} | {s['mean_coherence']} | {s['pass_rate']}% |"
        )

    lines += [
        "",
        "## Silent quality regressions",
        "",
        f"**{len(regressions)}** caught.",
        "",
        "> A run where the request **succeeded** -- no error, no exception, within the latency "
        "> SLO, green on every ops dashboard -- but whose composite score fell below the "
        f"> {threshold} gate, on a question the primary (`{primary}`) passed.",
        "",
        "This is the failure mode that failover introduces and that monitoring cannot see: "
        "the request works, the answer is worse. Nothing in an error rate, a latency histogram "
        "or an uptime check registers it.",
        "",
    ]
    if regressions:
        lines += [
            "| question | fallback provider | primary score | fallback score | delta |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(regressions, key=lambda x: x["delta"]):
            lines.append(
                f"| {r['question_id']} | {r['provider']} | {r['primary_composite']:.2f} | "
                f"{r['fallback_composite']:.2f} | {r['delta']:+.2f} |"
            )
        lines.append("")

    lines += [
        "## Limitations",
        "",
        f"* n = {len(questions)} questions. Per-provider means carry real uncertainty; treat "
        "small gaps between providers as noise, not ranking.",
        "* A single judge model scores every report. `results/judge_agreement.md` records how "
        "well it tracks hand labels -- read that before trusting any number here.",
        "* Citation sub-scores are deterministic and do not depend on the judge at all.",
        "",
    ]
    (results / "provider_matrix.md").write_text("\n".join(lines))
    print(f"\nsilent quality regressions: {len(regressions)}")
    print(f"wrote {results / 'provider_matrix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
