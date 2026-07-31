"""Recompute results/provider_matrix.md over the subset of questions with real evidence.

The 2026-07-29 run of run_eval.py --questions 30 hit Tavily's free-tier quota partway
through (search started raising ForbiddenError, swallowed silently by researcher.py's
failure handling). Any run with succeeded=True and n_sources=0 measured "no evidence
was retrievable", not provider quality -- and since a zero-evidence report scores the
same regardless of which model wrote it, those rows collapse the provider comparison
rather than informing it. A question is dropped entirely (for all providers, not just
the affected one) if any row shows the symptom, since the matrix's fairness claim
("all providers saw identical evidence") is void for that question either way.

Reads the raw rows already collected -- no live calls, no quota spent -- and rewrites
provider_matrix.md over the clean subset with the exclusion disclosed up front.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    raw_path = ROOT / "results" / "provider_matrix.json"
    data = json.loads(raw_path.read_text())
    rows = data["rows"]
    threshold = data["threshold"]
    judge_model = data.get("judge")

    by_question: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["provider"]] = row

    excluded = sorted(
        qid for qid, per in by_question.items()
        if any(r.get("succeeded") and r.get("n_sources") == 0 for r in per.values())
    )
    clean = sorted(set(by_question) - set(excluded), key=lambda x: int(x[1:]))
    clean_rows = [r for r in rows if r["question_id"] in clean]

    providers = list(dict.fromkeys(r["provider"] for r in rows))
    primary = providers[0]

    regressions = []
    for qid in clean:
        per_provider = by_question[qid]
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
        got = [r for r in clean_rows if r["provider"] == provider and r["succeeded"]]
        scored = [r["composite"] for r in got]
        total = len([r for r in clean_rows if r["provider"] == provider])
        return {
            "runs": total,
            "succeeded": len(got),
            "mean_composite": round(statistics.mean(scored), 3) if scored else None,
            "mean_citation": round(statistics.mean([r["citation"] for r in got]), 3) if got else None,
            "mean_depth": round(statistics.mean([r["depth"] for r in got]), 3) if got else None,
            "mean_coherence": round(statistics.mean([r["coherence"] for r in got]), 3) if got else None,
            "pass_rate": round(100 * sum(bool(r["passed"]) for r in got) / len(got), 1) if got else None,
        }

    summary = {p: stats(p) for p in providers}

    lines = [
        "# Phase 7 - Provider quality matrix",
        "",
        f"**Re-scoped 2026-07-29.** The original run covered 30 questions x {len(providers)} "
        f"providers (120 runs); Tavily's free-tier quota ran out partway through, and "
        f"**{len(excluded)} questions** had at least one provider return zero search results "
        "(a data-collection failure, not a quality signal -- a zero-evidence report scores the "
        "same regardless of which model wrote it, which would have measured 'no evidence' "
        "instead of comparing providers). This report covers the **9 questions with intact "
        "evidence across all rows** (`" + ", ".join(clean) + "`). "
        f"Excluded: `{', '.join(excluded)}`. See `results/provider_matrix_raw_30q.json` for "
        "the full unfiltered data.",
        "",
        f"{len(clean)} questions x {len(providers)} providers, each run **pinned** to one "
        "provider with failover disabled, so every row measures that provider alone. "
        f"Gate threshold **{threshold}**. "
        + (f"Judge: `{judge_model}`, temperature 0, a different model from every routed one."
           if judge_model else "Deterministic scoring only."),
        "",
        "All providers saw **identical evidence** on the questions kept here: the search cache "
        "is shared, so this compares models, not search luck.",
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

    groq_errors = [r for r in rows if r["provider"] == "groq" and not r["succeeded"]]
    lines += [
        "## Limitations",
        "",
        f"* **n = {len(clean)} questions** after excluding {len(excluded)}/30 to the Tavily "
        "quota gap above -- too small for the per-provider means to mean much beyond direction. "
        "Re-run once quota resets (cache already covers the excluded questions' non-failed "
        "queries) to recover the full n=30.",
        f"* Groq errored on {len(groq_errors)}/30 runs across the full (unfiltered) run with "
        "`RateLimitError` -- a real finding, not a data artifact: its free tier could not sustain "
        "pinned single-provider load with fallback disabled. Kept in the raw JSON; not counted "
        "toward the regression set here since a provider that never returned isn't a silent "
        "regression, it's a visible failure.",
        "* A single judge model scores every report. `results/judge_agreement.md` records how "
        "well it tracks hand labels -- read that before trusting any number here.",
        "* Citation sub-scores are deterministic and do not depend on the judge at all.",
        "",
    ]

    (ROOT / "results" / "provider_matrix.md").write_text("\n".join(lines))
    if not (ROOT / "results" / "provider_matrix_raw_30q.json").exists():
        (ROOT / "results" / "provider_matrix_raw_30q.json").write_text(json.dumps(data, indent=2))
    print(f"clean questions: {len(clean)}  excluded: {len(excluded)}")
    print(f"silent quality regressions (clean subset): {len(regressions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
