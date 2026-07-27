"""Phase 6 hard gate: does the judge track a human?

Two modes, deliberately separated so the labelling is not contaminated:

    python scripts/judge_agreement.py generate --n 20
        Produces reports and writes datasets/human_labels.jsonl with EMPTY score fields.
        The judge's scores are **not** computed here and do not appear in the sheet --
        seeing them first would anchor the labeller and the resulting agreement would
        measure compliance, not agreement.

    python scripts/judge_agreement.py score
        Reads the filled sheet, scores the same reports with the judge, and writes
        results/judge_agreement.md.

If Spearman rho comes out below ~0.6 the rubric is broken and the provider matrix is
measuring noise. Fix the rubric before running Phase 7 -- that is what makes this a gate
rather than a formality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gateway.agents.orchestrator import build_runner, run_research  # noqa: E402
from gateway.eval.judge import Judge, JudgeUnavailable  # noqa: E402
from gateway.llm.router import build_gateway  # noqa: E402
from gateway.settings import get_settings  # noqa: E402

SHEET = ROOT / "datasets" / "human_labels.jsonl"
REPORTS = ROOT / "results" / "label_reports.json"


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties don't distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def cmd_generate(args) -> int:
    settings = get_settings()
    gateway = build_gateway(settings=settings)
    questions = [
        json.loads(line)
        for line in (ROOT / "datasets" / "research_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # Spread across providers so the sheet contains a real quality range -- labelling 20
    # reports that are all good tells you nothing about whether the judge discriminates.
    providers = list(gateway.chain)
    picked, reports = [], []
    for i in range(args.n):
        question = questions[i % len(questions)]
        provider = providers[i % len(providers)]
        print(f"[{i+1}/{args.n}] {question['id']} on {provider}", flush=True)
        try:
            runner = build_runner(tavily_key=settings.tavily_api_key)
            result, _ = run_research(question["question"], gateway, runner, model=provider)
        except Exception as exc:
            print(f"   skipped: {type(exc).__name__}")
            continue
        item = {
            "label_id": f"L{len(picked)+1:02d}",
            "question_id": question["id"],
            "question": question["question"],
            "provider": provider,
            "report": result.report,
        }
        reports.append(item)
        picked.append({
            "label_id": item["label_id"],
            "question_id": question["id"],
            "human_depth": None,      # <- you fill these, 1-5
            "human_coherence": None,  # <- you fill these, 1-5
        })

    REPORTS.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.write_text(json.dumps(reports, indent=2))
    SHEET.write_text("\n".join(json.dumps(r) for r in picked) + "\n")

    md = ROOT / "results" / "label_sheet.md"
    md.write_text(
        "# Hand-labelling sheet\n\n"
        "Score each report 1-5 for DEPTH and COHERENCE using the rubrics in "
        "`src/gateway/eval/judge.py`, then fill `datasets/human_labels.jsonl` and run\n"
        "`python scripts/judge_agreement.py score`.\n\n"
        "The judge's own scores are deliberately absent here -- seeing them first would "
        "anchor you, and the agreement figure would then measure compliance rather than "
        "agreement.\n\n"
        + "\n\n---\n\n".join(
            f"## {r['label_id']} — {r['question_id']} ({r['provider']})\n\n"
            f"**Question:** {r['question']}\n\n{r['report']}\n\n"
            f"`human_depth`: ___   `human_coherence`: ___"
            for r in reports
        )
    )
    print(f"\nwrote {SHEET}  ({len(picked)} reports to label)")
    print(f"wrote {md}  <- read this, fill the sheet, then run: judge_agreement.py score")
    return 0


def cmd_score(args) -> int:
    if not SHEET.exists():
        print("no label sheet; run `judge_agreement.py generate` first", file=sys.stderr)
        return 2
    labels = [json.loads(line) for line in SHEET.read_text().splitlines() if line.strip()]
    filled = [x for x in labels if x.get("human_depth") and x.get("human_coherence")]
    if len(filled) < 3:
        print(f"only {len(filled)} rows labelled; fill datasets/human_labels.jsonl first",
              file=sys.stderr)
        return 2

    reports = {r["label_id"]: r for r in json.loads(REPORTS.read_text())}
    settings = get_settings()
    try:
        judge = Judge(settings)
    except JudgeUnavailable as exc:
        print(f"judge unavailable: {exc}", file=sys.stderr)
        return 2

    rows = []
    for item in filled:
        report = reports[item["label_id"]]
        print(f"judging {item['label_id']}...", flush=True)
        rows.append({
            **item,
            "judge_depth": judge.depth(report["question"], report["report"]).score,
            "judge_coherence": judge.coherence(report["question"], report["report"]).score,
        })

    out = {}
    for dim in ("depth", "coherence"):
        human = [float(r[f"human_{dim}"]) for r in rows]
        model = [float(r[f"judge_{dim}"]) for r in rows]
        within1 = sum(abs(h - m) <= 1 for h, m in zip(human, model)) / len(rows)
        out[dim] = {
            "spearman": round(spearman(human, model), 3),
            "within_1": round(100 * within1, 1),
            "mean_human": round(statistics.mean(human), 2),
            "mean_judge": round(statistics.mean(model), 2),
            "bias": round(statistics.mean(model) - statistics.mean(human), 2),
        }

    verdict = all(
        out[d]["spearman"] >= 0.6 for d in out if out[d]["spearman"] == out[d]["spearman"]
    )
    lines = [
        "# Phase 6 - Judge agreement with hand labels",
        "",
        f"`n = {len(rows)}` reports, hand-labelled before the judge was run. Judge: "
        f"`{settings.judge_model}`, temperature 0.",
        "",
        "The labeller did not see the judge's scores. Had they been visible, this would "
        "measure compliance rather than agreement.",
        "",
        "| dimension | Spearman rho | within +/-1 | mean human | mean judge | judge bias |",
        "|---|---|---|---|---|---|",
    ]
    for dim, s in out.items():
        lines.append(
            f"| {dim} | **{s['spearman']}** | {s['within_1']}% | {s['mean_human']} | "
            f"{s['mean_judge']} | {s['bias']:+} |"
        )
    lines += [
        "",
        f"## Verdict: {'PASS' if verdict else 'FAIL'}",
        "",
        "The gate is Spearman rho >= 0.6 on both dimensions. "
        + (
            "Both clear it, so the provider matrix is measuring something real."
            if verdict
            else "It is not met. The rubric needs work before Phase 7 is worth running -- "
            "a matrix built on a judge that does not track human judgement measures noise "
            "with extra steps."
        ),
        "",
        "`judge bias` is the mean judge score minus the mean human score. A consistent "
        "offset does not hurt provider *comparison* (it cancels), but it does move the "
        "absolute gate threshold, so the 0.70 cutoff should be read against it.",
        "",
        "## Per-report",
        "",
        "| id | question | provider | human depth | judge depth | human coh | judge coh |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep = reports[r["label_id"]]
        lines.append(
            f"| {r['label_id']} | {r['question_id']} | {rep['provider']} | "
            f"{r['human_depth']} | {r['judge_depth']:.0f} | "
            f"{r['human_coherence']} | {r['judge_coherence']:.0f} |"
        )
    lines.append("")

    path = ROOT / "results" / "judge_agreement.md"
    path.write_text("\n".join(lines))
    for dim, s in out.items():
        print(f"{dim:<10} rho={s['spearman']:<6} within1={s['within_1']}%  bias={s['bias']:+}")
    print(f"\nverdict: {'PASS' if verdict else 'FAIL'}   wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="produce reports and an unlabelled sheet")
    g.add_argument("--n", type=int, default=20)
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("score", help="compute agreement from the filled sheet")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
