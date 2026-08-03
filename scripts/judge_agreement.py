"""Phase 6: does the judge track an independent reference rater?

Two modes, deliberately separated so the labelling is not contaminated:

    python scripts/judge_agreement.py generate --n 20
        Produces reports and writes datasets/human_labels.jsonl with EMPTY score fields.
        The judge's scores are **not** computed here and do not appear in the sheet --
        seeing them first would anchor the labeller and the resulting agreement would
        measure compliance, not agreement.

    python scripts/judge_agreement.py score
        Reads the filled sheet, scores the same reports with the judge, and writes
        results/judge_agreement.md.

The reference rater may be a human or a second model, and **which one it was changes what
the number means**:

  human   -- validates the judge against human judgement. This is the real Phase 6 gate.
  model   -- inter-rater agreement between two models. Useful, and a strong second rater
             catches a weak judge, but two models can agree because they share a blind
             spot. Passing does NOT close the human gate, and the report says so.

If Spearman rho comes out below ~0.6 the rubric is broken and the provider matrix is
measuring noise, regardless of rater type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
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
            runner = build_runner(tavily_key=settings.tavily_api_key, parallel_key=settings.parallel_api_key)
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

    write_sheet(reports)
    print(f"\nwrote {SHEET}  ({len(picked)} reports to label)")
    print(f"wrote {ROOT/'results'/'label_sheet.md'}  <- read this, fill the sheet, then run: judge_agreement.py score")
    return 0


def write_sheet(reports: list[dict]) -> None:
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


def cmd_regenerate(args) -> int:
    """Re-run the reports that came back empty, then rebuild the sheet.

    An empty report is not a bad report -- it is a broken one, and it renders in the sheet
    as a bare question with nothing under it. Labelling that measures nothing. Any row
    regenerated here has its labels reset to null, because scores attached to the old
    (blank) text do not describe the new text.
    """
    reports = json.loads(REPORTS.read_text())
    targets = [r for r in reports
               if (args.ids and r["label_id"] in args.ids)
               or (not args.ids and len(r["report"].strip()) < args.min_chars)]
    if not targets:
        print("nothing to regenerate")
        return 0

    settings = get_settings()
    gateway = build_gateway(settings=settings)
    by_id = {r["label_id"]: r for r in reports}
    regenerated = []

    for item in targets:
        print(f"regenerating {item['label_id']} ({item['question_id']} on {item['provider']}) "
              f"-- was {len(item['report'])} chars", flush=True)
        try:
            runner = build_runner(tavily_key=settings.tavily_api_key, parallel_key=settings.parallel_api_key)
            result, _ = run_research(
                item["question"], gateway, runner, model=item["provider"]
            )
        except Exception as exc:
            print(f"   failed: {type(exc).__name__}: {str(exc)[:70]}")
            continue
        if not result.report.strip():
            print("   still empty -- leaving as is")
            continue
        by_id[item["label_id"]]["report"] = result.report
        regenerated.append(item["label_id"])
        print(f"   now {len(result.report)} chars")

    if not regenerated:
        print("\nnothing changed")
        return 1

    REPORTS.write_text(json.dumps(list(by_id.values()), indent=2))
    write_sheet(list(by_id.values()))

    labels = [json.loads(line) for line in SHEET.read_text().splitlines() if line.strip()]
    for row in labels:
        if row["label_id"] in regenerated:
            for key in list(row):
                if key.endswith(("_depth", "_coherence")):
                    row[key] = None
    SHEET.write_text("\n".join(json.dumps(r) for r in labels) + "\n")

    print(f"\nregenerated {len(regenerated)}: {', '.join(regenerated)}")
    print("their labels were reset to null -- the old scores described the blank text")
    return 0


def cmd_score(args) -> int:
    if not SHEET.exists():
        print("no label sheet; run `judge_agreement.py generate` first", file=sys.stderr)
        return 2
    labels = [json.loads(line) for line in SHEET.read_text().splitlines() if line.strip()]

    # Accept either key naming, and infer the rater type from it unless overridden. The
    # distinction is not cosmetic: it decides whether this closes the Phase 6 human gate.
    def pick(row, dim):
        return row.get(f"human_{dim}") or row.get(f"model_{dim}")

    inferred = "model" if any("model_depth" in x for x in labels) else "human"
    rater_type = args.rater_type or inferred
    rater_name = args.rater or ("reference model" if rater_type == "model" else "human labeller")

    filled = [x for x in labels if pick(x, "depth") and pick(x, "coherence")]
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

    def score_once(pass_idx: int) -> list[dict]:
        scored = []
        for i, item in enumerate(filled):
            report = reports[item["label_id"]]
            # Pace the loop. Scoring n reports is 2n judge calls back to back, and the judge
            # sits on a free tier that rate-limits sustained bursts -- 36 calls with no gap
            # returned a 429 that outlasted Judge._ask's whole retry ladder, so the run died
            # after having already spent most of its calls. The ladder is for a blip; this is
            # for continuous pressure. --pace 0 disables it when the judge is a paid tier.
            if (i or pass_idx) and args.pace:
                time.sleep(args.pace)
            label = f"[pass {pass_idx + 1}/{args.repeat}] " if args.repeat > 1 else ""
            print(f"{label}judging {item['label_id']}... ({i + 1}/{len(filled)})", flush=True)
            scored.append({
                **item,
                "ref_depth": float(pick(item, "depth")),
                "ref_coherence": float(pick(item, "coherence")),
                "judge_depth": judge.depth(report["question"], report["report"]).score,
                "judge_coherence": judge.coherence(report["question"], report["report"]).score,
            })
        return scored

    def summarise(scored: list[dict]) -> dict:
        stats = {}
        for dim in ("depth", "coherence"):
            human = [float(r[f"ref_{dim}"]) for r in scored]
            model = [float(r[f"judge_{dim}"]) for r in scored]
            within1 = sum(abs(h - m) <= 1 for h, m in zip(human, model)) / len(scored)
            stats[dim] = {
                "spearman": round(spearman(human, model), 3),
                "within_1": round(100 * within1, 1),
                "mean_human": round(statistics.mean(human), 2),
                "mean_judge": round(statistics.mean(model), 2),
                "bias": round(statistics.mean(model) - statistics.mean(human), 2),
            }
        return stats

    # --repeat scores the identical reports N times. The judge is nominally deterministic at
    # temperature 0 and is not: a single run therefore reports a point estimate with unknown
    # error, and the provider matrix inherits that error. Repeating turns "rho = 0.801" into
    # "rho = 0.80 +/- sd", which is the honest form given the instrument.
    # Checkpoint each pass. 5 passes is 180 judge calls against a free tier that intermittently
    # returns "service temporarily overloaded"; one such burst outlasted the retry ladder in
    # pass 4 and took the three completed passes with it, because they were only in memory.
    # Same failure the provider matrix had, same fix. The fingerprint covers the judge model
    # and the exact reports+labels being scored, so a pass recorded against different inputs
    # is never silently reused.
    fingerprint = hashlib.sha256(json.dumps({
        "judge": settings.judge_model,
        "items": [(i["label_id"], pick(i, "depth"), pick(i, "coherence"),
                   reports[i["label_id"]]["report"]) for i in filled],
    }, sort_keys=True, default=str).encode()).hexdigest()[:16]
    passes_path = ROOT / "results" / "judge_passes.jsonl"

    passes: list[list[dict]] = []
    if passes_path.exists() and not args.fresh:
        for line in passes_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("fingerprint") == fingerprint:
                passes.append(rec["rows"])
        if passes:
            print(f"reusing {len(passes)} completed pass(es) from {passes_path.name}")
    elif args.fresh:
        passes_path.unlink(missing_ok=True)

    passes = passes[:args.repeat]
    for k in range(len(passes), args.repeat):
        scored = score_once(k)
        passes.append(scored)
        with passes_path.open("a") as fh:
            fh.write(json.dumps({"fingerprint": fingerprint, "rows": scored}) + "\n")
    rows = passes[-1]
    per_pass = [summarise(p) for p in passes]
    out = per_pass[-1]

    spread = {}
    if args.repeat > 1:
        for dim in ("depth", "coherence"):
            rhos = [s[dim]["spearman"] for s in per_pass]
            biases = [s[dim]["bias"] for s in per_pass]
            spread[dim] = {
                "rho_mean": round(statistics.mean(rhos), 3),
                "rho_sd": round(statistics.stdev(rhos), 3) if len(rhos) > 1 else 0.0,
                "rho_min": min(rhos), "rho_max": max(rhos),
                "bias_mean": round(statistics.mean(biases), 2),
                "bias_sd": round(statistics.stdev(biases), 3) if len(biases) > 1 else 0.0,
            }
        # How many individual scores moved between the first and last pass, on identical input.
        flips = sum(
            1 for a, b in zip(passes[0], passes[-1])
            for k in ("judge_depth", "judge_coherence") if a[k] != b[k]
        )
        spread["flips"] = flips
        spread["cells"] = len(rows) * 2

    # Judge the threshold on the mean when repeated, not on whichever pass happened to be last.
    def _rho(dim):
        return spread[dim]["rho_mean"] if spread else out[dim]["spearman"]
    verdict = all(_rho(d) >= 0.6 for d in ("depth", "coherence") if _rho(d) == _rho(d))
    human_gate = rater_type == "human"
    lines = [
        f"# Phase 6 - Judge agreement with an independent rater",
        "",
        f"`n = {len(rows)}` reports. Judge under test: `{settings.judge_model}`, temperature 0. "
        f"Reference rater: **{rater_name}** ({rater_type}).",
        "",
        "The reference rater scored the reports without seeing the judge's scores. Had they "
        "been visible, this would measure compliance rather than agreement.",
        "",]
    if not human_gate:
        lines += [
            "> **This is inter-model agreement, not human validation.** The reference rater is "
            f"a model (`{rater_name}`), not a person. A strong reference model is a real check "
            "-- it will catch a judge that is simply wrong -- but two language models can agree "
            "because they share a blind spot, and no amount of agreement between them detects "
            "that. The Phase 6 human gate in PLAN.md remains **open**; this does not close it.",
            "",
            "What it does establish: the judge is not idiosyncratic relative to a stronger model, "
            "so the provider matrix is not being ordered by one small model's private tastes.",
            "",
        ]
    # Emitted from here rather than written into the .md by hand, because this file is fully
    # regenerated on every `score` run -- a hand-added caveat would be silently erased by the
    # next one, which is exactly how the stale gpt-oss-120b numbers survived as long as they
    # did. The figures below are a fixed record of the 2026-08-02 re-run, not recomputed:
    # measuring drift needs two runs and this command only has one.
    if spread:
        lines += [
            f"> **Reproducibility, measured over {args.repeat} passes of the identical reports.** "
            f"The judge is nominally deterministic at temperature 0 and is not: "
            f"**{spread['flips']} of {spread['cells']}** individual scores differed between the "
            f"first and last pass. Spearman rho came out "
            f"**{spread['depth']['rho_mean']} +/- {spread['depth']['rho_sd']}** on depth "
            f"(range {spread['depth']['rho_min']}-{spread['depth']['rho_max']}) and "
            f"**{spread['coherence']['rho_mean']} +/- {spread['coherence']['rho_sd']}** on "
            f"coherence (range {spread['coherence']['rho_min']}-{spread['coherence']['rho_max']}). "
            "Quote the mean with its spread, not a single run's third decimal. Phase 6's "
            "*\"scores reproduce across two runs\"* acceptance criterion is **not met**; this "
            "block is what replaces it -- the instrument has error bars and they are stated.",
            "",
        ]
    else:
        lines += [
            "> **These scores do not reproduce exactly.** Re-scoring byte-identical reports at "
            "temperature 0 on 2026-08-02 changed **6 of 34** judge scores, moving depth rho "
            "0.903 -> 0.844 and coherence rho 0.879 -> 0.799 with no input change at all. Read "
            "any single run as +/-0.08 rather than exact. Run with `--repeat 5` to measure the "
            "spread on this judge rather than relying on this fixed record.",
            "",
        ]
    lines += [
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
        f"## Verdict: {'PASS' if verdict else 'FAIL'}"
        + ("" if human_gate else " (against a model rater -- human gate still open)"),
        "",
        "The threshold is Spearman rho >= 0.6 on both dimensions. "
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
        (f"Per-report scores below are from the final pass of {args.repeat}; other passes differ "
         f"on {spread['flips']} cell(s)." if spread else ""),
        "",
        f"| id | question | provider | {rater_type} depth | judge depth | {rater_type} coh | judge coh |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep = reports[r["label_id"]]
        lines.append(
            f"| {r['label_id']} | {r['question_id']} | {rep['provider']} | "
            f"{r['ref_depth']:.0f} | {r['judge_depth']:.0f} | "
            f"{r['ref_coherence']:.0f} | {r['judge_coherence']:.0f} |"
        )
    lines.append("")

    path = ROOT / "results" / "judge_agreement.md"
    path.write_text("\n".join(lines))
    for dim, s in out.items():
        if spread:
            sp = spread[dim]
            print(f"{dim:<10} rho={sp['rho_mean']} +/- {sp['rho_sd']} "
                  f"(range {sp['rho_min']}-{sp['rho_max']}, n={args.repeat} passes)  "
                  f"bias={sp['bias_mean']:+} +/- {sp['bias_sd']}")
        else:
            print(f"{dim:<10} rho={s['spearman']:<6} within1={s['within_1']}%  bias={s['bias']:+}")
    if spread:
        print(f"{'drift':<10} {spread['flips']}/{spread['cells']} scores differed between "
              f"first and last pass on identical input")
    print(f"\nverdict: {'PASS' if verdict else 'FAIL'}   wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="produce reports and an unlabelled sheet")
    g.add_argument("--n", type=int, default=20)
    g.set_defaults(func=cmd_generate)
    r = sub.add_parser("regenerate", help="re-run reports that came back empty")
    r.add_argument("--ids", nargs="*", default=None, help="specific label ids")
    r.add_argument("--min-chars", type=int, default=50)
    r.set_defaults(func=cmd_regenerate)

    s = sub.add_parser("score", help="compute agreement from the filled sheet")
    s.add_argument("--rater", default=None, help="who or what produced the labels")
    s.add_argument("--rater-type", choices=["human", "model"], default=None,
                   help="overrides the type inferred from the key naming")
    s.add_argument("--fresh", action="store_true",
                   help="discard checkpointed passes and re-score from scratch")
    s.add_argument("--repeat", type=int, default=1,
                   help="score the same reports N times and report rho as mean +/- sd; the "
                        "judge is not deterministic at temperature 0, so N=1 hides its error")
    s.add_argument("--pace", type=float, default=3.0,
                   help="seconds to wait between reports; 0 to disable (paid judge tier)")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
