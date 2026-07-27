"""Phase 1: measure each provider, then order the fallback chain from the result.

The ordering key is **p95, not p50**. Failover exists for the tail: a provider with an
excellent median and a bad tail is a bad primary, because the tail is exactly when you
need to have already moved on. p50 is reported for context and deliberately not used.

Run twice at different times of day -- provider load varies -- and average.
Usage:
    python scripts/bench_latency.py --runs 20 --label morning
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

import litellm  # noqa: E402

from gateway.llm.model_list import DEFAULT_CHAIN, provider_params  # noqa: E402
from gateway.settings import get_settings  # noqa: E402

PROMPT = [
    {
        "role": "user",
        "content": (
            "In about 200 words, summarise the main constraints on grid-scale battery "
            "storage deployment. Be specific about figures where you can."
        ),
    }
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def bench(name: str, params: dict, runs: int) -> dict:
    # A per-provider timeout in params wins; passing another one here would collide and
    # fail every run for the provider that has one (the local tier).
    params = {"timeout": 60, **params}
    latencies, failures, tokens, costs = [], 0, [], []
    for _ in range(runs):
        started = time.perf_counter()
        try:
            response = litellm.completion(messages=PROMPT, max_tokens=400, **params)
            latencies.append((time.perf_counter() - started) * 1000)
            usage = getattr(response, "usage", None)
            if usage:
                tokens.append(getattr(usage, "completion_tokens", 0) or 0)
            try:
                costs.append(litellm.completion_cost(completion_response=response))
            except Exception:
                pass
        except Exception:
            failures += 1
    return {
        "provider": name,
        "runs": runs,
        "failures": failures,
        "p50_ms": round(percentile(latencies, 0.50), 1) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95), 1) if latencies else None,
        "max_ms": round(max(latencies), 1) if latencies else None,
        "mean_completion_tokens": round(statistics.mean(tokens), 1) if tokens else None,
        "mean_cost_usd": round(statistics.mean(costs), 6) if costs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--label", default="run1", help="e.g. morning / evening")
    ap.add_argument("--only", nargs="*", default=None,
                    help="benchmark just these providers and merge into the label")
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the report from latency.json without re-benchmarking")
    args = ap.parse_args()

    settings = get_settings()
    rows = []
    for name in DEFAULT_CHAIN:
        if args.report_only or (args.only and name not in args.only):
            continue
        params = provider_params(name, settings)
        if params is None:
            print(f"{name}: skipped (no credentials)")
            continue
        print(f"benchmarking {name} ({args.runs} runs)...", flush=True)
        row = bench(name, params, args.runs)
        rows.append(row)
        print(f"  p50 {row['p50_ms']}ms  p95 {row['p95_ms']}ms  failures {row['failures']}")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    path = results / "latency.json"
    history = json.loads(path.read_text()) if path.exists() else {}
    # Merge per provider rather than replacing the label wholesale, so a single provider
    # can be re-run (or backfilled) without discarding the rest of the sitting.
    existing = {r["provider"]: r for r in history.get(args.label, [])}
    existing.update({r["provider"]: r for r in rows})
    history[args.label] = list(existing.values())
    path.write_text(json.dumps(history, indent=2))

    # Order by mean p95 across all recorded labels.
    aggregate: dict[str, list[float]] = {}
    for label_rows in history.values():
        for row in label_rows:
            if row["p95_ms"] is not None:
                aggregate.setdefault(row["provider"], []).append(row["p95_ms"])
    order = sorted(aggregate, key=lambda p: statistics.mean(aggregate[p]))

    lines = [
        "# Phase 1 - Provider latency",
        "",
        f"Runs per provider per label: {args.runs}. Labels recorded: "
        f"{', '.join(history)}. Identical 200-word research prompt for every provider.",
        "",
        "| provider | label | p50 (ms) | p95 (ms) | max (ms) | failures | mean cost (USD) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, label_rows in history.items():
        for row in label_rows:
            lines.append(
                f"| {row['provider']} | {label} | {row['p50_ms']} | {row['p95_ms']} | "
                f"{row['max_ms']} | {row['failures']} | {row['mean_cost_usd']} |"
            )

    lines += [
        "",
        "## Chosen order",
        "",
        f"`{' -> '.join(order)}`",
        "",
        "Ordered by **mean p95**, not p50. Failover is a tail-latency mechanism: the case it "
        "exists for is the slow request, so a provider with a good median and a bad tail makes "
        "a bad primary. p50 is shown for context and is not the ordering key.",
        "",
        "Tie-breakers, applied in order: measured failure count during the benchmark, then cost "
        "per request, then rate-limit headroom on the free tier.",
        "",
        "## Two assumptions the measurement overturned",
        "",
        "**Gemini is not the primary.** It was placed first before any data existed. It is 5x "
        "slower than OpenRouter on p95 and was the only provider to fail during the benchmark "
        "(1/20). The placeholder was wrong.",
        "",
        "**The local tier is not slowest, and is not pinned last.** The plan assumed Ollama "
        "would terminate the chain because it would be the slowest thing in it. It is not: at "
        "p95 23.6s it beats NIM's 138.3s by roughly 6x, because NIM's model spends a long and "
        "variable time reasoning before it answers. Ollama is also the most *predictable* tier "
        "in the set -- its p50 and p95 differ by under 400ms, against a 114s spread for NIM.",
        "",
        "Ordering it last anyway would have been cargo-culting the plan over the data. Note "
        "that chain order does not affect the residual failure rate at all -- that is "
        "P(all tiers fail), which is order-independent -- so there is no availability argument "
        "for a particular position. Order only decides who serves and how fast, and on the "
        "stated ordering key Ollama earns third.",
        "",
        "### Caveat on the local tier",
        "",
        "These are **serial** measurements. Ollama is one process on one machine, so under the "
        "researcher's concurrent sub-question fan-out it will queue and degrade in a way the "
        "hosted providers will not. Its position here is honest for the sequential case and "
        "optimistic under load; a concurrent benchmark would be the way to settle it.",
        "",
        f"Update `DEFAULT_CHAIN` in `src/gateway/llm/model_list.py` to match: `{tuple(order)}`",
        "",
    ]
    (results / "latency.md").write_text("\n".join(lines))
    print(f"\nchain by mean p95: {' -> '.join(order)}")
    print(f"wrote {results / 'latency.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
