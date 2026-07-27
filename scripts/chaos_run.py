"""Phase 3: the X% -> Y% measurement.

Two arms, identical seeded faults, identical requests:

  Arm A (baseline)  single provider + retries, no failover   -> X%
  Arm B (gateway)   full chain + retries + cooldowns         -> Y%

Two deliberate choices keep this from being a strawman:

* **Arm A retries.** Comparing failover against a baseline with *no* retries would
  inflate the delta for free. The baseline is what a competent single-provider
  implementation looks like.
* **Sticky faults by default.** Real provider failures are correlated in time -- a 503
  during a deploy, a rate limit that persists for a window. Modelling faults as
  independent per attempt would make retries look far more effective than they are, and
  would understate the case for multi-provider failover. ``--fault-model independent``
  is available for comparison.

Latency here is not wall-clock (the transport is mocked); the cost of failover is
reported as *added attempts* and mean fallback depth, which combine with the real p95
figures from Phase 1 to estimate added latency.

Usage:
    python scripts/chaos_run.py --n 200 --p 0.1 0.3 0.5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gateway.llm.model_list import DEFAULT_CHAIN, PRIMARY_ALIAS  # noqa: E402
from gateway.llm.router import AllProvidersExhausted, Gateway  # noqa: E402
from tests.conftest import FAIL_429, FAIL_500, make_settings  # noqa: E402

# Imported, never hardcoded: a chaos number that describes a chain the system does not
# actually ship is worse than no number. Phase 1 reordered this and the study must follow.
CHAIN = DEFAULT_CHAIN
LOCAL_TIER = "ollama"
QUESTION = [{"role": "user", "content": "Summarise recent progress in grid-scale storage."}]

# Retry backoff is real time spent sleeping. Against a mocked transport it adds hours of
# wall clock to a simulation whose latency figures are explicitly *not* wall clock (see the
# report footer). Retry and fallback *counts* -- the things being measured -- are unchanged.
import litellm.router as _litellm_router  # noqa: E402

_litellm_router.Router._time_to_sleep_before_retry = lambda *a, **k: 0.0


# --------------------------------------------------------------------------- faults
def sticky_schedule(n: int, p: float, mean_len: float, rng: random.Random) -> list[bool]:
    """Markov on/off schedule with marginal down-probability ~p and mean outage ~mean_len."""
    if p <= 0:
        return [False] * n
    p_recover = 1.0 / mean_len
    p_fail = (p * p_recover) / max(1e-9, (1.0 - p))
    down, out = False, []
    for _ in range(n):
        down = (rng.random() > p_recover) if down else (rng.random() < p_fail)
        out.append(down)
    return out


def independent_schedule(n: int, p: float, rng: random.Random) -> list[bool]:
    return [rng.random() < p for _ in range(n)]


def build_schedules(n, p, model, rng, local_fail: float) -> dict[str, list[bool]]:
    """Ollama is local: it cannot rate-limit, but it is not infallible.

    Giving the terminal tier a 0% failure rate would be a modelling gift that drives the
    residual rate to exactly zero and makes the headline number unfalsifiable. Local
    inference fails for its own reasons -- OOM, model not pulled, server not running -- so
    it gets a small independent failure rate instead. ``--local-fail 0`` reproduces the
    optimistic variant.
    """
    sched = {}
    for name in CHAIN:
        if name == LOCAL_TIER:
            sched[name] = independent_schedule(n, local_fail, rng)
        elif model == "sticky":
            sched[name] = sticky_schedule(n, p, mean_len=8.0, rng=rng)
        else:
            sched[name] = independent_schedule(n, p, rng)
    return sched


# --------------------------------------------------------------------------- gateway
def make_arm(chain: tuple[str, ...], *, fallbacks: bool) -> Gateway:
    settings = make_settings(num_retries=2, allowed_fails=3, cooldown_time_s=30)

    def params(name: str) -> dict:
        return {"model": "openai/gpt-4o-mini", "api_key": "sk-test", "mock_response": "ok"}

    entries = [
        {
            "model_name": name,
            "litellm_params": params(name),
            "model_info": {"id": name, "provider": name, "chain_index": i},
        }
        for i, name in enumerate(chain)
    ]
    entries.append(
        {
            "model_name": PRIMARY_ALIAS,
            "litellm_params": params(chain[0]),
            "model_info": {"id": f"primary::{chain[0]}", "provider": chain[0], "chain_index": 0},
        }
    )
    gw = Gateway(settings=settings, model_list=entries)
    if not fallbacks:
        gw.router.fallbacks = []
    return gw


def apply_faults(gw: Gateway, down: dict[str, bool], rng: random.Random) -> None:
    """Mutate each deployment's mock behaviour in place, preserving Router cooldown state."""
    for entry in gw.router.model_list:
        provider = entry["model_info"]["provider"]
        if down.get(provider):
            entry["litellm_params"]["mock_response"] = rng.choice([FAIL_500, FAIL_429])
        else:
            entry["litellm_params"]["mock_response"] = f"answer from {provider}"


# --------------------------------------------------------------------------- stats
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, (centre - half) * 100), min(100.0, (centre + half) * 100))


@dataclass
class ArmResult:
    name: str
    n: int
    failures: int = 0
    depths: list[int] = field(default_factory=list)
    served: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return 100.0 * self.failures / self.n

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.failures, self.n)

    @property
    def mean_depth(self) -> float:
        return statistics.mean(self.depths) if self.depths else 0.0


def run_arm(name: str, gw: Gateway, schedules, n: int, rng: random.Random) -> ArmResult:
    res = ArmResult(name=name, n=n)
    for i in range(n):
        apply_faults(gw, {p: s[i] for p, s in schedules.items()}, rng)
        try:
            out = gw.complete(QUESTION, model=PRIMARY_ALIAS)
        except (AllProvidersExhausted, Exception):
            res.failures += 1
            continue
        res.depths.append(out.fallback_depth)
        res.served[out.served_by or "?"] = res.served.get(out.served_by or "?", 0) + 1
    return res


# --------------------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--p", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fault-model", choices=["sticky", "independent"], default="sticky")
    ap.add_argument("--local-fail", type=float, default=0.02,
                    help="independent failure rate for the local terminal tier")
    args = ap.parse_args()

    rows, raw = [], []
    for p in args.p:
        schedules = build_schedules(
            args.n, p, args.fault_model, random.Random(args.seed), args.local_fail
        )
        base = run_arm("baseline", make_arm((CHAIN[0],), fallbacks=False), schedules, args.n,
                       random.Random(args.seed + 1))
        gwy = run_arm("gateway", make_arm(CHAIN, fallbacks=True), schedules, args.n,
                      random.Random(args.seed + 1))
        rows.append((p, base, gwy))
        raw.append(
            {
                "p": p,
                "n": args.n,
                "fault_model": args.fault_model,
                "local_fail": args.local_fail,
                "baseline_failure_pct": round(base.failure_rate, 2),
                "baseline_ci": [round(v, 2) for v in base.ci],
                "gateway_failure_pct": round(gwy.failure_rate, 2),
                "gateway_ci": [round(v, 2) for v in gwy.ci],
                "gateway_mean_fallback_depth": round(gwy.mean_depth, 3),
                "gateway_provider_share": gwy.served,
            }
        )
        print(
            f"p={p:<5} baseline {base.failure_rate:6.2f}%  ->  gateway {gwy.failure_rate:6.2f}%"
            f"   (mean depth {gwy.mean_depth:.2f})"
        )

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "chaos_raw.json").write_text(json.dumps(raw, indent=2))

    headline = raw[len(raw) // 2]
    lines = [
        "# Phase 3 - Forced-failure study",
        "",
        f"`n = {args.n}` requests per arm per fault rate · fault model = **{args.fault_model}** "
        f"· seed = {args.seed} · chain = {' -> '.join(CHAIN)}",
        "",
        f"The local terminal tier is given an independent **{args.local_fail:.0%}** failure rate "
        "(OOM, model not loaded, server down). Setting it to zero would drive the residual rate "
        "to exactly zero and make the headline unfalsifiable.",
        "",
        "Both arms see **identical** seeded faults and identical requests. The baseline is a "
        "single provider *with* retries (`num_retries=2`), not a strawman without them.",
        "",
        "## Headline",
        "",
        f"At a **{headline['p']:.0%} provider failure rate**, request failures fell from "
        f"**{headline['baseline_failure_pct']:.1f}%** "
        f"(95% CI {headline['baseline_ci'][0]:.1f}-{headline['baseline_ci'][1]:.1f}) to "
        f"**{headline['gateway_failure_pct']:.1f}%** "
        f"(95% CI {headline['gateway_ci'][0]:.1f}-{headline['gateway_ci'][1]:.1f}).",
        "",
        "## Sweep",
        "",
        "| provider failure rate | baseline failures | 95% CI | gateway failures | 95% CI | mean fallback depth |",
        "|---|---|---|---|---|---|",
    ]
    for r in raw:
        lines.append(
            f"| {r['p']:.0%} | {r['baseline_failure_pct']:.1f}% | "
            f"{r['baseline_ci'][0]:.1f}-{r['baseline_ci'][1]:.1f} | "
            f"{r['gateway_failure_pct']:.1f}% | "
            f"{r['gateway_ci'][0]:.1f}-{r['gateway_ci'][1]:.1f} | "
            f"{r['gateway_mean_fallback_depth']:.2f} |"
        )
    lines += [
        "",
        "## Which tier served",
        "",
        "| provider failure rate | " + " | ".join(CHAIN) + " |",
        "|---" * (len(CHAIN) + 1) + "|",
    ]
    for r in raw:
        share = r["gateway_provider_share"]
        lines.append(
            f"| {r['p']:.0%} | " + " | ".join(str(share.get(c, 0)) for c in CHAIN) + " |"
        )
    lines += [
        "",
        "## What this does and does not show",
        "",
        "* Faults are injected at the transport layer against real `litellm.Router` fallback "
        "logic -- the routing, retry and cooldown machinery under test is the production code "
        "path. Only the network is simulated.",
        "* Latency here is not wall-clock. The cost of failover is reported as mean fallback "
        "depth; multiply by the per-provider p95 in `results/latency.md` for the real added "
        "latency.",
        "* Provider outages are modelled as independent across providers. Correlated failure "
        "(a shared upstream) is mitigated by choosing an OpenRouter model from a different "
        "vendor than Gemini/NIM, not by this experiment.",
        f"* The residual {headline['gateway_failure_pct']:.1f}% is dominated by the assumed "
        f"{args.local_fail:.0%} local-tier failure rate. That assumption is doing real work in "
        "these numbers; it is stated rather than hidden, and `--local-fail 0` shows the "
        "optimistic variant.",
        "",
    ]
    (results / "chaos_report.md").write_text("\n".join(lines))
    print(f"\nwrote {results / 'chaos_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
