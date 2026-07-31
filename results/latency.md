# Phase 1 - Provider latency

Runs per provider per label: 20. Labels recorded: morning. Identical 200-word research prompt for every provider.

| provider | label | p50 (ms) | p95 (ms) | max (ms) | failures | mean cost (USD) |
|---|---|---|---|---|---|---|
| gemini | morning | 9287.3 | 23095.8 | 24897.5 | 1 | 0.000438 |
| nim | morning | 24116.1 | 138277.7 | 141763.5 | 0 | None |
| openrouter | morning | 3320.6 | 4588.9 | 35764.6 | 0 | 0.000109 |

## Chosen order

`openrouter -> gemini -> nim`

Ordered by **mean p95**, not p50. Failover is a tail-latency mechanism: the case it exists for is the slow request, so a provider with a good median and a bad tail makes a bad primary. p50 is shown for context and is not the ordering key.

Tie-breakers, applied in order: measured failure count during the benchmark, then cost per request, then rate-limit headroom on the free tier.

## Two assumptions the measurement overturned

**Gemini is not the primary.** It was placed first before any data existed. It is 5x slower than OpenRouter on p95 and was the only provider to fail during the benchmark (1/20). The placeholder was wrong.

**The local tier is not slowest, and is not pinned last** was true of the original plan, which assumed Ollama would terminate the chain. It does not apply to the shipped system: Ollama was dropped from `DEFAULT_CHAIN` on 2026-07-28 (see PLAN.md D3 — a single local synthesis call ran ~214s, making the Phase 7 matrix impractical) and replaced by Groq. This section is kept for the historical record but no longer describes production; see the note below.

## Groq's position is unmeasured — flagged, not fixed

The shipped chain is `openrouter -> gemini -> groq -> nim` (`src/gateway/llm/model_list.py`), but
**Groq has never been cleanly benchmarked**. It was substituted for Ollama on practical grounds
(speed of iteration), not because a measurement put it ahead of NIM. A benchmark attempt on
2026-07-30 failed outright: Groq's own **daily token cap** (100,000 TPD on the free tier) was at
99,977/100,000 from cumulative testing that day, so 19/20 calls returned `RateLimitError` before
producing a single real latency sample — that run's data was discarded rather than reported, since
p50=p95=1530ms off one lucky call is not a measurement.

**Consequence:** the chain order past Gemini is not fully data-driven. OpenRouter-first and
Gemini-second are genuinely measured and hold. Groq-third-ahead-of-NIM is a placeholder carried
over from Ollama's old slot, unverified. Re-run `python scripts/bench_latency.py --runs 20 --label
morning --only groq` once Groq's daily quota resets, on a day with no concurrent load on the same
key (garak or matrix runs saturate it), before citing this chain order as measured in an interview.

Update `DEFAULT_CHAIN` in `src/gateway/llm/model_list.py` to match once Groq is measured: currently
`('openrouter', 'gemini', 'groq', 'nim')` is unverified for the `groq` position.
