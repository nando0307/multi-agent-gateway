# Phase 1 - Provider latency

Runs per provider per label: 20. Labels recorded: morning. Identical 200-word research prompt for every provider.

| provider | label | p50 (ms) | p95 (ms) | max (ms) | failures | mean cost (USD) |
|---|---|---|---|---|---|---|
| gemini | morning | 9287.3 | 23095.8 | 24897.5 | 1 | 0.000438 |
| nim | morning | 24116.1 | 138277.7 | 141763.5 | 0 | None |
| openrouter | morning | 3320.6 | 4588.9 | 35764.6 | 0 | 0.000109 |
| ollama | morning | 23209.3 | 23584.3 | 31894.2 | 0 | 0.0 |

## Chosen order

`openrouter -> gemini -> ollama -> nim`

Ordered by **mean p95**, not p50. Failover is a tail-latency mechanism: the case it exists for is the slow request, so a provider with a good median and a bad tail makes a bad primary. p50 is shown for context and is not the ordering key.

Tie-breakers, applied in order: measured failure count during the benchmark, then cost per request, then rate-limit headroom on the free tier.

## Two assumptions the measurement overturned

**Gemini is not the primary.** It was placed first before any data existed. It is 5x slower than OpenRouter on p95 and was the only provider to fail during the benchmark (1/20). The placeholder was wrong.

**The local tier is not slowest, and is not pinned last.** The plan assumed Ollama would terminate the chain because it would be the slowest thing in it. It is not: at p95 23.6s it beats NIM's 138.3s by roughly 6x, because NIM's model spends a long and variable time reasoning before it answers. Ollama is also the most *predictable* tier in the set -- its p50 and p95 differ by under 400ms, against a 114s spread for NIM.

Ordering it last anyway would have been cargo-culting the plan over the data. Note that chain order does not affect the residual failure rate at all -- that is P(all tiers fail), which is order-independent -- so there is no availability argument for a particular position. Order only decides who serves and how fast, and on the stated ordering key Ollama earns third.

### Caveat on the local tier

These are **serial** measurements. Ollama is one process on one machine, so under the researcher's concurrent sub-question fan-out it will queue and degrade in a way the hosted providers will not. Its position here is honest for the sequential case and optimistic under load; a concurrent benchmark would be the way to settle it.

Update `DEFAULT_CHAIN` in `src/gateway/llm/model_list.py` to match: `('openrouter', 'gemini', 'ollama', 'nim')`
