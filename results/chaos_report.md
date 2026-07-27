# Phase 3 - Forced-failure study

`n = 1000` requests per arm per fault rate · fault model = **sticky** · seed = 7 · chain = openrouter -> gemini -> ollama -> nim

The local terminal tier is given an independent **2%** failure rate (OOM, model not loaded, server down). Setting it to zero would drive the residual rate to exactly zero and make the headline unfalsifiable.

Both arms see **identical** seeded faults and identical requests. The baseline is a single provider *with* retries (`num_retries=2`), not a strawman without them.

## Headline

At a **30% provider failure rate**, request failures fell from **31.2%** (95% CI 28.4-34.1) to **0.1%** (95% CI 0.0-0.6).

## Sweep

| provider failure rate | baseline failures | 95% CI | gateway failures | 95% CI | mean fallback depth |
|---|---|---|---|---|---|
| 10% | 8.6% | 7.0-10.5 | 0.0% | 0.0-0.4 | 0.10 |
| 30% | 31.2% | 28.4-34.1 | 0.1% | 0.0-0.6 | 0.39 |
| 50% | 52.4% | 49.3-55.5 | 0.1% | 0.0-0.6 | 0.78 |

## Which tier served

| provider failure rate | openrouter | gemini | ollama | nim |
|---|---|---|---|---|
| 10% | 914 | 73 | 13 | 0 |
| 30% | 688 | 235 | 76 | 0 |
| 50% | 476 | 267 | 253 | 3 |

## What this does and does not show

* Faults are injected at the transport layer against real `litellm.Router` fallback logic -- the routing, retry and cooldown machinery under test is the production code path. Only the network is simulated.
* Latency here is not wall-clock. The cost of failover is reported as mean fallback depth; multiply by the per-provider p95 in `results/latency.md` for the real added latency.
* Provider outages are modelled as independent across providers. Correlated failure (a shared upstream) is mitigated by choosing an OpenRouter model from a different vendor than Gemini/NIM, not by this experiment.
* The residual 0.1% is dominated by the assumed 2% local-tier failure rate. That assumption is doing real work in these numbers; it is stated rather than hidden, and `--local-fail 0` shows the optimistic variant.
