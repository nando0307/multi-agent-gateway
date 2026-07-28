# Phase 3 - Forced-failure study

`n = 1000` requests per arm per fault rate · fault model = **sticky** · seed = 7 · chain = openrouter -> gemini -> groq -> nim

Every tier is cloud-hosted and gets the same sticky fault model. There is no longer a local tier immune to rate limiting -- Groq replaced Ollama for speed -- so the residual rate here is genuinely bounded by all four providers failing together, not by an assumption about a backstop.

Both arms see **identical** seeded faults and identical requests. The baseline is a single provider *with* retries (`num_retries=2`), not a strawman without them.

## Headline

At a **30% provider failure rate**, request failures fell from **31.2%** (95% CI 28.4-34.1) to **0.2%** (95% CI 0.1-0.7).

## Sweep

| provider failure rate | baseline failures | 95% CI | gateway failures | 95% CI | mean fallback depth |
|---|---|---|---|---|---|
| 10% | 8.6% | 7.0-10.5 | 0.0% | 0.0-0.4 | 0.10 |
| 30% | 31.2% | 28.4-34.1 | 0.2% | 0.1-0.7 | 0.41 |
| 50% | 52.4% | 49.3-55.5 | 6.6% | 5.2-8.3 | 0.74 |

## Which tier served

| provider failure rate | openrouter | gemini | groq | nim |
|---|---|---|---|---|
| 10% | 914 | 73 | 12 | 1 |
| 30% | 688 | 235 | 54 | 21 |
| 50% | 476 | 267 | 150 | 41 |

## What this does and does not show

* Faults are injected at the transport layer against real `litellm.Router` fallback logic -- the routing, retry and cooldown machinery under test is the production code path. Only the network is simulated.
* Latency here is not wall-clock. The cost of failover is reported as mean fallback depth; multiply by the per-provider p95 in `results/latency.md` for the real added latency.
* Provider outages are modelled as independent across providers. Correlated failure (a shared upstream) is mitigated by choosing an OpenRouter model from a different vendor than Gemini/NIM, not by this experiment.
* All four tiers are cloud providers, so a correlated outage (shared upstream, shared region) would take more of the chain than this independent-failure model assumes. The previous version of this study had a local tier that could not rate-limit, which made the residual rate structurally lower; that is gone.
