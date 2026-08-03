# Phase 7 - Provider quality matrix

30 questions x 4 providers, each run **pinned** to one provider with failover disabled, so every row measures that provider alone. Gate threshold **0.7**. Judge: `nvidia_nim/deepseek-ai/deepseek-v4-flash`, temperature 0, a different model from every routed one.

> **Independence caveat.** The judge (`nvidia_nim/deepseek-ai/deepseek-v4-flash`) is a different model from every routed one, so there is no self-preference bias, but it is reached through the same `nim` account that also serves requests. A wholly separate vendor would be a stronger guarantee.

All providers saw **identical evidence**: the search cache is shared across the matrix, so this compares models, not search luck.

| provider | runs | succeeded | mean composite | citation | depth | coherence | pass rate |
|---|---|---|---|---|---|---|---|
| openrouter | 30 | 30 | 0.688 | 0.849 | 0.427 | 0.733 | 50.0% |
| gemini | 30 | 30 | 0.758 | 0.877 | 0.489 | 0.867 | 73.3% |
| groq | 30 | 30 | 0.66 | 0.836 | 0.436 | 0.65 | 50.0% |
| nim | 30 | 30 | 0.707 | 0.797 | 0.469 | 0.825 | 60.0% |

## Silent quality regressions

**11** caught across 15 comparable question(s).

> A run where the request **succeeded** -- no error, no exception, within the latency > SLO, green on every ops dashboard -- but whose composite score fell below the > 0.7 gate, on a question the primary (`openrouter`) passed.

This is the failure mode that failover introduces and that monitoring cannot see: the request works, the answer is worse. Nothing in an error rate, a latency histogram or an uptime check registers it.

| question | fallback provider | primary score | fallback score | delta |
|---|---|---|---|---|
| q12 | nim | 0.74 | 0.52 | -0.23 |
| q12 | groq | 0.74 | 0.54 | -0.20 |
| q29 | nim | 0.73 | 0.54 | -0.19 |
| q12 | gemini | 0.74 | 0.56 | -0.18 |
| q29 | gemini | 0.73 | 0.55 | -0.17 |
| q25 | groq | 0.75 | 0.61 | -0.14 |
| q06 | groq | 0.77 | 0.63 | -0.14 |
| q16 | nim | 0.78 | 0.67 | -0.12 |
| q22 | nim | 0.73 | 0.63 | -0.10 |
| q16 | groq | 0.78 | 0.69 | -0.09 |
| q29 | groq | 0.73 | 0.65 | -0.08 |

## Limitations

* n = 30 questions. Per-provider means carry real uncertainty; treat small gaps between providers as noise, not ranking.
* A single judge model scores every report. `results/judge_agreement.md` records how well it tracks hand labels -- read that before trusting any number here.
* Citation sub-scores are deterministic and do not depend on the judge at all.
