# Phase 7 - Provider quality matrix

30 questions x 4 providers, each run **pinned** to one provider with failover disabled, so every row measures that provider alone. Gate threshold **0.7**. Judge: `openrouter/openai/gpt-oss-120b`, temperature 0, a different model from every routed one.

> **Independence caveat.** The judge (`openrouter/openai/gpt-oss-120b`) is a different model from every routed one, so there is no self-preference bias, but it is reached through the same `openrouter` account that also serves requests. A wholly separate vendor would be a stronger guarantee.

All providers saw **identical evidence**: the search cache is shared across the matrix, so this compares models, not search luck.

| provider | runs | succeeded | mean composite | citation | depth | coherence | pass rate |
|---|---|---|---|---|---|---|---|
| openrouter | 30 | 30 | 0.73 | 0.892 | 0.419 | 0.825 | 73.3% |
| gemini | 30 | 30 | 0.782 | 0.904 | 0.453 | 0.95 | 90.0% |
| groq | 30 | 19 | 0.703 | 0.854 | 0.456 | 0.75 | 57.9% |
| nim | 30 | 30 | 0.778 | 0.814 | 0.54 | 0.967 | 86.7% |

## Silent quality regressions

**11** caught.

> A run where the request **succeeded** -- no error, no exception, within the latency > SLO, green on every ops dashboard -- but whose composite score fell below the > 0.7 gate, on a question the primary (`openrouter`) passed.

This is the failure mode that failover introduces and that monitoring cannot see: the request works, the answer is worse. Nothing in an error rate, a latency histogram or an uptime check registers it.

| question | fallback provider | primary score | fallback score | delta |
|---|---|---|---|---|
| q16 | nim | 0.77 | 0.51 | -0.25 |
| q12 | groq | 0.81 | 0.60 | -0.21 |
| q01 | groq | 0.82 | 0.64 | -0.18 |
| q26 | nim | 0.74 | 0.57 | -0.17 |
| q11 | groq | 0.71 | 0.54 | -0.17 |
| q16 | groq | 0.77 | 0.67 | -0.10 |
| q29 | gemini | 0.77 | 0.69 | -0.08 |
| q11 | gemini | 0.71 | 0.63 | -0.08 |
| q29 | groq | 0.77 | 0.69 | -0.08 |
| q16 | gemini | 0.77 | 0.70 | -0.07 |
| q30 | nim | 0.70 | 0.67 | -0.03 |

## Limitations

* n = 30 questions. Per-provider means carry real uncertainty; treat small gaps between providers as noise, not ranking.
* A single judge model scores every report. `results/judge_agreement.md` records how well it tracks hand labels -- read that before trusting any number here.
* Citation sub-scores are deterministic and do not depend on the judge at all.
