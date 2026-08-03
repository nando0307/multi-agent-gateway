# Phase 6 - Judge agreement with an independent rater

`n = 18` reports. Judge under test: `nvidia_nim/deepseek-ai/deepseek-v4-flash`, temperature 0. Reference rater: **claude-opus-5** (model).

The reference rater scored the reports without seeing the judge's scores. Had they been visible, this would measure compliance rather than agreement.

> **This is inter-model agreement, not human validation.** The reference rater is a model (`claude-opus-5`), not a person. A strong reference model is a real check -- it will catch a judge that is simply wrong -- but two language models can agree because they share a blind spot, and no amount of agreement between them detects that. The Phase 6 human gate in PLAN.md remains **open**; this does not close it.

What it does establish: the judge is not idiosyncratic relative to a stronger model, so the provider matrix is not being ordered by one small model's private tastes.

> **Reproducibility, measured over 5 passes of the identical reports.** The judge is nominally deterministic at temperature 0 and is not: **3 of 36** individual scores differed between the first and last pass. Spearman rho came out **0.859 +/- 0.026** on depth (range 0.814-0.883) and **0.743 +/- 0.011** on coherence (range 0.734-0.761). Quote the mean with its spread, not a single run's third decimal. Phase 6's *"scores reproduce across two runs"* acceptance criterion is **not met**; this block is what replaces it -- the instrument has error bars and they are stated.

| dimension | Spearman rho | within +/-1 | mean human | mean judge | judge bias |
|---|---|---|---|---|---|
| depth | **0.866** | 100.0% | 2.56 | 2.83 | +0.28 |
| coherence | **0.74** | 83.3% | 2.78 | 3.33 | +0.56 |

## Verdict: PASS (against a model rater -- human gate still open)

The threshold is Spearman rho >= 0.6 on both dimensions. Both clear it, so the provider matrix is measuring something real.

`judge bias` is the mean judge score minus the mean human score. A consistent offset does not hurt provider *comparison* (it cancels), but it does move the absolute gate threshold, so the 0.70 cutoff should be read against it.

## Per-report

Per-report scores below are from the final pass of 5; other passes differ on 3 cell(s).

| id | question | provider | model depth | judge depth | model coh | judge coh |
|---|---|---|---|---|---|---|
| L01 | q01 | openrouter | 3 | 4 | 3 | 3 |
| L02 | q02 | gemini | 2 | 3 | 4 | 4 |
| L03 | q03 | ollama | 1 | 2 | 2 | 2 |
| L04 | q04 | nim | 4 | 4 | 3 | 4 |
| L05 | q05 | openrouter | 3 | 4 | 3 | 4 |
| L06 | q06 | gemini | 1 | 1 | 4 | 5 |
| L07 | q07 | ollama | 3 | 2 | 4 | 2 |
| L08 | q08 | nim | 3 | 4 | 2 | 4 |
| L09 | q09 | openrouter | 3 | 3 | 3 | 3 |
| L10 | q10 | gemini | 4 | 4 | 4 | 5 |
| L11 | q11 | ollama | 1 | 1 | 1 | 2 |
| L12 | q12 | nim | 2 | 2 | 2 | 2 |
| L13 | q13 | openrouter | 3 | 4 | 3 | 5 |
| L14 | q15 | ollama | 1 | 2 | 1 | 1 |
| L15 | q16 | nim | 4 | 4 | 4 | 5 |
| L16 | q17 | openrouter | 2 | 2 | 2 | 3 |
| L17 | q19 | ollama | 1 | 1 | 1 | 1 |
| L18 | q20 | nim | 5 | 4 | 4 | 5 |
