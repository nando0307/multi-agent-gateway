# Phase 6 - Judge agreement with an independent rater

`n = 18` reports. Judge under test: `nvidia_nim/deepseek-ai/deepseek-v4-flash`, temperature 0. Reference rater: **Nando (project author)** (human).

The reference rater scored the reports without seeing the judge's scores. Had they been visible, this would measure compliance rather than agreement.

> **Label provenance.** Scored by the project author, but **not blind**: the sheet already contained a prior rater's (claude-opus-5) scores and these labels were produced by reviewing and adjusting them, changing 3 of 36 cells. Anchoring inflates agreement, so read this as a human review of a model's labels rather than independent human labelling. A clean pass would re-score from `results/label_sheet.md`, which deliberately carries no scores at all.

> **Reproducibility, measured over 5 passes of the identical reports.** The judge is nominally deterministic at temperature 0 and is not: **4 of 36** individual scores differed between the first and last pass. Spearman rho came out **0.846 +/- 0.035** on depth (range 0.799-0.895) and **0.79 +/- 0.037** on coherence (range 0.748-0.839). Quote the mean with its spread, not a single run's third decimal. Phase 6's *"scores reproduce across two runs"* acceptance criterion is **not met**; this block is what replaces it -- the instrument has error bars and they are stated.

| dimension | Spearman rho | within +/-1 | mean human | mean judge | judge bias |
|---|---|---|---|---|---|
| depth | **0.839** | 94.4% | 2.61 | 2.83 | +0.22 |
| coherence | **0.839** | 88.9% | 2.78 | 3.39 | +0.61 |

## Verdict: PASS

The threshold is Spearman rho >= 0.6 on both dimensions. Both clear it, so the provider matrix is measuring something real.

`judge bias` is the mean judge score minus the mean human score. A consistent offset does not hurt provider *comparison* (it cancels), but it does move the absolute gate threshold, so the 0.70 cutoff should be read against it.

## Per-report

Per-report scores below are from the final pass of 5; other passes differ on 4 cell(s).

| id | question | provider | human depth | judge depth | human coh | judge coh |
|---|---|---|---|---|---|---|
| L01 | q01 | openrouter | 3 | 4 | 3 | 3 |
| L02 | q02 | gemini | 2 | 4 | 4 | 4 |
| L03 | q03 | ollama | 2 | 2 | 2 | 2 |
| L04 | q04 | nim | 4 | 4 | 3 | 5 |
| L05 | q05 | openrouter | 3 | 4 | 3 | 4 |
| L06 | q06 | gemini | 1 | 1 | 4 | 5 |
| L07 | q07 | ollama | 3 | 2 | 4 | 3 |
| L08 | q08 | nim | 3 | 4 | 2 | 4 |
| L09 | q09 | openrouter | 3 | 3 | 2 | 3 |
| L10 | q10 | gemini | 4 | 4 | 4 | 5 |
| L11 | q11 | ollama | 1 | 1 | 1 | 1 |
| L12 | q12 | nim | 2 | 2 | 2 | 2 |
| L13 | q13 | openrouter | 3 | 4 | 4 | 5 |
| L14 | q15 | ollama | 1 | 1 | 1 | 1 |
| L15 | q16 | nim | 4 | 4 | 4 | 5 |
| L16 | q17 | openrouter | 2 | 2 | 2 | 3 |
| L17 | q19 | ollama | 1 | 1 | 1 | 1 |
| L18 | q20 | nim | 5 | 4 | 4 | 5 |
