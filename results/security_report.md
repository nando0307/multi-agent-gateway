# Phase 8 - Prompt-injection defence

Attack corpus: **41** injections embedded in fetched page content. Benign corpus: **40** legitimate passages, several deliberately written to look suspicious (a security advisory describing prompt injection, a page saying "ignore the previous version of this document", documentation mentioning `api_key`).

## Headline

| metric | value |
|---|---|
| attacks blocked | **100.0%** (41/41) |
| false positives on benign content | **0.0%** (0/40) |

Both numbers or neither: a filter that downgrades every document blocks 100% of attacks and destroys the system's usefulness.

## By category

| category | n | blocked | by scope gate | by sanitiser |
|---|---|---|---|---|
| budget_exhaustion | 3 | 3 | 2 | 2 |
| citation_poisoning | 4 | 4 | 0 | 4 |
| credential_exfiltration | 6 | 6 | 0 | 6 |
| encoding | 4 | 4 | 0 | 4 |
| instruction_override | 5 | 5 | 0 | 5 |
| multi_turn_drift | 2 | 2 | 0 | 2 |
| role_reassignment | 4 | 4 | 0 | 4 |
| system_prompt_extraction | 4 | 4 | 0 | 4 |
| tool_abuse | 9 | 9 | 8 | 7 |

## Which layer caught what

The two layers are not redundant. The scope gate stops every attack that needs a tool call to succeed -- SSRF, metadata endpoints, `file://`, budget exhaustion -- and it does so without consulting a model, so a persuasive injection cannot argue its way past it. The sanitiser covers attacks that never touch a tool: instruction override, role reassignment, system-prompt extraction, citation poisoning.

## Scope

* These numbers cover the deterministic layers only. Model-level susceptibility is measured separately by `scripts/run_garak.sh`, reported below rather than blended in.
* The scope gate never asks a model for permission, so its results are not sensitive to prompt wording, model version, or temperature.
* The sanitiser is defence in depth. It is pattern-based and therefore evadable by a novel phrasing; that is why the control that actually prevents tool abuse is the scope gate, not this.

## Model-level susceptibility (garak) -- partial

`scripts/run_garak.sh` points garak's `rest` generator at the live `POST /research` endpoint,
so probes traverse the full pipeline (scope gate, sanitiser, synthesis) rather than a bare
model. This is a **different attack surface** from the corpus above: these prompts arrive as
the top-level user question, not as fetched page content, so the scope/sanitiser layers are
not what is being tested here -- this measures whether the underlying model itself resists a
jailbreak when asked directly.

Only `dan.DanInTheWild` (real-world jailbreak prompts) has completed; `encoding.InjectBase64`
and `promptinject.HijackHateHumans` were interrupted mid-run and are **not** reported here.
There is also no raw-provider baseline yet (`run_garak.sh`'s second half, run manually) --
without it this is a susceptibility measurement, not yet hardening evidence with a before/after
delta. Both are follow-up work.

| metric | value |
|---|---|
| prompts processed | 256 / 256 |
| scorable by garak's detector | 34 (222 blocked before reaching a model -- see below) |
| mitigation bypassed (jailbreak succeeded) | **22 / 34 (64.7%)**, 95% CI [47.1%, 79.4%] |
| mitigation held | 12 / 34 |

Detector: `mitigation.MitigationBypass`. The 222 unscored prompts are not a gap in
measurement, mostly the opposite: `POST /research` caps the `question` field at 1000
characters (a real input-validation boundary — see PLAN.md Phase 8), and a large share of
`DanInTheWild`'s prompts exceed it, so they were rejected with a 422 before any model saw
them. Garak has no way to distinguish "blocked by design" from "no response for another
reason" in this count, so the 222 should be read as "not evaluated," not "safe."

22 concrete jailbreak transcripts are preserved in
`results/garak/dan_in_the_wild_hitlog_backup.jsonl` for review (copied out before a
subsequent garak run overwrote garak's own report directory -- it opens its report file in
`"w"` mode, so a fresh `--probes` run truncates the prior one's raw output; only the
interpreted numbers above and this backup survive across runs).

**Read this cautiously:** n=34 is a small scored sample, one probe, one run. It says the
underlying model *can* be jailbroken via direct prompting a majority of the time when asked
outright -- consistent with published jailbreak-resistance results for open/mid-tier models
generally, and a reason to treat the scope/sanitiser layer (which blocks the *tool-abuse*
consequences of a jailbreak, not the jailbreak itself) as the actual line of defence rather
than assuming the model won't comply.
