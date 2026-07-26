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
