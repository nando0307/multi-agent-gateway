# multi-agent-gateway

A multi-agent web-research system with provider failover, a quality gate on every
response, and a deterministic prompt-injection defence.

Three claims, each backed by a script in this repo that prints the number:

| claim | measured | evidence |
|---|---|---|
| Automatic failover across 4 providers, p95-ordered | request failures **31.2% → 0.2%** at a 30% provider-failure rate (n=1000/arm) | [`results/chaos_report.md`](results/chaos_report.md) |
| Every response scored before it reaches a user | citation / depth / coherence, gate at 0.70 | [`results/provider_matrix.md`](results/provider_matrix.md) |
| Prompt-injection hardening | **100%** of 41 attacks blocked, **0%** false positives on 40 benign passages | [`results/security_report.md`](results/security_report.md) |

Read the [Limitations](#limitations) before quoting any of these.

## Architecture

```
POST /research
      │
      ▼
LlamaIndex Workflow ── Planner ──► Researcher ──► Synthesizer
      │                              │                │
      │  every LLM call              │ every tool call│ no tools at all
      ▼                              ▼                ▼
  GatewayLLM                   ScopePolicy.check()   sanitised evidence in
      │                         allow / deny+reason  <untrusted_document>
      ▼                              │
 litellm.Router                      ▼
  fallback chain              web_search / fetch_url
  per-provider timeouts
  retries + cooldown
      │
  ┌──────────┴──┬─────────────┬──────────┐
  ▼             ▼             ▼          ▼
OpenRouter    Gemini      Groq      NVIDIA NIM
      │
      ▼
 QualityGate — score, retry once on another provider, never return a failure silently
```

### Three decisions that shaped it

**The LlamaIndex bridge is hand-written, not `llama-index-llms-litellm`.** That package
calls `litellm.completion()`, which has no Router and therefore no fallbacks. Dropping it
into an agent looks like it works and silently voids the entire failover story.
[`gateway_llm.py`](src/gateway/llm/gateway_llm.py) routes every agent call through the
Router instead, and [`test_gateway_llm.py`](tests/test_gateway_llm.py) goes red if that
ever regresses.

**Citation accuracy is checked deterministically.** An LLM asked "are these citations
accurate?" gives an unfalsifiable answer. Instead every `[n]` is cross-checked against the
run's actual tool-call trace: a URL the agent cited but never fetched is a hallucination
provable by string comparison. Only the final claim-support step uses a model, and only on
the sentences a lexical screen cannot settle.

**The injection control is not a prompt.** Prompt injection works by persuading a model, so
the layer that stops it must not itself be a prompt.
[`ScopePolicy`](src/gateway/security/scope.py) denies by default on role, URL and budget,
and never consults a model. The synthesizer — the step that reads the most
attacker-controlled text — has no tools at all.

## Quickstart

```bash
uv sync
cp .env.example .env      # fill in what you have; providers without keys are simply absent
uv run gateway providers  # the chain that will actually be used, and N
uv run gateway research "How much grid-scale storage was added in 2024?"
uv run gateway serve      # HTTP API and web UI on :8000
```

Or in a container — the image needs at least one provider key at runtime, or `build_gateway()`
fails fast on startup rather than serving a gateway with nothing behind it:

```bash
docker build -t multi-agent-gateway .
docker run --rm -p 8000:8000 --env-file .env multi-agent-gateway
# → http://localhost:8000  (web UI)   http://localhost:8000/health
```

## Reproducing the numbers

```bash
uv run pytest                              # 111 tests, fully offline, no keys needed
uv run python scripts/smoke_providers.py   # what N actually is
uv run python scripts/bench_latency.py     # → results/latency.md, sets the chain order
uv run python scripts/chaos_run.py --n 1000 --p 0.1 0.3 0.5   # → results/chaos_report.md
uv run python scripts/eval_injection.py    # → results/security_report.md
uv run python scripts/run_eval.py          # → results/provider_matrix.md (needs keys + judge)
bash scripts/run_security_scans.sh         # bandit, pip-audit, secret scan
bash scripts/run_garak.sh                  # model-level probes against the running API
```

## Limitations

- **The chaos study injects faults at the transport layer.** The routing, retry, cooldown
  and fallback machinery under test is the production code path; only the network is
  simulated. Latency there is structural (mean fallback depth), not wall-clock.
- **The residual failure rate no longer rests on a backstop assumption.** It used to: a
  local tier modelled at a 2% independent failure rate, where setting it to 0 drove the
  residual to exactly zero and made the number unfalsifiable. Since Ollama was dropped
  (2026-07-28) every tier is cloud-hosted under the same sticky fault model, so the residual
  is bounded by all four providers failing together. `--local-fail` still exists but applies
  to nothing.
- **The injection block rate is in-sample.** The sanitiser's patterns were iterated against
  the corpus they are scored on, so 100% means "the known attack classes are covered", not
  "novel phrasings will be caught". The scope-gate half *does* generalise — it matches on
  URL, role and budget, never on attack wording. `results/security_report.md` separates the
  two.
- **garak probes the model, not the tool layer.** Its results and the scope-gate results are
  reported separately rather than blended into one flattering figure.
- **The judge has been checked against both a model and a human — the human pass is anchored,
  not blind.** Claude Opus 5 scored 18 reports independently: depth ρ=0.86 ± 0.03, coherence
  ρ=0.74 ± 0.01 (5 passes). The project author then scored the same 18 using
  `human_depth`/`human_coherence` keys, which is what closes Phase 6's gate — but did so by
  reviewing and adjusting the model rater's already-visible scores rather than labelling
  blind, changing 3 of 36 cells: depth ρ=**0.85 ± 0.04**, coherence ρ=**0.79 ± 0.04**.
  Anchoring inflates agreement, so read this as a human *review* of a model's labels, not
  independent human judgement — `results/judge_agreement.md` records the provenance
  explicitly rather than presenting it as a clean validation. A genuinely blind pass is still
  open: `results/label_sheet.md` carries no scores at all and is ready for one. The judge also
  runs +0.59 ± 0.06 high on coherence against the human labels, which cancels for provider
  comparison but inflates the absolute gate threshold. Citation sub-scores do not depend on
  the judge at all.
- **The judge does not reproduce its own scores, and the error bars above are why.** At
  temperature 0, 3 of 36 scores differed between the first and last pass against the model
  rater, 4 of 36 against the human rater. Two runs taken days apart drifted further — 6 of
  34 — so a within-session spread of ±0.03–0.04 likely understates day-to-day variation.
  `--repeat` on `scripts/judge_agreement.py` re-measures it; any figure quoted from a single
  run is a point estimate with unstated error.
- **Half the matrix cannot produce a regression verdict.** A silent regression is defined
  relative to the primary, so a question the primary failed yields no verdict either way.
  OpenRouter passed 15 of 30, so the 11 regressions are counted across 15 comparable
  questions, not 30. The report says so rather than printing a bare count.
- **The primary is not the strongest provider.** The chain is p95-latency-ordered, which puts
  OpenRouter (mean composite 0.688, 50% pass) ahead of Gemini (0.758, 73.3%). That is the
  documented cost of optimising for tail latency, not an oversight — but it does mean the
  best provider only serves when the fastest one fails. The 0.02 gap to NVIDIA NIM is inside
  the judge's ±0.08 noise band; the 0.07 gap to Gemini is probably not.
- **N = 4, not 5.** Azure OpenAI was planned and dropped for lack of access. Four providers
  — OpenRouter, Gemini, Groq and NVIDIA NIM, four independent cloud vendors — is the honest
  count. It was three cloud vendors plus a local Ollama tier until 2026-07-28; see the next
  bullet for what that swap cost.
- **The chain no longer terminates in something that cannot rate-limit.** Ollama held that
  role until it proved too slow to evaluate (~214s per synthesis, generation-bound) and was
  replaced by Groq. All four tiers are now cloud providers, so a correlated outage would
  take more of the chain than the independent-failure model assumes. The cost is visible at
  high fault rates: the residual was 0.1% with a local backstop and is 6.6% without one at
  a 50% provider-failure rate. At 30% it is essentially unchanged.

## Layout

```
src/gateway/
  llm/         router, fallback chain, LlamaIndex bridge, run tracing
  agents/      planner, researcher, synthesizer, LlamaIndex Workflow
  tools/       web search, SSRF-aware fetch, the tool-registry choke point
  security/    scope validation, untrusted-content handling, secret redaction
  eval/        deterministic citation checks, judge, rubric, quality gate
scripts/       every number above is printed by one of these
datasets/      30 research questions, 41 attacks, 40 benign lookalikes
results/       committed evidence
PLAN.md        the build plan and its progress log
```
