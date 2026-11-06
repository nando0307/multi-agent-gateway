# multi-agent-gateway

A generation gateway for Boss Assistant, plus the existing multi-agent web-research
application. Boss retains retrieval, authorization, citations and embeddings.

Historical web-research measurements (not measurements of the new serving chain):

| claim | measured | evidence |
|---|---|---|
| Historical p95-ordered provider failover | request failures **31.2% → 0.2%** at a 30% provider-failure rate (n=1000/arm) | [`results/chaos_report.md`](results/chaos_report.md) |
| Research responses scored before delivery | citation / depth / coherence, gate at 0.70 | [`results/provider_matrix.md`](results/provider_matrix.md) |
| Prompt-injection hardening | **100%** of 41 attacks blocked, **0%** false positives on 40 benign passages | [`results/security_report.md`](results/security_report.md) |

Read the [Limitations](#limitations) before quoting any of these.

## Architecture

`GET /` serves a single-page browser UI over this same endpoint — question in, report and
scores out, including the citation sub-scores below, since those are checked against the
run's tool-call trace rather than by a judge.

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
  bounded attempts + cooldown
      │
  ┌──────────┴──┬─────────────┬──────────┐
  ▼             ▼             ▼          ▼
Gemini    OpenRouter      Groq      DeepSeek
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

## Boss generation deployment

The approved serving chain is **Gemini -> OpenRouter -> Groq -> direct DeepSeek**.
NVIDIA and subscription Codex are not generation fallbacks. Each provider's `*_MODEL`
and `*_API_KEY` in `.env.example` selects its server-held model and credentials.
All four Boss aliases currently share this configured chain: `boss-default`,
`boss-deep`, `boss-graph-map`, `boss-graph-reduce`. Caller model IDs, upstream URLs
and upstream credentials are not accepted.

1. Set the four provider keys and a separate, randomly generated
   `GATEWAY_SERVICE_TOKEN`. Do not reuse a user JWT or a provider key.
2. Run `uv run gateway providers`. Missing keys appear as disabled. This is
   configuration inventory, not live health. Startup requires at least one provider.
3. Run `uv run python scripts/smoke_providers.py --generation-only` in the deployment
   environment. This sends only synthetic text and a tiny JSON schema to each configured
   provider, with no search, judge, embedding or private-document calls. It exits nonzero
   for any failed or missing tier. A configured key alone does not establish model access,
   quota, schema compatibility or permission to send private documents.
4. Run `uv run gateway serve --host 127.0.0.1 --port 8000` behind an internal TLS
   reverse proxy/private network. Allow Boss only `POST /v1/chat/completions`.
   Do **not** expose the legacy `/research`, UI, docs or operational endpoints publicly:
   those pre-existing surfaces do not use the generation service token.
5. Configure Boss `GENERATION_BASE_URL=https://gateway.internal/v1`,
   `GENERATION_API_KEY=<same service token>`, zero client retries and a transport
   timeout greater than the gateway's total deadline (65 seconds for the default 60).
   Keep NVIDIA embedding configuration separate; no corpus reindexing is involved.

### Request and output contract

`POST /v1/chat/completions` requires `Authorization: Bearer <GATEWAY_SERVICE_TOKEN>`.
Unset service token disables generation with 503. Wrong/missing token returns 401.
The body accepts a Boss alias, a nonempty `messages` array of string `role`/`content`
pairs, optional `temperature` (0-2), either positive `max_tokens` or
`max_completion_tokens`, and `stream` (default false). Roles: system, developer, user,
assistant. Token limits are preserved, with `max_completion_tokens` translated to
`max_tokens` for the approved adapters. Unknown fields, tools, multimodal content,
caller routing credentials and unsupported required formats are rejected.

Optional structured output:

```json
{"response_format":{"type":"json_schema","json_schema":{
  "name":"RouteQuery","strict":true,"schema":{
    "type":"object","properties":{"department":{"enum":["hr","finance","both"]}},
    "required":["department"],"additionalProperties":false
  }
}}}
```

Schemas use JSON Schema 2020-12 with an object root and local JSON-pointer references.
Remote references, `$id`, dynamic references and other dialects are rejected.
Native schema mode is used when LiteLLM reports support. Otherwise an adapter must
support JSON mode: the schema is supplied as a system instruction and its result is
locally validated against the same schema. This is validated output, not a claim that
every vendor offers native constrained decoding. Invalid output fails explicitly.
Structured streams are buffered until validated; ordinary text streams are incremental.
No `/v1/models` endpoint is needed by the Boss OpenAI adapter.

Non-streaming results are OpenAI `choices[].message` envelopes, with usage where
supplied, plus `X-Gateway-Provider` and `X-Gateway-Fallback-Depth` headers.
Streaming results use OpenAI `data:` delta frames, a verified `finish_reason: stop`,
then `[DONE]`. Output truncation and missing upstream terminal evidence are errors,
never successful partial answers. Before content, transient failure advances to the
next provider. After any content, failure emits an OpenAI `error` frame and closes
without `[DONE]`; consumers must not publish that partial answer as complete.
No continuation from another model is appended. Caller disconnect cancels upstream
work at this HTTP boundary; Boss must also close its gateway request on cancellation.

One `GENERATION_DEADLINE_S` wall-clock budget (default 60 seconds) covers the entire
request. Each provider gets at most `GENERATION_ATTEMPT_S` (default 15 seconds), its
existing provider-specific timeout, and the remaining total budget, whichever is least.
There are no nested retries or per-provider full-deadline resets. Connection failures,
timeouts, 429 and 500/502/503/504 advance to backups; invalid requests, context errors,
safety refusals, invalid schema output and credential errors do not. Blank completions
and pre-content incomplete streams also advance. All-blank exhaustion is 503.
Lower timeouts can manufacture failover under normal provider latency; tune with
synthetic probes. Four enabled providers are not four independent infrastructure paths.

Errors never include upstream response bodies, credentials or prompt contents:

| HTTP | `error.code` | Meaning |
|---|---|---|
| 400 | `invalid_request` | Unsupported/invalid input or schema |
| 400 | `generation_refused` | Provider safety refusal; no fallback |
| 401 | `authentication_error` | Invalid service Bearer token |
| 502 | `provider_configuration` | Upstream credential/permission error |
| 502 | `invalid_completion` | Output failed required schema |
| 502 | `stream_interrupted` | Truncated or interrupted answer |
| 502 | `generation_failed` | Other non-retryable provider failure |
| 503 | `providers_exhausted` | All available approved tiers failed |
| 503 | `service_unavailable` | Service token not configured |
| 504 | `generation_deadline` | Shared total budget expired |

After SSE headers, these errors are data frames rather than new HTTP statuses.
LiteLLM is pinned because its 1.93 stream wrapper otherwise synthesizes success on
premature EOF and can continue another model's answer even with fallbacks disabled.
The small Router override preserves its provider adapters while enforcing this contract.
Run `uv run pytest tests/test_generation_http.py` before upgrading LiteLLM.
That regression file exercises real loopback HTTP faults, including cancellation,
schema validation, sync/async blank consistency, streaming EOF and token preservation.

Approve data processing with **every** enabled provider before routing private RAG
context. Embeddings, business-action tools, autonomous execution and public-web
research orchestration are outside this generation endpoint. Provider failover does
not make the gateway process, network, retriever or vector store highly available.


## Reproducing the numbers

```bash
uv run pytest                              # offline, no provider keys needed
uv run python scripts/smoke_providers.py   # what N actually is
uv run python scripts/bench_latency.py     # benchmarks, does not change approved routing order
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
- **The historical primary was not the strongest provider.** The old p95 ordering put
  OpenRouter (mean composite 0.688, 50% pass) ahead of Gemini (0.758, 73.3%). Those
  measurements do not rank the current Gemini-first serving chain.
- **The historical matrix used OpenRouter, Gemini, Groq and NVIDIA NIM.** Current serving
  uses Gemini, OpenRouter, Groq and direct DeepSeek. Its effective provider count is the
  number of configured keys, and live availability requires the deployment probe.
- **The chain no longer terminates in something that cannot rate-limit.** Ollama held that
  role until it proved too slow to evaluate (~214s per synthesis, generation-bound) and was
  replaced by Groq. All four tiers are now cloud providers, so a correlated outage would
  take more of the chain than the independent-failure model assumes. The cost is visible at
  high fault rates: the residual was 0.1% with a local backstop and is 6.6% without one at
  a 50% provider-failure rate. At 30% it is essentially unchanged.

## Layout

```
src/gateway/
  llm/           router, fallback chain, LlamaIndex bridge, run tracing
  agents/        planner, researcher, synthesizer, LlamaIndex Workflow
  tools/         web search, SSRF-aware fetch, the tool-registry choke point
  security/      scope validation, untrusted-content handling, secret redaction
  eval/          deterministic citation checks, judge, rubric, quality gate
  static/        the web UI (index.html) served at GET /, self-contained, no build step
Dockerfile       multi-stage, non-root, uv.lock-pinned — see Quickstart
scripts/         every number above is printed by one of these
datasets/        30 research questions, 41 attacks, 40 benign lookalikes
results/         committed evidence
PLAN.md          the build plan and its progress log
```
