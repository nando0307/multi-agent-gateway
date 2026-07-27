# Multi-Agent Research System — Build Plan

**Repo:** `~/dev/multi-agent-gateway/multi-agent-gateway/`
**Stack:** LiteLLM (Router/failover) · LlamaIndex (agent workflow) · pytest · garak + bandit
**Timeline:** 10 working days (1–2 weeks)
**Status:** Phases 0–5, 8, 9 done · Phases 6-validation, 7, garak remaining
**Last updated:** 2026-07-27
**Providers (N=4):** Gemini · NVIDIA NIM · OpenRouter · Ollama (local) — *Azure OpenAI dropped, no access*
**Search:** Tavily · **Judge:** `openrouter/openai/gpt-oss-120b` (different model from every routed one)

---

## 0. What this project has to prove

This build exists to make three resume claims *literally true and reproducible*. Every phase below
terminates in an artifact under `results/` that fills a `[bracket]` in these bullets:

| # | Claim | Number to produce | Artifact |
|---|-------|-------------------|----------|
| 1 | Automatic failover across N providers, latency-ordered, verified by forced-failure tests | `N`, `X%` → `Y%` | `results/chaos_report.md` + `results/latency.md` |
| 2 | Eval harness gating every response on depth / citation accuracy / coherence | count of silent regressions | `results/provider_matrix.md` |
| 3 | Prompt-injection hardening via pre-tool-call scope validation; garak + bandit clean | block rate, FP rate, 0 leaks | `results/security_report.md` |

**Rule for the whole build: never write a number into the resume that isn't printed by a script in
this repo.** If a phase doesn't produce a measurement, it isn't done.

---

## 1. Architecture

```
                      ┌─────────────────────────────────────────────┐
   POST /research ──► │  FastAPI  (api.py)                          │
                      └───────────────┬─────────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────────┐
                      │  LlamaIndex Workflow (agents/orchestrator)  │
                      │   Planner ──► Researcher(×k) ──► Synthesizer│
                      └───────┬──────────────────┬──────────────────┘
                              │ every LLM call   │ every tool call
                              ▼                  ▼
                   ┌────────────────────┐  ┌──────────────────────┐
                   │ GatewayLLM         │  │ ScopePolicy.check()  │  ◄── security/scope.py
                   │ (llm/gateway_llm)  │  │ allow / deny+reason  │
                   └─────────┬──────────┘  └──────────┬───────────┘
                             ▼                        ▼
                   ┌────────────────────┐  ┌──────────────────────┐
                   │ litellm.Router     │  │ web_search / fetch   │
                   │  fallback chain    │  │ → sanitize() wrap    │
                   │  retries/cooldown  │  └──────────────────────┘
                   └─────────┬──────────┘
        ┌──────────┬─────────┴─────────┬────────────┐
        ▼          ▼                   ▼            ▼
     Gemini    NVIDIA NIM         OpenRouter    Ollama
                                               (local, free)
                             │
                             ▼
                   ┌─────────────────────────────────────┐
                   │ eval/gate.py — score before return  │
                   │  citation · depth · coherence       │
                   │  fail → retry on alt provider       │
                   └─────────────────────────────────────┘
```

### Key design decisions (and why)

**D1. In-process `litellm.Router`, not the LiteLLM proxy server.**
The proxy (`litellm --config models.yaml`) is the easy path and gives you an OpenAI-compatible
endpoint, but fallback events are then invisible to your test suite — you can only observe them
through logs. In-process Router lets `test_failover.py` assert *exactly which provider served each
request and at what fallback depth*, which is the entire point of bullet #1. Keep a `--proxy` deploy
mode as a footnote in the README; don't build the project on it.

**D2. A custom `GatewayLLM(CustomLLM)` bridging LiteLLM Router → LlamaIndex.**
`llama-index-llms-litellm` calls `litellm.completion()`, which has **no Router, therefore no
fallbacks**. Using it silently voids the whole failover story inside the agent. You must write ~80
lines subclassing LlamaIndex's `CustomLLM` whose `chat()`/`complete()` delegate to
`router.completion()` and stamp the served provider onto the response metadata.
*This is the single highest-risk integration point in the project — build it on Day 2 and prove it
with a test that forces the primary down and asserts the agent still answers.*

**D3. Ollama (`qwen3.5:9b`, already installed) is the last fallback tier.**
It is free, local, and cannot rate-limit — so the chain's tail always terminates in something that
works, which is what drives `Y%` toward 0. It is also *measurably worse*, which is what makes
bullet #2's "silent quality regression" real rather than hypothetical. Do not treat that as a
problem to hide; it's the finding.

**D4. Citation accuracy is deterministic first, LLM-judged second.**
An LLM judging "are these citations accurate?" is unfalsifiable. Instead: cross-check every `[n]`
marker against the *actual tool-call trace* of that run. A URL the agent never fetched but cited is
a hallucination you can prove with a string comparison, no judge needed. Only the final
claim-support step uses a model.

**D5. The judge must not be a model under test.**
If Gemini judges Gemini's output you get self-preference bias and the matrix is worthless.

**Revised 2026-07-27** — the guard is now *model*-level, not *provider*-level. The bias that
matters is a model rating its own output; a different model reached through a shared account
does not have it. Anthropic was the original plan but the account has no credits, so the judge
is `openrouter/openai/gpt-oss-120b`: vendor-distinct from all four routed models
(gemini-3.1-flash-lite, nemotron-super-49b, mistral-small, qwen3.5), ~$0.03 for a full
120-run matrix, and it reuses `OPENROUTER_API_KEY` rather than duplicating a secret.

Sharing an account with a routed provider *is* a weaker claim than a wholly separate vendor.
`settings.judge_independence_caveat()` says so, and `run_eval.py` prints it into
`results/provider_matrix.md` rather than passing over it. `assert_judge_is_independent()` still
hard-fails on the same model, including the same model reached by a different route
(`openrouter/google/gemini-...` vs `gemini/gemini-...`), so switching route is not a bypass.

**D6. N=4, and say 4.** Azure OpenAI is out (no access). The fallback chain is Gemini → NVIDIA NIM →
OpenRouter → Ollama, and it is *better shaped* than a 5th cloud provider would make it: three
independent cloud vendors plus a local terminal tier that cannot rate-limit. Nothing in the resume
bullet gets weaker. If Azure access appears mid-build, add it as a 5th entry in `model_list.py` and
re-run Phases 1 and 3 — those are the only two phases whose numbers depend on N.

---

## 2. Repository layout (target)

```
multi-agent-gateway/
├── PLAN.md                      ← this file; update the Progress Log every phase
├── README.md                    ← written last, from results/
├── .env.example
├── pyproject.toml
├── src/gateway/
│   ├── settings.py              # pydantic-settings; loads keys, fails loud if missing
│   ├── llm/
│   │   ├── model_list.py        # the 5 provider entries + fallback chain
│   │   ├── router.py            # build_router() → configured litellm.Router
│   │   ├── gateway_llm.py       # D2: LlamaIndex CustomLLM over the Router
│   │   └── events.py            # FallbackEvent records: attempt, provider, error, latency
│   ├── agents/
│   │   ├── planner.py           # question → 3-5 sub-questions (JSON)
│   │   ├── researcher.py        # sub-question → searched+fetched evidence w/ source ids
│   │   ├── synthesizer.py       # evidence → report with inline [n] markers
│   │   └── orchestrator.py      # LlamaIndex Workflow wiring + run trace
│   ├── tools/
│   │   ├── web_search.py
│   │   ├── fetch_url.py
│   │   └── registry.py          # name → (callable, args schema, allowed roles)
│   ├── security/
│   │   ├── scope.py             # ScopePolicy: the pre-tool-call gate
│   │   ├── sanitize.py          # untrusted-content wrapping + injection heuristics
│   │   └── redaction.py         # structlog processor scrubbing secrets
│   ├── eval/
│   │   ├── citations.py         # deterministic: resolvable / fetched / supported
│   │   ├── depth.py
│   │   ├── coherence.py
│   │   ├── judge.py             # fixed judge model, structured 1-5 output
│   │   ├── rubric.py            # composite score + weights
│   │   ├── gate.py              # threshold, retry-on-alt-provider, warning banner
│   │   └── runner.py            # batch over dataset × provider
│   ├── observability/
│   │   ├── logging.py
│   │   └── trace.py             # RunTrace: every llm call, tool call, decision
│   ├── api.py
│   └── cli.py
├── datasets/
│   ├── research_questions.jsonl # 30 questions
│   ├── injection_corpus.jsonl   # 40 attacks
│   └── benign_corpus.jsonl      # 40 lookalikes (for false-positive rate)
├── tests/
│   ├── conftest.py              # fake providers, deterministic fixtures
│   ├── test_failover.py
│   ├── test_gateway_llm.py
│   ├── test_scope.py
│   ├── test_sanitize.py
│   ├── test_citations.py
│   ├── test_redaction.py
│   └── test_gate.py
├── scripts/
│   ├── smoke_providers.py
│   ├── bench_latency.py
│   ├── chaos_run.py
│   ├── run_eval.py
│   ├── run_garak.sh
│   └── run_security_scans.sh
└── results/                     ← committed. This is the evidence.
    ├── latency.md / latency.json
    ├── chaos_report.md / chaos_raw.json
    ├── provider_matrix.md
    ├── judge_agreement.md
    └── security_report.md
```

---

## 3. Phases

Each phase lists: **goal → files → acceptance criteria → the number it produces.**
Do not advance until acceptance criteria pass. Tick the boxes as you go.

---

### Phase 0 — Scaffold & provider smoke test (Day 0, ~3h)

**Goal:** all five providers answer "hello" through LiteLLM. Nothing else.

- [x] `git add -A && git commit` — repo currently has **zero commits**; fix that first.
- [x] Add deps: `uv add llama-index-core llama-index-llms-openai-like pydantic-settings structlog httpx tenacity rich` and dev: `uv add --dev pytest-asyncio respx bandit pip-audit`
- [x] `.env.example` + `src/gateway/settings.py`:
  ```
  # --- routed providers (N=4) ---
  GEMINI_API_KEY=
  NVIDIA_NIM_API_KEY=
  OPENROUTER_API_KEY=
  OLLAMA_API_BASE=http://localhost:11434
  # --- tools ---
  TAVILY_API_KEY=
  # --- eval judge: MUST NOT be a routed provider (D5) ---
  JUDGE_API_KEY=
  JUDGE_MODEL=claude-sonnet-5     # or an OpenAI equivalent; pin it and never change it mid-project
  ```
  `settings.py` asserts at import time that `JUDGE_MODEL`'s provider prefix is absent from the
  router's model list — a one-line guard that makes D5 impossible to violate by accident.
- [ ] `scripts/smoke_providers.py`: loop all four, print `provider | ok/fail | latency_ms | error`.
      Add a 5th line that smoke-tests the judge key separately.
- [ ] Confirm LiteLLM model-string format for each against current docs (these change; verify, don't trust):
  `gemini/gemini-2.5-flash` · `nvidia_nim/meta/llama-3.3-70b-instruct` · `openrouter/<vendor>/<model>` ·
  `ollama_chat/qwen3.5:9b`
- [ ] Tavily: sign up, key in `.env`, one smoke query. Confirm the free-tier monthly quota against
      your Phase 7 budget — 30 questions × 4 sub-questions ≈ 120 searches per full matrix run, and
      you'll run the matrix more than once. **Build the search cache (Phase 7) on Day 0 if the quota
      looks tight**, since you need it for fairness anyway.

**Acceptance:** 4/4 providers green + judge key green + one successful Tavily query.
**Produces:** `N = 4`.

---

### Phase 1 — Latency benchmark → routing order (Day 1, ~4h)

**Goal:** pick the fallback order from data, so the resume's "after measuring [reasoning]" is real.

- [ ] `scripts/bench_latency.py`: for each provider, 20 runs of a fixed 300-token research-style
      prompt. Record TTFT, total latency, output tokens, tokens/sec, cost/1M (from
      `litellm.completion_cost`), failures.
- [ ] Report p50 / p95 / max. **p95 is the ordering key, not p50** — failover is a tail-latency
      story; a provider with a great median and a terrible tail is a bad primary.
- [ ] Run it at two times of day (e.g. 10:00 and 22:00 local) — provider load varies. Average them.
- [ ] Write `results/latency.md` with the table **and one paragraph justifying the chosen order**,
      naming the tie-breakers (cost, rate limit headroom, context window).
- [ ] Encode the order in `llm/model_list.py` as an explicit `FALLBACK_CHAIN` list with a comment
      linking to `results/latency.md`.

**Expected shape** (verify, don't assume): Gemini Flash fastest → NIM / OpenRouter mid, with
OpenRouter's tail depending heavily on which upstream you pick → Ollama slowest but never fails.
For OpenRouter, choose a model on a *different* underlying vendor than Gemini and NIM, otherwise a
single upstream outage takes down two of your four tiers and the chain is less independent than it
looks. Note that reasoning in `results/latency.md`.

**Acceptance:** `results/latency.md` exists, chain in code matches it.
**Produces:** the `[reasoning]` clause.

---

### Phase 2 — Router, failover, and the LlamaIndex bridge (Day 2, ~6h)

**Goal:** one call site that is provider-agnostic and self-healing.

- [x] `llm/model_list.py` — model list with `rpm`/`tpm` limits per deployment.
- [x] `llm/router.py` — `build_router()`:
  ```python
  Router(
      model_list=MODEL_LIST,
      fallbacks=[{"research-primary": FALLBACK_CHAIN}],
      context_window_fallbacks=[{"research-primary": ["long-ctx"]}],
      num_retries=2, timeout=45.0,
      allowed_fails=3, cooldown_time=60,      # circuit breaker
      routing_strategy="latency-based-routing",
      set_verbose=False,
  )
  ```
  All agents request the alias `"research-primary"` and never name a provider.
- [x] `llm/events.py` — a `FallbackEvent` per attempt (`attempt_idx, model, error_class, latency_ms,
      served: bool`), appended to the `RunTrace`. Hook via Router's success/failure callbacks.
      **Without this, you have failover but no evidence of failover.**
- [x] `llm/gateway_llm.py` — D2 bridge. Implement `metadata`, `complete`, `stream_complete`, `chat`,
      `achat`. Stamp `response.additional_kwargs["served_by"]`.
- [x] `tests/test_gateway_llm.py` — with the primary monkeypatched to always raise, a LlamaIndex
      agent still returns an answer, and `served_by` is the second provider.

**Acceptance:** kill the primary → agent still answers → trace shows depth ≥ 1.

---

### Phase 3 — Chaos harness: the X% → Y% number (Day 3, ~5h)

**Goal:** the headline metric. This is bullet #1's payload.

Design it properly, because a sloppy version of this is the most common way a resume bullet
collapses under interview questioning:

- [x] **Fault injector** (`tests/conftest.py` + `scripts/chaos_run.py`): wrap each provider's
      transport and inject, at probability `p`, one of:
      `503 ServiceUnavailable` · `429 RateLimitError` · connection timeout · malformed/empty response.
      Sample from a fixed seed so runs are reproducible.
- [x] **Two arms, same faults, same seed, same questions:**
  - **Arm A (baseline):** single provider, `num_retries=0`, no fallbacks → failure rate = **X%**
  - **Arm B (gateway):** full chain + retries + cooldown → failure rate = **Y%**
- [x] `n = 200` requests per arm. Report Wilson 95% CI on both rates — with n=200 you can honestly
      say "reduced from X% to Y%"; with n=20 you cannot.
- [x] Sweep `p ∈ {0.1, 0.3, 0.5}` and produce a small table. Quote the middle one on the resume, and
      have the sweep ready for the interview follow-up ("at what failure rate does it break down?").
- [x] Also record: mean fallback depth, added p95 latency (failover isn't free — knowing the cost
      is a strong interview answer), % of requests served by each tier.
- [x] `tests/test_failover.py` — deterministic unit cases, no probability:
      each single provider forced down; the first two down; **all four down** (must raise a clean
      `AllProvidersExhausted`, not hang); a 429 triggers cooldown so the next request skips that
      provider; context-window overflow routes to the long-context alias.
- [x] Write `results/chaos_report.md`.

**Acceptance:** `pytest tests/test_failover.py` green; report has both arms, CIs, and the sweep.
**Produces:** `X%`, `Y%`.

---

### Phase 4 — Tools + the scope-validation gate (Day 4, ~5h)

Build the security layer *before* the agents, so no tool ever ships unguarded. Order matters here.

- [x] `tools/web_search.py`, `tools/fetch_url.py` (httpx, 10s timeout, 1MB cap, `trafilatura` or
      `readability-lxml` for text extraction), `tools/registry.py`.
- [x] `security/scope.py` — **every tool call passes through `ScopePolicy.check(role, tool, args,
      run_state) -> Allow | Deny(reason)` before execution.** Deny by default. Rules:
  1. Role→tool allowlist (planner: ∅; researcher: search+fetch; synthesizer: ∅ — the writer never
     touches the network, so injected text can't trigger a call at write time)
  2. Args validated against a pydantic schema; query length cap
  3. URL policy: `https` only; **block private/loopback/link-local IPs and DNS-rebind** (SSRF);
     domain denylist; block non-HTML content types
  4. Per-run budgets: ≤12 searches, ≤25 fetches, ≤200KB total fetched, ≤90s tool wall-clock
  5. No filesystem / shell / code-exec tool exists in the registry at all
  6. Every decision → `RunTrace` (allow *and* deny)
- [x] `security/sanitize.py`:
  - Wrap all fetched content in `<untrusted_document id="...">…</untrusted_document>` with a
    neutralizing preamble; **never** concatenate it into a system prompt
  - Heuristic injection scan: "ignore previous/above", "you are now", "system prompt", "reveal your
    instructions", tool names appearing in fetched body, invisible/zero-width chars, long base64,
    markdown image exfil (`![](http://evil/?d=…)`)
  - On flag: downgrade the document to quote-only (can be cited, cannot instruct) + log event.
    **Downgrade, don't drop** — dropping tanks recall and inflates false positives.
- [x] `tests/test_scope.py`, `tests/test_sanitize.py` — table-driven, ≥25 cases.

**Acceptance:** a researcher tool call with a `file://` or `http://169.254.169.254` URL is denied and
traced.

---

### Phase 5 — The agent workflow (Day 5, ~6h)

- [x] `agents/planner.py` — question → 3–5 sub-questions, structured JSON output, validated.
- [x] `agents/researcher.py` — per sub-question: search → pick top-k → fetch → extract →
      emit `Evidence{source_id, url, title, quote, retrieved_at}`. Runs k sub-questions concurrently
      (`asyncio.gather`) — also stresses the Router's rate limiting, which is useful.
- [x] `agents/synthesizer.py` — evidence → markdown report. **Hard contract:** every claim-bearing
      sentence carries ≥1 `[n]`; `[n]` indexes the numbered source list; sources it wasn't given are
      forbidden. State this contract in the prompt *and* enforce it in `eval/citations.py` — prompts
      are requests, the eval is the enforcement.
- [x] `agents/orchestrator.py` — LlamaIndex `Workflow`; emits a complete `RunTrace`
      (every LLM call w/ served_by + fallback depth, every tool call w/ scope decision, timings).
      Persist to `results/runs/<run_id>.json` — the trace is what everything downstream measures.
- [x] `datasets/research_questions.jsonl` — 30 questions across: factual-recent, comparative,
      multi-hop, quantitative, contested/ambiguous. Include 3 with **no good answer** to test
      honest abstention. Each with `id, question, category, notes`.

**Acceptance:** 5 end-to-end runs produce reports with resolvable citations and a full trace.

---

### Phase 6 — Eval harness (Day 6, ~6h)

- [x] `eval/citations.py` — **deterministic, no LLM:**
  - `resolvable` = % of `[n]` markers that map to a listed source
  - `grounded` = % of cited URLs that appear in the run's fetch trace ← catches fabricated URLs
  - `supported` = % of cited sentences entailed by the cited chunk. Two-stage: token-recall overlap
    screen → LLM NLI check only on the low-overlap remainder (keeps cost sane)
  - `citation_score = 0.3·resolvable + 0.3·grounded + 0.4·supported`
- [x] `eval/depth.py` — deterministic (unique domains, source count, sub-questions answered,
      specificity density = numbers/dates/proper nouns per 100 words, conflict acknowledged y/n)
      + judge 1–5 on coverage & non-genericity. 50/50 blend.
- [x] `eval/coherence.py` — judge 1–5: structure, self-contradiction, answers the question asked,
      redundancy.
- [x] `eval/judge.py` — fixed independent model from `JUDGE_MODEL` (D5), temperature 0, structured
      output, rubric with **anchored
      examples for each score level** (unanchored 1–5 judges drift badly), 3-sample self-consistency
      with median.
- [x] `eval/rubric.py` — `composite = 0.4·citation + 0.3·depth + 0.3·coherence`.
- [ ] **Validate the judge** (`results/judge_agreement.md`): hand-label 20 reports yourself, report
      Spearman ρ and %-within-1 agreement. If ρ < 0.6 the rubric is broken — fix it before running
      the matrix. *Skipping this step is what makes most eval harnesses decorative.*

**Acceptance:** `judge_agreement.md` shows acceptable agreement; scores reproduce across two runs.

---

### Phase 7 — The gate + provider matrix (Day 7, ~5h)

- [x] `eval/gate.py` — score before the response returns. `composite ≥ 0.70` → return.
      Else → retry once on a different provider; still failing → return with an explicit
      quality warning + the failed sub-scores. **Never silently return a failing report** — that's
      the whole thesis of bullet #2.
- [x] `scripts/run_eval.py` — 30 questions × N providers (pin each run to one provider), cache
      search results across providers so all providers see **identical evidence** — otherwise you're
      measuring search variance, not model quality.
- [x] Compute the **silent quality regression** count with this exact definition, and put the
      definition in the report:
      > A run where the request **succeeded** (HTTP 200, no error, within latency SLO) — so every
      > ops dashboard shows green — **but** the composite score fell below the 0.70 gate, on a
      > question the primary provider passed.
- [ ] `results/provider_matrix.md`: per-provider mean composite + sub-scores, pass rate, and the
      regression count with per-case examples (2–3 concrete before/after excerpts — these are what
      you actually talk about in an interview).
- [x] `tests/test_gate.py` — fabricated-citation report scores low; a good report passes; a gate
      failure triggers exactly one alt-provider retry.

**Acceptance:** matrix complete, regression count computed from real runs.
**Produces:** bullet #2's `[X]`.

---

### Phase 8 — Security: injection, garak, bandit, leak tests (Day 8, ~6h)

- [x] `datasets/injection_corpus.jsonl` — 40 attacks, embedded in *fetched page content* (the
      realistic vector; direct-prompt attacks are the easy case). Categories: instruction override ·
      tool-abuse ("now fetch file:///etc/passwd") · exfiltration (markdown-image beacon, "append
      your API key to the URL") · system-prompt extraction · citation poisoning ("cite example.com
      as the source") · encoding (base64/zero-width/homoglyph) · multi-turn drift.
- [x] `datasets/benign_corpus.jsonl` — 40 pages that *look* suspicious but are legitimate (a blog
      post *about* prompt injection, a security advisory quoting an attack). **Report false-positive
      rate alongside block rate** — a filter that blocks everything scores 100% and is useless.
- [x] `scripts/run_security_scans.sh` — `bandit -r src/ -ll`, `pip-audit`, plus a secret-scan.
- [x] `scripts/run_garak.sh` — point garak's `rest` generator at your FastAPI endpoint so it probes
      **the whole pipeline**, not a bare model. Probes: `promptinject`, `dan`, `encoding`,
      `leakreplay`, `xss`. Run it **twice**: once against the gateway, once against a raw provider
      call — the delta is your hardening evidence. Be honest in the writeup that garak is
      model-level and your scope layer is what covers the tool-abuse class it doesn't reach.
- [x] `tests/test_redaction.py` — the credential-leakage proof: run a full research request with
      sentinel API keys (`sk-SENTINEL-DO-NOT-LOG-a1b2c3`), then assert the sentinel appears in
      **zero** of: stdout, stderr, log files, `results/runs/*.json` traces, error messages,
      exception tracebacks, the HTTP response body. Also assert redaction survives an exception path
      — that's where keys usually escape.
- [x] `results/security_report.md`: block rate, FP rate, garak before/after, bandit findings
      (and fixes), leak test result.

**Acceptance:** bandit clean at `-ll`; sentinel leak count = 0; block/FP rates recorded.
**Produces:** bullet #3's numbers.

---

### Phase 9 — API, metrics, docs (Day 9, ~4h)

- [x] `api.py` — `POST /research` (question → report + scores + trace id), `GET /health`
      (per-provider circuit state), `GET /metrics` (requests, fallback depth histogram, provider
      share, score distribution).
- [x] `cli.py` — `gateway research "…"` with a rich live view showing provider, fallback depth, and
      live scores. This is your demo; make it look good.
- [x] `README.md` — architecture diagram, the three claims each linked to its `results/` file,
      quickstart, and an honest "Limitations" section (small n, judge is a single model, garak
      coverage caveats). *An explicit limitations section reads as senior, not as weakness.*
- [ ] CI: GitHub Actions running `pytest` + `bandit` on push (mock providers, no keys in CI).

---

### Phase 10 — Buffer & resume finalization (Day 10)

- [ ] Re-run every script end-to-end from a clean clone; confirm all numbers reproduce.
- [ ] Fill the brackets from `results/` — see §5.
- [ ] Record a 60–90s screen capture of the CLI surviving a killed provider mid-run.

---

## 4. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ~~Azure OpenAI access~~ | — | **Resolved:** no access, N=4 (D6). Not a risk anymore. |
| `llama-index-llms-litellm` bypasses Router (D2) | Certain | Custom `GatewayLLM`; test on Day 2 |
| Free-tier rate limits break the 200-run chaos study | High | Fault injection is at the transport layer with mocked responses for the bulk of runs; reserve live calls for a smaller confirmation set. Say which is which in the report. |
| Judge unreliable → matrix meaningless | Low-Med | Independent frontier judge (D5) + Phase 6 agreement check as a hard gate |
| Judge spend creeps | Low | Temperature 0 + cache judgments by `hash(report)`; 3-sample self-consistency only on the 20-report validation set, single-sample for the matrix |
| Tavily free-tier quota exhausted mid-matrix | **Med-High** | Shared on-disk search cache keyed by query, built Day 0. Required for fairness anyway, so it costs nothing extra. Re-runs of the matrix then hit zero live searches. |
| OpenRouter routes to the same upstream as Gemini/NIM → correlated failure | Medium | Pick a distinct-vendor model at Phase 1; document the choice |
| Scope layer over-blocks | Medium | Benign corpus + reported FP rate |
| Scope creep → nothing finished | **High** | Phases 0–3 and 8 are the resume. If you're behind, cut Phase 9's metrics endpoint and the CI, not the measurements. |

---

## 5. Resume bullet → evidence map

Fill these only from `results/`:

- `[N]` ← **4** (Gemini, NVIDIA NIM, OpenRouter, Ollama) — confirm all four green in Phase 0 before
  committing to the number
- `[reasoning]` ← `results/latency.md` — the p95-ordering paragraph
- `[X]% → [Y]%` ← `results/chaos_report.md`, arm A vs arm B, n=200, with CI
- bullet 2's `[X]` ← `results/provider_matrix.md` silent-regression count
- bullet 3 ← `results/security_report.md` (block rate, FP rate, 0 sentinel leaks, bandit clean)

**Also fix the grammar in bullet 3** before sending: "by added scope validation" → "by adding scope
validation".

---

## 6. Resolved decisions

*(all Day-0 blockers answered 2026-07-26)*

- **Search API → Tavily.** Free tier, agent-oriented, returns pre-extracted clean text, which makes
  the Phase 6 citation-support check materially more reliable than snippet-only APIs.
  Consequence: build the shared search cache early (see risk register).
- **Azure OpenAI → dropped.** No access. `N=4`. See D6.
- **Judge → independent Anthropic/OpenAI key**, never added to the router's model list, guarded by
  an assertion in `settings.py`. Removes self-preference bias from the provider matrix entirely,
  which is the difference between a defensible eval harness and a decorative one.

### Remaining work

* **Phase 1 — running.** `bench_latency.py --runs 20 --label morning` is in flight.
  `DEFAULT_CHAIN` is still the *placeholder* order and must be re-derived from
  `results/latency.md` before the "after measuring" clause on the resume is true.
  First result in: Gemini p50 9.3s / p95 23.1s / 1 failure in 20 — a poor primary, and it
  currently sits first in the chain. The benchmark demoting it is the point of this phase.
* **Phase 6 validation — needs you, not me.** `scripts/judge_agreement.py generate --n 20`
  produces reports and an *empty* label sheet; the judge's scores are deliberately withheld
  so labelling is not anchored. Fill `datasets/human_labels.jsonl`, then run
  `judge_agreement.py score`. Gate is Spearman ρ ≥ 0.6 on both dimensions. I cannot
  fabricate the labels — invented labels would make this check worse than skipping it.
* **Phase 7** — `run_eval.py` ready; run after the judge clears the gate.
* **garak** — installed at `.venv-garak/`, config verified against 0.15.1. Needs the API up.

### Still open (non-blocking, decide by the phase noted)

- **Which OpenRouter model** — decide during Phase 1; pick a distinct upstream vendor from
  Gemini/NIM so the four tiers fail independently.
- **Gate threshold 0.70** — a placeholder. Recalibrate after Phase 6 once you can see the actual
  score distribution; a threshold nothing fails, or everything fails, measures nothing.
- **Long-context fallback alias** — only needed if Phase 5 runs hit context limits; skip until they do.

---

## 7. Progress log

*Append one entry per phase completed: date, what shipped, numbers produced, deviations from plan.*

| Date | Phase | Shipped | Numbers | Notes |
|---|---|---|---|---|
| 2026-07-26 | — | Plan written | — | — |
| 2026-07-26 | — | Day-0 decisions resolved | N=4 | Tavily search; Azure dropped (no access); independent judge key |
| 2026-07-26 | 0 | Scaffold, deps, settings, `.env.example`, smoke script, repo pushed | — | Judge-independence guard asserts at import, so D5 cannot be violated by accident |
| 2026-07-26 | 2 | Router + fallback chain + `GatewayLLM` bridge + `RunTrace` | — | `model_info.tier` collides with a reserved litellm field; renamed `chain_index`. Per-provider timeouts added after the 45s global killed the local tier outright |
| 2026-07-26 | 3 | Chaos harness, 12 deterministic failover tests | **31.2% → 0.0%** @ p=0.3, n=1000 (CI 0.0–0.4) | Baseline retries too, faults are sticky, local tier given a 2% failure rate — all three keep the number defensible |
| 2026-07-26 | 4 | `ScopePolicy`, sanitiser, tool registry choke point | 28 scope tests | Synthesizer holds zero tools by design |
| 2026-07-26 | 5 | Planner/researcher/synthesizer + LlamaIndex Workflow, 30-question dataset | — | Live end-to-end run against Ollama; with no search key it retrieved nothing and the gate refused to return a report rather than inventing one |
| 2026-07-26 | 6 | Deterministic citation checks, rubric, judge client | — | Judge client written but **unrun**: needs `JUDGE_API_KEY`. Refuses to fall back to a routed provider |
| 2026-07-26 | 8 | 41-attack + 40-benign corpora, injection eval, bandit/pip-audit/secret scan | **100% blocked, 0% FP**; bandit 0 issues; 0 sentinel leaks | Block rate is in-sample — patterns were tuned against this corpus, disclosed in the report. Both bandit findings fixed rather than suppressed |
| 2026-07-26 | 9 | FastAPI, CLI, README | — | 104 tests, all offline |
| 2026-07-27 | 0 | Smoke test against real keys | **N = 4** | Three stale defaults found: `gemini-2.5-flash` is 404 "no longer available to new users"; NIM `llama-3.3-70b` takes ~87s (timed out at 362s once); Anthropic judge account has no credits. Now `gemini-3.1-flash-lite`, `nemotron-super-49b`, judge on OpenRouter |
| 2026-07-27 | — | Judge guard reworked from provider-level to model-level | — | Provider-level blocked any judge sharing an account. Self-preference is a *model* scoring itself, so the check compares underlying model names — `openrouter/google/gemini-…` is still caught as `gemini/gemini-…`. Shared-account caveat is printed into the matrix report, not buried |
| 2026-07-27 | 6 | `judge_agreement.py` — two-phase harness | — | Sheet is generated *without* judge scores; seeing them first would measure compliance, not agreement. Spearman verified against perfect/inverse/tied cases |
| 2026-07-27 | 4 | On-disk page cache | — | Matrix would otherwise re-download every page once per provider. Scope check still runs on every call; the cache short-circuits the network, never the policy |
| 2026-07-27 | 1 | Latency benchmark, n=20/provider | chain **openrouter → gemini → ollama → nim** | Both placeholder assumptions overturned. Gemini was first by guess: it is 5x slower than OpenRouter on p95 and the only provider to fail (1/20). The local tier was assumed slowest and pinned last: it is not — p95 23.6s vs NIM's 138.3s, and the tightest p50/p95 spread in the set. Chain order does not affect P(all fail), so there was no availability reason to override the data |
| 2026-07-27 | 2 | Per-provider timeouts derived from measured p95 | — | The global 45s sat below NIM's 138s p95, so the gateway would have abandoned NIM on its own normal tail and manufactured failovers that were not outages. Each timeout is now ~2x that provider's measured p95 |
| 2026-07-27 | 3 | Chaos study re-run against the measured chain | **31.2% → 0.1%** @ p=0.3, n=1000 | `chaos_run.py` had its own hardcoded chain; it now imports `DEFAULT_CHAIN`, because a chaos number describing a chain the system does not ship is worse than no number |
| 2026-07-27 | 8 | garak 0.15.1 installed, REST config verified | — | `request_timeout` defaulted to 20s (too short for a research call); `promptinject.HijackHateHumansMini` does not exist. Probe set bounded — each probe is a full research run |
