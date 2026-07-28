"""Phase 0: does every configured provider actually answer?

N in the resume bullet is whatever this script prints green, not whatever is in .env.
A provider you cannot get working is one you drop, not one you claim.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import litellm  # noqa: E402

from gateway.llm.model_list import DEFAULT_CHAIN, provider_params  # noqa: E402
from gateway.settings import get_settings  # noqa: E402

PROMPT = [{"role": "user", "content": "Reply with the single word: ready"}]


def main() -> int:
    settings = get_settings()
    ok = 0

    print(f"{'provider':<12} {'status':<8} {'latency':>9}  detail")
    print("-" * 68)

    for name in DEFAULT_CHAIN:
        params = provider_params(name, settings)
        if params is None:
            print(f"{name:<12} {'skip':<8} {'-':>9}  no credentials configured")
            continue
        # provider_params may already carry a per-provider timeout (the local tier does);
        # passing another one here would collide.
        params.setdefault("timeout", 30)
        started = time.perf_counter()
        try:
            response = litellm.completion(messages=PROMPT, max_tokens=64, **params)
            elapsed = (time.perf_counter() - started) * 1000
            text = (response.choices[0].message.content or "").strip()[:40]
            print(f"{name:<12} {'ok':<8} {elapsed:>8.0f}ms  {text!r}")
            ok += 1
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            print(f"{name:<12} {'FAIL':<8} {elapsed:>8.0f}ms  {type(exc).__name__}: {str(exc)[:80]}")

    print("-" * 68)

    # The judge is checked separately: it is deliberately not in the router.
    if settings.resolve_judge_key():
        try:
            settings.assert_judge_is_independent()
            litellm.completion(
                model=settings.judge_model, api_key=settings.resolve_judge_key(),
                messages=PROMPT, max_tokens=16, timeout=30,
            )
            print(f"{'judge':<12} {'ok':<8} {'-':>9}  {settings.judge_model}")
        except Exception as exc:
            print(f"{'judge':<12} {'FAIL':<8} {'-':>9}  {type(exc).__name__}: {str(exc)[:80]}")
    else:
        print(f"{'judge':<12} {'skip':<8} {'-':>9}  JUDGE_API_KEY not set (eval will refuse to run)")

    if settings.tavily_api_key:
        try:
            from gateway.tools.web_search import search

            hits = search("solar capacity 2023", k=2, api_key=settings.tavily_api_key)
            print(f"{'tavily':<12} {'ok':<8} {'-':>9}  {len(hits)} results")
        except Exception as exc:
            print(f"{'tavily':<12} {'FAIL':<8} {'-':>9}  {type(exc).__name__}: {str(exc)[:80]}")
    else:
        print(f"{'tavily':<12} {'skip':<8} {'-':>9}  TAVILY_API_KEY not set")

    print(f"\nN = {ok} working provider(s). Use this number, not the aspirational one.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
