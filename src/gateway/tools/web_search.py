"""Tavily-backed web search, with Parallel.ai as a fallback, and an on-disk cache.

The cache is not only a cost control. Phase 7 runs the same questions through every
provider, and if each provider sees different search results the matrix measures search
variance rather than model quality. Sharing one cache across providers is what makes that
comparison fair -- it happens to also keep the run inside Tavily's free tier.

Tavily's free "Researcher" plan hard-caps at 1000 searches/month with no pay-as-you-go
overflow (confirmed via /usage on 2026-07-29, after it silently zeroed out mid-Phase-7-run).
Parallel.ai's REST search endpoint is the fallback so a single provider's quota can't stall
an eval run -- plain REST, no MCP client needed, since MCP is for AI-assistant connectors,
not backend-to-backend calls.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "search"


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    score: float = 0.0


def _cache_path(query: str, k: int) -> Path:
    key = hashlib.sha256(f"{query}|{k}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def _search_tavily(query: str, k: int, api_key: str) -> list[SearchHit]:
    from tavily import TavilyClient

    raw = TavilyClient(api_key=api_key).search(
        query=query, max_results=k, search_depth="advanced"
    )
    return [
        SearchHit(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", "")[:800],
            score=float(r.get("score", 0.0)),
        )
        for r in raw.get("results", [])
    ]


def _search_parallel(query: str, k: int, api_key: str) -> list[SearchHit]:
    import httpx

    resp = httpx.post(
        "https://api.parallel.ai/v1/search",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"search_queries": [query]},
        timeout=30.0,
    )
    resp.raise_for_status()
    raw = resp.json()
    return [
        SearchHit(
            title=r.get("title") or "",
            url=r.get("url", ""),
            snippet=" ".join(r.get("excerpts", []))[:800],
        )
        for r in raw.get("results", [])[:k]
    ]


def search(
    query: str, *, k: int = 5, api_key: str | None = None,
    parallel_api_key: str | None = None, use_cache: bool = True,
) -> list[SearchHit]:
    path = _cache_path(query, k)
    if use_cache and path.exists():
        return [SearchHit(**h) for h in json.loads(path.read_text())]

    if not api_key and not parallel_api_key:
        raise RuntimeError(
            "neither TAVILY_API_KEY nor PARALLEL_API_KEY is set and the query is not cached"
        )

    hits: list[SearchHit] | None = None
    last_exc: Exception | None = None
    if api_key:
        try:
            hits = _search_tavily(query, k, api_key)
        except Exception as exc:  # quota, outage -- fall through to Parallel
            last_exc = exc
    if hits is None and parallel_api_key:
        try:
            hits = _search_parallel(query, k, parallel_api_key)
        except Exception as exc:
            last_exc = exc
    if hits is None:
        assert last_exc is not None
        raise last_exc

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(h) for h in hits], indent=2))
    return hits
