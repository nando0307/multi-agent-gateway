"""Tavily-backed web search with an on-disk cache.

The cache is not only a cost control. Phase 7 runs the same questions through every
provider, and if each provider sees different search results the matrix measures search
variance rather than model quality. Sharing one cache across providers is what makes that
comparison fair -- it happens to also keep the run inside Tavily's free tier.
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


def search(query: str, *, k: int = 5, api_key: str | None = None, use_cache: bool = True) -> list[SearchHit]:
    path = _cache_path(query, k)
    if use_cache and path.exists():
        return [SearchHit(**h) for h in json.loads(path.read_text())]

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set and the query is not cached")

    from tavily import TavilyClient

    raw = TavilyClient(api_key=api_key).search(
        query=query, max_results=k, search_depth="advanced"
    )
    hits = [
        SearchHit(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", "")[:800],
            score=float(r.get("score", 0.0)),
        )
        for r in raw.get("results", [])
    ]
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(h) for h in hits], indent=2))
    return hits
