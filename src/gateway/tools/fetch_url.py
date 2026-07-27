"""HTTPS fetch with SSRF-aware redirect handling.

Redirects are followed *manually* and every hop is re-validated. Following redirects
automatically is the standard way an allowlisted URL turns into a request to
``169.254.169.254``: the first hop passes the scope check, the redirect target never gets
one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from gateway.security.scope import check_url

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "pages"

MAX_BYTES = 1_000_000
MAX_REDIRECTS = 3
ALLOWED_CONTENT = ("text/html", "text/plain", "application/xhtml+xml", "application/xml")


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    n_bytes: int
    truncated: bool = False


class FetchError(RuntimeError):
    pass


def _extract(html: str, url: str) -> tuple[str, str]:
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
        meta = trafilatura.extract_metadata(html)
        title = (getattr(meta, "title", None) or url) if meta else url
    except Exception:  # extraction is best-effort; a failure must not kill the run
        text, title = "", url
    return title, text


def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"


def fetch(
    url: str, *, timeout: float = 10.0, resolve_dns: bool = True, use_cache: bool = True
) -> FetchedPage:
    """Fetch and extract a page.

    Cached on disk for the same reason searches are: the provider matrix runs every
    question through every provider, and re-downloading each page once per provider would
    be slow, rude to the sites, and would let page drift between passes turn into apparent
    quality differences between models. The scope check still runs on every call -- the
    cache short-circuits the network, never the policy.
    """
    decision = check_url(url, resolve_dns=resolve_dns)
    if not decision.allowed:
        raise FetchError(f"blocked by scope policy ({decision.rule}): {decision.reason}")

    path = _cache_path(url)
    if use_cache and path.exists():
        try:
            return FetchedPage(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass  # corrupt entry: fall through and refetch

    page = _fetch_live(url, timeout=timeout, resolve_dns=resolve_dns)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(page)))
    return page


def _fetch_live(url: str, *, timeout: float, resolve_dns: bool) -> FetchedPage:
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        for _ in range(MAX_REDIRECTS + 1):
            decision = check_url(current, resolve_dns=resolve_dns)
            if not decision.allowed:
                raise FetchError(f"blocked by scope policy ({decision.rule}): {decision.reason}")

            response = client.get(current, headers={"User-Agent": "multi-agent-gateway/0.1"})
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise FetchError("redirect without Location header")
                current = str(httpx.URL(current).join(location))
                continue

            response.raise_for_status()
            ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and not any(ctype.startswith(a) for a in ALLOWED_CONTENT):
                raise FetchError(f"content-type {ctype!r} not permitted")

            body = response.content[:MAX_BYTES]
            truncated = len(response.content) > MAX_BYTES
            title, text = _extract(body.decode(response.encoding or "utf-8", "replace"), current)
            return FetchedPage(
                url=current, title=title, text=text, n_bytes=len(body), truncated=truncated
            )
    raise FetchError(f"too many redirects (>{MAX_REDIRECTS})")
