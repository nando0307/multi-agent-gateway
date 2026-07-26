"""Researcher: the only role that touches the network.

Every search and fetch goes through ``ToolRunner``, which consults ``ScopePolicy`` first.
Fetched text is sanitised and framed as untrusted data before it can reach any model
context. A denied tool call is a normal, traced outcome -- not an error that aborts the run.
"""

from __future__ import annotations

from gateway.agents.base import Evidence
from gateway.security.sanitize import sanitize
from gateway.llm.events import ToolCallRecord, get_trace
from gateway.tools.registry import ToolDenied, ToolRunner


def _record_fetch_failure(url: str, exc: Exception) -> None:
    trace = get_trace()
    if trace is not None:
        trace.tool_calls.append(
            ToolCallRecord(
                tool="fetch_url", role="researcher", allowed=True,
                reason=f"fetch failed: {type(exc).__name__}", args={"url": url}, url=None,
            )
        )


def _record_search_failure(query: str, exc: Exception) -> None:
    trace = get_trace()
    if trace is not None:
        trace.tool_calls.append(
            ToolCallRecord(
                tool="web_search", role="researcher", allowed=True,
                reason=f"search failed: {type(exc).__name__}", args={"query": query[:200]},
            )
        )


def research(
    sub_questions: list[str],
    runner: ToolRunner,
    *,
    hits_per_question: int = 3,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    seen_urls: set[str] = set()
    next_id = 1

    for sub_question in sub_questions:
        try:
            hits = runner.call("web_search", query=sub_question)
        except ToolDenied:
            continue
        except Exception as exc:
            # Search outages (quota exhausted, provider down) degrade the run to whatever
            # other sub-questions return. Crashing here would turn a partial answer into
            # no answer, and the eval gate already catches the resulting thin report.
            _record_search_failure(sub_question, exc)
            continue

        for hit in list(hits)[:hits_per_question]:
            url = getattr(hit, "url", None) or (hit.get("url") if isinstance(hit, dict) else None)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            try:
                page = runner.call("fetch_url", url=url)
            except ToolDenied:
                continue
            except Exception as exc:
                # A single dead link must not take down the run -- but swallowing the
                # reason silently would hide a systematic fetch failure behind a merely
                # thin report, so it goes on the trace.
                _record_fetch_failure(url, exc)
                continue

            body = getattr(page, "text", "") or ""
            if not body.strip():
                continue
            clean = sanitize(body)
            evidence.append(
                Evidence(
                    source_id=next_id,
                    url=getattr(page, "url", url),
                    title=getattr(page, "title", url),
                    text=clean.text,
                    sub_question=sub_question,
                    quote_only=clean.quote_only,
                    flags=clean.flags,
                )
            )
            next_id += 1

    return evidence
