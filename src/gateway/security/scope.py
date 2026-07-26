"""Pre-tool-call scope validation -- PLAN.md Phase 4.

Every tool call passes through ``ScopePolicy.check()`` *before* it executes, and the
default is deny. This is the layer the resume bullet refers to, and it is deliberately
independent of the model: prompt injection works by convincing the model to do something,
so the control that stops it must not itself be a prompt.

Threat model in scope here:
  * injected instructions in fetched page content telling the agent to call a tool
  * SSRF via cloud metadata endpoints, loopback, and private ranges
  * exfiltration through a tool argument (credentials appended to a URL)
  * runaway loops driven by injected "keep searching" instructions -> per-run budgets

Explicitly *not* in scope: anything the model says without calling a tool. That is what
the eval gate and the sanitizer handle.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

#: Which roles may call which tools. The synthesizer is the interesting entry: it writes
#: the final report from evidence and has *no* tools at all, so injected text that reaches
#: the writing step has nothing to trigger.
ROLE_TOOLS: dict[str, frozenset[str]] = {
    "planner": frozenset(),
    "researcher": frozenset({"web_search", "fetch_url"}),
    "synthesizer": frozenset(),
}

DENY_HOSTS = {
    "localhost",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
    "169.254.169.254",
}

DENY_HOST_SUFFIXES = (".local", ".internal", ".localdomain")

ALLOWED_SCHEMES = frozenset({"https"})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: str = "ok"
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


ALLOW = Decision(True)


@dataclass
class RunBudget:
    """Per-run ceilings. Injected instructions cannot spend more than this."""

    max_searches: int = 12
    max_fetches: int = 25
    max_bytes: int = 200_000
    max_query_chars: int = 400

    searches: int = 0
    fetches: int = 0
    bytes_fetched: int = 0
    denials: list[str] = field(default_factory=list)

    def record_search(self) -> None:
        self.searches += 1

    def record_fetch(self, nbytes: int) -> None:
        self.fetches += 1
        self.bytes_fetched += nbytes


def _is_private(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def resolved_addresses(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    return [i[4][0] for i in infos]


def check_url(url: str, *, resolve_dns: bool = True) -> Decision:
    """Validate a URL before any connection is opened.

    ``resolve_dns`` is on in production so a hostname pointing at 169.254.169.254 is
    caught. It is off in unit tests, which must not depend on a resolver. The fetch layer
    re-validates the address it actually connected to, so a DNS-rebind between this check
    and the connection is still caught.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return Decision(False, "url_unparseable", f"cannot parse {url!r}")

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return Decision(
            False, "scheme_not_allowed", f"scheme {parsed.scheme!r} not in {sorted(ALLOWED_SCHEMES)}"
        )

    host = (parsed.hostname or "").lower()
    if not host:
        return Decision(False, "no_host", "URL has no host")
    if host in DENY_HOSTS or host.endswith(DENY_HOST_SUFFIXES):
        return Decision(False, "host_denylisted", f"host {host!r} is denylisted")
    if _is_private(host):
        return Decision(False, "private_address", f"{host} is a private/loopback address")

    if resolve_dns:
        addrs = resolved_addresses(host)
        if not addrs:
            return Decision(False, "dns_failure", f"cannot resolve {host!r}")
        for addr in addrs:
            if _is_private(addr):
                return Decision(
                    False, "private_address", f"{host} resolves to private address {addr}"
                )
    return ALLOW


class ScopePolicy:
    """Deny by default. An unknown tool is not a bug to handle later, it is a denial."""

    def __init__(self, role_tools: dict[str, frozenset[str]] | None = None, *, resolve_dns: bool = True):
        self.role_tools = role_tools or ROLE_TOOLS
        self.resolve_dns = resolve_dns

    def check(self, role: str, tool: str, args: dict, budget: RunBudget) -> Decision:
        allowed_tools = self.role_tools.get(role)
        if allowed_tools is None:
            return Decision(False, "unknown_role", f"role {role!r} has no policy")
        if tool not in allowed_tools:
            return Decision(
                False, "tool_not_permitted", f"role {role!r} may not call {tool!r}"
            )

        if tool == "web_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return Decision(False, "bad_args", "query must be a non-empty string")
            if len(query) > budget.max_query_chars:
                return Decision(
                    False, "query_too_long", f"query is {len(query)} chars, max {budget.max_query_chars}"
                )
            if budget.searches >= budget.max_searches:
                return Decision(
                    False, "budget_searches", f"search budget exhausted ({budget.max_searches})"
                )
            return ALLOW

        if tool == "fetch_url":
            url = args.get("url")
            if not isinstance(url, str) or not url.strip():
                return Decision(False, "bad_args", "url must be a non-empty string")
            if budget.fetches >= budget.max_fetches:
                return Decision(
                    False, "budget_fetches", f"fetch budget exhausted ({budget.max_fetches})"
                )
            if budget.bytes_fetched >= budget.max_bytes:
                return Decision(
                    False, "budget_bytes", f"byte budget exhausted ({budget.max_bytes})"
                )
            return check_url(url, resolve_dns=self.resolve_dns)

        return Decision(False, "unknown_tool", f"no policy for tool {tool!r}")
