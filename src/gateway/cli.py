"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gateway.agents.orchestrator import build_runner, run_research
from gateway.eval.gate import QualityGate
from gateway.eval.judge import Judge, JudgeUnavailable
from gateway.llm.router import AllProvidersExhausted, build_gateway
from gateway.security.redaction import register_settings
from gateway.settings import get_settings

console = Console()


def cmd_research(args) -> int:
    settings = get_settings()
    register_settings(settings)

    judge = None
    if not args.no_judge:
        try:
            judge = Judge(settings)
        except JudgeUnavailable as exc:
            console.print(f"[yellow]judge unavailable:[/] {exc}")
            console.print("[yellow]continuing with deterministic scoring only[/]\n")

    gateway = build_gateway(settings=settings)
    console.print(f"[dim]chain:[/] {' -> '.join(gateway.chain)}\n")

    runner = build_runner(tavily_key=settings.tavily_api_key, parallel_key=settings.parallel_api_key)
    gate = QualityGate(args.threshold or settings.gate_threshold, judge)

    with console.status("researching..."):
        try:
            result, trace = run_research(args.question, gateway, runner, gate=gate)
        except AllProvidersExhausted as exc:
            console.print(f"[red]every provider failed:[/] {exc}")
            return 1

    console.print(Panel(result.report, title=args.question, border_style="cyan"))

    if result.evidence:
        console.print("\n[bold]Sources[/]")
        for e in result.evidence:
            mark = " [yellow](quote-only: flagged)[/]" if e.quote_only else ""
            console.print(f"  [{e.source_id}] {e.title} - [dim]{e.url}[/]{mark}")

    table = Table(title="\nRun", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("served by", ", ".join(result.served_by) or "-")
    table.add_row("max fallback depth", str(trace.max_fallback_depth))
    table.add_row("llm calls", str(len(trace.llm_calls)))
    table.add_row("tool calls", f"{len(trace.tool_calls)} ({sum(not t.allowed for t in trace.tool_calls)} denied)")
    if result.scores:
        for key in ("citation", "depth", "coherence", "composite"):
            table.add_row(key, f"{result.scores[key]:.3f}")
        table.add_row("gate", "[green]PASS[/]" if result.gate_passed else "[red]FAIL[/]")
    console.print(table)

    if result.warning:
        console.print(Panel(result.warning, border_style="red", title="quality gate"))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("gateway.api:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_providers(args) -> int:
    settings = get_settings()
    gateway = build_gateway(settings=settings)
    table = Table(title="Fallback chain")
    table.add_column("#")
    table.add_column("provider")
    table.add_column("model")
    for i, name in enumerate(gateway.chain):
        entry = next(e for e in gateway.model_list if e["model_name"] == name)
        table.add_row(str(i), name, entry["litellm_params"]["model"])
    console.print(table)
    console.print(f"\nN = {len(gateway.chain)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="gateway", description="Multi-agent research gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("research", help="run a research question")
    p.add_argument("question")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--no-judge", action="store_true")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("providers", help="show the configured fallback chain")
    p.set_defaults(func=cmd_providers)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
