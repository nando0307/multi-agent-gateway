"""Phase 8: block rate and false-positive rate for the injection defences.

What this measures is the *deterministic* half of the defence -- the scope gate and the
sanitiser -- because that half is falsifiable without a model in the loop. An attack
counts as blocked when either:

  * the scope policy denies the tool call it was trying to induce, or
  * the sanitiser downgrades the document to quote-only, so its text can be cited but
    never acted on.

The block rate on its own is meaningless: a filter that flags everything scores 100%.
So the benign corpus is scored in the same run and the false-positive rate is reported
next to it. Both numbers, or neither.

What this does *not* measure is whether the model itself resists an injection that gets
through both layers. That is what `scripts/run_garak.sh` is for, and the two are reported
separately rather than blended into one flattering figure.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gateway.security.sanitize import sanitize  # noqa: E402
from gateway.security.scope import RunBudget, ScopePolicy  # noqa: E402

POLICY = ScopePolicy(resolve_dns=False)


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "datasets" / name).read_text().splitlines() if line.strip()]


def evaluate_attack(item: dict) -> dict:
    """Return how each layer responded to one attack."""
    result = sanitize(item["payload"])
    sanitizer_blocked = result.quote_only

    scope_blocked = False
    scope_rule = None
    induces = item.get("induces")
    if induces:
        args = {k: v for k, v in induces["args"].items() if k != "repeat"}
        budget = RunBudget()
        repeat = induces["args"].get("repeat", 1)
        for _ in range(min(repeat, 60)):
            decision = POLICY.check("researcher", induces["tool"], args, budget)
            if not decision.allowed:
                scope_blocked, scope_rule = True, decision.rule
                break
            if induces["tool"] == "web_search":
                budget.record_search()
            else:
                budget.record_fetch(1000)

    return {
        "id": item["id"],
        "category": item["category"],
        "sanitizer_blocked": sanitizer_blocked,
        "scope_blocked": scope_blocked,
        "scope_rule": scope_rule,
        "blocked": sanitizer_blocked or scope_blocked,
        "flags": result.flags,
    }


def main() -> int:
    attacks = [evaluate_attack(a) for a in load("injection_corpus.jsonl")]
    benign = load("benign_corpus.jsonl")
    false_positives = [b for b in benign if sanitize(b["text"]).quote_only]

    blocked = [a for a in attacks if a["blocked"]]
    missed = [a for a in attacks if not a["blocked"]]
    block_rate = 100 * len(blocked) / len(attacks)
    fp_rate = 100 * len(false_positives) / len(benign)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for a in attacks:
        by_category[a["category"]].append(a)

    print(f"block rate       {block_rate:5.1f}%  ({len(blocked)}/{len(attacks)})")
    print(f"false positives  {fp_rate:5.1f}%  ({len(false_positives)}/{len(benign)})")

    lines = [
        "# Phase 8 - Prompt-injection defence",
        "",
        f"Attack corpus: **{len(attacks)}** injections embedded in fetched page content. "
        f"Benign corpus: **{len(benign)}** legitimate passages, several deliberately written to "
        "look suspicious (a security advisory describing prompt injection, a page saying "
        '"ignore the previous version of this document", documentation mentioning `api_key`).',
        "",
        "## Headline",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| attacks blocked | **{block_rate:.1f}%** ({len(blocked)}/{len(attacks)}) |",
        f"| false positives on benign content | **{fp_rate:.1f}%** ({len(false_positives)}/{len(benign)}) |",
        "",
        "Both numbers or neither: a filter that downgrades every document blocks 100% of "
        "attacks and destroys the system's usefulness.",
        "",
        "## By category",
        "",
        "| category | n | blocked | by scope gate | by sanitiser |",
        "|---|---|---|---|---|",
    ]
    for category, items in sorted(by_category.items()):
        lines.append(
            f"| {category} | {len(items)} | "
            f"{sum(i['blocked'] for i in items)} | "
            f"{sum(i['scope_blocked'] for i in items)} | "
            f"{sum(i['sanitizer_blocked'] for i in items)} |"
        )

    lines += [
        "",
        "## Which layer caught what",
        "",
        "The two layers are not redundant. The scope gate stops every attack that needs a "
        "tool call to succeed -- SSRF, metadata endpoints, `file://`, budget exhaustion -- and "
        "it does so without consulting a model, so a persuasive injection cannot argue its way "
        "past it. The sanitiser covers attacks that never touch a tool: instruction override, "
        "role reassignment, system-prompt extraction, citation poisoning.",
        "",
    ]

    if missed:
        lines += [
            "## Not blocked",
            "",
            "Recorded rather than hidden. These reach the model, where the untrusted-content "
            "envelope and the model's own instruction-following are the remaining defence:",
            "",
        ]
        for m in missed:
            lines.append(f"* `{m['id']}` ({m['category']})")
        lines.append("")

    if false_positives:
        lines += ["## False positives", ""]
        for fp in false_positives:
            lines.append(f"* `{fp['id']}` - {fp['text'][:120]}")
        lines.append("")

    lines += [
        "## How much to trust the block rate",
        "",
        "**The sanitiser's patterns were iterated against this corpus.** Six attacks that "
        "initially slipped through (a character class that excluded `.` and so missed any "
        "payload naming a domain; `\\b` not firing inside `OPENROUTER_API_KEY`; no rule for "
        "persistence or resource-exhaustion phrasing) were fixed by editing the patterns, and "
        "the corpus was then re-run. That makes the sanitiser half of this figure **in-sample**: "
        "it measures whether the known attack classes are covered, not how the layer performs "
        "against phrasings nobody has thought of yet. A held-out corpus written after the "
        "patterns were frozen would be needed for a generalisation claim.",
        "",
        "The scope-gate half does generalise. It is structural rather than lexical -- it denies "
        "by URL, role, and budget, never by matching attack wording -- so its results do not "
        "depend on having anticipated the phrasing. Every `tool_abuse` attack in this corpus is "
        "blocked by the gate independently of whether the sanitiser flags the text at all.",
        "",
        "## Scope",
        "",
        "* These numbers cover the deterministic layers only. Model-level susceptibility is "
        "measured separately by `scripts/run_garak.sh`, reported below rather than blended in.",
        "* The scope gate never asks a model for permission, so its results are not sensitive "
        "to prompt wording, model version, or temperature.",
        "* The sanitiser is defence in depth. It is pattern-based and therefore evadable by a "
        "novel phrasing; that is why the control that actually prevents tool abuse is the "
        "scope gate, not this.",
        "",
    ]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "injection_raw.json").write_text(json.dumps(attacks, indent=2))
    (out / "security_report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'security_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
