"""Planner: decompose a research question into sub-questions.

Has **no tools** (see ``ROLE_TOOLS``). It only ever sees the user's question, never
retrieved content, so there is no untrusted text in its context at all.
"""

from __future__ import annotations

from gateway.agents.base import parse_json
from gateway.llm.router import Gateway
from gateway.llm.model_list import PRIMARY_ALIAS

SYSTEM = """You plan web research. Break the user's question into 3-5 sub-questions that
together cover it: different facets, not restatements. Each must be independently
searchable. Prefer sub-questions whose answers are concrete (numbers, dates, named
organisations) over ones inviting general commentary.

Reply with a JSON array of strings and nothing else."""


def plan(question: str, gateway: Gateway, *, model: str = PRIMARY_ALIAS, max_sub: int = 5) -> list[str]:
    result = gateway.complete(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        model=model,
        temperature=0.2,
    )
    parsed = parse_json(result.text, default=None)
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), None)
    if not isinstance(parsed, list) or not parsed:
        # Degrade rather than fail: a single sub-question still produces a usable run,
        # and the eval gate will mark the thin result rather than the pipeline crashing.
        return [question]
    return [str(s).strip() for s in parsed if str(s).strip()][:max_sub]
