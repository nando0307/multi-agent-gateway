#!/usr/bin/env bash
# Phase 8: model-level probing with garak, against the whole pipeline.
#
# garak is pointed at the FastAPI endpoint via its `rest` generator rather than at a bare
# model, so the probes traverse the untrusted-content envelope, the scope gate and the
# eval gate -- the thing actually deployed. Run it twice: once against the gateway and
# once against a raw provider. The delta is the hardening evidence; a single number in
# isolation says nothing.
#
# Honest scope note: garak's probes are model-level. They do not exercise the tool-abuse
# class at all -- that is what scripts/eval_injection.py measures. The two are reported
# separately in results/security_report.md rather than blended into one flattering figure.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/garak

if ! command -v garak >/dev/null 2>&1; then
  echo "garak not installed. Install into a separate venv (it pins conflicting deps):"
  echo "  uv venv .venv-garak && .venv-garak/bin/pip install garak"
  exit 1
fi

cat > results/garak/rest_gateway.json <<'JSON'
{
  "rest": {
    "RestGenerator": {
      "name": "multi-agent-gateway",
      "uri": "http://127.0.0.1:8000/research",
      "method": "post",
      "headers": {"Content-Type": "application/json"},
      "req_template_json_object": {"question": "$INPUT"},
      "response_json": true,
      "response_json_field": "report"
    }
  }
}
JSON

PROBES="promptinject,dan,encoding,leakreplay"

echo "== garak against the gateway (start the API first: gateway serve) =="
garak --model_type rest --generator_option_file results/garak/rest_gateway.json \
      --probes "$PROBES" --report_prefix results/garak/gateway

echo
echo "== compare against a raw provider =="
echo "Re-run with --model_type litellm --model_name gemini/gemini-2.5-flash to get the"
echo "unprotected baseline, then diff the hit rates into results/security_report.md."
