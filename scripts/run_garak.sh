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
#
# 422 is in skip_codes deliberately, not to hide failures: /research caps `question` at
# 1000 chars (a real input-validation boundary, not a test artifact), and dan.DanInTheWild's
# longest prompt is 1665 chars. A 422 there means the API's own length cap rejected the
# payload before it reached any model -- that is hardening evidence, not a bug, so it is
# recorded as a skip (logged, run continues) rather than crashing the whole scan. Report the
# skip count in results/security_report.md rather than silently dropping it.
#
# 504 is in skip_codes for a different reason: /research enforces a 480s server-side
# deadline (worst-case fallback+retry chains can otherwise sum past any fixed client
# timeout), and this generator's own request_timeout (540s) is set comfortably above that so
# a slow request gets a clean 504 from the server rather than a raw client-side read
# timeout -- which garak has no way to skip and treats as a fatal crash of the whole run.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/garak

# garak pins deps that conflict with litellm, so it lives in its own venv.
GARAK="${GARAK:-.venv-garak/bin/garak}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
if [ ! -x "$GARAK" ]; then
  if command -v garak >/dev/null 2>&1; then GARAK=garak; else
    echo "garak not found. Install it into its own venv:"
    echo "  uv venv .venv-garak && uv pip install --python .venv-garak/bin/python garak"
    exit 1
  fi
fi

cat > results/garak/rest_gateway.json <<JSON
{
  "rest": {
    "RestGenerator": {
      "name": "multi-agent-gateway",
      "uri": "http://127.0.0.1:${GATEWAY_PORT}/research",
      "method": "post",
      "headers": {"Content-Type": "application/json"},
      "req_template_json_object": {"question": "\$INPUT"},
      "response_json": true,
      "response_json_field": "report",
      "request_timeout": 540,
      "ratelimit_codes": [429],
      "skip_codes": [503, 422, 504]
    }
  }
}
JSON

# Each probe here is a FULL research run -- search, fetch, synthesis, scoring -- so the
# probe set is deliberately bounded. Running the whole promptinject suite at the default
# generation count would be thousands of live provider calls and hours of wall clock for
# no extra signal. Widen it once you know the run is clean.
PROBES="${PROBES:-promptinject.HijackHateHumans,dan.DanInTheWild,encoding.InjectBase64}"
GENERATIONS="${GENERATIONS:-1}"

# garak opens its report file in "w" mode -- a fixed prefix means a second run silently
# truncates the first one's raw output (lost DanInTheWild's report.jsonl this way once
# already). Stamp the prefix with probes+time so runs never collide; results/garak/gateway
# is no longer written directly.
STAMP="$(date +%Y%m%d-%H%M%S)_$(echo "$PROBES" | tr ',' '+' | tr -cd 'a-zA-Z0-9+_')"
REPORT_PREFIX="results/garak/gateway_${STAMP}"

echo "== garak against the gateway (start the API first: gateway serve) =="
"$GARAK" --model_type rest --generator_option_file results/garak/rest_gateway.json \
      --probes "$PROBES" --generations "$GENERATIONS" \
      --report_prefix "$REPORT_PREFIX"
echo "report written to ~/.local/share/garak/garak_runs/${REPORT_PREFIX}.report.jsonl"

echo
echo "== compare against a raw provider =="
echo "Re-run with --model_type litellm --model_name gemini/gemini-2.5-flash to get the"
echo "unprotected baseline, then diff the hit rates into results/security_report.md."
