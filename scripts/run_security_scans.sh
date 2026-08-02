#!/usr/bin/env bash
# Phase 8: static analysis and dependency audit.
# bandit -ll reports medium/high confidence issues only; low-confidence noise on a
# codebase this size drowns the signal.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

echo "== bandit =="
.venv/bin/bandit -r src/ -ll -f screen | tee results/bandit.txt
bandit_status=${PIPESTATUS[0]}

echo
echo "== pip-audit =="
.venv/bin/pip-audit --progress-spinner off 2>&1 | tee results/pip_audit.txt
# Captured the same way as bandit's, rather than discarded with `|| true`. There is no
# `set -e` here, so the old `|| true` bought nothing except throwing the status away --
# which made results/pip_audit.txt evidence of a scan that could never fail.
audit_status=${PIPESTATUS[0]}

echo
echo "== secret scan of tracked files =="
# Catches a key committed by accident. .env is gitignored; this checks it stayed that way.
#
# Three paths legitimately contain key-shaped strings and are excluded by path, not by
# weakening the pattern: the redaction module (its patterns ARE key shapes), its test
# (sentinel keys are the point), and the attack corpus (payloads quote fake keys).
if git ls-files -z \
   | grep -zZv -e '^src/gateway/security/redaction.py$' \
                -e '^tests/test_redaction.py$' \
                -e '^scripts/run_security_scans.sh$' \
                -e '^.github/workflows/ci.yml$' \
                -e '^datasets/' \
   | xargs -0 grep -nEI '(sk-ant-|sk-or-v1-|nvapi-|AIza[0-9A-Za-z_-]{20,}|tvly-)' ; then
  echo "SECRET-LIKE STRING FOUND IN A TRACKED FILE"; exit 1
else
  echo "clean: no key-shaped strings in tracked files"
fi

# Both scans run to completion before either can fail the script -- a bandit finding must
# not hide a pip-audit finding. bandit's status wins when both are non-zero.
exit $(( bandit_status != 0 ? bandit_status : audit_status ))
