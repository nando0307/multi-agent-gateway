#!/usr/bin/env bash
# pip-audit with the accepted-findings list in exactly one place: CI and
# scripts/run_security_scans.sh both call this, so results/pip_audit.txt cannot disagree
# with the gate that produces it.
#
# The only accepted finding is filelock. LiteLLM (every release to date) requires
# filelock<4.0 and every fixed filelock is 4.x, so no installable version satisfies both;
# `uv lock --upgrade-package filelock` cannot move past 3.x. Delete these lines when
# LiteLLM lifts the cap. Anything else must fail the build.
set -uo pipefail
cd "$(dirname "$0")/.."

exec "${PYTHON:-python}" -m pip_audit --progress-spinner off \
  --ignore-vuln CVE-2026-112907 \
  --ignore-vuln CVE-2026-112908 \
  --ignore-vuln CVE-2026-112909 \
  --ignore-vuln CVE-2026-112910 \
  --ignore-vuln CVE-2026-112911 \
  --ignore-vuln CVE-2026-112913 \
  --ignore-vuln CVE-2026-112915 \
  --ignore-vuln CVE-2026-112935 \
  --ignore-vuln CVE-2026-112936 \
  --ignore-vuln CVE-2026-112938 \
  --ignore-vuln CVE-2026-112939 \
  --ignore-vuln CVE-2026-112940
