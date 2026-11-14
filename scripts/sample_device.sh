#!/usr/bin/env bash
# Record Pi power and temperature in parallel with one benchmark.
# Each invocation is archived under results/baseline/history/<timestamp>/.
#
#   ./scripts/sample_device.sh -- ./scripts/run_harness.sh --split long_extra --task-id long_extra_03_fund_after_dump
#   ./scripts/sample_device.sh -- ./scripts/run_harness.sh --split test --limit 3
#   ./scripts/sample_device.sh -- ./scripts/characterize_baseline.sh --lengths 512,2048,4096
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/sample_device.sh -- <benchmark command>" >&2
  echo "       ./scripts/sample_device.sh {start|stop|mark|status}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
if [[ "$1" == "--" ]]; then
  shift
  exec python "${ROOT}/scripts/sample_device.py" wrap -- "$@"
fi
exec python "${ROOT}/scripts/sample_device.py" "$@"
