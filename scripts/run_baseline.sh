#!/usr/bin/env bash
# Record the Week 1 baseline after llama-server is already running.
# Long task, held-out sample, and prefill/decode sweep, each with power and
# temperature, then refresh docs/data/dashboard.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BASE_URL="http://${HOST}:${PORT}"

if ! curl -fsS --max-time 5 "${BASE_URL}/health" >/dev/null; then
  echo "llama-server is not up at ${BASE_URL}/." >&2
  echo "Start it first and leave it running: ./scripts/start_server.sh" >&2
  exit 1
fi

echo "Server is up at ${BASE_URL}/"
echo "1/3 long task"
./scripts/sample_device.sh -- ./scripts/run_harness.sh --split long_extra --task-id long_extra_03_fund_after_dump

echo "2/3 held-out sample"
./scripts/sample_device.sh -- ./scripts/run_harness.sh --split test --limit 3

echo "3/3 prefill/decode sweep"
./scripts/sample_device.sh -- ./scripts/characterize_baseline.sh --lengths 512,2048,4096

echo "Exporting dashboard"
./scripts/export_dashboard.sh
echo "Wrote docs/data/dashboard.json"
echo "Commit that file and push to update the GitHub page."
