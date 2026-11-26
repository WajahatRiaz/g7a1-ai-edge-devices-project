#!/usr/bin/env bash
# Pi baseline after llama-server is already running.
# One short session plus a prefill/decode sweep, together around 10k tokens.
# Longer sessions stay out of this run: a stateless BF16 history of ~36k tokens
# at ~1.4 tok/s is about 7 hours, and that is what later experiments should fix.
# Full held-out accuracy is ./scripts/run_host_accuracy.sh on a host.
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
echo "Pi budget: about 10k tokens processed. Sessions well above that are prohibitively long for this BF16 stateless baseline."
echo "1/2 short TradingBot session"
./scripts/sample_device.sh -- ./scripts/run_harness.sh --split long_extra --task-id long_extra_04_watchlist

echo "2/2 prefill/decode sweep (512 + 2048 + 4096 prompt tokens)"
./scripts/sample_device.sh -- ./scripts/characterize_baseline.sh --lengths 512,2048,4096

echo "Exporting dashboard"
./scripts/export_dashboard.sh
echo "Wrote docs/data/dashboard.json"
echo "Anything that processes much more than 10k tokens (the fund-after-dump session reached ~36k at ~1.4 tok/s, about 7 hours) is left for later: FP16, prompt caching, then quantization."
echo "Score the full test split on a host with the same weights: ./scripts/run_host_accuracy.sh"
