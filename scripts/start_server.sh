#!/usr/bin/env bash
# Start llama-server with the Web UI for interactive chat in a browser.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SERVER="${ROOT}/build/llama.cpp/bin/llama-server"
if [[ ! -x "${SERVER}" ]]; then
  echo "Missing ${SERVER}. Run ./setup.sh first." >&2
  exit 1
fi

BF16="${ROOT}/models/qwen3-1.7b-bf16.gguf"
Q8="${ROOT}/models/qwen3-1.7b-q8_0.gguf"
if [[ -n "${MODEL:-}" ]]; then
  GGUF="${MODEL}"
elif [[ -f "${BF16}" ]]; then
  GGUF="${BF16}"
elif [[ -f "${Q8}" ]]; then
  GGUF="${Q8}"
else
  echo "No GGUF found. Set MODEL=path/to/model.gguf or run:" >&2
  echo "  ./scripts/download_and_convert_baseline.sh" >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"
NPROC="$(nproc)"
THREADS="${THREADS:-2}"
THREADS_BATCH="${THREADS_BATCH:-${NPROC}}"
# One slot on 8GB Pi: default `-np auto` reserves 4×8192 KV and slows BF16 runs.
PARALLEL="${PARALLEL:-1}"

export LD_LIBRARY_PATH="${ROOT}/build/llama.cpp/bin:${LD_LIBRARY_PATH:-}"

echo "Model:   ${GGUF}"
echo "Listen:  http://${HOST}:${PORT}/"
echo "CTX=${CTX} parallel=${PARALLEL} threads=${THREADS} threads_batch=${THREADS_BATCH}"
echo "Stop:    pkill -f llama-server"
exec "${SERVER}" \
  -m "${GGUF}" \
  -c "${CTX}" \
  --jinja \
  --webui \
  -t "${THREADS}" \
  -tb "${THREADS_BATCH}" \
  -np "${PARALLEL}" \
  --host "${HOST}" \
  --port "${PORT}"
