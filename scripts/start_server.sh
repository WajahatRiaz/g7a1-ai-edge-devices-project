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
CTX="${CTX:-16384}"
CTX_MAX="${CTX_MAX:-16384}"
if (( CTX > CTX_MAX )); then
  echo "CTX=${CTX} exceeds cap CTX_MAX=${CTX_MAX}. Course A1 baseline is 16k; 32k does not fit 8 GB." >&2
  exit 1
fi
NPROC="$(nproc)"
THREADS="${THREADS:-2}"
THREADS_BATCH="${THREADS_BATCH:-${NPROC}}"
# One slot on 8GB Pi: default `-np auto` reserves 4× n_ctx KV and slows BF16 runs.
PARALLEL="${PARALLEL:-1}"

BASELINE_CFG="${ROOT}/configs/baseline.yaml"
if [[ -z "${SERVER_TIMEOUT:-}" && -f "${BASELINE_CFG}" && -x "${ROOT}/.venv/bin/python" ]]; then
  SERVER_TIMEOUT="$("${ROOT}/.venv/bin/python" - "${BASELINE_CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
srv = (cfg.get("server") or {}).get("read_write_timeout_s")
cli = (cfg.get("client") or {}).get("timeout_s")
print(srv if srv is not None else (cli if cli is not None else 86400))
PY
)"
fi
SERVER_TIMEOUT="${SERVER_TIMEOUT:-${LLAMA_ARG_TIMEOUT:-86400}}"
export LLAMA_ARG_TIMEOUT="${SERVER_TIMEOUT}"

export LD_LIBRARY_PATH="${ROOT}/build/llama.cpp/bin:${LD_LIBRARY_PATH:-}"

echo "Model:   ${GGUF}"
echo "Listen:  http://${HOST}:${PORT}/"
echo "CTX=${CTX} parallel=${PARALLEL} threads=${THREADS} threads_batch=${THREADS_BATCH}"
echo "Timeout: ${SERVER_TIMEOUT}s (llama-server --timeout; see baseline.yaml)"
echo "Stop:    pkill -f llama-server"
exec "${SERVER}" \
  -m "${GGUF}" \
  -c "${CTX}" \
  --timeout "${SERVER_TIMEOUT}" \
  --jinja \
  --webui \
  -t "${THREADS}" \
  -tb "${THREADS_BATCH}" \
  -np "${PARALLEL}" \
  --host "${HOST}" \
  --port "${PORT}"
