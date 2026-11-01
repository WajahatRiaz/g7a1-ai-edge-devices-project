#!/usr/bin/env bash
# Single-turn chat via llama-cli (not multi-turn; use start_server.sh + browser for that).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLI="${ROOT}/build/llama.cpp/bin/llama-cli"
if [[ ! -x "${CLI}" ]]; then
  echo "Missing ${CLI}. Run ./setup.sh first." >&2
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
  echo "No GGUF found. Set MODEL=path/to/model.gguf" >&2
  exit 1
fi

N_PREDICT="${N_PREDICT:-512}"
CTX="${CTX:-8192}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"your prompt here\"" >&2
  echo "Optional env: MODEL, CTX, N_PREDICT" >&2
  exit 1
fi

PROMPT="$*"
export LD_LIBRARY_PATH="${ROOT}/build/llama.cpp/bin:${LD_LIBRARY_PATH:-}"

exec "${CLI}" \
  -m "${GGUF}" \
  -c "${CTX}" \
  --jinja \
  -p "${PROMPT}" \
  -n "${N_PREDICT}"
