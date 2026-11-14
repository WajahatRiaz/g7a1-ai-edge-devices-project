#!/usr/bin/env bash
# Optional HF→GGUF stack (torch, transformers). Not needed for harness / llama-server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REQ="${ROOT}/third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
if [[ ! -f "$REQ" ]]; then
  echo "Missing ${REQ}. Run: git submodule update --init --depth 1" >&2
  exit 1
fi

if [[ -x .venv-convert/bin/python ]] && .venv-convert/bin/python -c 'import torch, transformers' 2>/dev/null; then
  exit 0
fi

echo "==> Creating .venv-convert (GGUF conversion only; torch + transformers)"
python3 -m venv .venv-convert
.venv-convert/bin/pip install --upgrade pip
.venv-convert/bin/pip install -r "$REQ"