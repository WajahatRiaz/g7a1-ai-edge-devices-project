#!/usr/bin/env bash
# Full held-out test split for accuracy. Run this on a host with the same
# GGUF and pipeline. Do not use it for Pi timing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

exec "${ROOT}/scripts/run_harness.sh" --split test "$@"
