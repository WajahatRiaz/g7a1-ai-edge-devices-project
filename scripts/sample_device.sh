#!/usr/bin/env bash
# Sample memory (and Pi power/thermal when vcgencmd exists) while a command runs.
# Usage: ./scripts/sample_device.sh -- ./scripts/characterize_baseline.sh --lengths 512,2048,4096
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
exec python "${ROOT}/scripts/sample_device.py" "$@"
