#!/usr/bin/env bash
# Publish results/baseline into docs/data/dashboard.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
exec python "${ROOT}/scripts/export_dashboard.py" "$@"
