#!/usr/bin/env bash
# Score the full held-out test split on this Pi (same stack as baseline.yaml).
# Long — run overnight or one task at a time with --task-id / --limit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

exec "${ROOT}/scripts/sample_device.sh" -- "${ROOT}/scripts/run_harness.sh" --split test "$@"