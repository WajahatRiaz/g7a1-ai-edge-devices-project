#!/usr/bin/env bash
# Pi baseline scenarios (same stack as configs/baseline.yaml). llama-server must
# already be running. Run one scenario or all enabled steps from the YAML.
#
#   ./scripts/run_baseline.sh --only characterize
#   ./scripts/run_baseline.sh --only session --task-id long_extra_04_watchlist
#   ./scripts/run_baseline.sh --no-export
#   ./scripts/run_baseline.sh --no-device --only session
#
# Full held-out accuracy: ./scripts/run_host_accuracy.sh (host, not Pi timing).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_baseline.sh [options]

Pi baseline scenarios (see configs/baseline.yaml → pi_scenarios):
  session       Tool-calling harness + optional device trace
  characterize  Prefill/decode sweep + optional device trace
  export        Update docs/data/dashboard.json from results/history

Options:
  --only SCENARIO     Run only this scenario (repeatable; disables other steps)
  --session           Enable session scenario
  --no-session        Skip session scenario
  --characterize      Enable characterize scenario
  --no-characterize   Skip characterize scenario
  --export            Enable dashboard export
  --no-export         Skip dashboard export
  --no-device         Do not wrap the command in sample_device.sh
  --split SPLIT       Override session split (default from YAML)
  --task-id ID        Override session task id (default from YAML)
  --lengths LIST      Override characterize lengths, e.g. 512,2048,4096
  --config PATH       Baseline YAML (default: configs/baseline.yaml)
  -h, --help          Show this help

With no step flags, every pi_scenarios.*.enabled: true entry runs in order.
EOF
}

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

CONFIG="${CONFIG:-${ROOT}/configs/baseline.yaml}"
RUN_SESSION=""
RUN_CHARACTERIZE=""
RUN_EXPORT=""
ONLY_PICKED=false
SPLIT_OVERRIDE=""
TASK_OVERRIDE=""
LENGTHS_OVERRIDE=""
FORCE_NO_DEVICE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || { echo "Missing value for --only" >&2; exit 2; }
      if [[ "${ONLY_PICKED}" != true ]]; then
        ONLY_PICKED=true
        RUN_SESSION=false
        RUN_CHARACTERIZE=false
        RUN_EXPORT=false
      fi
      case "$2" in
        session) RUN_SESSION=true ;;
        characterize) RUN_CHARACTERIZE=true ;;
        export) RUN_EXPORT=true ;;
        *)
          echo "Unknown scenario: $2 (use session, characterize, or export)" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --session) RUN_SESSION=true ;;
    --no-session) RUN_SESSION=false ;;
    --characterize) RUN_CHARACTERIZE=true ;;
    --no-characterize) RUN_CHARACTERIZE=false ;;
    --export) RUN_EXPORT=true ;;
    --no-export) RUN_EXPORT=false ;;
    --no-device) FORCE_NO_DEVICE=true ;;
    --split)
      [[ $# -ge 2 ]] || { echo "Missing value for --split" >&2; exit 2; }
      SPLIT_OVERRIDE="$2"
      shift 2
      ;;
    --task-id)
      [[ $# -ge 2 ]] || { echo "Missing value for --task-id" >&2; exit 2; }
      TASK_OVERRIDE="$2"
      shift 2
      ;;
    --lengths)
      [[ $# -ge 2 ]] || { echo "Missing value for --lengths" >&2; exit 2; }
      LENGTHS_OVERRIDE="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

read -r CLIENT_TIMEOUT_S SERVER_RW_TIMEOUT_S <<EOF
$(python - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
client = cfg.get("client") or {}
server = cfg.get("server") or {}
print(
    client.get("timeout_s", 86400),
    server.get("read_write_timeout_s", client.get("timeout_s", 86400)),
)
PY
)
EOF
export A1_CLIENT_TIMEOUT_S="${A1_CLIENT_TIMEOUT_S:-${CLIENT_TIMEOUT_S}}"
export A1_COMPLETION_TIMEOUT_S="${A1_COMPLETION_TIMEOUT_S:-${CLIENT_TIMEOUT_S}}"
export LLAMA_ARG_TIMEOUT="${LLAMA_ARG_TIMEOUT:-${SERVER_RW_TIMEOUT_S}}"

read -r SESSION_ENABLED SESSION_SAMPLE SESSION_SPLIT SESSION_TASK \
  CHAR_ENABLED CHAR_SAMPLE CHAR_LENGTHS EXPORT_ENABLED <<EOF
$(python - "${CONFIG}" <<'PY'
import json, sys
import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
pi = cfg.get("pi_scenarios") or {}
sess = pi.get("session") or {}
char = pi.get("characterize") or {}
export = pi.get("export") or {}
print(
    "true" if sess.get("enabled", True) else "false",
    "true" if sess.get("sample_device", True) else "false",
  sess.get("split", "long_extra"),
  sess.get("task_id", "long_extra_04_watchlist"),
    "true" if char.get("enabled", True) else "false",
    "true" if char.get("sample_device", True) else "false",
  char.get("lengths", "512,2048,4096"),
    "true" if export.get("enabled", True) else "false",
)
PY
)
EOF

if [[ "${ONLY_PICKED}" != true ]]; then
  [[ -z "${RUN_SESSION}" ]] && RUN_SESSION="${SESSION_ENABLED}"
  [[ -z "${RUN_CHARACTERIZE}" ]] && RUN_CHARACTERIZE="${CHAR_ENABLED}"
  [[ -z "${RUN_EXPORT}" ]] && RUN_EXPORT="${EXPORT_ENABLED}"
else
  [[ -z "${RUN_SESSION}" ]] && RUN_SESSION=false
  [[ -z "${RUN_CHARACTERIZE}" ]] && RUN_CHARACTERIZE=false
  [[ -z "${RUN_EXPORT}" ]] && RUN_EXPORT=false
fi

SESSION_SPLIT="${SPLIT_OVERRIDE:-${SESSION_SPLIT}}"
SESSION_TASK="${TASK_OVERRIDE:-${SESSION_TASK}}"
CHAR_LENGTHS="${LENGTHS_OVERRIDE:-${CHAR_LENGTHS}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BASE_URL="http://${HOST}:${PORT}"

if [[ "${RUN_SESSION}" == true || "${RUN_CHARACTERIZE}" == true ]]; then
  if ! curl -fsS --max-time 5 "${BASE_URL}/health" >/dev/null; then
    echo "llama-server is not up at ${BASE_URL}/." >&2
    echo "Start it first: ./scripts/start_server.sh" >&2
    exit 1
  fi
  echo "Server is up at ${BASE_URL}/"
fi

run_wrapped() {
  local use_device="$1"
  shift
  if [[ "${FORCE_NO_DEVICE}" == true || "${use_device}" != true ]]; then
    "$@"
  else
    ./scripts/sample_device.sh -- "$@"
  fi
}

step=0
total=0
[[ "${RUN_SESSION}" == true ]] && total=$((total + 1))
[[ "${RUN_CHARACTERIZE}" == true ]] && total=$((total + 1))
[[ "${RUN_EXPORT}" == true ]] && total=$((total + 1))

if [[ "${total}" -eq 0 ]]; then
  echo "No scenarios selected. Use --only, --session, --characterize, or --export." >&2
  usage >&2
  exit 2
fi

if [[ "${RUN_SESSION}" == true ]]; then
  step=$((step + 1))
  echo "${step}/${total} session (${SESSION_SPLIT} / ${SESSION_TASK})"
  run_wrapped "${SESSION_SAMPLE}" \
    ./scripts/run_harness.sh \
    --split "${SESSION_SPLIT}" \
    --task-id "${SESSION_TASK}"
fi

if [[ "${RUN_CHARACTERIZE}" == true ]]; then
  step=$((step + 1))
  echo "${step}/${total} characterize (lengths ${CHAR_LENGTHS})"
  run_wrapped "${CHAR_SAMPLE}" \
    ./scripts/characterize_baseline.sh \
    --config "${CONFIG}" \
    --lengths "${CHAR_LENGTHS}"
fi

if [[ "${RUN_EXPORT}" == true ]]; then
  step=$((step + 1))
  echo "${step}/${total} export dashboard"
  ./scripts/export_dashboard.sh
  echo "Wrote docs/data/dashboard.json"
fi

echo "Done. Scenarios: session=${RUN_SESSION} characterize=${RUN_CHARACTERIZE} export=${RUN_EXPORT}"
if [[ "${RUN_SESSION}" == true ]]; then
  echo "Held-out accuracy (host): ./scripts/run_host_accuracy.sh"
fi