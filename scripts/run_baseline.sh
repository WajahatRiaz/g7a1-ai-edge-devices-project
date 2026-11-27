#!/usr/bin/env bash
# Pi baseline scenarios (same stack as configs/baseline.yaml). llama-server must
# already be running. Run one scenario or all enabled steps from the YAML.
#
#   ./scripts/run_baseline.sh --only characterize
#   ./scripts/run_baseline.sh --only session --limit 1
#   Full run order: session (harness) → characterize (sweep) → export after each step
#   ./scripts/run_baseline.sh --no-export
#   ./scripts/run_baseline.sh --no-device --only session
#
# Full held-out test split on the Pi: ./scripts/run_test_accuracy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_baseline.sh [options]

Pi baseline scenarios (see configs/baseline.yaml → pi_scenarios):
  session       Tool-calling harness + optional device trace
  characterize  Prefill/decode sweep + optional device trace
  export        Refresh docs/data/dashboard.json (after each step, or alone)

Options:
  --only SCENARIO     Run only this scenario (repeatable; disables other steps)
  --session           Enable session scenario
  --no-session        Skip session scenario
  --characterize      Enable characterize scenario
  --no-characterize   Skip characterize scenario
  --export            Enable dashboard export
  --no-export         Skip dashboard export
  --git-push          Commit and push docs/data/dashboard.json after each export
  --no-git-push       Export locally only
  --no-device         Do not wrap the command in sample_device.sh
  --split SPLIT       Override session split (default from YAML)
  --limit N           Override session task limit (default from YAML, usually 1)
  --task-id ID        Run one task by id (disables limit unless you pass both)
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
RUN_GIT_PUSH=""
ONLY_PICKED=false
SPLIT_OVERRIDE=""
LIMIT_OVERRIDE=""
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
    --git-push) RUN_GIT_PUSH=true ;;
    --no-git-push) RUN_GIT_PUSH=false ;;
    --no-device) FORCE_NO_DEVICE=true ;;
    --split)
      [[ $# -ge 2 ]] || { echo "Missing value for --split" >&2; exit 2; }
      SPLIT_OVERRIDE="$2"
      shift 2
      ;;
    --limit)
      [[ $# -ge 2 ]] || { echo "Missing value for --limit" >&2; exit 2; }
      LIMIT_OVERRIDE="$2"
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

read -r SESSION_ENABLED SESSION_SAMPLE SESSION_SPLIT SESSION_LIMIT SESSION_TASK \
  CHAR_ENABLED CHAR_SAMPLE CHAR_LENGTHS EXPORT_ENABLED EXPORT_GIT_PUSH <<EOF
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
task_id = sess.get("task_id")
limit = sess.get("limit", 1)
print(
    "true" if sess.get("enabled", True) else "false",
    "true" if sess.get("sample_device", True) else "false",
    sess.get("split", "test"),
    limit if limit is not None else "",
    task_id if task_id else "none",
    "true" if char.get("enabled", True) else "false",
    "true" if char.get("sample_device", True) else "false",
    char.get("lengths", "512,2048,4096"),
    "true" if export.get("enabled", True) else "false",
    "true" if export.get("git_push", False) else "false",
)
PY
)
EOF

if [[ "${ONLY_PICKED}" != true ]]; then
  [[ -z "${RUN_SESSION}" ]] && RUN_SESSION="${SESSION_ENABLED}"
  [[ -z "${RUN_CHARACTERIZE}" ]] && RUN_CHARACTERIZE="${CHAR_ENABLED}"
  [[ -z "${RUN_EXPORT}" ]] && RUN_EXPORT="${EXPORT_ENABLED}"
  [[ -z "${RUN_GIT_PUSH}" ]] && RUN_GIT_PUSH="${EXPORT_GIT_PUSH}"
else
  [[ -z "${RUN_SESSION}" ]] && RUN_SESSION=false
  [[ -z "${RUN_CHARACTERIZE}" ]] && RUN_CHARACTERIZE=false
  [[ -z "${RUN_EXPORT}" ]] && RUN_EXPORT=false
  [[ -z "${RUN_GIT_PUSH}" ]] && RUN_GIT_PUSH=false
fi

SESSION_SPLIT="${SPLIT_OVERRIDE:-${SESSION_SPLIT}}"
SESSION_LIMIT="${LIMIT_OVERRIDE:-${SESSION_LIMIT}}"
SESSION_TASK="${TASK_OVERRIDE:-${SESSION_TASK}}"
[[ "${SESSION_TASK}" == "none" ]] && SESSION_TASK=""
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

export_dashboard() {
  ./scripts/export_dashboard.sh
  echo "Wrote docs/data/dashboard.json"
}

publish_dashboard() {
  local message="$1"
  if [[ "${RUN_GIT_PUSH}" != true ]]; then
    return 0
  fi
  if [[ "${RUN_EXPORT}" != true ]]; then
    echo "Export disabled; skipping git push" >&2
    return 0
  fi
  if ! git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Not a git repository; skipping push" >&2
    return 0
  fi
  git -C "${ROOT}" add docs/data/dashboard.json
  if git -C "${ROOT}" diff --cached --quiet; then
    echo "Dashboard unchanged; nothing to commit"
    return 0
  fi
  echo "  → commit and push ${message}"
  git -C "${ROOT}" commit -m "${message}"
  git -C "${ROOT}" push origin HEAD
}

STANDALONE_EXPORT=false
if [[ "${RUN_EXPORT}" == true && "${RUN_SESSION}" != true && "${RUN_CHARACTERIZE}" != true ]]; then
  STANDALONE_EXPORT=true
fi

step=0
total=0
[[ "${RUN_SESSION}" == true ]] && total=$((total + 1))
[[ "${RUN_CHARACTERIZE}" == true ]] && total=$((total + 1))
[[ "${STANDALONE_EXPORT}" == true ]] && total=$((total + 1))

if [[ "${total}" -eq 0 ]]; then
  echo "No scenarios selected. Use --only, --session, --characterize, or --export." >&2
  usage >&2
  exit 2
fi

if [[ "${RUN_SESSION}" == true ]]; then
  step=$((step + 1))
  session_label="${SESSION_SPLIT}"
  harness_args=(./scripts/run_harness.sh --split "${SESSION_SPLIT}")
  if [[ -n "${SESSION_TASK}" ]]; then
    session_label="${session_label} / ${SESSION_TASK}"
    harness_args+=(--task-id "${SESSION_TASK}")
  elif [[ -n "${SESSION_LIMIT}" ]]; then
    session_label="${session_label} / limit ${SESSION_LIMIT}"
    harness_args+=(--limit "${SESSION_LIMIT}")
  fi
  echo "${step}/${total} session (${session_label})"
  run_wrapped "${SESSION_SAMPLE}" "${harness_args[@]}"
  if [[ "${RUN_EXPORT}" == true ]]; then
    echo "  → export dashboard (held-out run)"
    export_dashboard
    publish_dashboard "Update dashboard after held-out harness run."
  fi
fi

if [[ "${RUN_CHARACTERIZE}" == true ]]; then
  step=$((step + 1))
  echo "${step}/${total} characterize (lengths ${CHAR_LENGTHS})"
  run_wrapped "${CHAR_SAMPLE}" \
    ./scripts/characterize_baseline.sh \
    --config "${CONFIG}" \
    --lengths "${CHAR_LENGTHS}"
  if [[ "${RUN_EXPORT}" == true ]]; then
    echo "  → export dashboard (sweep run)"
    export_dashboard
    publish_dashboard "Update dashboard after characterization sweep."
  fi
fi

if [[ "${STANDALONE_EXPORT}" == true ]]; then
  step=$((step + 1))
  echo "${step}/${total} export dashboard"
  export_dashboard
  publish_dashboard "Update dashboard from export."
fi

echo "Done. Scenarios: session=${RUN_SESSION} characterize=${RUN_CHARACTERIZE} export=${RUN_EXPORT}"
if [[ "${RUN_SESSION}" == true ]]; then
  echo "Full test split on Pi: ./scripts/run_test_accuracy.sh"
fi