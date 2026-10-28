#!/usr/bin/env bash
# Initialize the llama.cpp and gorilla submodules, install Python dependencies,
# and build the llama.cpp tools used by the baseline (server, quantize, imatrix).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f third_party/llama.cpp/CMakeLists.txt || ! -f third_party/gorilla/berkeley-function-call-leaderboard/pyproject.toml ]]; then
  git submodule update --init --depth 1
fi

missing=()
command -v git >/dev/null 2>&1 || missing+=(git)
command -v cmake >/dev/null 2>&1 || missing+=(cmake)
command -v g++ >/dev/null 2>&1 || missing+=(build-essential)
command -v pkg-config >/dev/null 2>&1 || missing+=(pkg-config)
if ! pkg-config --exists libcurl 2>/dev/null; then
  missing+=(libcurl4-openssl-dev)
fi
if ! python3 -c 'import venv' >/dev/null 2>&1; then
  missing+=(python3-venv python3-dev)
fi

if ((${#missing[@]})); then
  if ! sudo -n true 2>/dev/null; then
    printf 'Install these packages, then rerun ./setup.sh:\n  sudo apt-get install -y %s\n' "${missing[*]}"
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# BFCL backends and the local harness package are imported via .pth entries
# so we avoid the full bfcl-eval dependency tree (torch, cloud SDKs, etc.).
BFCL_PATH="${ROOT}/third_party/gorilla/berkeley-function-call-leaderboard"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "${BFCL_PATH}" > "${SITE_PACKAGES}/a1_bfcl.pth"
echo "${ROOT}/src" > "${SITE_PACKAGES}/a1_agent_src.pth"
python - <<'PY'
import bfcl_eval
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot
from a1_agent.schemas import load_bfcl_tool_docs
print("bfcl_eval OK:", bfcl_eval.__file__)
print("TradingBot OK:", TradingBot)
print("tools:", len(load_bfcl_tool_docs(["TradingBot", "MessageAPI"])))
PY

# Converter pins a different numpy than a full BFCL install would, and we keep
# conversion out of .venv entirely.
python3 -m venv .venv-convert
.venv-convert/bin/pip install --upgrade pip
.venv-convert/bin/pip install -r third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

cmake -S third_party/llama.cpp -B build/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_OPENSSL=ON

cmake --build build/llama.cpp -j "$(nproc)" \
  --target llama-server llama-cli llama-quantize llama-imatrix

echo "llama-server: ${ROOT}/build/llama.cpp/bin/llama-server"
echo "Python env:  ${ROOT}/.venv"
echo "GGUF convert env: ${ROOT}/.venv-convert"
