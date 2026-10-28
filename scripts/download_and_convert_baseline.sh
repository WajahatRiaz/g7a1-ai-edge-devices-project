#!/usr/bin/env bash
# Download Qwen3-1.7B from Hugging Face and convert it to the BF16 GGUF baseline.
# The large safetensors shard often stalls under huggingface_hub alone, so this
# script always finishes that shard with curl -C - (resume-capable).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HF_DIR="${ROOT}/models/hf/Qwen3-1.7B"
GGUF="${ROOT}/models/qwen3-1.7b-bf16.gguf"
SHARD1="model-00001-of-00002.safetensors"
SHARD1_URL="https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/${SHARD1}"
# Published size from HF (bytes).
SHARD1_BYTES=3441185608

if [[ ! -x .venv-convert/bin/python ]]; then
  echo "Missing .venv-convert. Run ./setup.sh first." >&2
  exit 1
fi
command -v curl >/dev/null 2>&1 || {
  echo "curl is required to resume the large model shard." >&2
  exit 1
}

mkdir -p models/hf

echo "==> Downloading Qwen/Qwen3-1.7B metadata and small files"
# Ignore failure if the large shard times out; curl finishes it below.
.venv-convert/bin/python - <<'PY' || true
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="models/hf/Qwen3-1.7B",
    ignore_patterns=["*.gguf"],
)
print("snapshot_download finished (or partial)")
PY

echo "==> Ensuring ${SHARD1} with curl resume"
mkdir -p "${HF_DIR}"
if [[ -f "${HF_DIR}/${SHARD1}" ]]; then
  have="$(stat -c%s "${HF_DIR}/${SHARD1}")"
else
  have=0
fi
if [[ "${have}" -lt "${SHARD1_BYTES}" ]]; then
  echo "Have ${have} / ${SHARD1_BYTES} bytes; resuming..."
  curl -L --retry 50 --retry-all-errors --retry-delay 3 \
    -C - \
    -o "${HF_DIR}/${SHARD1}" \
    "${SHARD1_URL}"
else
  echo "${SHARD1} already complete (${have} bytes)"
fi

have="$(stat -c%s "${HF_DIR}/${SHARD1}")"
if [[ "${have}" -ne "${SHARD1_BYTES}" ]]; then
  echo "ERROR: ${SHARD1} size ${have} != expected ${SHARD1_BYTES}" >&2
  exit 1
fi

if [[ ! -f "${HF_DIR}/model-00002-of-00002.safetensors" ]]; then
  echo "Missing model-00002-of-00002.safetensors; retrying snapshot_download" >&2
  .venv-convert/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-1.7B", local_dir="models/hf/Qwen3-1.7B")
PY
fi

echo "==> Converting to BF16 GGUF: ${GGUF}"
.venv-convert/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  "${HF_DIR}" \
  --outfile "${GGUF}" \
  --outtype bf16

ls -lh "${GGUF}"
echo "Done. Serve with:"
echo "  ./build/llama.cpp/bin/llama-server -m ${GGUF} -c 16384 --jinja -t \"\$(nproc)\" --host 127.0.0.1 --port 8080"
