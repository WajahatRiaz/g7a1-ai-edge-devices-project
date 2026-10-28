# Long Context Tool-Calling Agent

AI624 AI on Edge Devices, Fall 2026. Track A, Project A1, on an 8 GB Raspberry Pi 5.

An on-device agent completes multi-step tasks by calling tools. Every token is processed on the Pi through [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`. The model family is Qwen3 (full attention in every layer). Each turn the model emits a tool call, the tool runs locally, and the tool output is appended to the context. Sessions grow to 10k–20k tokens, and the agent still has to choose a valid tool and arguments at the end.

The tool set is 20–30 functions from the BFCL multi-turn backends, in long-context mode, and it includes TradingBot. Text only. Multi-agent orchestration and a model served from another machine are out of scope. Hybrid families such as Qwen3.5 are out of scope, because their KV cache is too small for the memory side of this project.

Latency, memory, and power are measured on the Pi. Task accuracy may be scored on a host that runs the same weights and pipeline. Until the Pi arrives, develop and debug the harness on this Linux host. Rebuild on the Pi before reporting device metrics.

## Milestones

Each change is applied on top of the last accepted configuration. Every step is measured on the same held-out split and marked kept or discarded. Discarded steps stay in the lineage.

| When | Due | Status | What has to be working |
|---|---|---|---|
| Week 1 presentation | 29 November 2026 | In progress (host) | Tool subset, trace set, and harness. Baseline characterized on the Pi. |
| Week 2 presentation | 6 December 2026 | Not started | Prompt caching. Weight quantization and KV-cache quantization. |
| Final presentation, demo, and viva | 20 December 2026 | Not started | One further compression step, one context-management technique, and a live demo on a task outside the trace set. |
| Code and report | 24 December 2026 | Not started | Documented code and the written report. |

### Week 1 checklist

Legend: `[x]` done on this Linux host · `[ ]` still open · `[P]` must be redone or measured on the Pi

**Infrastructure**

- [x] Repo layout, `setup.sh`, lean `.venv`, `.venv-convert`
- [x] Submodules: `third_party/llama.cpp`, `third_party/gorilla`
- [x] Native `llama-server` / `llama-cli` / `llama-quantize` / `llama-imatrix` build (x86 host)
- [ ] `[P]` Rebuild llama.cpp on the Pi (`./setup.sh` there)
- [ ] Download `Qwen/Qwen3-1.7B` HF weights (`./scripts/download_and_convert_baseline.sh` or `DOWNLOAD_MODEL=1 ./setup.sh`; large shard via `curl -C -`)
- [ ] Convert to `models/qwen3-1.7b-bf16.gguf` (same script does this after the download)

**Task 1 — tool subset and long session**

- [x] Fixed subset: TradingBot (20) + MessageAPI (10) = 30 tools (`configs/tool_subset.json`)
- [x] BFCL long-context tasks restricted to that subset (33 tasks)
- [x] Train / test hold-out (`data/tasks/train.jsonl`, `data/tasks/test.jsonl`)
- [x] Five project long tasks aiming at 16k–32k tokens (`data/tasks/long_extra.jsonl`)
- [x] At least two early-detail long tasks held out in the test split

**Task 2 — client harness**

- [x] `src/a1_agent` harness: llama-server client, BFCL backends, tool-call parse/validate
- [x] Logs per-turn prompt/completion tokens, prefill/decode/tool time, call validity
- [x] Baseline client sends `cache_prompt: false` (`configs/baseline.yaml`)
- [x] All tool schemas placed in the system prompt
- [ ] End-to-end smoke against a live `llama-server` (after GGUF is ready)

**Task 3 — baseline characterization**

- [ ] Serve BF16 model at 16k context on the host and run harness smoke
- [ ] Host prefill/decode sweep at ≥3 context lengths (`scripts/characterize_baseline.py`)
- [ ] `[P]` Peak memory breakdown on the Pi
- [ ] `[P]` Bandwidth-bound decode check on the Pi
- [ ] `[P]` Power / energy / thermal (`vcgencmd`) on the Pi
- [ ] `[P]` Full held-out test split on the Pi before any optimization

Week 1 baseline, before any optimization:

- Qwen3-1.7B, or the 2–4B member of the family, in BF16
- 16k context, which fits in 8 GB
- Stateless client that resends the full history every turn, with prefix caching off
- All tool schemas in the system prompt
- Unconstrained sampling and default llama.cpp build settings

`llama-server` reuses cached prompt prefixes by default. The baseline client must send `cache_prompt: false`.

Metrics on every change: session success, call validity, per-turn latency split into prefill, decode, and tool time, cold and warm time to the first call, peak memory, energy per session, and thermal state. On the Pi, power comes from `vcgencmd pmic_read_adc` (sum of voltage times current across rails), integrated over a steady-state window, with `vcgencmd measure_temp` and `vcgencmd get_throttled` logged beside it. Throttled runs are flagged. Record the llama.cpp commit, build flags, and GGUF formats with every result.

## Current tool subset

Configured in `configs/tool_subset.json`:

| Backend | Tools | Role |
|---|---|---|
| TradingBot | 20 | Required. Long-context dumps of order and transaction history grow sessions past 16k tokens. |
| MessageAPI | 10 | Second BFCL backend. Brings the set to 30 tools. |

Tasks are taken from `BFCL_v4_multi_turn_long_context.json` when `involved_classes ⊆ {TradingBot, MessageAPI}` and TradingBot is present, plus five project-authored long tasks in `data/tasks/long_extra.jsonl`. At least two of those require a detail planted early in the session.

## Requirements

Hardware for the measured results is an 8 GB Raspberry Pi 5. This repository builds on any Linux machine. A binary built on x86 will not run on the Pi, so run `./setup.sh` again on the Pi.

System packages (Debian or Ubuntu, including Raspberry Pi OS):

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
  libcurl4-openssl-dev python3-venv python3-dev
```

Python 3.10 or newer. The agent venv stays lean (requests, PyYAML, BFCL backends via submodule path). The GGUF converter pins `numpy~=2.2.6` and `torch==2.11.0`, so conversion uses a second virtualenv.

Submodules, both shallow:

| Path | Repository | Role |
|---|---|---|
| `third_party/llama.cpp` | https://github.com/ggml-org/llama.cpp | `llama-server`, quantize, importance-matrix calibration, HF to GGUF |
| `third_party/gorilla` | https://github.com/ShishirPatil/gorilla | BFCL. Only `berkeley-function-call-leaderboard` is used. |

Weights are not in git. The baseline BF16 GGUF is converted locally from [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). The official GGUF repo publishes Q8_0 only, which is not the baseline.

## Setup

Clone with the submodules:

```bash
git clone --recurse-submodules --shallow-submodules \
  git@github.com:WajahatRiaz/g7a1-ai-edge-devices-project.git
cd g7a1-ai-edge-devices-project
```

If the repo is already cloned without them:

```bash
git submodule update --init --depth 1
```

Install Python dependencies and build llama.cpp:

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` does four things by default:

1. Initializes `third_party/llama.cpp` and `third_party/gorilla` if those checkouts are missing.
2. Creates `.venv`, installs the lean harness deps, and puts the BFCL submodule on `PYTHONPATH` (no torch / sentence-transformers).
3. Creates `.venv-convert` and installs `third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt`.
4. Configures a native Release build and compiles `llama-server`, `llama-cli`, `llama-quantize`, and `llama-imatrix` into `build/llama.cpp/bin/`.

Weights are opt-in (several GB). Either:

```bash
DOWNLOAD_MODEL=1 ./setup.sh
```

or, after a normal `./setup.sh`:

```bash
./scripts/download_and_convert_baseline.sh
```

## Week 1 commands (host)

Prepare the tool subset split and the five long extra tasks:

```bash
source .venv/bin/activate
python scripts/prepare_tasks.py
```

That writes `data/tasks/train.jsonl`, `data/tasks/test.jsonl`, `data/tasks/long_extra.jsonl`, and `data/tasks/split_summary.json`.

### Download and convert the BF16 baseline

Preferred (resume-safe). Uses `huggingface_hub` for small files, then `curl -C -` for the ~3.3 GB `model-00001` shard that often stalls, then converts to GGUF:

```bash
./scripts/download_and_convert_baseline.sh
```

Manual equivalent:

```bash
mkdir -p models/hf/Qwen3-1.7B

# Small files / second shard (may partially fail on the large shard; that is OK)
.venv-convert/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-1.7B", local_dir="models/hf/Qwen3-1.7B")
PY

# Finish or resume the large shard (~3.3 GB). Safe to rerun.
curl -L --retry 50 --retry-all-errors --retry-delay 3 \
  -C - \
  -o models/hf/Qwen3-1.7B/model-00001-of-00002.safetensors \
  "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model-00001-of-00002.safetensors"

.venv-convert/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  models/hf/Qwen3-1.7B \
  --outfile models/qwen3-1.7b-bf16.gguf \
  --outtype bf16
```

Optional: `export HF_TOKEN=...` before downloading for higher Hugging Face rate limits.

Start `llama-server` in another terminal. `-c 16384` is the baseline cache. `--jinja` turns on the chat template Qwen3 tool calling needs.

```bash
./build/llama.cpp/bin/llama-server \
  -m models/qwen3-1.7b-bf16.gguf \
  -c 16384 \
  --jinja \
  -t "$(nproc)" \
  --host 127.0.0.1 \
  --port 8080
```

Run the harness. Baseline config forces `cache_prompt: false` and puts every tool schema in the system prompt.

```bash
source .venv/bin/activate
# smoke one long task
python scripts/run_harness.py --split long_extra --task-id long_extra_03_fund_after_dump
# held-out BFCL + early-detail long tasks
python scripts/run_harness.py --split test --limit 3
```

Prefill and decode sweep at several prompt lengths (host timings only; memory, energy, and thermal stay null until the Pi):

```bash
python scripts/characterize_baseline.py --lengths 512,2048,8192
```

Quantize only after the baseline is characterized:

```bash
./build/llama.cpp/bin/llama-quantize \
  models/qwen3-1.7b-bf16.gguf \
  models/qwen3-1.7b-q4_k_m.gguf \
  Q4_K_M
```

Record the pinned llama.cpp commit with every result:

```bash
git -C third_party/llama.cpp rev-parse HEAD
```

On the Pi, sample power, temperature, and throttling from a separate process while a session runs:

```bash
vcgencmd pmic_read_adc
vcgencmd measure_temp
vcgencmd get_throttled
```

## Layout

```
setup.sh                  submodule init, both virtualenvs, llama.cpp build
requirements.txt          BFCL + harness deps
configs/tool_subset.json  TradingBot + MessageAPI (30 tools)
configs/baseline.yaml     BF16, 16k, cache_prompt=false
src/a1_agent/             client, parser, BFCL backends, harness
scripts/prepare_tasks.py  BFCL subset + long extras + train/test split
scripts/run_harness.py    multi-turn agent loop against llama-server
scripts/characterize_baseline.py
                          prefill/decode sweep
scripts/download_and_convert_baseline.sh
                          HF download + curl resume + BF16 GGUF
data/tasks/               generated splits
third_party/llama.cpp     inference runtime
third_party/gorilla/...   BFCL backends and long-context tasks
build/llama.cpp/bin/      llama-server and related tools
models/                   local HF weights and GGUF files, not committed
results/                  per-run metrics, not committed
docs/                     course project description
```

`.venv/`, `.venv-convert/`, `build/`, `models/`, and `results/` are gitignored.
