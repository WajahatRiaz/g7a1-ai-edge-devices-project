#!/usr/bin/env python3
"""
Baseline characterization helpers (Task 3).

On this host: measure prefill/decode timing and prompt sizes at several
context lengths via llama-server. Peak memory, power, and thermal state
require the Pi and are left as recorded fields to fill later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_prompt(n_tokens_approx: int) -> str:
    # Roughly 1 token ~ 4 chars for English-ish filler.
    unit = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    need = max(n_tokens_approx * 4, 16)
    return (unit * ((need // len(unit)) + 1))[:need]


def completion(base_url: str, prompt: str, cache_prompt: bool, n_predict: int = 32):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/completion", json=body, timeout=600)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    timings = payload.get("timings") or {}
    return {
        "wall_s": wall,
        "tokens_evaluated": payload.get("tokens_evaluated"),
        "tokens_predicted": payload.get("tokens_predicted"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_n": timings.get("prompt_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_n": timings.get("predicted_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--lengths",
        default="512,2048,8192",
        help="Approximate prompt token lengths to sweep",
    )
    parser.add_argument("--cache-prompt", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_url = args.base_url or (
        f"http://{cfg['server']['host']}:{cfg['server']['port']}"
    )
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    health = requests.get(f"{base_url}/health", timeout=5)
    health.raise_for_status()

    rows = []
    for n in lengths:
        prompt = make_prompt(n)
        print(f"length~{n} cache_prompt={args.cache_prompt} ...", flush=True)
        row = completion(base_url, prompt, cache_prompt=args.cache_prompt)
        row["requested_tokens_approx"] = n
        row["cache_prompt"] = args.cache_prompt
        # Pi-only fields left null on the host.
        row["peak_memory_bytes"] = None
        row["energy_j"] = None
        row["temp_c"] = None
        row["throttled"] = None
        rows.append(row)
        print(json.dumps(row, indent=2))

    out = ROOT / cfg["metrics_dir"] / "characterization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "host_note": cfg.get("host_note"),
                "cache_prompt": args.cache_prompt,
                "context_size": cfg["context_size"],
                "measurements": rows,
            },
            f,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
