#!/usr/bin/env python3
"""
Baseline characterization helpers (Task 3).

Measure prefill and decode at several prompt lengths via llama-server.
On the Pi, scripts/sample_device.sh records power, temperature, and memory (default every 10 min)
around this sweep.
"""

from __future__ import annotations

import argparse
import json
import os
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


def request_timeout_s(cfg: dict | None = None) -> float:
    raw = os.environ.get("A1_COMPLETION_TIMEOUT_S")
    if raw:
        return float(raw)
    if cfg:
        client = cfg.get("client") or {}
        server = cfg.get("server") or {}
        if client.get("timeout_s") is not None:
            return float(client["timeout_s"])
        if server.get("read_write_timeout_s") is not None:
            return float(server["read_write_timeout_s"])
    return 86400.0


def completion(
    base_url: str,
    prompt: str,
    cache_prompt: bool,
    n_predict: int = 32,
    *,
    timeout_s: float,
):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/completion", json=body, timeout=timeout_s)
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
        default="512,2048,4096",
        help="Approximate prompt token lengths to sweep (stay under server -c)",
    )
    parser.add_argument("--cache-prompt", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_url = args.base_url or (
        f"http://{cfg['server']['host']}:{cfg['server']['port']}"
    )
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    http_timeout_s = request_timeout_s(cfg)

    health = requests.get(f"{base_url}/health", timeout=5)
    health.raise_for_status()

    rows = []
    for n in lengths:
        prompt = make_prompt(n)
        print(f"length~{n} cache_prompt={args.cache_prompt} ...", flush=True)
        row = completion(
            base_url,
            prompt,
            cache_prompt=args.cache_prompt,
            timeout_s=http_timeout_s,
        )
        row["requested_tokens_approx"] = n
        row["cache_prompt"] = args.cache_prompt
        # Filled when characterize runs inside sample_device.sh on the Pi.
        row["peak_memory_bytes"] = None
        row["energy_j"] = None
        row["temp_c"] = None
        row["throttled"] = None
        rows.append(row)
        print(json.dumps(row, indent=2))

    metrics_root = (
        Path(os.environ["A1_RESULTS_DIR"])
        if os.environ.get("A1_RESULTS_DIR")
        else ROOT / cfg["metrics_dir"]
    )
    out = metrics_root / "characterization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pi_note": cfg.get("pi_note"),
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
