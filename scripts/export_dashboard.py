#!/usr/bin/env python3
"""Fold results/baseline into docs/data/dashboard.json for GitHub Pages.

Reads harness session logs, the prefill/decode sweep, and any device samples
written by scripts/sample_device.py. Missing inputs stay empty so the page
can be published before the first run.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "baseline"
OUT = ROOT / "docs" / "data" / "dashboard.json"
SPLITS = ("long_extra", "test", "train")


def llama_cpp_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT / "third_party" / "llama.cpp"), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def server_snapshot(base_url: str) -> dict | None:
    try:
        response = requests.get(f"{base_url}/props", timeout=2)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None
    settings = payload.get("default_generation_settings") or {}
    model_path = payload.get("model_path") or ""
    return {
        "model_file": Path(model_path).name if model_path else None,
        "model_ftype": payload.get("model_ftype"),
        "n_ctx": settings.get("n_ctx"),
        "total_slots": payload.get("total_slots"),
        "build_info": payload.get("build_info"),
    }


def load_sessions() -> list[dict]:
    sessions = []
    for split in SPLITS:
        split_dir = RESULTS / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            turns = payload.get("turns") or []
            ok_calls = sum(
                1
                for turn in turns
                for verdict in turn.get("call_validity") or []
                if verdict.get("ok")
            )
            total_calls = sum(len(turn.get("call_validity") or []) for turn in turns)
            sessions.append(
                {
                    "split": split,
                    "task_id": payload.get("task_id"),
                    "success": payload.get("success"),
                    "error": payload.get("error"),
                    "n_messages": payload.get("n_messages"),
                    "calls_ok": ok_calls,
                    "calls_total": total_calls,
                    "turns": turns,
                }
            )
    return sessions


def load_characterization() -> dict | None:
    path = RESULTS / "characterization.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_device_runs() -> list[dict]:
    runs: list[dict] = []
    jsonl = RESULTS / "device_runs.jsonl"
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
        return runs
    latest = RESULTS / "device.json"
    if latest.is_file():
        with open(latest, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    return runs


def main() -> None:
    with open(ROOT / "configs" / "baseline.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    base_url = f"http://{host}:{port}"
    characterization = load_characterization()
    payload = {
        "schema": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": "Long Context Tool-Calling Agent",
            "course": "AI624 · AI on Edge Devices · Fall 2026",
            "track": "Track A — Project A1",
        },
        "baseline": {
            "name": cfg.get("name"),
            "model_family": cfg.get("model_family"),
            "model_size": cfg.get("model_size"),
            "gguf": Path(str(cfg.get("gguf") or "")).name or None,
            "context_size": cfg.get("context_size"),
            "threads": cfg["server"].get("threads"),
            "cache_prompt": cfg["client"].get("cache_prompt"),
            "temperature": cfg["client"].get("temperature"),
            "schemas_in_system_prompt": cfg["client"].get("schemas_in_system_prompt"),
            "llama_cpp_commit": llama_cpp_commit(),
        },
        "server": server_snapshot(base_url),
        "sessions": load_sessions(),
        "characterization": characterization,
        "device_runs": load_device_runs(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {OUT}")
    print(
        f"sessions={len(payload['sessions'])} "
        f"sweep={'yes' if characterization else 'no'} "
        f"device_runs={len(payload['device_runs'])}"
    )


if __name__ == "__main__":
    main()
