#!/usr/bin/env python3
"""Fold every dated run into docs/data/dashboard.json for GitHub Pages.

Reads results/baseline/history/<timestamp>/ only. Removing a history folder
and re-exporting drops that run from docs/data/dashboard.json.
"""

from __future__ import annotations

import importlib.util
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


def session_from_payload(payload: dict, split: str) -> dict:
    turns = payload.get("turns") or []
    ok_calls = sum(
        1
        for turn in turns
        for verdict in turn.get("call_validity") or []
        if verdict.get("ok")
    )
    total_calls = sum(len(turn.get("call_validity") or []) for turn in turns)
    return {
        "split": split,
        "task_id": payload.get("task_id"),
        "success": payload.get("success"),
        "error": payload.get("error"),
        "n_messages": payload.get("n_messages"),
        "calls_ok": ok_calls,
        "calls_total": total_calls,
        "turns": turns,
    }


def load_sessions_from(directory: Path) -> list[dict]:
    sessions = []
    for split in SPLITS:
        split_dir = directory / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            with open(path, encoding="utf-8") as handle:
                sessions.append(session_from_payload(json.load(handle), split))
    return sessions


def load_characterization_from(directory: Path) -> dict | None:
    path = directory / "characterization.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_sample_device():
    path = ROOT / "scripts" / "sample_device.py"
    spec = importlib.util.spec_from_file_location("sample_device", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_history_run(directory: Path, sample_device) -> dict | None:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    return {
        "id": meta.get("id") or directory.name,
        "started_at": meta.get("started_at"),
        "ended_at": meta.get("ended_at"),
        "kind": meta.get("kind") or "power",
        "command": meta.get("command") or [],
        "exit_code": meta.get("exit_code"),
        "sessions": load_sessions_from(directory),
        "characterization": load_characterization_from(directory),
        "device": sample_device.load_experiment(directory),
    }


def load_runs() -> list[dict]:
    found: dict[str, dict] = {}
    sample_device = load_sample_device()
    history = RESULTS / "history"
    if history.is_dir():
        for directory in sorted(path for path in history.iterdir() if path.is_dir()):
            run = load_history_run(directory, sample_device)
            if run is not None:
                found[run["id"]] = run
    return sorted(found.values(), key=lambda run: run.get("started_at") or run.get("id") or "")


def main() -> None:
    with open(ROOT / "configs" / "baseline.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    base_url = f"http://{host}:{port}"
    runs = load_runs()
    payload = {
        "schema": 2,
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
            "top_p": cfg["client"].get("top_p"),
            "top_k": cfg["client"].get("top_k"),
            "enable_thinking": cfg["client"].get("enable_thinking"),
            "schemas_in_system_prompt": cfg["client"].get("schemas_in_system_prompt"),
            "llama_cpp_commit": llama_cpp_commit(),
        },
        "server": server_snapshot(base_url),
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {OUT}")
    print(
        f"runs={len(runs)} " + " ".join(
            f"{kind}={sum(1 for run in runs if run.get('kind') == kind)}"
            for kind in ("long_task", "held_out", "sweep")
        )
    )


if __name__ == "__main__":
    main()
