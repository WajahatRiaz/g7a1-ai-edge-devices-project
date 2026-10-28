#!/usr/bin/env python3
"""Run the agent harness on a task JSONL split against llama-server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(
    0,
    str(ROOT / "third_party" / "gorilla" / "berkeley-function-call-leaderboard"),
)

from a1_agent.client import LlamaServerClient
from a1_agent.harness import AgentHarness, write_session_log


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument(
        "--split",
        choices=["train", "test", "long_extra"],
        default="test",
    )
    parser.add_argument("--task-id", default=None, help="Run a single task id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / cfg["tool_subset"], encoding="utf-8") as f:
        subset = json.load(f)

    tasks_path = ROOT / cfg["tasks"][args.split]
    tasks = load_jsonl(tasks_path)
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"No tasks selected from {tasks_path}")

    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    base_url = args.base_url or f"http://{host}:{port}"
    client_cfg = cfg["client"]
    client = LlamaServerClient(
        base_url=base_url,
        cache_prompt=bool(client_cfg["cache_prompt"]),
        temperature=float(client_cfg["temperature"]),
        top_p=float(client_cfg["top_p"]),
        max_tokens=int(client_cfg["max_tokens"]),
    )
    if not client.health():
        raise SystemExit(
            f"llama-server not healthy at {base_url}. Start it first "
            "(see README Commands)."
        )

    harness = AgentHarness(
        client,
        subset["backends"],
        long_context=True,
        put_schemas_in_system=bool(client_cfg["schemas_in_system_prompt"]),
        pass_tools_api=False,
    )

    out_dir = ROOT / cfg["metrics_dir"] / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in tasks:
        print(f"running {task['id']} ...", flush=True)
        result = harness.run_task(task)
        write_session_log(result, out_dir / f"{task['id']}.json")
        ok_calls = sum(
            1
            for turn in result.turns
            for v in turn.call_validity
            if v.get("ok")
        )
        total_calls = sum(len(turn.call_validity) for turn in result.turns)
        summary.append(
            {
                "task_id": result.task_id,
                "success": result.success,
                "error": result.error,
                "n_turns": len(result.turns),
                "call_validity": f"{ok_calls}/{total_calls}",
            }
        )
        print(
            f"  success={result.success} calls={ok_calls}/{total_calls} "
            f"error={result.error}",
            flush=True,
        )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
