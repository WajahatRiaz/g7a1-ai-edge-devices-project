"""Load BFCL tool schemas for the fixed backend subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

BFCL_DATA = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "gorilla"
    / "berkeley-function-call-leaderboard"
    / "bfcl_eval"
    / "data"
)


def load_tool_subset_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_bfcl_tool_docs(backends: list[str]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    doc_dir = BFCL_DATA / "multi_turn_func_doc"
    for backend in backends:
        filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING[backend]
        path = doc_dir / filename
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    return docs


def tools_as_openai(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert BFCL function docs to OpenAI-style tool definitions."""
    tools = []
    for doc in docs:
        params = doc.get("parameters", {"type": "object", "properties": {}, "required": []})
        # BFCL uses "dict"; OpenAI/JSON Schema expects "object".
        if params.get("type") == "dict":
            params = {**params, "type": "object"}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": doc["name"],
                    "description": doc.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools


def schemas_system_prompt(docs: list[dict[str, Any]]) -> str:
    """Baseline: every tool schema lives in the system prompt."""
    lines = [
        "You are a tool-calling agent. Solve the user request by calling tools.",
        "Emit only well-formed tool calls using the available functions.",
        "Do not invent tools. Prefer the minimum sequence of calls that completes the turn.",
        "",
        "Available tools (JSON schemas):",
        json.dumps(docs, indent=2),
    ]
    return "\n".join(lines)
