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


_TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "list": "array",
}


def _normalize_json_schema(node: Any) -> Any:
    """Map BFCL schema types (float/dict/...) to JSON Schema for llama-server."""
    if isinstance(node, list):
        return [_normalize_json_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            out[key] = _TYPE_MAP.get(value, value)
        else:
            out[key] = _normalize_json_schema(value)
    return out


def tools_as_openai(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert BFCL function docs to OpenAI-style tool definitions."""
    tools = []
    for doc in docs:
        params = doc.get("parameters", {"type": "object", "properties": {}, "required": []})
        params = _normalize_json_schema(params)
        if params.get("type") != "object":
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
    """Baseline: every tool schema lives in the system prompt.

    Response schemas are omitted from the dump; they are not needed to call a tool
    and inflate the prompt by several thousand tokens.
    """
    compact = []
    for doc in docs:
        compact.append(
            {
                "name": doc["name"],
                "description": doc.get("description", ""),
                "parameters": _normalize_json_schema(
                    doc.get("parameters", {"type": "object", "properties": {}, "required": []})
                ),
            }
        )
    lines = [
        "You are a tool-calling agent. Solve the user request by calling tools.",
        "When a tool is needed, emit a native tool call (do not only describe the call in prose).",
        "Do not invent tools. Prefer the minimum sequence of calls that completes the turn.",
        "Do not use thinking/reasoning traces; call tools directly.",
        "",
        "Available tools (JSON schemas):",
        json.dumps(compact, indent=2),
    ]
    return "\n".join(lines)
