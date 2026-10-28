"""Parse tool calls from llama-server / Qwen chat responses."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_CALL_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"(?P<name>[^\"]+)\"\s*,\s*\"arguments\"\s*:\s*(?P<args>\{.*?\})\s*\}",
    re.DOTALL,
)


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": raw}


def parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI-style chat message."""
    calls: list[dict[str, Any]] = []

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append(
            {
                "id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": _coerce_arguments(fn.get("arguments")),
            }
        )
    if calls:
        return [c for c in calls if c.get("name")]

    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    # Qwen / hermes XML style
    for match in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.DOTALL
    ):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if name:
            calls.append(
                {
                    "id": None,
                    "name": name,
                    "arguments": _coerce_arguments(obj.get("arguments")),
                }
            )
    if calls:
        return calls

    # Fallback: bare {"name","arguments"} JSON objects in the text
    for match in _JSON_CALL_RE.finditer(content):
        try:
            args = json.loads(match.group("args"))
        except json.JSONDecodeError:
            continue
        calls.append(
            {
                "id": None,
                "name": match.group("name"),
                "arguments": _coerce_arguments(args),
            }
        )
    return calls


def validate_call(
    call: dict[str, Any],
    allowed_names: set[str],
    schemas_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Check that a call names an existing tool and supplies required args."""
    name = call.get("name")
    args = call.get("arguments") or {}
    result = {
        "name": name,
        "ok": False,
        "existing_tool": False,
        "schema_valid": False,
        "missing_required": [],
    }
    if name not in allowed_names:
        return result
    result["existing_tool"] = True
    schema = schemas_by_name.get(name, {})
    params = schema.get("parameters") or {}
    required = params.get("required") or []
    missing = [key for key in required if key not in args]
    result["missing_required"] = missing
    result["schema_valid"] = len(missing) == 0
    result["ok"] = result["existing_tool"] and result["schema_valid"]
    return result
