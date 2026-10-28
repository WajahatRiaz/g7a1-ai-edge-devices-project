"""Execute tool calls against BFCL multi-turn backends in long-context mode."""

from __future__ import annotations

import json
from typing import Any

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)


def format_python_call(name: str, arguments: dict[str, Any] | None) -> str:
    """Turn a parsed tool call into the BFCL `name(arg=...)` string form."""
    arguments = arguments or {}
    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={json.dumps(value)}")
    return f"{name}({', '.join(parts)})"


class ToolRuntime:
    """Stateful wrapper around BFCL's multi-turn executor."""

    def __init__(
        self,
        initial_config: dict[str, Any],
        involved_classes: list[str],
        *,
        model_name: str = "local",
        test_entry_id: str = "session",
        long_context: bool = True,
    ) -> None:
        self.initial_config = initial_config
        self.involved_classes = involved_classes
        self.model_name = model_name
        self.test_entry_id = test_entry_id
        self.long_context = long_context
        self._instances: dict[str, Any] = {}

    def execute(self, calls: list[dict[str, Any]]) -> list[str]:
        call_strings = [
            format_python_call(c["name"], c.get("arguments") or {}) for c in calls
        ]
        results, instances = execute_multi_turn_func_call(
            call_strings,
            self.initial_config,
            self.involved_classes,
            model_name=self.model_name,
            test_entry_id=self.test_entry_id,
            long_context=self.long_context,
        )
        self._instances = instances
        return results

    @property
    def instances(self) -> dict[str, Any]:
        return self._instances
