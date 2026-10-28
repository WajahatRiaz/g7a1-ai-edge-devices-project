"""Multi-turn agent loop with per-turn metric logging."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from a1_agent.backends import ToolRuntime
from a1_agent.client import LlamaServerClient
from a1_agent.parse import parse_tool_calls, validate_call
from a1_agent.schemas import (
    load_bfcl_tool_docs,
    schemas_system_prompt,
    tools_as_openai,
)


@dataclass
class TurnMetrics:
    turn_index: int
    user_chars: int
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_sent_est: int | None
    tokens_reprocessed: int | None
    prefill_ms: float | None
    decode_ms: float | None
    tool_ms: float | None
    wall_ms: float
    n_tool_calls: int
    call_validity: list[dict[str, Any]]
    cache_prompt: bool


@dataclass
class SessionResult:
    task_id: str
    success: bool | None
    turns: list[TurnMetrics] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class AgentHarness:
    def __init__(
        self,
        client: LlamaServerClient,
        backends: list[str],
        *,
        long_context: bool = True,
        put_schemas_in_system: bool = True,
        pass_tools_api: bool = True,
        tool_choice: str | None = "auto",
        max_tool_iters: int = 8,
    ) -> None:
        self.client = client
        self.backends = backends
        self.long_context = long_context
        self.put_schemas_in_system = put_schemas_in_system
        self.pass_tools_api = pass_tools_api
        self.tool_choice = tool_choice
        self.max_tool_iters = max_tool_iters

        self.docs = load_bfcl_tool_docs(backends)
        self.openai_tools = tools_as_openai(self.docs)
        self.schemas_by_name = {d["name"]: d for d in self.docs}
        self.allowed_names = set(self.schemas_by_name)
        self.system_prompt = (
            schemas_system_prompt(self.docs)
            if put_schemas_in_system
            else "You are a tool-calling agent. Use the provided tools."
        )
        # Per-backend OpenAI tool lists (API can expose only involved backends).
        self._tools_by_backend = {
            backend: tools_as_openai(load_bfcl_tool_docs([backend]))
            for backend in backends
        }

    def _tools_for_task(self, involved_classes: list[str]) -> list[dict[str, Any]] | None:
        if not self.pass_tools_api:
            return None
        tools: list[dict[str, Any]] = []
        for cls in involved_classes:
            tools.extend(self._tools_by_backend.get(cls, []))
        return tools or self.openai_tools

    def run_task(self, task: dict[str, Any]) -> SessionResult:
        task_id = task["id"]
        runtime = ToolRuntime(
            task.get("initial_config") or {},
            task["involved_classes"],
            model_name="local",
            test_entry_id=task_id,
            long_context=self.long_context,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        turns: list[TurnMetrics] = []
        prev_prompt_tokens = 0
        tools = self._tools_for_task(task["involved_classes"])

        try:
            for turn_index, user_bundle in enumerate(task["question"]):
                # BFCL stores each user turn as a list of message dicts.
                for user_msg in user_bundle:
                    messages.append(
                        {"role": user_msg["role"], "content": user_msg["content"]}
                    )
                user_chars = sum(len(m.get("content") or "") for m in user_bundle)

                for _ in range(self.max_tool_iters):
                    resp = self.client.chat(
                        messages,
                        tools=tools,
                        tool_choice=self.tool_choice if tools else None,
                    )
                    message = resp["message"]
                    messages.append(message)

                    usage = resp.get("usage") or {}
                    timings = resp.get("timings") or {}
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    tokens_reprocessed = None
                    if prompt_tokens is not None:
                        # Baseline disables prefix cache, so each turn re-prefills
                        # the entire prompt; the incremental new suffix is estimated
                        # against the previous prompt length.
                        tokens_reprocessed = prompt_tokens
                    tokens_sent_est = (
                        None
                        if prompt_tokens is None
                        else max(prompt_tokens - prev_prompt_tokens, 0)
                    )
                    if prompt_tokens is not None:
                        prev_prompt_tokens = prompt_tokens

                    prefill_ms = None
                    decode_ms = None
                    if "prompt_ms" in timings:
                        prefill_ms = float(timings["prompt_ms"])
                    if "predicted_ms" in timings:
                        decode_ms = float(timings["predicted_ms"])

                    calls = parse_tool_calls(message)
                    validity = [
                        validate_call(c, self.allowed_names, self.schemas_by_name)
                        for c in calls
                    ]
                    if calls:
                        print(
                            f"  turn {turn_index}: tool_calls="
                            f"{[c.get('name') for c in calls]}",
                            flush=True,
                        )
                    else:
                        preview = (message.get("content") or "")[:160].replace("\n", " ")
                        print(
                            f"  turn {turn_index}: no tool_calls "
                            f"content_preview={preview!r}",
                            flush=True,
                        )

                    tool_ms = None
                    if calls:
                        t0 = time.perf_counter()
                        results = runtime.execute(calls)
                        tool_ms = (time.perf_counter() - t0) * 1000.0
                        for call, result in zip(calls, results):
                            # Keep long dumps out of the next prompt if they would
                            # explode the context; still execute for side effects.
                            text = (
                                result
                                if isinstance(result, str)
                                else json.dumps(result)
                            )
                            if len(text) > 12000:
                                text = (
                                    text[:12000]
                                    + f"\n...[truncated {len(text) - 12000} chars]"
                                )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id") or call["name"],
                                    "content": text,
                                }
                            )

                    turns.append(
                        TurnMetrics(
                            turn_index=turn_index,
                            user_chars=user_chars,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            tokens_sent_est=tokens_sent_est,
                            tokens_reprocessed=tokens_reprocessed,
                            prefill_ms=prefill_ms,
                            decode_ms=decode_ms,
                            tool_ms=tool_ms,
                            wall_ms=resp["wall_s"] * 1000.0,
                            n_tool_calls=len(calls),
                            call_validity=validity,
                            cache_prompt=self.client.cache_prompt,
                        )
                    )

                    if not calls:
                        break
                    # Continue the inner loop so the model can chain tools
                    # within the same user turn.

            success = self._check_success(task, runtime)
            return SessionResult(
                task_id=task_id, success=success, turns=turns, messages=messages
            )
        except Exception as exc:  # noqa: BLE001 - harness must keep going
            return SessionResult(
                task_id=task_id,
                success=False,
                turns=turns,
                messages=messages,
                error=str(exc),
            )

    def _check_success(self, task: dict[str, Any], runtime: ToolRuntime) -> bool | None:
        check = task.get("success_check")
        if not check:
            return None
        kind = check.get("type")
        if kind == "account_balance_equals":
            bot = runtime.instances.get("TradingBot")
            if bot is None:
                return False
            expected = float(check["value"])
            actual = float(bot.account_info.get("balance", 0.0))
            return abs(actual - expected) < 1e-6
        if kind == "watchlist_contains":
            bot = runtime.instances.get("TradingBot")
            if bot is None:
                return False
            return check["symbol"] in list(getattr(bot, "watch_list", []) or [])
        if kind == "early_detail_recalled":
            needle = str(check["needle"])
            field_name = check.get("field")
            bot = runtime.instances.get("TradingBot")
            if bot is None:
                return False
            if field_name == "binding_card":
                return needle == str(bot.account_info.get("binding_card"))
            if field_name == "order_id":
                return needle in {str(k) for k in getattr(bot, "orders", {})}
            if field_name == "watch_symbol":
                return needle in list(getattr(bot, "watch_list", []) or [])
            return needle in json.dumps(bot.__dict__, default=str)
        return None


def write_session_log(result: SessionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": result.task_id,
        "success": result.success,
        "error": result.error,
        "turns": [asdict(t) for t in result.turns],
        "n_messages": len(result.messages),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
