"""HTTP client for llama-server chat completions."""

from __future__ import annotations

import time
from typing import Any

import requests


class LlamaServerClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        cache_prompt: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_prompt = cache_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        """
        POST /v1/chat/completions.

        Baseline: cache_prompt=false so the full history is re-prefilled every turn.
        Schemas may also be in the system prompt; tools= here is optional.
        """
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "cache_prompt": self.cache_prompt,
            "stream": False,
        }
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout_s,
        )
        wall_s = time.perf_counter() - t0
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        timings = payload.get("timings") or {}
        message = payload["choices"][0]["message"]
        return {
            "message": message,
            "usage": usage,
            "timings": timings,
            "wall_s": wall_s,
            "raw": payload,
        }
