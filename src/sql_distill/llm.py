"""OpenAI-compatible chat client with usage and cost accounting.

Works against llama.cpp's ``llama-server``, DeepSeek, or any ``/chat/completions`` endpoint.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key_env: str = ""
    extra_body: dict[str, Any] = field(default_factory=dict)
    # USD per million tokens: cached input, uncached input, output.
    price: tuple[float, float, float] = (0.0, 0.0, 0.0)


ENDPOINTS: dict[str, Endpoint] = {
    # DeepSeek off-peak prices (peak is 2x); thinking off so it answers directly.
    "deepseek-flash": Endpoint(
        "deepseek-flash", "https://api.deepseek.com", "deepseek-flash", "DEEPSEEK_API_KEY",
        {"thinking": {"type": "disabled"}}, (0.003, 0.15, 0.6)),
    "qwen3.5-4b-gguf": Endpoint(
        "qwen3.5-4b-gguf", os.environ.get("LLAMA_URL", "http://127.0.0.1:8091/v1"),
        "qwen3.5-4b", "",
        {"chat_template_kwargs": {"enable_thinking": False}}),
}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    uncached: int = 0
    output: int = 0
    seconds: float = 0.0

    def cost(self, ep: Endpoint) -> float:
        c, u, o = ep.price
        return (self.cached * c + self.uncached * u + self.output * o) / 1e6


@dataclass
class Reply:
    text: str
    cached: int
    uncached: int
    output: int
    seconds: float


def load_env(path: Path = Path("../.env")) -> None:
    """Read KEY=VALUE lines from a local .env (kept outside the repo) into the environment."""
    for candidate in (Path(".env"), path):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip() and value.strip() and key.strip() not in os.environ:
                    os.environ[key.strip()] = value.strip()


class ChatClient:
    def __init__(self, ep: Endpoint, concurrency: int = 4, timeout: float = 120.0) -> None:
        load_env()
        headers = {}
        if ep.api_key_env:
            key = os.environ.get(ep.api_key_env)
            if not key:
                raise RuntimeError(f"{ep.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self.ep = ep
        self.usage = Usage()
        self._sem = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(base_url=ep.base_url, headers=headers, timeout=timeout)

    async def complete(self, messages: list[dict[str, str]], temperature: float = 0.0,
                       max_tokens: int = 512, **extra: Any) -> Reply:
        body = {"model": self.ep.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **self.ep.extra_body, **extra}
        async with self._sem:
            for attempt in range(4):
                start = time.perf_counter()
                try:
                    resp = await self._http.post("/chat/completions", json=body)
                    if resp.status_code in (429, 500, 502, 503) and attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    break
                except httpx.TransportError:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 ** attempt)
            seconds = time.perf_counter() - start
        data = resp.json()
        u = data.get("usage") or {}
        prompt = int(u.get("prompt_tokens", 0))
        cached = int(u.get("prompt_cache_hit_tokens",
                           (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0)
        reply = Reply(data["choices"][0]["message"].get("content") or "", cached,
                      prompt - cached, int(u.get("completion_tokens", 0)), seconds)
        self.usage.calls += 1
        self.usage.cached += reply.cached
        self.usage.uncached += reply.uncached
        self.usage.output += reply.output
        self.usage.seconds += seconds
        return reply

    async def aclose(self) -> None:
        await self._http.aclose()
