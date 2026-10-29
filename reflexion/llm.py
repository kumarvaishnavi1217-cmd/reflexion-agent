"""LLM backends. ``ClaudeCodeLLM`` runs on a Claude subscription through the Claude Code CLI, so experiments
cost nothing beyond the subscription; ``AnthropicLLM`` uses the pay-per-token API."""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> Completion: ...


class ClaudeCodeLLM:
    def __init__(self, model: str = "haiku", timeout_s: int = 180, retries: int = 2) -> None:
        self.name = f"{model} (Claude Code CLI)"
        self._model = model
        self._timeout_s = timeout_s
        self._retries = retries

    def complete(self, system: str, prompt: str) -> Completion:
        cmd = [
            "claude", "-p", prompt,
            "--system-prompt", system,  # replaces the coding-agent prompt: a plain model call
            "--model", self._model,
            "--tools", "",  # no tools: the agent loop lives in LangGraph, not in the CLI
            "--output-format", "json",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--setting-sources", "",
        ]  # fmt: skip
        # never bill an API key that happens to be exported; use the logged-in subscription
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        last_error = ""
        for attempt in range(self._retries + 1):
            started = time.perf_counter()
            try:
                proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    cmd, env=env, capture_output=True, text=True, timeout=self._timeout_s, check=False
                )
                out = json.loads(proc.stdout)
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if not out.get("is_error"):
                    usage = out.get("usage") or {}
                    return Completion(
                        text=out.get("result") or "",
                        input_tokens=usage.get("input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                last_error = str(out.get("result"))[:300]
            time.sleep(2 * (attempt + 1))
        raise LLMError(f"Claude Code CLI failed after {self._retries + 1} attempts: {last_error}")


class AnthropicLLM:
    def __init__(self, model: str = "claude-haiku-5-5") -> None:
        import anthropic

        self.name = model
        self._model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> Completion:
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=16000,
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"stop_reason={response.stop_reason}")
        return Completion(
            text="".join(b.text for b in response.content if b.type == "text"),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
