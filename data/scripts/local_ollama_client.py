#!/usr/bin/env python3
"""Minimal Ollama client (stdlib only).

Ollama runs locally and exposes an HTTP API on http://localhost:11434.
We use it as an offline-friendly alternative to Gemini for paper reproducibility.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 900
    num_ctx: int = 8192
    num_predict: int = 4096
    seed: int = 42

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        model = os.environ.get("REQ_OLLAMA_MODEL", "").strip() or "qwen2.5:7b-instruct"
        base_url = os.environ.get("REQ_OLLAMA_BASE_URL", "http://localhost:11434").strip()
        timeout_raw = os.environ.get("REQ_OLLAMA_TIMEOUT_SECONDS", "900").strip()
        num_ctx_raw = os.environ.get("REQ_OLLAMA_NUM_CTX", "16384").strip()
        num_predict_raw = os.environ.get("REQ_OLLAMA_NUM_PREDICT", "4096").strip()
        seed_raw = os.environ.get("REQ_OLLAMA_SEED", "42").strip()
        return cls(
            model=model,
            base_url=base_url.rstrip("/"),
            timeout_seconds=int(timeout_raw),
            num_ctx=int(num_ctx_raw),
            num_predict=int(num_predict_raw),
            seed=int(seed_raw),
        )


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self.chat_url = f"{self.config.base_url}/api/chat"

    def chat(self, prompt: str, *, temperature: float = 0.2, response_schema: dict | None = None) -> dict:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_ctx": int(self.config.num_ctx),
                "num_predict": int(self.config.num_predict),
                "seed": int(self.config.seed),
            },
        }
        if response_schema is not None:
            # Ollama supports structured output via `format`.
            payload["format"] = response_schema
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.config.base_url}. Is Ollama running?"
            ) from exc

        data = json.loads(raw)
        msg = data.get("message", {}) if isinstance(data, dict) else {}
        text = msg.get("content", "") if isinstance(msg, dict) else ""
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Ollama response missing message.content: {raw}")
        usage = {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
        }
        return {"text": text, "usage": usage, "raw_response": data}
