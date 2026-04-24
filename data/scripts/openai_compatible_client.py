#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat client using only the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


def resolve_chat_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


def extract_text_content(message_content: object) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    raise ValueError("Unsupported message content format in API response")


def extract_first_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON object in model response")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Extracted JSON payload is not an object")
    return payload


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: int = 90

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        base_url = os.environ.get("REQ_LLM_BASE_URL", "").strip()
        model = os.environ.get("REQ_LLM_MODEL", "").strip()
        api_key = os.environ.get("REQ_LLM_API_KEY", "").strip() or None
        timeout_raw = os.environ.get("REQ_LLM_TIMEOUT_SECONDS", "90").strip()

        if not base_url:
            raise ValueError("Missing REQ_LLM_BASE_URL")
        if not model:
            raise ValueError("Missing REQ_LLM_MODEL")

        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=int(timeout_raw),
        )


class OpenAICompatibleClient:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self.chat_url = resolve_chat_url(config.base_url)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        payload = {
            "model": self.config.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(
            self.chat_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from API: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach API endpoint {self.chat_url}: {exc}") from exc

        payload = json.loads(raw)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("API response does not contain choices")

        message = choices[0].get("message", {})
        content = extract_text_content(message.get("content", ""))
        return {
            "raw_response": payload,
            "text": content,
            "usage": payload.get("usage", {}),
        }
