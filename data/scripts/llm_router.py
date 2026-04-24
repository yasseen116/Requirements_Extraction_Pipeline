#!/usr/bin/env python3
"""Small provider router for LLM calls (Gemini or local Ollama).

Env:
- REQ_LLM_PROVIDER=gemini|ollama  (default: gemini)
- For ollama: REQ_OLLAMA_MODEL, REQ_OLLAMA_BASE_URL
- For gemini: REQ_GEMINI_API_KEY, REQ_GEMINI_MODEL, ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from gemini_native_client import GeminiConfig, GeminiNativeClient, extract_first_json_object
from local_ollama_client import OllamaClient, OllamaConfig


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: dict
    raw_response: dict


def provider() -> str:
    return (os.environ.get("REQ_LLM_PROVIDER", "gemini") or "gemini").strip().lower()


def has_gemini_env() -> bool:
    return bool(os.environ.get("REQ_GEMINI_API_KEY")) and bool(os.environ.get("REQ_GEMINI_MODEL"))


def get_client() -> object:
    prov = provider()
    if prov == "ollama":
        return OllamaClient(OllamaConfig.from_env())
    return GeminiNativeClient(GeminiConfig.from_env())


def generate_json(prompt: str, schema: dict, *, temperature: float = 0.0) -> LLMResponse:
    """Generate JSON for the given schema from the selected provider."""
    prov = provider()
    if prov == "ollama":
        client = OllamaClient(OllamaConfig.from_env())
        wrapped = (
            prompt
            + "\n\nOutput rules:\n- Return JSON only.\n- Must match this JSON schema exactly:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        resp = client.chat(wrapped, temperature=temperature, response_schema=schema)
        return LLMResponse(text=resp["text"], usage=resp.get("usage", {}), raw_response=resp["raw_response"])

    client = GeminiNativeClient(GeminiConfig.from_env())
    resp = client.generate_json(prompt, schema, temperature=temperature)
    return LLMResponse(text=resp["text"], usage=resp.get("usage", {}), raw_response=resp["raw_response"])


def parse_first_json_object(text: str) -> dict:
    return extract_first_json_object(text)
