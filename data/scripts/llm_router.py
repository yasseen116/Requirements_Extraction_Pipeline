#!/usr/bin/env python3
"""Small provider router for LLM calls (Gemini, OpenAI, or local Ollama).

Env:
- REQ_LLM_PROVIDER=gemini|openai|ollama  (default: gemini)
- For ollama: REQ_OLLAMA_MODEL, REQ_OLLAMA_BASE_URL
- For gemini: REQ_GEMINI_API_KEY, REQ_GEMINI_MODEL, ...
- For openai: REQ_OPENAI_API_KEY, REQ_OPENAI_MODEL, ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from gemini_native_client import GeminiConfig, GeminiNativeClient, extract_first_json_object
from local_ollama_client import OllamaClient, OllamaConfig
from openai_compatible_client import OpenAICompatibleClient, OpenAICompatibleConfig


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: dict
    raw_response: dict


def provider() -> str:
    return (os.environ.get("REQ_LLM_PROVIDER", "gemini") or "gemini").strip().lower()


def has_gemini_env() -> bool:
    return bool(os.environ.get("REQ_GEMINI_API_KEY")) and bool(os.environ.get("REQ_GEMINI_MODEL"))


def has_openai_env() -> bool:
    return bool(os.environ.get("REQ_OPENAI_API_KEY") or os.environ.get("REQ_LLM_API_KEY")) and bool(
        os.environ.get("REQ_OPENAI_MODEL") or os.environ.get("REQ_LLM_MODEL")
    )


def active_model_name() -> str | None:
    prov = provider()
    if prov == "ollama":
        return (os.environ.get("REQ_OLLAMA_MODEL", "").strip() or None)
    if prov == "openai":
        return (os.environ.get("REQ_OPENAI_MODEL", "").strip() or os.environ.get("REQ_LLM_MODEL", "").strip() or None)
    return (os.environ.get("REQ_GEMINI_MODEL", "").strip() or None)


def get_client() -> object:
    prov = provider()
    if prov == "ollama":
        return OllamaClient(OllamaConfig.from_env())
    if prov == "openai":
        return OpenAICompatibleClient(OpenAICompatibleConfig.from_env())
    return GeminiNativeClient(GeminiConfig.from_env())


def generate_json(
    prompt: str,
    schema: dict,
    *,
    temperature: float = 0.0,
    cache_prefix: str | None = None,
    cache_namespace: str | None = None,
    max_output_tokens: int | None = None,
    timeout_seconds: int | None = None,
) -> LLMResponse:
    """Generate JSON for the given schema from the selected provider."""
    prov = provider()
    if prov == "ollama":
        client = OllamaClient(OllamaConfig.from_env())
        wrapped = (
            prompt
            + "\n\nOutput rules:\n- Return JSON only.\n- Must match this JSON schema exactly:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        resp = client.chat(
            wrapped,
            temperature=temperature,
            response_schema=schema,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        return LLMResponse(text=resp["text"], usage=resp.get("usage", {}), raw_response=resp["raw_response"])

    if prov == "openai":
        client = OpenAICompatibleClient(OpenAICompatibleConfig.from_env())
        resp = client.chat(
            "Return JSON only. Match the provided JSON schema exactly.",
            prompt,
            temperature=temperature,
            response_schema=schema,
            max_output_tokens=max_output_tokens,
        )
        return LLMResponse(text=resp["text"], usage=resp.get("usage", {}), raw_response=resp["raw_response"])

    client = GeminiNativeClient(GeminiConfig.from_env())
    resp = client.generate_json(
        prompt,
        schema,
        temperature=temperature,
        cache_prefix=cache_prefix,
        cache_namespace=cache_namespace,
    )
    return LLMResponse(text=resp["text"], usage=resp.get("usage", {}), raw_response=resp["raw_response"])


def parse_first_json_object(text: str) -> dict:
    return extract_first_json_object(text)
