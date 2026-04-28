#!/usr/bin/env python3
"""Generate a direct Gemini baseline: dialogue -> requirements (pilot schema).

This is intentionally simple: it bypasses slots/frames and generates requirement items
directly from the dialogue. It exists to provide a fair LLM baseline in the pilot track.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import llm_router as llm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "manual_gold"
DEFAULT_OUTPUT = ROOT / "outputs" / "b3_pilot_direct_requirements"
DEFAULT_PROMPT = ROOT / "prompts" / "pilot_dialogue_to_requirements_gemini.txt"

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "object",
            "additionalProperties": False,
            "required": ["functional", "non_functional"],
            "properties": {
                "functional": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "text", "source_slots"],
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "source_slots": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "non_functional": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "category", "text", "source_slots"],
                        "properties": {
                            "id": {"type": "string"},
                            "category": {"type": "string", "enum": ["performance", "security"]},
                            "text": {"type": "string"},
                            "source_slots": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        }
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def render_dialogue(sample: dict) -> str:
    return "\n".join(f"{turn['turn_id']}. {turn['role']}: {turn['text']}" for turn in sample["dialogue"])


def build_prompt(template: str, sample: dict) -> str:
    return template.replace("{{DIALOGUE}}", render_dialogue(sample))


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "dialogue" not in payload:
            continue
        samples.append(payload)
    return samples


def normalize_out(payload: dict) -> dict:
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        raise ValueError("Missing requirements")

    def norm_list(items: list[dict], *, nfr: bool) -> list[dict]:
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("text", ""))
            if not text:
                continue
            out = {
                "id": clean_text(item.get("id", "")) or ("NFR" if nfr else "FR") + str(len(normalized) + 1),
                "text": text.rstrip(".") + ".",
                "source_slots": item.get("source_slots") if isinstance(item.get("source_slots"), list) else ["dialogue"],
            }
            if not out["source_slots"]:
                out["source_slots"] = ["dialogue"]
            if nfr:
                out["category"] = clean_text(item.get("category", "performance")).lower()
                if out["category"] not in {"performance", "security"}:
                    out["category"] = "performance"
            normalized.append(out)
        return normalized

    return {
        "requirements": {
            "functional": norm_list(reqs.get("functional", []), nfr=False),
            "non_functional": norm_list(reqs.get("non_functional", []), nfr=True),
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    template = args.prompt_path.read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    schema_hash = sha256_text(json.dumps(RESPONSE_SCHEMA, sort_keys=True))

    model_name = llm.active_model_name()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for sample in samples:
        prompt = build_prompt(template, sample)
        response = llm.generate_json(prompt, RESPONSE_SCHEMA, temperature=0.0)
        parsed = llm.parse_first_json_object(response.text)
        normalized = normalize_out(parsed)

        out_payload = {
            "sample_id": sample["sample_id"],
            "method": "b3_pilot_direct_dialogue_to_requirements_v1",
            "model": model_name,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "requirements": normalized["requirements"],
        }
        out_path = args.output_dir / f"{sample['sample_id']}.json"
        out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append({"sample_id": sample["sample_id"], "path": str(out_path)})

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
