#!/usr/bin/env python3
"""Generate expanded elicitation dialogues from PURE benchmark requirements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gemini_native_client import GeminiConfig, GeminiNativeClient, extract_first_json_object


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "expanded_dialogues"
DEFAULT_PROMPT = ROOT / "prompts" / "pure_requirements_to_dialogue.txt"
DEFAULT_SCHEMA = ROOT / "schemas" / "gemini_expanded_dialogue_response.schema.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "ground_truth_requirements" not in payload:
            continue
        samples.append(payload)
    return samples


def render_source_requirements(sample: dict, max_requirements: int) -> str:
    lines = []
    for item in sample["ground_truth_requirements"][:max_requirements]:
        lines.append(f"- {item['req_id']}: {item['text']}")
    return "\n".join(lines)

def render_requirement_count(sample: dict, max_requirements: int) -> int:
    return min(len(sample.get("ground_truth_requirements", [])), max_requirements)


def build_prompt(template: str, sample: dict, max_requirements: int) -> str:
    return (
        template.replace("{{SAMPLE_ID}}", sample["sample_id"])
        .replace("{{DOCUMENT_ID}}", sample["source"]["document_id"])
        .replace("{{TITLE}}", sample["source"]["title"])
        .replace("{{REQUIREMENT_COUNT}}", str(render_requirement_count(sample, max_requirements)))
        .replace("{{SOURCE_REQUIREMENTS}}", render_source_requirements(sample, max_requirements))
    )


def normalize_dialogue(payload: dict) -> list[dict]:
    dialogue = payload.get("dialogue", [])
    if not isinstance(dialogue, list) or not dialogue:
        raise ValueError("Missing dialogue array in Gemini response")

    normalized = []
    expected_turn_id = 1
    expected_role = "bot"
    for item in dialogue:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = clean_text(item.get("text", ""))
        if role not in {"bot", "user"} or not text:
            continue
        normalized.append(
            {
                "turn_id": expected_turn_id,
                "role": role,
                "text": text,
            }
        )
        expected_turn_id += 1
        expected_role = "user" if expected_role == "bot" else "bot"

    if len(normalized) < 8:
        raise ValueError("Dialogue is too short after normalization")
    if normalized[0]["role"] != "bot":
        raise ValueError("Dialogue must start with bot turn")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--max-samples", type=int, default=None)
    # Use a higher default so smaller PURE docs are not truncated in v1 runs.
    parser.add_argument("--max-source-requirements", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_repair_prompt(schema_text: str, invalid_text: str) -> str:
    return (
        "Repair the following invalid JSON so it matches the schema exactly.\n"
        "Return JSON only.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Invalid JSON:\n{invalid_text}\n"
    )


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    prompt_hash = sha256_text(prompt_template)
    schema_hash = sha256_text(schema_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        if not samples:
            print("No samples available for dry run.")
            return 0
        preview = {
            "sample_id": samples[0]["sample_id"],
            "prompt": build_prompt(prompt_template, samples[0], args.max_source_requirements),
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "schema": schema,
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    config = GeminiConfig.from_env()
    client = GeminiNativeClient(config)
    summary = []

    for sample in samples:
        prompt = build_prompt(prompt_template, sample, args.max_source_requirements)
        repair_attempted = False
        invalid_text = ""
        raw_response = None

        try:
            response = client.generate_json(prompt, schema)
            invalid_text = response["text"]
            parsed = extract_first_json_object(response["text"])
            dialogue = normalize_dialogue(parsed)
            raw_response = response["raw_response"]
            parse_status = "ok"
            usage = response.get("usage", {})
        except Exception as first_error:  # noqa: BLE001
            repair_attempted = True
            try:
                repair_prompt = build_repair_prompt(schema_text, invalid_text or str(first_error))
                response = client.generate_json(repair_prompt, schema)
                parsed = extract_first_json_object(response["text"])
                dialogue = normalize_dialogue(parsed)
                raw_response = response["raw_response"]
                parse_status = "repaired"
                usage = response.get("usage", {})
            except Exception as second_error:  # noqa: BLE001
                dialogue = []
                parse_status = "failed"
                usage = {}
                error = f"{type(first_error).__name__}: {first_error}; repair failed with {type(second_error).__name__}: {second_error}"

        output_payload = {
            "sample_id": sample["sample_id"],
            "metadata": {
                "domain": "pure_document",
                "source_type": "synthetic",
                "parent_id": sample["sample_id"],
                "split": "benchmark",
                "dialogue_style": "expanded",
            },
            "source": sample["source"],
            "dialogue": dialogue,
            "dialogue_generation": {
                "method": "g_full_source_to_dialogue_v1",
                "model": config.model,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "parse_status": parse_status,
                "repair_attempted": repair_attempted,
                "usage": usage,
                "error": error if parse_status == "failed" else None,
                "max_source_requirements": args.max_source_requirements,
            },
        }
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if raw_response is not None:
            raw_path = args.output_dir / f"{sample['sample_id']}.raw_response.json"
            raw_path.write_text(json.dumps(raw_response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(output_path.relative_to(ROOT)),
                "parse_status": parse_status,
                "turn_count": len(dialogue),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
