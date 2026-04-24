#!/usr/bin/env python3
"""Generate canonical requirement frames with native Gemini structured output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from frame_slot_utils import canonicalize_frames, normalize_frame
from gemini_native_client import GeminiConfig, GeminiNativeClient, extract_first_json_object


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "synthetic" / "pilot_noisy"
DEFAULT_OUTPUT = ROOT / "outputs" / "g1_gemini_frames_noisy"
DEFAULT_PROMPT = ROOT / "prompts" / "dialogue_to_frames_gemini.txt"
DEFAULT_SCHEMA = ROOT / "schemas" / "gemini_requirement_frames_response.schema.json"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        payload = load_json(path)
        if not isinstance(payload, dict) or "sample_id" not in payload or "dialogue" not in payload:
            continue
        samples.append(payload)
    return samples


def render_dialogue(sample: dict) -> str:
    lines = []
    for turn in sample["dialogue"]:
        lines.append(f"{turn['turn_id']}. {turn['role']}: {turn['text']}")
    return "\n".join(lines)


def build_prompt(template_text: str, sample: dict) -> str:
    return (
        template_text.replace("{{SAMPLE_ID}}", sample["sample_id"])
        .replace("{{DOMAIN}}", sample["metadata"]["domain"])
        .replace("{{DIALOGUE}}", render_dialogue(sample))
    )


def normalize_frames_payload(payload: dict) -> list[dict]:
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("Gemini payload does not contain a frames array")
    normalized = [normalize_frame(frame, default_index=index) for index, frame in enumerate(frames, start=1) if isinstance(frame, dict)]
    if not normalized:
        raise ValueError("Gemini payload contained no usable frames")
    return canonicalize_frames(normalized)


def build_repair_prompt(schema_text: str, invalid_text: str) -> str:
    return (
        "Repair the following invalid or incomplete JSON so it matches the schema exactly.\n"
        "Return JSON only.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Invalid JSON:\n{invalid_text}\n"
    )


def build_failure_prediction(sample_id: str, model: str, prompt_hash: str, schema_hash: str, error: str, repair_attempted: bool) -> dict:
    return {
        "sample_id": sample_id,
        "method": "g1_gemini_dialogue_to_frames_v1",
        "model": model,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "parse_status": "failed",
        "repair_attempted": repair_attempted,
        "usage": {},
        "error": error,
        "frames": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    schema = load_json(args.schema_path)
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    prompt_hash = sha256_text(prompt_template)
    schema_hash = sha256_text(schema_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        if not samples:
            print("No samples found for dry run.")
            return 0
        sample = samples[0]
        preview = {
            "sample_id": sample["sample_id"],
            "prompt_path": display_path(args.prompt_path),
            "schema_path": display_path(args.schema_path),
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "prompt": build_prompt(prompt_template, sample),
            "response_schema": schema,
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    config = GeminiConfig.from_env()
    client = GeminiNativeClient(config)
    debug_raw = os.environ.get("REQ_GEMINI_DEBUG_RAW", "").strip().lower() == "true"
    summary = []

    for sample in samples:
        prompt = build_prompt(prompt_template, sample)
        repair_attempted = False
        invalid_output = ""
        raw_response = None
        try:
            response = client.generate_json(prompt, schema)
            invalid_output = response["text"]
            parsed = extract_first_json_object(response["text"])
            frames = normalize_frames_payload(parsed)
            prediction = {
                "sample_id": sample["sample_id"],
                "method": "g1_gemini_dialogue_to_frames_v1",
                "model": config.model,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "parse_status": "ok",
                "repair_attempted": False,
                "usage": response["usage"],
                "frames": frames,
            }
            raw_response = response["raw_response"]
        except Exception as first_error:  # noqa: BLE001
            repair_attempted = True
            try:
                repair_prompt = build_repair_prompt(schema_text, invalid_output or str(first_error))
                response = client.generate_json(repair_prompt, schema)
                parsed = extract_first_json_object(response["text"])
                frames = normalize_frames_payload(parsed)
                prediction = {
                    "sample_id": sample["sample_id"],
                    "method": "g1_gemini_dialogue_to_frames_v1",
                    "model": config.model,
                    "prompt_hash": prompt_hash,
                    "schema_hash": schema_hash,
                    "parse_status": "repaired",
                    "repair_attempted": True,
                    "usage": response["usage"],
                    "frames": frames,
                }
                raw_response = response["raw_response"]
            except Exception as repair_error:  # noqa: BLE001
                prediction = build_failure_prediction(
                    sample_id=sample["sample_id"],
                    model=config.model,
                    prompt_hash=prompt_hash,
                    schema_hash=schema_hash,
                    error=f"{type(first_error).__name__}: {first_error}; repair failed with {type(repair_error).__name__}: {repair_error}",
                    repair_attempted=repair_attempted,
                )

        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if debug_raw and raw_response is not None:
            raw_path = args.output_dir / f"{sample['sample_id']}.raw_response.json"
            raw_path.write_text(json.dumps(raw_response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": display_path(output_path),
                "parse_status": prediction["parse_status"],
                "repair_attempted": prediction["repair_attempted"],
                "frame_count": len(prediction["frames"]),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
