#!/usr/bin/env python3
"""Generate full structured requirements directly from source PURE requirements.

Baseline condition:
source requirements -> Gemini -> full requirements
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import llm_router as llm

from generate_pure_full_requirements import build_repair_prompt, merge_normalized_payloads, normalize_payload, requirement_count


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "direct_generated_requirements"
DEFAULT_PROMPT = ROOT / "prompts" / "pure_source_to_full_requirements_gemini.txt"
DEFAULT_SCHEMA = ROOT / "schemas" / "gemini_full_requirements_response.schema.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "ground_truth_requirements" not in payload:
            continue
        samples.append(payload)
    return samples


def render_source_requirements(sample: dict, max_requirements: int) -> str:
    lines = []
    for item in sample.get("ground_truth_requirements", [])[:max_requirements]:
        req_id = item.get("req_id", "REQ")
        text = clean_text(item.get("text", ""))
        if text:
            lines.append(f"- {req_id}: {text}")
    return "\n".join(lines)


def build_prompt(template: str, sample: dict, max_requirements: int) -> str:
    source = sample.get("source", {})
    return (
        template.replace("{{SAMPLE_ID}}", sample.get("sample_id", ""))
        .replace("{{DOCUMENT_ID}}", str(source.get("document_id", "")))
        .replace("{{TITLE}}", str(source.get("title", "")))
        .replace("{{SOURCE_REQUIREMENTS}}", render_source_requirements(sample, max_requirements))
    )


def build_chunked_source_samples(
    sample: dict,
    *,
    max_requirements: int,
    chunk_size: int,
    chunk_char_budget: int,
) -> list[dict]:
    items = sample.get("ground_truth_requirements", [])[:max_requirements]
    if not items:
        return [sample]

    chunks = []
    current = []
    current_chars = 0
    for item in items:
        text = clean_text(item.get("text", ""))
        item_chars = len(text)
        if current and (len(current) >= chunk_size or current_chars + item_chars > chunk_char_budget):
            chunk_sample = dict(sample)
            chunk_sample["ground_truth_requirements"] = current
            chunks.append(chunk_sample)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        chunk_sample = dict(sample)
        chunk_sample["ground_truth_requirements"] = current
        chunks.append(chunk_sample)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-source-requirements", type=int, default=250)
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=1,
        help="Number of independent generations to run per sample (merged by dedup).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Keep this at 0.0 for deterministic extraction.",
    )
    parser.add_argument("--chunk-source-requirements", action="store_true")
    parser.add_argument("--source-chunk-size", type=int, default=25)
    parser.add_argument("--source-chunk-char-budget", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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

    model_name = llm.active_model_name()
    summary = []

    for sample in samples:
        repair_attempted = False
        error_messages = []
        segment_payloads = []
        segment_statuses = []
        raw_responses = []
        usages = []

        segments = [sample]
        if args.chunk_source_requirements:
            segments = build_chunked_source_samples(
                sample,
                max_requirements=args.max_source_requirements,
                chunk_size=args.source_chunk_size,
                chunk_char_budget=args.source_chunk_char_budget,
            )

        for segment_index, chunk_sample in enumerate(segments, start=1):
            prompt = build_prompt(prompt_template, chunk_sample, args.max_source_requirements)
            invalid_text = ""
            try:
                runs = max(1, int(args.self_consistency))
                candidates = []
                segment_usages = []
                segment_raw_responses = []
                for _run_idx in range(runs):
                    response = llm.generate_json(prompt, schema, temperature=args.temperature)
                    invalid_text = response.text
                    parsed = llm.parse_first_json_object(response.text)
                    candidates.append(normalize_payload(parsed))
                    segment_usages.append(response.usage)
                    segment_raw_responses.append(response.raw_response)

                merged_segment = merge_normalized_payloads(candidates)
                segment_payloads.append(merged_segment)
                status = "ok" if runs == 1 else f"ok_self_consistency_{runs}"
                segment_statuses.append(f"segment_{segment_index}:{status}")
                usages.append({"segment_index": segment_index, "runs": runs, "per_run": segment_usages})
                raw_responses.append({"segment_index": segment_index, "responses": segment_raw_responses})
            except Exception as first_error:  # noqa: BLE001
                repair_attempted = True
                try:
                    repair_prompt = build_repair_prompt(schema_text, invalid_text or str(first_error))
                    response = llm.generate_json(repair_prompt, schema, temperature=0.0)
                    parsed = llm.parse_first_json_object(response.text)
                    repaired_payload = normalize_payload(parsed)
                    segment_payloads.append(repaired_payload)
                    segment_statuses.append(f"segment_{segment_index}:repaired")
                    usages.append({"segment_index": segment_index, "repair": response.usage})
                    raw_responses.append({"segment_index": segment_index, "responses": [response.raw_response]})
                except Exception as second_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"segment_{segment_index}: {type(first_error).__name__}: {first_error}; "
                        f"repair failed with {type(second_error).__name__}: {second_error}"
                    ) from second_error

        normalized_payload = merge_normalized_payloads(segment_payloads) if segment_payloads else None
        if args.chunk_source_requirements:
            parse_status = f"chunked_{len(segments)}_segments"
        else:
            parse_status = segment_statuses[0].split(":", 1)[-1] if segment_statuses else "failed"
        raw_response = {"segments": raw_responses} if raw_responses else None
        usage = {"segments": usages}
        error = " | ".join(error_messages) if error_messages else None

        output_payload = {
            "sample_id": sample["sample_id"],
            "source": sample.get("source"),
            "method": "g_direct_source_to_full_requirements_v1",
            "model": model_name,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "parse_status": parse_status,
            "repair_attempted": repair_attempted,
            "usage": usage,
            "error": error,
            "chunking": {
                "enabled": bool(args.chunk_source_requirements),
                "segment_count": len(segments),
                "segment_statuses": segment_statuses,
                "source_chunk_size": args.source_chunk_size if args.chunk_source_requirements else None,
                "source_chunk_char_budget": args.source_chunk_char_budget if args.chunk_source_requirements else None,
            },
            "project_summary": normalized_payload["project_summary"] if normalized_payload else "",
            "requirements": normalized_payload["requirements"] if normalized_payload else {
                "functional": [],
                "non_functional": [],
                "data": [],
                "business_rules": [],
                "interfaces": [],
                "constraints": [],
            },
        }
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if raw_response is not None:
            raw_path = args.output_dir / f"{sample['sample_id']}.raw_response.json"
            raw_path.write_text(json.dumps(raw_response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        req_count = requirement_count(normalized_payload) if normalized_payload else 0
        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(output_path),
                "parse_status": parse_status,
                "segment_count": len(segments),
                "requirement_count": req_count,
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
