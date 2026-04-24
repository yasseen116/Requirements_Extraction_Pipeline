#!/usr/bin/env python3
"""Postprocess generated PURE requirements without new LLM calls.

This is intentionally simple:
- fix common text normalization issues (e.g., "The system shall X shall ...")
- drop meta/disclaimer lines that are not requirements
- deduplicate exact/near-exact repeats

It exists so we can improve measured coverage/precision when the original run
contained formatting artifacts, without re-running Gemini.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_pure_full_requirements as norm


ROOT = Path(__file__).resolve().parent.parent


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "requirements" not in payload:
            continue
        samples.append(payload)
    return samples


def postprocess_payload(payload: dict) -> dict:
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        reqs = {}

    processed = {
        "sample_id": payload.get("sample_id", ""),
        "source": payload.get("source"),
        "method": f"{payload.get('method','')}_postprocess_v2".strip("_"),
        "model": payload.get("model"),
        "prompt_hash": payload.get("prompt_hash"),
        "schema_hash": payload.get("schema_hash"),
        "parse_status": payload.get("parse_status"),
        "repair_attempted": payload.get("repair_attempted"),
        "usage": payload.get("usage", {}),
        "error": payload.get("error"),
        "project_summary": norm.clean_text(payload.get("project_summary", "")) or "Postprocessed.",
        "requirements": norm.finalize_requirements(reqs),
    }
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.input_dir)
    summary = []

    for sample in samples:
        processed = postprocess_payload(sample)
        out_path = args.output_dir / f"{processed['sample_id']}.json"
        out_path.write_text(json.dumps(processed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        req_count = sum(len(processed["requirements"][key]) for key in norm.REQUIREMENT_KEYS)
        # Use string path as-is; avoid brittle relative_to() behavior when paths are already relative.
        summary.append({"sample_id": processed["sample_id"], "path": str(out_path), "count": req_count})

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
