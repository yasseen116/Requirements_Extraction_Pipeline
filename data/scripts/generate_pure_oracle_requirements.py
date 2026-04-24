#!/usr/bin/env python3
"""Build an oracle baseline by projecting source PURE requirements into output format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "oracle_requirements"


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def infer_nfr_category(text: str) -> str | None:
    lower = text.lower()
    if any(token in lower for token in ["latency", "throughput", "response time", "seconds", "performance"]):
        return "performance"
    if any(token in lower for token in ["security", "encrypt", "authentication", "authorization", "password", "access control"]):
        return "security"
    if any(token in lower for token in ["available", "availability", "uptime"]):
        return "availability"
    if any(token in lower for token in ["reliable", "reliability", "recover", "failure"]):
        return "reliability"
    if any(token in lower for token in ["compliance", "policy", "regulation", "standard"]):
        return "compliance"
    return None


def convert_sample(sample: dict) -> dict:
    functional = []
    non_functional = []
    data = []
    business_rules = []
    interfaces = []
    constraints = []
    fr_index = 1
    nfr_index = 1

    for item in sample["ground_truth_requirements"]:
        text = normalize_text(item["text"])
        if not text:
            continue
        category = infer_nfr_category(text)
        if category is not None:
            non_functional.append(
                {
                    "id": f"NFR-{nfr_index:03d}",
                    "category": category,
                    "text": text if text.endswith(".") else f"{text}.",
                    "priority": "medium",
                    "evidence_turns": [2],
                }
            )
            nfr_index += 1
            continue
        functional.append(
            {
                "id": f"FR-{fr_index:03d}",
                "text": text if text.endswith(".") else f"{text}.",
                "priority": "medium",
                "evidence_turns": [2],
            }
        )
        fr_index += 1

    return {
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "method": "oracle_source_projection_v1",
        "model": None,
        "prompt_hash": None,
        "schema_hash": None,
        "parse_status": "ok",
        "repair_attempted": False,
        "usage": {},
        "error": None,
        "project_summary": f"Oracle baseline projection from source PURE requirements for {sample['source']['title']}.",
        "requirements": {
            "functional": functional,
            "non_functional": non_functional,
            "data": data,
            "business_rules": business_rules,
            "interfaces": interfaces,
            "constraints": constraints,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in args.input_dir.glob("*.json") if path.name not in {"summary.json", "evaluation.json"})
    if args.max_samples is not None:
        paths = paths[: args.max_samples]

    summary = []
    for path in paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        output = convert_sample(sample)
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        req_count = sum(
            len(output["requirements"][key])
            for key in ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]
        )
        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(output_path.relative_to(ROOT)),
                "requirement_count": req_count,
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
