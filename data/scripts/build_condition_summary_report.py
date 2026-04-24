#!/usr/bin/env python3
"""Build a compact markdown summary for one evaluation condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_failures(frames_dir: Path) -> dict:
    parse_status_counts: dict[str, int] = {}
    for path in sorted(frames_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = load_json(path)
        status = payload.get("parse_status", "unknown")
        parse_status_counts[status] = parse_status_counts.get(status, 0) + 1
    failed = parse_status_counts.get("failed", 0)
    return {
        "parse_status_counts": parse_status_counts,
        "failed_samples": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-eval", type=Path, required=True)
    parser.add_argument("--slot-eval", type=Path, required=True)
    parser.add_argument("--req-eval", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--system-label", required=True)
    args = parser.parse_args()

    frame_eval = load_json(args.frame_eval)
    slot_eval = load_json(args.slot_eval)
    req_eval = load_json(args.req_eval)
    failure_stats = count_failures(args.frames_dir)

    payload = {
        "system_label": args.system_label,
        "condition": args.condition,
        "sample_count": frame_eval["sample_count"],
        "failed_samples": failure_stats["failed_samples"],
        "parse_status_counts": failure_stats["parse_status_counts"],
        "metrics": {
            "frame_f1": frame_eval["aggregate"]["overall_f1_macro"],
            "frame_consistency": frame_eval["aggregate"]["dialogue_frame_consistency_macro"],
            "slot_f1": slot_eval["aggregate"]["overall_f1_macro"],
            "requirement_f1": req_eval["aggregate"]["overall_f1_macro"],
            "coverage": req_eval["aggregate"]["coverage_macro"],
            "hallucination_rate": req_eval["aggregate"]["hallucination_rate_macro"],
        },
    }

    lines = [
        f"# {args.system_label} {args.condition.capitalize()} Summary",
        "",
        f"- Sample count: `{payload['sample_count']}`",
        f"- Failed samples: `{payload['failed_samples']}`",
        f"- Parse status counts: `{payload['parse_status_counts']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Frame F1 | {payload['metrics']['frame_f1']:.4f} |",
        f"| Frame Consistency | {payload['metrics']['frame_consistency']:.4f} |",
        f"| Slot F1 | {payload['metrics']['slot_f1']:.4f} |",
        f"| Requirement F1 | {payload['metrics']['requirement_f1']:.4f} |",
        f"| Coverage | {payload['metrics']['coverage']:.4f} |",
        f"| Hallucination Rate | {payload['metrics']['hallucination_rate']:.4f} |",
        "",
    ]

    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
