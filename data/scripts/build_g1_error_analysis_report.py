#!/usr/bin/env python3
"""Build a readable error analysis report for the promoted G1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frame_slot_utils import canonicalize_frames, flatten_frame, normalize_frame


ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def normalize_requirement_item(item: dict) -> str:
    if "category" in item:
        return f"{item['category']}::{normalize_text(item['text'])}"
    return normalize_text(item["text"])


def slot_items(slots: dict) -> dict[str, set[str]]:
    return {
        "system_type": {normalize_text(slots["system_type"]["value"])},
        "user_roles": {normalize_text(item["value"]) for item in slots["user_roles"]},
        "functional_capabilities": {
            f"{normalize_text(item['actor'])}::{normalize_text(item['action'])}"
            for item in slots["functional_capabilities"]
        },
        "authentication_required": {str(bool(slots["authentication"]["required"])).lower()},
        "authentication_methods": {normalize_text(item) for item in slots["authentication"]["methods"]},
        "performance_constraints": {normalize_text(item["text"]) for item in slots["performance_constraints"]},
        "security_constraints": {normalize_text(item["text"]) for item in slots["security_constraints"]},
    }


def requirement_items(requirements: dict) -> dict[str, set[str]]:
    return {
        "functional": {normalize_requirement_item(item) for item in requirements["functional"]},
        "non_functional": {normalize_requirement_item(item) for item in requirements["non_functional"]},
    }


def frame_items(frames: list[dict]) -> set[str]:
    canonical = canonicalize_frames(
        [normalize_frame(frame, default_index=index) for index, frame in enumerate(frames, start=1)]
    )
    return {flatten_frame(frame) for frame in canonical}


def diff_sets(gold: set[str], pred: set[str]) -> dict[str, list[str]]:
    return {
        "missing": sorted(gold - pred),
        "extra": sorted(pred - gold),
    }


def build_condition_report(condition: str, gold_dir: Path, run_dir: Path) -> dict:
    combined_dir = run_dir / condition / "combined"
    frame_eval = load_json(run_dir / condition / "metrics" / "frame_evaluation.json")
    slot_eval = load_json(run_dir / condition / "metrics" / "slot_evaluation.json")
    req_eval = load_json(run_dir / condition / "metrics" / "requirement_evaluation.json")

    per_sample = []
    for gold_path in sorted(gold_dir.glob("*.json")):
        if gold_path.name in {"summary.json", "evaluation.json"} or gold_path.name.startswith("template_"):
            continue
        gold = load_json(gold_path)
        pred = load_json(combined_dir / gold_path.name)

        frame_sample = next(item for item in frame_eval["per_sample"] if item["sample_id"] == gold["sample_id"])
        slot_sample = next(item for item in slot_eval["per_sample"] if item["sample_id"] == gold["sample_id"])
        req_sample = next(item for item in req_eval["per_sample"] if item["sample_id"] == gold["sample_id"])

        gold_slot_items = slot_items(gold["slots"])
        pred_slot_items = slot_items(pred["slots"])
        gold_req_items = requirement_items(gold["requirements"])
        pred_req_items = requirement_items(pred["requirements"])

        per_sample.append(
            {
                "sample_id": gold["sample_id"],
                "frame_f1": frame_sample["overall"]["f1"],
                "slot_f1": slot_sample["overall"]["f1"],
                "requirement_f1": req_sample["overall"]["f1"],
                "coverage": req_sample["overall"]["coverage"],
                "hallucination_rate": req_sample["overall"]["hallucination_rate"],
                "frame_diff": diff_sets(frame_items(gold["frames"]), frame_items(pred["frames"])),
                "slot_diff": {
                    key: diff_sets(gold_slot_items[key], pred_slot_items[key])
                    for key in gold_slot_items
                    if gold_slot_items[key] != pred_slot_items[key]
                },
                "requirement_diff": {
                    key: diff_sets(gold_req_items[key], pred_req_items[key])
                    for key in gold_req_items
                    if gold_req_items[key] != pred_req_items[key]
                },
            }
        )

    return {
        "condition": condition,
        "metrics": {
            "frame_f1": frame_eval["aggregate"]["overall_f1_macro"],
            "slot_f1": slot_eval["aggregate"]["overall_f1_macro"],
            "requirement_f1": req_eval["aggregate"]["overall_f1_macro"],
            "coverage": req_eval["aggregate"]["coverage_macro"],
            "hallucination_rate": req_eval["aggregate"]["hallucination_rate_macro"],
        },
        "samples": per_sample,
    }


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# G1 Error Analysis")
    lines.append("")
    lines.append(f"Run: `{payload['run_id']}`")
    lines.append("")

    for condition_data in payload["conditions"]:
        lines.append(f"## {condition_data['condition'].capitalize()}")
        lines.append("")
        lines.append(
            f"- Frame F1: `{condition_data['metrics']['frame_f1']:.4f}`"
        )
        lines.append(
            f"- Slot F1: `{condition_data['metrics']['slot_f1']:.4f}`"
        )
        lines.append(
            f"- Requirement F1: `{condition_data['metrics']['requirement_f1']:.4f}`"
        )
        lines.append(
            f"- Coverage: `{condition_data['metrics']['coverage']:.4f}`"
        )
        lines.append(
            f"- Hallucination rate: `{condition_data['metrics']['hallucination_rate']:.4f}`"
        )
        lines.append("")

        for sample in condition_data["samples"]:
            if (
                sample["frame_f1"] == 1.0
                and sample["slot_f1"] == 1.0
                and sample["requirement_f1"] == 1.0
            ):
                continue
            lines.append(f"### {sample['sample_id']}")
            lines.append("")
            lines.append(f"- Frame F1: `{sample['frame_f1']:.4f}`")
            lines.append(f"- Slot F1: `{sample['slot_f1']:.4f}`")
            lines.append(f"- Requirement F1: `{sample['requirement_f1']:.4f}`")
            lines.append(f"- Coverage: `{sample['coverage']:.4f}`")
            lines.append(f"- Hallucination rate: `{sample['hallucination_rate']:.4f}`")
            lines.append("")

            if sample["frame_diff"]["missing"] or sample["frame_diff"]["extra"]:
                lines.append("Frame mismatches:")
                for item in sample["frame_diff"]["missing"]:
                    lines.append(f"- Missing frame: `{item}`")
                for item in sample["frame_diff"]["extra"]:
                    lines.append(f"- Extra frame: `{item}`")
                lines.append("")

            if sample["slot_diff"]:
                lines.append("Slot mismatches:")
                for slot_name, diff in sample["slot_diff"].items():
                    for item in diff["missing"]:
                        lines.append(f"- Missing {slot_name}: `{item}`")
                    for item in diff["extra"]:
                        lines.append(f"- Extra {slot_name}: `{item}`")
                lines.append("")

            if sample["requirement_diff"]:
                lines.append("Requirement mismatches:")
                for req_name, diff in sample["requirement_diff"].items():
                    for item in diff["missing"]:
                        lines.append(f"- Missing {req_name}: `{item}`")
                    for item in diff["extra"]:
                        lines.append(f"- Extra {req_name}: `{item}`")
                lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=ROOT / "outputs" / "g1_error_analysis.md")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "g1_error_analysis.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_dir is None:
        latest = load_json(ROOT / "outputs" / "g1_gemini_latest_run.json")
        run_dir = Path(latest["run_dir"])
        run_id = latest["run_id"]
    else:
        run_dir = args.run_dir.resolve()
        run_id = run_dir.name

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "conditions": [
            build_condition_report("clean", ROOT / "raw_sources" / "manual_gold", run_dir),
            build_condition_report("noisy", ROOT / "synthetic" / "pilot_noisy", run_dir),
        ],
    }
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.output_md.relative_to(ROOT)} and {args.output_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
