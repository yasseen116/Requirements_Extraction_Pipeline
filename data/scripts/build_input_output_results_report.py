#!/usr/bin/env python3
"""Create a readable file showing system inputs and generated requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a readable report of dialogue inputs and generated requirements."
    )
    parser.add_argument(
        "--input-dir",
        default="synthetic/pilot_noisy",
        help="Directory containing the input dialogue samples.",
    )
    parser.add_argument(
        "--combined-dir",
        default=None,
        help="Directory containing combined predictions with frames, slots, and requirements.",
    )
    parser.add_argument(
        "--requirements-dir",
        default="outputs/b2_keyword_normalized_requirements_noisy",
        help="Directory containing generated requirement predictions.",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/input_output_results_b2_noisy.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/input_output_results_b2_noisy.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--system-label",
        default="B2 keyword-normalized baseline on noisy pilot dialogues",
        help="Human-readable label for the system used.",
    )
    return parser.parse_args()


def build_markdown(samples: list[dict], system_label: str) -> str:
    lines: list[str] = []
    lines.append("# Input To Requirements Results")
    lines.append("")
    lines.append(f"System: `{system_label}`")
    lines.append("")
    lines.append("This file shows exactly what was fed into the system and the requirements it produced.")
    lines.append("")

    for sample in samples:
        lines.append(f"## {sample['sample_id']}")
        lines.append("")
        lines.append(f"- Domain: `{sample['domain']}`")
        lines.append(f"- Split: `{sample['split']}`")
        lines.append(f"- Source dataset: `{sample['source_dataset']}`")
        lines.append(f"- Source document: `{sample['source_document_id']}`")
        lines.append("")
        lines.append("### Input Dialogue")
        lines.append("")
        for turn in sample["dialogue"]:
            lines.append(f"- `{turn['turn_id']}` `{turn['role']}`: {turn['text']}")
        lines.append("")
        if sample.get("frames") is not None:
            lines.append("### Output Frames")
            lines.append("")
            if sample["frames"]:
                for frame in sample["frames"]:
                    lines.append(
                        "- "
                        + f"`{frame['frame_id']}` `{frame['kind']}` "
                        + f"actor={frame['actor']!r} action={frame['action']!r} "
                        + f"object={frame['object']!r} constraint={frame['constraint']!r} "
                        + f"turns={frame['evidence_turns']}"
                    )
            else:
                lines.append("- none")
            lines.append("")
        if sample.get("slots") is not None:
            lines.append("### Derived Slots")
            lines.append("")
            slots = sample["slots"]
            lines.append(f"- System type: `{slots['system_type']['value']}`")
            lines.append(
                "- User roles: "
                + (", ".join(item["value"] for item in slots["user_roles"]) if slots["user_roles"] else "none")
            )
            lines.append(
                "- Functional capabilities: "
                + (str(len(slots["functional_capabilities"])) if slots["functional_capabilities"] else "0")
            )
            for capability in slots["functional_capabilities"]:
                lines.append(f"- {capability['actor']} -> {capability['action']}")
            lines.append(
                "- Authentication: "
                + (
                    ", ".join(slots["authentication"]["methods"])
                    if slots["authentication"]["required"]
                    else "not required"
                )
            )
            lines.append(
                "- Performance constraints: "
                + ("; ".join(item["text"] for item in slots["performance_constraints"]) if slots["performance_constraints"] else "none")
            )
            lines.append(
                "- Security constraints: "
                + ("; ".join(item["text"] for item in slots["security_constraints"]) if slots["security_constraints"] else "none")
            )
            lines.append("")
        lines.append("### Output Functional Requirements")
        lines.append("")
        if sample["requirements"]["functional"]:
            for item in sample["requirements"]["functional"]:
                lines.append(f"- `{item['id']}` {item['text']}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("### Output Non-Functional Requirements")
        lines.append("")
        if sample["requirements"]["non_functional"]:
            for item in sample["requirements"]["non_functional"]:
                lines.append(f"- `{item['id']}` `{item['category']}` {item['text']}")
        else:
            lines.append("- none")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    input_dir = ROOT / args.input_dir
    combined_dir = ROOT / args.combined_dir if args.combined_dir else None
    requirements_dir = ROOT / args.requirements_dir if args.requirements_dir else None
    output_md = ROOT / args.output_md
    output_json = ROOT / args.output_json

    samples: list[dict] = []
    for input_path in sorted(input_dir.glob("*.json")):
        input_payload = load_json(input_path)
        if not isinstance(input_payload, dict) or "sample_id" not in input_payload:
            continue

        sample_id = input_payload["sample_id"]
        combined_payload = None
        requirements_payload = None
        if combined_dir is not None:
            combined_path = combined_dir / f"{sample_id}.json"
            if combined_path.exists():
                combined_payload = load_json(combined_path)
        if requirements_dir is not None:
            requirements_path = requirements_dir / f"{sample_id}.json"
            if requirements_path.exists():
                requirements_payload = load_json(requirements_path)

        if combined_payload is None and requirements_payload is None:
            continue

        requirements = (
            combined_payload["requirements"]
            if combined_payload is not None and "requirements" in combined_payload
            else requirements_payload["requirements"]
        )
        sample_record = {
            "sample_id": sample_id,
            "system_label": args.system_label,
            "domain": input_payload["metadata"]["domain"],
            "split": input_payload["metadata"]["split"],
            "source_dataset": input_payload["source"]["dataset"],
            "source_document_id": input_payload["source"]["document_id"],
            "dialogue": input_payload["dialogue"],
            "frames": combined_payload.get("frames") if combined_payload is not None else None,
            "slots": combined_payload.get("slots") if combined_payload is not None else None,
            "requirements": requirements,
        }
        samples.append(sample_record)

    output_payload = {
        "system_label": args.system_label,
        "sample_count": len(samples),
        "samples": samples,
    }

    output_md.write_text(build_markdown(samples, args.system_label), encoding="utf-8")
    output_json.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {display_path(output_md)} and {display_path(output_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
