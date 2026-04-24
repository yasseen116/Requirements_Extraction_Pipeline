#!/usr/bin/env python3
"""Generate the B0 baseline: gold slots -> template requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from requirements_generation_utils import generate_requirements_from_slots


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "manual_gold"
DEFAULT_OUTPUT = ROOT / "outputs" / "b0_template_requirements"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())

def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("template_"):
            continue
        samples.append(json.loads(path.read_text()))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for sample in load_samples(args.input_dir):
        generated = generate_requirements_from_slots(
            sample_id=sample["sample_id"],
            slots=sample["slots"],
            method="b0_template_from_gold_slots_v1",
        )
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(generated, indent=2))
        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": display_path(output_path),
                "functional_requirements": len(generated["requirements"]["functional"]),
                "non_functional_requirements": len(generated["requirements"]["non_functional"]),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
