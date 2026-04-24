#!/usr/bin/env python3
"""Generate templated requirements from predicted slot files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from requirements_generation_utils import generate_requirements_from_slots


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "b1_rule_based_slots"
DEFAULT_OUTPUT = ROOT / "outputs" / "b1_rule_based_requirements"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_predictions(input_dir: Path) -> list[dict]:
    predictions = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        payload = json.loads(path.read_text())
        if "sample_id" not in payload or "slots" not in payload:
            continue
        predictions.append(payload)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method-suffix",
        default="template_generation_v1",
        help="Suffix appended to the prediction method name.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for prediction in load_predictions(args.input_dir):
        generated = generate_requirements_from_slots(
            sample_id=prediction["sample_id"],
            slots=prediction["slots"],
            method=f"{prediction['method']}__{args.method_suffix}",
        )
        output_path = args.output_dir / f"{prediction['sample_id']}.json"
        output_path.write_text(json.dumps(generated, indent=2))
        summary.append(
            {
                "sample_id": prediction["sample_id"],
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
