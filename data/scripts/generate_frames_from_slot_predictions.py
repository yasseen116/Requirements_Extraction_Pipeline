#!/usr/bin/env python3
"""Project slot predictions into canonical requirement frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frame_slot_utils import derive_frames_from_slots


ROOT = Path(__file__).resolve().parent.parent


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "slots" not in payload:
            continue
        predictions.append(payload)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-suffix", default="derived_frames_v1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for prediction in load_predictions(args.input_dir):
        frames = derive_frames_from_slots(prediction["slots"])
        output_payload = {
            "sample_id": prediction["sample_id"],
            "method": f"{prediction['method']}__{args.method_suffix}",
            "frames": frames,
        }
        output_path = args.output_dir / f"{prediction['sample_id']}.json"
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append(
            {
                "sample_id": prediction["sample_id"],
                "path": display_path(output_path),
                "frame_count": len(frames),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
