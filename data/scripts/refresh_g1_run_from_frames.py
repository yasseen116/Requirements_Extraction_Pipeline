#!/usr/bin/env python3
"""Rebuild a saved G1 run from existing frame outputs using the current normalizer."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--promote-latest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = load_json(manifest_path)

    for condition, config in manifest["conditions"].items():
        effective_input_dir = Path(config["effective_input_dir"])
        frames_dir = Path(config["frames_dir"])
        combined_dir = Path(config["combined_dir"])
        slots_dir = Path(config["slots_dir"])
        requirements_dir = Path(config["requirements_dir"])
        metrics_dir = Path(config["metrics_dir"])
        reports_dir = Path(config["reports_dir"])

        run(
            [
                "python3",
                "scripts/evaluate_frame_predictions.py",
                "--gold-dir",
                str(effective_input_dir),
                "--pred-dir",
                str(frames_dir),
                "--source-dir",
                str(effective_input_dir),
                "--output",
                str(metrics_dir / "frame_evaluation.json"),
            ]
        )
        run(
            [
                "python3",
                "scripts/generate_slots_requirements_from_frames.py",
                "--input-dir",
                str(frames_dir),
                "--source-dir",
                str(effective_input_dir),
                "--combined-dir",
                str(combined_dir),
                "--output-slot-dir",
                str(slots_dir),
                "--output-req-dir",
                str(requirements_dir),
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_slot_predictions.py",
                "--gold-dir",
                str(effective_input_dir),
                "--pred-dir",
                str(slots_dir),
                "--output",
                str(metrics_dir / "slot_evaluation.json"),
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_requirement_predictions.py",
                "--gold-dir",
                str(effective_input_dir),
                "--pred-dir",
                str(requirements_dir),
                "--output",
                str(metrics_dir / "requirement_evaluation.json"),
            ]
        )
        run(
            [
                "python3",
                "scripts/build_input_output_results_report.py",
                "--input-dir",
                str(effective_input_dir),
                "--combined-dir",
                str(combined_dir),
                "--output-md",
                str(reports_dir / f"input_output_{condition}.md"),
                "--output-json",
                str(reports_dir / f"input_output_{condition}.json"),
                "--system-label",
                f"G1 Gemini frame-slot pipeline on {condition} pilot dialogues",
            ]
        )
        run(
            [
                "python3",
                "scripts/build_input_output_results_report.py",
                "--input-dir",
                str(effective_input_dir),
                "--combined-dir",
                str(combined_dir),
                "--output-md",
                str(ROOT / "outputs" / f"input_output_results_g1_{condition}.md"),
                "--output-json",
                str(ROOT / "outputs" / f"input_output_results_g1_{condition}.json"),
                "--system-label",
                f"G1 Gemini frame-slot pipeline on {condition} pilot dialogues",
            ]
        )
        run(
            [
                "python3",
                "scripts/build_condition_summary_report.py",
                "--condition",
                condition,
                "--frames-dir",
                str(frames_dir),
                "--frame-eval",
                str(metrics_dir / "frame_evaluation.json"),
                "--slot-eval",
                str(metrics_dir / "slot_evaluation.json"),
                "--req-eval",
                str(metrics_dir / "requirement_evaluation.json"),
                "--output-md",
                str(reports_dir / f"summary_{condition}.md"),
                "--output-json",
                str(reports_dir / f"summary_{condition}.json"),
                "--system-label",
                "G1 Gemini frame-slot pipeline",
            ]
        )
        run(
            [
                "python3",
                "scripts/build_condition_summary_report.py",
                "--condition",
                condition,
                "--frames-dir",
                str(frames_dir),
                "--frame-eval",
                str(metrics_dir / "frame_evaluation.json"),
                "--slot-eval",
                str(metrics_dir / "slot_evaluation.json"),
                "--req-eval",
                str(metrics_dir / "requirement_evaluation.json"),
                "--output-md",
                str(ROOT / "outputs" / f"g1_summary_{condition}.md"),
                "--output-json",
                str(ROOT / "outputs" / f"g1_summary_{condition}.json"),
                "--system-label",
                "G1 Gemini frame-slot pipeline",
            ]
        )

    manifest["refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.promote_latest:
        latest_path = ROOT / "outputs" / "g1_gemini_latest_run.json"
        latest_path.write_text(
            json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, indent=2) + "\n",
            encoding="utf-8",
        )

    run(["python3", "scripts/summarize_baseline_comparison.py"])
    run(["python3", "scripts/build_pilot_flow_report.py"])
    print(
        json.dumps(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "promoted_latest": args.promote_latest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
