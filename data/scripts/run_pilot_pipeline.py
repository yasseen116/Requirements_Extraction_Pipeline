#!/usr/bin/env python3
"""Run the pilot benchmark pipeline end-to-end in a readable order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    run(["python3", "scripts/validate_source_grounded_samples.py"])
    run(["python3", "scripts/generate_b0_template_requirements.py"])
    run(["python3", "scripts/evaluate_requirement_predictions.py"])
    run(["python3", "scripts/generate_pilot_noisy_variants.py"])

    # Direct LLM baseline on the clean pilot set (requirements-only).
    # This provides a fair comparison baseline against slot/frames pipelines.
    run(["python3", "scripts/generate_pilot_direct_requirements.py"])
    run(
        [
            "python3",
            "scripts/evaluate_requirement_predictions.py",
            "--gold-dir",
            "raw_sources/manual_gold",
            "--pred-dir",
            "outputs/b3_pilot_direct_requirements",
            "--output",
            "outputs/b3_pilot_direct_requirements/evaluation.json",
        ]
    )

    for name, input_dir, pred_slots_dir, pred_frames_dir, pred_req_dir in [
        ("b1_clean", "raw_sources/manual_gold", "outputs/b1_rule_based_slots", "outputs/b1_rule_based_frames", "outputs/b1_rule_based_requirements"),
        ("b1_noisy", "synthetic/pilot_noisy", "outputs/b1_rule_based_slots_noisy", "outputs/b1_rule_based_frames_noisy", "outputs/b1_rule_based_requirements_noisy"),
        ("b2_clean", "raw_sources/manual_gold", "outputs/b2_keyword_normalized_slots", "outputs/b2_keyword_normalized_frames", "outputs/b2_keyword_normalized_requirements"),
        ("b2_noisy", "synthetic/pilot_noisy", "outputs/b2_keyword_normalized_slots_noisy", "outputs/b2_keyword_normalized_frames_noisy", "outputs/b2_keyword_normalized_requirements_noisy"),
    ]:
        generator = (
            "scripts/generate_b1_rule_based_slots.py"
            if name.startswith("b1")
            else "scripts/generate_b2_keyword_normalized_slots.py"
        )
        run(["python3", generator, "--input-dir", input_dir, "--output-dir", pred_slots_dir])
        run(
            [
                "python3",
                "scripts/generate_frames_from_slot_predictions.py",
                "--input-dir",
                pred_slots_dir,
                "--output-dir",
                pred_frames_dir,
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_frame_predictions.py",
                "--gold-dir",
                input_dir,
                "--pred-dir",
                pred_frames_dir,
                "--output",
                f"{pred_frames_dir}/evaluation.json",
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_slot_predictions.py",
                "--gold-dir",
                input_dir,
                "--pred-dir",
                pred_slots_dir,
                "--output",
                f"{pred_slots_dir}/evaluation.json",
            ]
        )
        run(
            [
                "python3",
                "scripts/generate_requirements_from_slot_predictions.py",
                "--input-dir",
                pred_slots_dir,
                "--output-dir",
                pred_req_dir,
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_requirement_predictions.py",
                "--gold-dir",
                input_dir,
                "--pred-dir",
                pred_req_dir,
                "--output",
                f"{pred_req_dir}/evaluation.json",
            ]
        )

    run(["python3", "scripts/summarize_pilot_results.py"])
    run(["python3", "scripts/summarize_baseline_comparison.py"])
    run(["python3", "scripts/build_pilot_flow_report.py"])
    run(["python3", "scripts/build_input_output_results_report.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
