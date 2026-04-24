#!/usr/bin/env python3
"""Run the OpenAI-compatible API baseline end-to-end."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def require_env(name: str) -> None:
    if not os.environ.get(name, "").strip():
        raise SystemExit(
            f"Missing {name}. Set REQ_LLM_BASE_URL and REQ_LLM_MODEL first. "
            "REQ_LLM_API_KEY is optional if your endpoint does not need it."
        )


def main() -> int:
    require_env("REQ_LLM_BASE_URL")
    require_env("REQ_LLM_MODEL")

    for input_dir, slot_dir, req_dir, output_md, output_json, label in [
        (
            "raw_sources/manual_gold",
            "outputs/b3_openai_compatible_slots",
            "outputs/b3_openai_compatible_requirements",
            "outputs/input_output_results_b3_clean.md",
            "outputs/input_output_results_b3_clean.json",
            "B3 openai-compatible API baseline on clean pilot dialogues",
        ),
        (
            "synthetic/pilot_noisy",
            "outputs/b3_openai_compatible_slots_noisy",
            "outputs/b3_openai_compatible_requirements_noisy",
            "outputs/input_output_results_b3_noisy.md",
            "outputs/input_output_results_b3_noisy.json",
            "B3 openai-compatible API baseline on noisy pilot dialogues",
        ),
    ]:
        run(
            [
                "python3",
                "scripts/generate_b3_openai_compatible_slots.py",
                "--input-dir",
                input_dir,
                "--output-dir",
                slot_dir,
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_slot_predictions.py",
                "--gold-dir",
                input_dir,
                "--pred-dir",
                slot_dir,
                "--output",
                f"{slot_dir}/evaluation.json",
            ]
        )
        run(
            [
                "python3",
                "scripts/generate_requirements_from_slot_predictions.py",
                "--input-dir",
                slot_dir,
                "--output-dir",
                req_dir,
            ]
        )
        run(
            [
                "python3",
                "scripts/evaluate_requirement_predictions.py",
                "--gold-dir",
                input_dir,
                "--pred-dir",
                req_dir,
                "--output",
                f"{req_dir}/evaluation.json",
            ]
        )
        run(
            [
                "python3",
                "scripts/build_input_output_results_report.py",
                "--input-dir",
                input_dir,
                "--requirements-dir",
                req_dir,
                "--output-md",
                output_md,
                "--output-json",
                output_json,
                "--system-label",
                label,
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
