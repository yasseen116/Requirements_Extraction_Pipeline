#!/usr/bin/env python3
"""Run the native Gemini frame-first benchmark pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name}")
    return value


def prepare_input_dir(source_dir: Path, target_dir: Path, max_samples: int | None) -> Path:
    if max_samples is None:
        return source_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = []
    for path in sorted(source_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        sample_paths.append(path)
    for path in sample_paths[:max_samples]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        (target_dir / path.name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def build_manifest(run_id: str, run_dir: Path, prompt_path: Path, schema_path: Path, model: str, max_samples: int | None) -> dict:
    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "gemini",
        "model": model,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_hash": sha256_file(prompt_path),
        "schema_path": str(schema_path.resolve()),
        "schema_hash": sha256_file(schema_path),
        "temperature": float(os.environ.get("REQ_GEMINI_TEMPERATURE", "0.0")),
        "debug_raw": os.environ.get("REQ_GEMINI_DEBUG_RAW", "").strip().lower() == "true",
        "max_samples": max_samples,
        "conditions": {},
    }


def main() -> int:
    args = parse_args()
    require_env("REQ_GEMINI_API_KEY")
    model = require_env("REQ_GEMINI_MODEL")

    prompt_path = ROOT / "prompts" / "dialogue_to_frames_gemini.txt"
    schema_path = ROOT / "schemas" / "gemini_requirement_frames_response.schema.json"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{sanitize(model)}"
    run_dir = ROOT / "outputs" / "g1_gemini_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(run_id, run_dir, prompt_path, schema_path, model, args.max_samples)

    for condition, input_dir in [("clean", "raw_sources/manual_gold"), ("noisy", "synthetic/pilot_noisy")]:
        source_input_dir = ROOT / input_dir
        effective_input_dir = prepare_input_dir(
            source_input_dir,
            run_dir / "input_snapshots" / condition,
            args.max_samples,
        )
        condition_dir = run_dir / condition
        frames_dir = condition_dir / "frames"
        combined_dir = condition_dir / "combined"
        slots_dir = condition_dir / "slots"
        requirements_dir = condition_dir / "requirements"
        metrics_dir = condition_dir / "metrics"
        reports_dir = condition_dir / "reports"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        generate_cmd = [
                "python3",
                "scripts/generate_g1_gemini_frames.py",
                "--input-dir",
                str(effective_input_dir),
                "--output-dir",
                str(frames_dir),
                "--prompt-path",
            str(prompt_path),
            "--schema-path",
            str(schema_path),
        ]
        if args.max_samples is not None:
            generate_cmd.extend(["--max-samples", str(args.max_samples)])

        run(generate_cmd)
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

        frame_summary = json.loads((frames_dir / "summary.json").read_text(encoding="utf-8"))
        failed_samples = sum(1 for item in frame_summary if item.get("parse_status") == "failed")
        sample_count = len(frame_summary)

        manifest["conditions"][condition] = {
            "input_dir": str(source_input_dir.resolve()),
            "effective_input_dir": str(effective_input_dir.resolve()),
            "frames_dir": str(frames_dir.resolve()),
            "combined_dir": str(combined_dir.resolve()),
            "slots_dir": str(slots_dir.resolve()),
            "requirements_dir": str(requirements_dir.resolve()),
            "metrics_dir": str(metrics_dir.resolve()),
            "reports_dir": str(reports_dir.resolve()),
            "sample_count": sample_count,
            "failed_samples": failed_samples,
        }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest_path = ROOT / "outputs" / "g1_gemini_latest_run.json"
    latest_path.write_text(json.dumps({"run_id": run_id, "run_dir": str(run_dir.resolve())}, indent=2) + "\n", encoding="utf-8")
    run(["python3", "scripts/summarize_baseline_comparison.py"])
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
