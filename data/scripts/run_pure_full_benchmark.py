#!/usr/bin/env python3
"""Run comprehensive PURE benchmark: source extraction -> controlled dialogue -> full requirements -> coverage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "outputs" / "pure_full_runs"
LATEST_POINTER = ROOT / "outputs" / "pure_full_latest_run.json"

PYTHON_BIN = "python3"

def check_dependencies() -> None:
    missing = []
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    try:
        import torch
    except ImportError:
        missing.append("torch")
        
    if missing:
        print(f"[error] Missing required dependencies: {', '.join(missing)}")
        print(f"Please run: {PYTHON_BIN} -m pip install {' '.join(missing)}")
        sys.exit(1)


def run(cmd: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    if cmd and cmd[0] == "python3":
        cmd = [PYTHON_BIN, *cmd[1:]]
    print(f"[run] {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--min-requirements", type=int, default=10)
    parser.add_argument(
        "--python-bin",
        type=str,
        default=os.environ.get("REQ_PYTHON_BIN", "python3"),
        help="Python interpreter used for child benchmark scripts.",
    )
    parser.add_argument(
        "--reuse-source-dir",
        type=Path,
        default=None,
        help="Reuse an existing source_requirements directory instead of rebuilding PURE XML extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic document sampling (0 keeps stable default ordering).",
    )
    # Keep high enough so typical PURE docs are not truncated during dialogue generation.
    parser.add_argument("--max-source-requirements", type=int, default=200)
    parser.add_argument("--match-threshold", type=float, default=0.55)
    parser.add_argument("--max-reqs-per-answer", type=int, default=3)
    parser.add_argument("--max-chars-per-answer", type=int, default=650)
    parser.add_argument("--dialogue-coverage-threshold", type=float, default=0.55)
    parser.add_argument(
        "--clarification-rounds",
        type=int,
        default=0,
        help="Additional dialogue clarification passes for uncovered requirements after the initial controlled pass.",
    )
    parser.add_argument(
        "--dialogue-variant",
        choices=["controlled", "transcript_paraphrase", "partial_information"],
        default="controlled",
        help="Dialogue condition to evaluate.",
    )
    parser.add_argument(
        "--partial-drop-rate",
        type=float,
        default=0.2,
        help="Only for partial_information dialogue variant.",
    )
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=3,
        help="Number of independent generations per sample (merged by dedup).",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=0.3,
        help="Temperature for self-consistency runs (ignored when self-consistency=1).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "ollama"],
        default=None,
        help="Override REQ_LLM_PROVIDER for this benchmark invocation.",
    )
    parser.add_argument(
        "--dialogue-chunking",
        choices=["auto", "always", "never"],
        default="auto",
        help="Dialogue chunking policy for dialogue->requirements extraction. 'auto' chunks for Ollama only.",
    )
    parser.add_argument("--dialogue-chunk-max-turns", type=int, default=8)
    parser.add_argument("--dialogue-chunk-max-chars", type=int, default=2600)
    parser.add_argument("--dialogue-chunk-overlap-turns", type=int, default=2)
    parser.add_argument("--memory-max-items", type=int, default=24)
    parser.add_argument("--memory-max-chars", type=int, default=3500)
    return parser.parse_args()


def llm_provider(provider_override: str | None = None) -> str:
    if provider_override:
        return provider_override.strip().lower()
    return (os.environ.get("REQ_LLM_PROVIDER", "gemini") or "gemini").strip().lower()


def has_llm_env(provider_override: str | None = None) -> bool:
    if llm_provider(provider_override) == "ollama":
        return True
    return bool(os.environ.get("REQ_GEMINI_API_KEY")) and bool(os.environ.get("REQ_GEMINI_MODEL"))


def should_chunk_dialogues(provider_name: str, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return provider_name == "ollama"


def main() -> int:
    check_dependencies()
    global PYTHON_BIN
    args = parse_args()
    provider_name = llm_provider(args.llm_provider)
    extra_env = {"REQ_LLM_PROVIDER": provider_name} if args.llm_provider else None
    dialogue_chunking_enabled = should_chunk_dialogues(provider_name, args.dialogue_chunking)
    PYTHON_BIN = args.python_bin or os.environ.get("REQ_PYTHON_BIN") or sys.executable or "python3"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_pure_full"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_dir = run_dir / "source_requirements"
    source_summary = run_dir / "source_summary.json"
    oracle_dir = run_dir / "oracle_requirements"
    dialogue_dir = run_dir / "expanded_dialogues"
    variant_dialogue_dir = run_dir / "expanded_dialogues_variant"
    generated_dir = run_dir / "generated_requirements"
    direct_generated_dir = run_dir / "direct_generated_requirements"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_source_dir is not None:
        reuse_dir = args.reuse_source_dir.resolve()
        if not reuse_dir.exists():
            raise FileNotFoundError(f"Reuse source dir does not exist: {reuse_dir}")
        source_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in sorted(reuse_dir.glob("*.json")):
            if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
                continue
            dest = source_dir / path.name
            shutil.copy2(path, dest)
            copied.append(dest)
        summary_payload = {
            "dataset": "pure",
            "reused": True,
            "source_dir": str(reuse_dir),
            "generated_samples": len(copied),
            "samples": [
                {
                    "sample_id": path.stem,
                    "path": str(path.relative_to(ROOT)),
                    "status": "ok_reused",
                }
                for path in copied
            ],
        }
        source_summary.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        run(
            [
                "python3",
                "scripts/build_pure_requirements_benchmark.py",
                "--output-dir",
                str(source_dir),
                "--summary-path",
                str(source_summary),
                "--min-requirements",
                str(args.min_requirements),
                "--max-docs",
                str(args.max_samples),
                "--seed",
                str(args.seed),
            ],
            extra_env=extra_env,
        )

    run(
        [
            "python3",
            "scripts/generate_pure_oracle_requirements.py",
            "--input-dir",
            str(source_dir),
            "--output-dir",
            str(oracle_dir),
        ],
        extra_env=extra_env,
    )

    oracle_eval_path = metrics_dir / "oracle_coverage.json"
    run(
        [
            "python3",
            "scripts/evaluate_pure_requirements_coverage.py",
            "--gold-dir",
            str(source_dir),
            "--pred-dir",
            str(oracle_dir),
            "--match-threshold",
            str(args.match_threshold),
            "--output",
            str(oracle_eval_path),
        ],
        extra_env=extra_env,
    )

    dialogue_eval_path = metrics_dir / "dialogue_coverage_user_only.json"
    controlled_cmd = [
        "python3",
        "scripts/generate_pure_controlled_dialogues.py",
        "--input-dir",
        str(source_dir),
        "--output-dir",
        str(dialogue_dir),
        "--max-reqs-per-answer",
        str(args.max_reqs_per_answer),
        "--max-chars-per-answer",
        str(args.max_chars_per_answer),
        "--coverage-threshold",
        str(args.dialogue_coverage_threshold),
        "--clarification-rounds",
        str(args.clarification_rounds),
    ]
    run(controlled_cmd, extra_env=extra_env)
    effective_dialogue_dir = dialogue_dir
    if args.dialogue_variant != "controlled":
        variant_cmd = [
            "python3",
            "scripts/generate_pure_dialogue_variants.py",
            "--input-dir",
            str(dialogue_dir),
            "--output-dir",
            str(variant_dialogue_dir),
            "--variant",
            args.dialogue_variant,
            "--seed",
            str(args.seed if args.seed != 0 else 1),
            "--partial-drop-rate",
            str(args.partial_drop_rate),
        ]
        run(variant_cmd, extra_env=extra_env)
        effective_dialogue_dir = variant_dialogue_dir
    run(
        [
            "python3",
            "scripts/evaluate_pure_dialogue_coverage.py",
            "--gold-dir",
            str(source_dir),
            "--dialogue-dir",
            str(effective_dialogue_dir),
            "--match-threshold",
            str(args.match_threshold),
            "--user-only",
            "--output",
            str(dialogue_eval_path),
        ],
        extra_env=extra_env,
    )

    direct_eval_path = None
    direct_error_analysis_path = None
    pipeline_eval_path = None
    pipeline_error_analysis_path = None
    validation_report_path = None
    legacy_gemini_eval_path = None
    legacy_gemini_error_analysis_path = None
    pipeline_status = "skipped_missing_env"
    if has_llm_env(provider_name):
        pipeline_status = "ran"
        run(
            [
                "python3",
                "scripts/generate_pure_direct_requirements.py",
                "--input-dir",
                str(source_dir),
                "--output-dir",
                str(direct_generated_dir),
                "--max-source-requirements",
                str(args.max_source_requirements),
                "--self-consistency",
                str(args.self_consistency),
                "--temperature",
                str(args.self_consistency_temperature),
            ]
            + (
                [
                    "--chunk-source-requirements",
                    "--source-chunk-size",
                    "25",
                    "--source-chunk-char-budget",
                    "6000",
                ]
                if provider_name == "ollama"
                else []
            ),
            extra_env=extra_env,
        )
        direct_eval_path = metrics_dir / "direct_coverage.json"
        run(
            [
                "python3",
                "scripts/evaluate_pure_requirements_coverage.py",
                "--gold-dir",
                str(source_dir),
                "--pred-dir",
                str(direct_generated_dir),
                "--match-threshold",
                str(args.match_threshold),
                "--output",
                str(direct_eval_path),
            ],
            extra_env=extra_env,
        )
        direct_error_analysis_path = metrics_dir / "direct_error_analysis.json"
        run(
            [
                "python3",
                "scripts/analyze_pure_errors.py",
                "--gold-dir",
                str(source_dir),
                "--pred-dir",
                str(direct_generated_dir),
                "--threshold",
                str(args.match_threshold),
                "--output",
                str(direct_error_analysis_path),
            ],
            extra_env=extra_env,
        )

        raw_generated_dir = run_dir / "raw_generated_requirements"
        run(
            [
                "python3",
                "scripts/generate_pure_full_requirements.py",
                "--input-dir",
                str(effective_dialogue_dir),
                "--output-dir",
                str(raw_generated_dir),
                "--self-consistency",
                str(args.self_consistency),
                "--temperature",
                str(args.self_consistency_temperature),
            ]
            + (
                [
                    "--chunk-dialogues",
                    "--dialogue-chunk-max-turns",
                    str(args.dialogue_chunk_max_turns),
                    "--dialogue-chunk-max-chars",
                    str(args.dialogue_chunk_max_chars),
                    "--dialogue-chunk-overlap-turns",
                    str(args.dialogue_chunk_overlap_turns),
                    "--memory-max-items",
                    str(args.memory_max_items),
                    "--memory-max-chars",
                    str(args.memory_max_chars),
                ]
                if dialogue_chunking_enabled
                else []
            ),
            extra_env=extra_env,
        )

        validation_report_path = metrics_dir / "pipeline_validation_report.json"
        run(
            [
                "python3",
                "scripts/validate_pure_extracted_requirements.py",
                "--pred-dir",
                str(raw_generated_dir),
                "--dialogue-dir",
                str(effective_dialogue_dir),
                "--output-dir",
                str(generated_dir),
                "--report-path",
                str(validation_report_path),
            ],
            extra_env=extra_env,
        )

        pipeline_eval_path = metrics_dir / "pipeline_coverage.json"
        run(
            [
                "python3",
                "scripts/evaluate_pure_requirements_coverage.py",
                "--gold-dir",
                str(source_dir),
                "--pred-dir",
                str(generated_dir),
                "--match-threshold",
                str(args.match_threshold),
                "--output",
                str(pipeline_eval_path),
            ],
            extra_env=extra_env,
        )
        pipeline_error_analysis_path = metrics_dir / "pipeline_error_analysis.json"
        run(
            [
                "python3",
                "scripts/analyze_pure_errors.py",
                "--gold-dir",
                str(source_dir),
                "--pred-dir",
                str(generated_dir),
                "--threshold",
                str(args.match_threshold),
                "--output",
                str(pipeline_error_analysis_path),
            ],
            extra_env=extra_env,
        )
        legacy_gemini_eval_path = metrics_dir / "gemini_coverage.json"
        legacy_gemini_error_analysis_path = metrics_dir / "gemini_error_analysis.json"
        shutil.copy2(pipeline_eval_path, legacy_gemini_eval_path)
        shutil.copy2(pipeline_error_analysis_path, legacy_gemini_error_analysis_path)

    comparison = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "max_samples": args.max_samples,
            "min_requirements": args.min_requirements,
            "seed": args.seed,
            "max_source_requirements": args.max_source_requirements,
            "match_threshold": args.match_threshold,
            "max_reqs_per_answer": args.max_reqs_per_answer,
            "max_chars_per_answer": args.max_chars_per_answer,
            "dialogue_coverage_threshold": args.dialogue_coverage_threshold,
            "clarification_rounds": args.clarification_rounds,
            "dialogue_variant": args.dialogue_variant,
            "partial_drop_rate": args.partial_drop_rate if args.dialogue_variant == "partial_information" else None,
            "self_consistency": args.self_consistency,
            "self_consistency_temperature": args.self_consistency_temperature,
            "llm_provider": provider_name,
            "dialogue_chunking": args.dialogue_chunking,
            "dialogue_chunking_enabled": dialogue_chunking_enabled,
            "dialogue_chunk_max_turns": args.dialogue_chunk_max_turns,
            "dialogue_chunk_max_chars": args.dialogue_chunk_max_chars,
            "dialogue_chunk_overlap_turns": args.dialogue_chunk_overlap_turns,
            "memory_max_items": args.memory_max_items,
            "memory_max_chars": args.memory_max_chars,
            "python_bin": PYTHON_BIN,
        },
        "oracle": load_json(oracle_eval_path)["aggregate"],
        "llm_provider": provider_name,
        "pipeline_status": pipeline_status,
        "gemini_status": pipeline_status,
        "direct": load_json(direct_eval_path)["aggregate"] if direct_eval_path else None,
        "pipeline": load_json(pipeline_eval_path)["aggregate"] if pipeline_eval_path else None,
        "gemini": load_json(pipeline_eval_path)["aggregate"] if pipeline_eval_path else None,
        "dialogue_lower_bound": load_json(dialogue_eval_path)["aggregate"],
        "dialogue_upper_bound": load_json(dialogue_eval_path)["aggregate"],
        "paths": {
            "source_summary": str(source_summary.relative_to(ROOT)),
            "oracle_coverage": str(oracle_eval_path.relative_to(ROOT)),
            "direct_coverage": str(direct_eval_path.relative_to(ROOT)) if direct_eval_path else None,
            "direct_error_analysis": str(direct_error_analysis_path.relative_to(ROOT)) if direct_error_analysis_path else None,
            "pipeline_coverage": str(pipeline_eval_path.relative_to(ROOT)) if pipeline_eval_path else None,
            "pipeline_error_analysis": str(pipeline_error_analysis_path.relative_to(ROOT)) if pipeline_error_analysis_path else None,
            "pipeline_validation_report": str(validation_report_path.relative_to(ROOT)) if has_llm_env(provider_name) else None,
            "gemini_coverage": str(legacy_gemini_eval_path.relative_to(ROOT)) if legacy_gemini_eval_path else None,
            "gemini_error_analysis": str(legacy_gemini_error_analysis_path.relative_to(ROOT)) if legacy_gemini_error_analysis_path else None,
            "dialogue_coverage_user_only": str(dialogue_eval_path.relative_to(ROOT)),
        },
    }

    comparison_path = run_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LATEST_POINTER.write_text(
        json.dumps({"run_id": run_id, "run_dir": str(run_dir.resolve())}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
