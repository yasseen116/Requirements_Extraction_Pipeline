#!/usr/bin/env python3
"""Run comprehensive PURE benchmark: source extraction -> controlled dialogue -> requirements -> coverage."""

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
PIPELINE_SLICE_SOURCE_DIR = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
PIPELINE_SLICES = {
    "hard_single_doc": ["pure_0000_cctns"],
    "paper_regression_3doc": ["pure_0000_cctns", "pure_0000_gamma_j", "pure_1999_dii"],
}
PYTHON_BIN = "python3"


def check_dependencies() -> None:
    missing = []
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")
    try:
        import torch  # noqa: F401
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
    parser.add_argument("--python-bin", type=str, default=os.environ.get("REQ_PYTHON_BIN", "python3"))
    parser.add_argument("--reuse-source-dir", type=Path, default=None)
    parser.add_argument("--benchmark-slice", choices=["hard_single_doc", "paper_regression_3doc"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-source-requirements", type=int, default=200)
    parser.add_argument("--match-threshold", type=float, default=0.55)
    parser.add_argument("--max-reqs-per-answer", type=int, default=4)
    parser.add_argument("--max-chars-per-answer", type=int, default=900)
    parser.add_argument("--dialogue-coverage-threshold", type=float, default=0.55)
    parser.add_argument("--clarification-rounds", type=int, default=2)
    parser.add_argument("--dialogue-variant", choices=["controlled", "transcript_paraphrase", "partial_information"], default="controlled")
    parser.add_argument("--partial-drop-rate", type=float, default=0.2)
    parser.add_argument("--self-consistency", type=int, default=3)
    parser.add_argument("--self-consistency-temperature", type=float, default=0.3)
    parser.add_argument("--llm-provider", choices=["gemini", "ollama"], default=None)
    parser.add_argument("--pipeline-preset", choices=["auto", "local_recall", "gemini_full_context", "gemini_evidence_bank"], default="auto")
    parser.add_argument("--dialogue-chunking", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--dialogue-chunk-overlap-turns", type=int, default=2)
    parser.add_argument("--memory-max-items", type=int, default=24)
    parser.add_argument("--memory-max-chars", type=int, default=3500)
    parser.add_argument("--retrieval-top-k", type=int, default=14)
    parser.add_argument("--gap-pass-top-k", type=int, default=10)
    parser.add_argument("--theme-max-exchanges", type=int, default=3)
    parser.add_argument("--target-dialogue-recall", type=float, default=0.82)
    parser.add_argument("--max-turns", type=int, default=28)
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


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def resolve_python_bin(raw: str | None) -> str:
    candidate = (raw or "").strip()
    if not candidate:
        return sys.executable or "python3"
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str((Path.cwd() / path).absolute())
    return candidate


def copy_source_slice(slice_name: str, source_dir: Path, source_summary: Path) -> None:
    sample_ids = PIPELINE_SLICES[slice_name]
    source_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sample_id in sample_ids:
        src = PIPELINE_SLICE_SOURCE_DIR / f"{sample_id}.json"
        if not src.exists():
            raise FileNotFoundError(f"Missing slice source file: {src}")
        dest = source_dir / src.name
        shutil.copy2(src, dest)
        payload = load_json(src)
        summary.append(
            {
                "sample_id": sample_id,
                "document_id": payload["source"]["document_id"],
                "path": str(dest.relative_to(ROOT)),
                "requirement_count": len(payload["ground_truth_requirements"]),
                "status": "ok_slice_copy",
            }
        )
    source_summary.write_text(
        json.dumps(
            {
                "dataset": "pure",
                "slice": slice_name,
                "generated_samples": len(summary),
                "samples": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_source_dir(args: argparse.Namespace, source_dir: Path, source_summary: Path, *, extra_env: dict[str, str] | None) -> None:
    if args.benchmark_slice:
        copy_source_slice(args.benchmark_slice, source_dir, source_summary)
        return
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
        source_summary.write_text(
            json.dumps(
                {
                    "dataset": "pure",
                    "reused": True,
                    "source_dir": str(reuse_dir),
                    "generated_samples": len(copied),
                    "samples": [{"sample_id": path.stem, "path": str(path.relative_to(ROOT)), "status": "ok_reused"} for path in copied],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return
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


def preset_config(provider_name: str, preset_name: str, args: argparse.Namespace) -> dict:
    effective = preset_name
    if preset_name == "auto":
        effective = "local_recall" if provider_name == "ollama" else "gemini_evidence_bank"

    config = {
        "name": effective,
        "chunk_dialogues": True,
        "extraction_mode": "evidence_bank",
        "enable_gap_pass": True,
        "retrieval_top_k": args.retrieval_top_k,
        "overlap": args.dialogue_chunk_overlap_turns,
    }
    if effective == "gemini_full_context":
        config.update(
            {
                "chunk_dialogues": False,
                "extraction_mode": "full_context",
                "retrieval_top_k": max(args.retrieval_top_k, 24),
                "overlap": 0,
            }
        )
    elif effective == "gemini_evidence_bank":
        config.update(
            {
                "chunk_dialogues": True,
                "extraction_mode": "evidence_bank",
                "retrieval_top_k": max(args.retrieval_top_k, 24),
                "overlap": 0,
            }
        )
    return config


def run_pipeline_variant(
    variant_name: str,
    preset: dict,
    *,
    source_dir: Path,
    dialogue_dir: Path,
    run_dir: Path,
    metrics_dir: Path,
    args: argparse.Namespace,
    extra_env: dict[str, str] | None,
) -> dict:
    raw_generated_dir = run_dir / f"raw_generated_requirements_{variant_name}"
    validated_dir = run_dir / f"generated_requirements_{variant_name}"
    validation_report_path = metrics_dir / f"pipeline_validation_report_{variant_name}.json"
    eval_path = metrics_dir / f"pipeline_coverage_{variant_name}.json"
    error_path = metrics_dir / f"pipeline_error_analysis_{variant_name}.json"

    generate_cmd = [
        "python3",
        "scripts/generate_pure_full_requirements.py",
        "--input-dir",
        str(dialogue_dir),
        "--output-dir",
        str(raw_generated_dir),
        "--self-consistency",
        str(args.self_consistency),
        "--temperature",
        str(args.self_consistency_temperature),
        "--memory-max-items",
        str(args.memory_max_items),
        "--memory-max-chars",
        str(args.memory_max_chars),
        "--retrieval-top-k",
        str(preset["retrieval_top_k"]),
        "--gap-pass-top-k",
        str(args.gap_pass_top_k),
        "--dialogue-chunk-overlap-turns",
        str(preset["overlap"]),
        "--extraction-mode",
        preset["extraction_mode"],
    ]
    if preset["chunk_dialogues"]:
        generate_cmd.append("--chunk-dialogues")
    if preset["enable_gap_pass"]:
        generate_cmd.append("--enable-gap-pass")
    run(generate_cmd, extra_env=extra_env)

    run(
        [
            "python3",
            "scripts/validate_pure_extracted_requirements.py",
            "--pred-dir",
            str(raw_generated_dir),
            "--dialogue-dir",
            str(dialogue_dir),
            "--output-dir",
            str(validated_dir),
            "--report-path",
            str(validation_report_path),
        ],
        extra_env=extra_env,
    )
    run(
        [
            "python3",
            "scripts/evaluate_pure_requirements_coverage.py",
            "--gold-dir",
            str(source_dir),
            "--pred-dir",
            str(validated_dir),
            "--match-threshold",
            str(args.match_threshold),
            "--output",
            str(eval_path),
        ],
        extra_env=extra_env,
    )
    run(
        [
            "python3",
            "scripts/analyze_pure_errors.py",
            "--gold-dir",
            str(source_dir),
            "--pred-dir",
            str(validated_dir),
            "--threshold",
            str(args.match_threshold),
            "--output",
            str(error_path),
        ],
        extra_env=extra_env,
    )
    return {
        "variant_name": variant_name,
        "preset": preset["name"],
        "raw_generated_dir": raw_generated_dir,
        "validated_dir": validated_dir,
        "validation_report_path": validation_report_path,
        "eval_path": eval_path,
        "error_path": error_path,
        "aggregate": load_json(eval_path)["aggregate"],
    }


def load_hard_slice_metrics(eval_payload: dict) -> dict | None:
    for item in eval_payload.get("per_sample", []):
        if item.get("sample_id") == "pure_0000_cctns":
            return item
    return None


def choose_gemini_variant(full_variant: dict, evidence_variant: dict) -> tuple[dict, dict]:
    full_payload = load_json(full_variant["eval_path"])
    evidence_payload = load_json(evidence_variant["eval_path"])
    full_hard = load_hard_slice_metrics(full_payload)
    evidence_hard = load_hard_slice_metrics(evidence_payload)

    if full_hard and evidence_hard:
        full_recall = float(full_hard.get("coverage_recall", 0.0))
        evidence_recall = float(evidence_hard.get("coverage_recall", 0.0))
        full_precision = float(full_hard.get("precision", 0.0))
    else:
        full_recall = float(full_variant["aggregate"].get("micro_coverage_recall", 0.0))
        evidence_recall = float(evidence_variant["aggregate"].get("micro_coverage_recall", 0.0))
        full_precision = float(full_variant["aggregate"].get("micro_precision", 0.0))

    if (full_recall - evidence_recall) >= 0.02 and full_precision >= 0.55:
        return full_variant, {
            "selector": "gemini_full_context",
            "reason": "full_context_beats_evidence_bank",
            "full_context_recall": full_recall,
            "evidence_bank_recall": evidence_recall,
            "full_context_precision": full_precision,
        }
    return evidence_variant, {
        "selector": "gemini_evidence_bank",
        "reason": "evidence_bank_default_or_better",
        "full_context_recall": full_recall,
        "evidence_bank_recall": evidence_recall,
        "full_context_precision": full_precision,
    }


def copy_selected_variant(selected: dict, generated_dir: Path, metrics_dir: Path) -> tuple[Path, Path, Path]:
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    shutil.copytree(selected["validated_dir"], generated_dir)
    pipeline_eval_path = metrics_dir / "pipeline_coverage.json"
    pipeline_error_path = metrics_dir / "pipeline_error_analysis.json"
    validation_report_path = metrics_dir / "pipeline_validation_report.json"
    shutil.copy2(selected["eval_path"], pipeline_eval_path)
    shutil.copy2(selected["error_path"], pipeline_error_path)
    shutil.copy2(selected["validation_report_path"], validation_report_path)
    return pipeline_eval_path, pipeline_error_path, validation_report_path


def extract_dialogue_method_metadata(dialogue_dir: Path) -> dict:
    for path in sorted(dialogue_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = load_json(path)
        dialogue_generation = payload.get("dialogue_generation", {})
        if not isinstance(dialogue_generation, dict):
            continue
        return {
            "method": dialogue_generation.get("method"),
            "question_algorithm_version": dialogue_generation.get("question_algorithm_version"),
            "question_algorithm_summary": dialogue_generation.get("question_algorithm_summary"),
        }
    return {}


def collect_pipeline_diagnostics(generated_dir: Path, dialogue_dir: Path) -> dict:
    per_sample = []
    theme_coverage = {}
    totals = {
        "evidence_bank_count": 0,
        "proposition_count": 0,
        "gap_pass_added_count": 0,
        "grounded_after_rewrite_count": 0,
    }
    for path in sorted(generated_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = load_json(path)
        diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics"), dict) else {}
        item = {"sample_id": payload.get("sample_id"), **diagnostics}
        per_sample.append(item)
        for key in totals:
            value = diagnostics.get(key)
            if isinstance(value, int):
                totals[key] += value
        dialogue_path = dialogue_dir / path.name
        if dialogue_path.exists():
            dialogue_payload = load_json(dialogue_path)
            coverage_summary = dialogue_payload.get("dialogue_generation", {}).get("coverage_summary", {})
            if isinstance(coverage_summary, dict):
                theme_coverage[payload.get("sample_id")] = coverage_summary.get("theme_coverage", {})
    return {
        "aggregate": totals,
        "per_sample": per_sample,
        "per_theme_dialogue_coverage": theme_coverage,
    }


def main() -> int:
    check_dependencies()
    global PYTHON_BIN
    args = parse_args()
    provider_name = llm_provider(args.llm_provider)
    extra_env = {"REQ_LLM_PROVIDER": provider_name} if args.llm_provider else None
    dialogue_chunking_enabled = should_chunk_dialogues(provider_name, args.dialogue_chunking)
    PYTHON_BIN = resolve_python_bin(args.python_bin or os.environ.get("REQ_PYTHON_BIN")) or sys.executable or "python3"

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

    prepare_source_dir(args, source_dir, source_summary, extra_env=extra_env)

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
        "--max-turns",
        str(args.max_turns),
        "--target-dialogue-recall",
        str(args.target_dialogue_recall),
        "--theme-max-exchanges",
        str(args.theme_max_exchanges),
    ]
    run(controlled_cmd, extra_env=extra_env)
    controlled_dialogue_method = extract_dialogue_method_metadata(dialogue_dir)
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
    pipeline_diagnostics = None
    gemini_comparison = None
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
                ["--chunk-source-requirements", "--source-chunk-size", "25", "--source-chunk-char-budget", "6000"]
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

        effective_preset = args.pipeline_preset
        if provider_name == "gemini" and effective_preset == "auto":
            full_variant = run_pipeline_variant(
                "gemini_full_context",
                preset_config(provider_name, "gemini_full_context", args),
                source_dir=source_dir,
                dialogue_dir=effective_dialogue_dir,
                run_dir=run_dir,
                metrics_dir=metrics_dir,
                args=args,
                extra_env=extra_env,
            )
            evidence_variant = run_pipeline_variant(
                "gemini_evidence_bank",
                preset_config(provider_name, "gemini_evidence_bank", args),
                source_dir=source_dir,
                dialogue_dir=effective_dialogue_dir,
                run_dir=run_dir,
                metrics_dir=metrics_dir,
                args=args,
                extra_env=extra_env,
            )
            selected_variant, gemini_comparison = choose_gemini_variant(full_variant, evidence_variant)
            pipeline_eval_path, pipeline_error_analysis_path, validation_report_path = copy_selected_variant(
                selected_variant,
                generated_dir,
                metrics_dir,
            )
            pipeline_diagnostics = collect_pipeline_diagnostics(generated_dir, effective_dialogue_dir)
        else:
            selected_variant = run_pipeline_variant(
                llm_provider(args.llm_provider) + "_" + preset_config(provider_name, effective_preset, args)["name"],
                preset_config(provider_name, effective_preset, args),
                source_dir=source_dir,
                dialogue_dir=effective_dialogue_dir,
                run_dir=run_dir,
                metrics_dir=metrics_dir,
                args=args,
                extra_env=extra_env,
            )
            pipeline_eval_path, pipeline_error_analysis_path, validation_report_path = copy_selected_variant(
                selected_variant,
                generated_dir,
                metrics_dir,
            )
            pipeline_diagnostics = collect_pipeline_diagnostics(generated_dir, effective_dialogue_dir)

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
            "benchmark_slice": args.benchmark_slice,
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
            "pipeline_preset": args.pipeline_preset,
            "dialogue_chunking": args.dialogue_chunking,
            "dialogue_chunking_enabled": dialogue_chunking_enabled,
            "dialogue_chunk_overlap_turns": args.dialogue_chunk_overlap_turns,
            "memory_max_items": args.memory_max_items,
            "memory_max_chars": args.memory_max_chars,
            "retrieval_top_k": args.retrieval_top_k,
            "gap_pass_top_k": args.gap_pass_top_k,
            "theme_max_exchanges": args.theme_max_exchanges,
            "target_dialogue_recall": args.target_dialogue_recall,
            "max_turns": args.max_turns,
            "python_bin": PYTHON_BIN,
        },
        "oracle": load_json(oracle_eval_path)["aggregate"],
        "llm_provider": provider_name,
        "pipeline_status": pipeline_status,
        "gemini_status": pipeline_status,
        "controlled_dialogue_method": controlled_dialogue_method,
        "direct": load_json(direct_eval_path)["aggregate"] if direct_eval_path else None,
        "pipeline": load_json(pipeline_eval_path)["aggregate"] if pipeline_eval_path else None,
        "gemini": load_json(pipeline_eval_path)["aggregate"] if pipeline_eval_path else None,
        "dialogue_lower_bound": load_json(dialogue_eval_path)["aggregate"],
        "dialogue_upper_bound": load_json(dialogue_eval_path)["aggregate"],
        "pipeline_diagnostics": pipeline_diagnostics,
        "gemini_preset_comparison": gemini_comparison,
        "paths": {
            "source_summary": str(source_summary.relative_to(ROOT)),
            "oracle_coverage": str(oracle_eval_path.relative_to(ROOT)),
            "direct_coverage": str(direct_eval_path.relative_to(ROOT)) if direct_eval_path else None,
            "direct_error_analysis": str(direct_error_analysis_path.relative_to(ROOT)) if direct_error_analysis_path else None,
            "pipeline_coverage": str(pipeline_eval_path.relative_to(ROOT)) if pipeline_eval_path else None,
            "pipeline_error_analysis": str(pipeline_error_analysis_path.relative_to(ROOT)) if pipeline_error_analysis_path else None,
            "pipeline_validation_report": str(validation_report_path.relative_to(ROOT)) if validation_report_path else None,
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
