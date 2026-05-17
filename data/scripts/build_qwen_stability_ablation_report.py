#!/usr/bin/env python3
"""Build a paper-facing Qwen stability and anchor-ablation report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_reports" / "20260517_qwen_stability_ablation"
WEIGHTED_NOTE = "N/A - Gemini validator disabled today"


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    run_dir: str
    seed: int | None
    dialogue_recall: float
    semantic_precision: float
    semantic_recall: float
    semantic_f1: float
    pipeline_only: bool
    anchor_preservation_enabled: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen stability and no-anchor ablation runs.")
    parser.add_argument("--stability-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--ablation-run-dir", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def mean_std(values: list[float]) -> str:
    return f"{mean(values):.4f} +/- {std(values):.4f}"


def load_run(run_dir: Path) -> RunMetrics:
    run_dir = run_dir.resolve()
    summary = load_json(run_dir / "comparison_summary.json")
    dialogue = summary.get("dialogue_lower_bound") or summary.get("dialogue_upper_bound") or {}
    pipeline = summary.get("pipeline") or {}
    settings = summary.get("settings") or {}
    return RunMetrics(
        run_id=str(summary.get("run_id") or run_dir.name),
        run_dir=str(run_dir),
        seed=settings.get("seed"),
        dialogue_recall=float(dialogue["micro_coverage_recall"]),
        semantic_precision=float(pipeline["micro_precision"]),
        semantic_recall=float(pipeline["micro_coverage_recall"]),
        semantic_f1=float(pipeline["micro_f1"]),
        pipeline_only=bool(summary.get("pipeline_only") or settings.get("pipeline_only")),
        anchor_preservation_enabled=bool(
            summary.get("anchor_preservation_enabled", settings.get("anchor_preservation_enabled", True))
        ),
    )


def run_to_dict(run: RunMetrics) -> dict:
    return {
        "run_id": run.run_id,
        "run_dir": safe_rel(Path(run.run_dir)),
        "seed": run.seed,
        "dialogue_recall": run.dialogue_recall,
        "semantic_precision": run.semantic_precision,
        "semantic_recall": run.semantic_recall,
        "semantic_f1": run.semantic_f1,
        "pipeline_only": run.pipeline_only,
        "anchor_preservation_enabled": run.anchor_preservation_enabled,
    }


def aggregate(runs: list[RunMetrics]) -> dict:
    return {
        "n": len(runs),
        "dialogue_recall": {
            "mean": mean([run.dialogue_recall for run in runs]),
            "std": std([run.dialogue_recall for run in runs]),
        },
        "semantic_precision": {
            "mean": mean([run.semantic_precision for run in runs]),
            "std": std([run.semantic_precision for run in runs]),
        },
        "semantic_recall": {
            "mean": mean([run.semantic_recall for run in runs]),
            "std": std([run.semantic_recall for run in runs]),
        },
        "semantic_f1": {
            "mean": mean([run.semantic_f1 for run in runs]),
            "std": std([run.semantic_f1 for run in runs]),
        },
        "weighted_f1": WEIGHTED_NOTE,
    }


def ablation_comparison(stability_runs: list[RunMetrics], ablation: RunMetrics) -> dict:
    baseline = stability_runs[0]
    return {
        "baseline_run_id": baseline.run_id,
        "ablation_run_id": ablation.run_id,
        "semantic_precision_delta": ablation.semantic_precision - baseline.semantic_precision,
        "semantic_recall_delta": ablation.semantic_recall - baseline.semantic_recall,
        "semantic_f1_delta": ablation.semantic_f1 - baseline.semantic_f1,
        "dialogue_recall_delta": ablation.dialogue_recall - baseline.dialogue_recall,
    }


def render_markdown(
    stability_runs: list[RunMetrics],
    ablation: RunMetrics,
    reference: RunMetrics | None,
    payload: dict,
) -> str:
    stability_dialogue = [run.dialogue_recall for run in stability_runs]
    stability_precision = [run.semantic_precision for run in stability_runs]
    stability_recall = [run.semantic_recall for run in stability_runs]
    stability_f1 = [run.semantic_f1 for run in stability_runs]
    delta = payload["ablation_comparison"]

    lines = [
        "# Qwen 4-Document Stability and Anchor Ablation",
        "",
        f"Generated at UTC: `{payload['generated_at_utc']}`",
        "",
        "## Stability Summary",
        "",
        "| Condition | Docs | Runs | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 | Weighted F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            "| Qwen full pipeline | 4 | "
            f"{len(stability_runs)} | {mean_std(stability_dialogue)} | {mean_std(stability_precision)} | "
            f"{mean_std(stability_recall)} | {mean_std(stability_f1)} | {WEIGHTED_NOTE} |"
        ),
        "",
        "## Per-Run Stability",
        "",
        "| Run | Seed | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in stability_runs:
        lines.append(
            f"| `{run.run_id}` | {run.seed if run.seed is not None else 'N/A'} | "
            f"{metric(run.dialogue_recall)} | {metric(run.semantic_precision)} | "
            f"{metric(run.semantic_recall)} | {metric(run.semantic_f1)} |"
        )

    lines.extend(
        [
            "",
            "## Anchor Preservation Ablation",
            "",
            "| Variant | Docs | Runs | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 | Main interpretation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            (
                "| Full pipeline | 4 | 3 | "
                f"{mean_std(stability_dialogue)} | {mean_std(stability_precision)} | "
                f"{mean_std(stability_recall)} | {mean_std(stability_f1)} | Baseline repeated Qwen pipeline |"
            ),
            (
                "| No anchor preservation | 4 | 1 | "
                f"{metric(ablation.dialogue_recall)} | {metric(ablation.semantic_precision)} | "
                f"{metric(ablation.semantic_recall)} | {metric(ablation.semantic_f1)} | "
                "Tests loss of requirement-specific anchors |"
            ),
            "",
            "## Ablation Delta",
            "",
            f"- Baseline comparison run: `{delta['baseline_run_id']}`",
            f"- Ablation run: `{delta['ablation_run_id']}`",
            f"- Semantic F1 delta: {metric(delta['semantic_f1_delta'])}",
            f"- Semantic recall delta: {metric(delta['semantic_recall_delta'])}",
            f"- Semantic precision delta: {metric(delta['semantic_precision_delta'])}",
            "",
        ]
    )

    if reference is not None:
        lines.extend(
            [
                "## Existing Qwen Reference Run",
                "",
                "| Run | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
                (
                    f"| `{reference.run_id}` | {metric(reference.dialogue_recall)} | "
                    f"{metric(reference.semantic_precision)} | {metric(reference.semantic_recall)} | "
                    f"{metric(reference.semantic_f1)} |"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- New runs intentionally disabled the Gemini validator.",
            "- Weighted F1 is therefore not computed for stability or ablation rows.",
            "- Existing reference weighted scores remain available in the original 4-document comparison artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    stability_runs = [load_run(path) for path in args.stability_run_dirs]
    ablation = load_run(args.ablation_run_dir)
    reference = load_run(args.reference_run_dir) if args.reference_run_dir else None

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weighted_metric_policy": WEIGHTED_NOTE,
        "stability_runs": [run_to_dict(run) for run in stability_runs],
        "stability_aggregate": aggregate(stability_runs),
        "ablation_run": run_to_dict(ablation),
        "ablation_comparison": ablation_comparison(stability_runs, ablation),
        "reference_run": run_to_dict(reference) if reference else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "qwen_stability_ablation_summary.json"
    md_path = args.output_dir / "qwen_stability_ablation_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(stability_runs, ablation, reference, payload), encoding="utf-8")
    print(json.dumps({"json": safe_rel(json_path), "markdown": safe_rel(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
