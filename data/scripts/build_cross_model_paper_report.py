#!/usr/bin/env python3
"""Build a paper-facing comparison report across two PURE benchmark runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    colors = None  # type: ignore[assignment]
    landscape = None  # type: ignore[assignment]
    letter = None  # type: ignore[assignment]
    getSampleStyleSheet = None  # type: ignore[assignment]
    inch = None  # type: ignore[assignment]
    PageBreak = Paragraph = SimpleDocTemplate = Spacer = Table = TableStyle = None  # type: ignore[assignment]
    _REPORTLAB_IMPORT_ERROR = exc
else:
    _REPORTLAB_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cross-model PURE comparison report.")
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default=None)
    parser.add_argument("--label-b", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def metric_fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def int_fmt(value: int | None) -> str:
    if value is None:
        return "N/A"
    return str(value)


def bold(text: str) -> str:
    return f"**{text}**"


def highlight_pair_markdown(left: float | None, right: float | None) -> tuple[str, str]:
    left_text = metric_fmt(left)
    right_text = metric_fmt(right)
    if left is None or right is None:
        return left_text, right_text
    if left > right:
        return bold(left_text), right_text
    if right > left:
        return left_text, bold(right_text)
    return bold(left_text), bold(right_text)


def resolve_metric_path(run_dir: Path, comparison_summary: dict, key: str) -> Path | None:
    rel = comparison_summary.get("paths", {}).get(key)
    if not rel:
        fallback = run_dir / "metrics" / f"{key}.json"
        return fallback if fallback.exists() else None
    path = ROOT / rel
    if path.exists():
        return path
    alt = run_dir / Path(rel).name
    if alt.exists():
        return alt
    fallback = run_dir / "metrics" / f"{key}.json"
    return fallback if fallback.exists() else None


def index_by_sample(rows: list[dict] | None) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows or []:
        sample_id = row.get("sample_id")
        if sample_id:
            indexed[sample_id] = row
    return indexed


def load_run(run_dir: Path, explicit_label: str | None) -> dict:
    run_dir = run_dir.resolve()
    comparison_summary = load_json(run_dir / "comparison_summary.json")
    provider = comparison_summary.get("llm_provider") or "unknown"
    model = comparison_summary.get("model_metadata", {}).get("generation", {}).get("model") or "unknown"
    label = explicit_label or f"{provider}:{model}"

    direct_cov_path = resolve_metric_path(run_dir, comparison_summary, "direct_coverage")
    pipeline_cov_path = resolve_metric_path(run_dir, comparison_summary, "pipeline_coverage")
    dialogue_cov_path = resolve_metric_path(run_dir, comparison_summary, "dialogue_coverage_user_only")
    direct_llm_path = resolve_metric_path(run_dir, comparison_summary, "direct_coverage_llm")
    pipeline_llm_path = resolve_metric_path(run_dir, comparison_summary, "pipeline_coverage_llm")
    pipeline_validation_path = resolve_metric_path(run_dir, comparison_summary, "pipeline_validation_report")

    direct_cov = load_json(direct_cov_path) if direct_cov_path and direct_cov_path.exists() else {}
    pipeline_cov = load_json(pipeline_cov_path) if pipeline_cov_path and pipeline_cov_path.exists() else {}
    dialogue_cov = load_json(dialogue_cov_path) if dialogue_cov_path and dialogue_cov_path.exists() else {}
    direct_llm = load_json(direct_llm_path) if direct_llm_path and direct_llm_path.exists() else {}
    pipeline_llm = load_json(pipeline_llm_path) if pipeline_llm_path and pipeline_llm_path.exists() else {}
    pipeline_validation = load_json(pipeline_validation_path) if pipeline_validation_path and pipeline_validation_path.exists() else {}

    direct_by_sample = index_by_sample(direct_cov.get("per_sample"))
    pipeline_by_sample = index_by_sample(pipeline_cov.get("per_sample"))
    dialogue_by_sample = index_by_sample(dialogue_cov.get("per_sample"))
    direct_llm_by_sample = index_by_sample(direct_llm.get("per_sample"))
    pipeline_llm_by_sample = index_by_sample(pipeline_llm.get("per_sample"))
    pipeline_validation_by_sample = index_by_sample(pipeline_validation.get("per_sample"))

    diagnostics_by_sample = {
        row.get("sample_id"): row
        for row in comparison_summary.get("pipeline_diagnostics", {}).get("per_sample", [])
        if row.get("sample_id")
    }

    sample_ids = sorted(
        {
            *direct_by_sample.keys(),
            *pipeline_by_sample.keys(),
            *dialogue_by_sample.keys(),
            *direct_llm_by_sample.keys(),
            *pipeline_llm_by_sample.keys(),
            *pipeline_validation_by_sample.keys(),
        }
    )

    samples = {}
    for sample_id in sample_ids:
        direct_row = direct_by_sample.get(sample_id, {})
        pipeline_row = pipeline_by_sample.get(sample_id, {})
        dialogue_row = dialogue_by_sample.get(sample_id, {})
        direct_llm_row = direct_llm_by_sample.get(sample_id, {})
        pipeline_llm_row = pipeline_llm_by_sample.get(sample_id, {})
        validation_row = pipeline_validation_by_sample.get(sample_id, {})
        diagnostics_row = diagnostics_by_sample.get(sample_id, {})

        validation_summary = validation_row.get("validation", {}) if isinstance(validation_row, dict) else {}

        samples[sample_id] = {
            "sample_id": sample_id,
            "document_id": (
                direct_row.get("document_id")
                or pipeline_row.get("document_id")
                or dialogue_row.get("document_id")
                or sample_id.replace("pure_", "", 1)
            ),
            "source_requirement_count": (
                direct_row.get("source_requirement_count")
                or pipeline_row.get("source_requirement_count")
                or dialogue_row.get("source_requirement_count")
            ),
            "dialogue": {
                "coverage_recall": dialogue_row.get("coverage_recall"),
                "covered_count": dialogue_row.get("covered_count"),
                "support_unit_count": dialogue_row.get("support_unit_count"),
            },
            "direct_semantic": {
                "precision": direct_row.get("precision"),
                "recall": direct_row.get("coverage_recall"),
                "f1": direct_row.get("f1"),
                "hallucination_rate": direct_row.get("hallucination_rate"),
                "matched_count": direct_row.get("matched_count"),
                "generated_count": direct_row.get("generated_requirement_count"),
            },
            "pipeline_semantic": {
                "precision": pipeline_row.get("precision"),
                "recall": pipeline_row.get("coverage_recall"),
                "f1": pipeline_row.get("f1"),
                "hallucination_rate": pipeline_row.get("hallucination_rate"),
                "matched_count": pipeline_row.get("matched_count"),
                "generated_count": pipeline_row.get("generated_requirement_count"),
            },
            "direct_llm": {
                "strict_precision": direct_llm_row.get("precision"),
                "strict_recall": direct_llm_row.get("coverage_recall"),
                "strict_f1": direct_llm_row.get("f1"),
                "weighted_precision": direct_llm_row.get("weighted_precision"),
                "weighted_recall": direct_llm_row.get("weighted_coverage_recall"),
                "weighted_f1": direct_llm_row.get("weighted_f1"),
                "full_match_count": direct_llm_row.get("full_match_count"),
                "partial_match_count": direct_llm_row.get("partial_match_count"),
            },
            "pipeline_llm": {
                "strict_precision": pipeline_llm_row.get("precision"),
                "strict_recall": pipeline_llm_row.get("coverage_recall"),
                "strict_f1": pipeline_llm_row.get("f1"),
                "weighted_precision": pipeline_llm_row.get("weighted_precision"),
                "weighted_recall": pipeline_llm_row.get("weighted_coverage_recall"),
                "weighted_f1": pipeline_llm_row.get("weighted_f1"),
                "full_match_count": pipeline_llm_row.get("full_match_count"),
                "partial_match_count": pipeline_llm_row.get("partial_match_count"),
            },
            "pipeline_validation": {
                "grounded": validation_summary.get("grounded"),
                "hallucinated": validation_summary.get("hallucinated"),
                "hallucination_rate": validation_summary.get("hallucination_rate"),
            },
            "pipeline_diagnostics": {
                "evidence_bank_count": diagnostics_row.get("evidence_bank_count"),
                "proposition_count": diagnostics_row.get("proposition_count"),
                "rewrite_candidate_count": diagnostics_row.get("rewrite_candidate_count"),
                "gap_pass_added_count": diagnostics_row.get("gap_pass_added_count"),
                "grounded_after_rewrite_count": diagnostics_row.get("grounded_after_rewrite_count"),
            },
        }

    return {
        "label": label,
        "provider": provider,
        "model": model,
        "validator_enabled": bool(
            comparison_summary.get("validation_layers", {}).get("standard_llm_validator", {}).get("enabled")
            or direct_llm
            or pipeline_llm
        ),
        "validator_model": comparison_summary.get("model_metadata", {}).get("standard_validator", {}).get("model"),
        "run_dir": str(run_dir),
        "run_id": comparison_summary.get("run_id"),
        "comparison_summary": comparison_summary,
        "direct_aggregate": comparison_summary.get("direct"),
        "pipeline_aggregate": comparison_summary.get("pipeline"),
        "dialogue_aggregate": comparison_summary.get("dialogue_lower_bound"),
        "direct_llm_aggregate": comparison_summary.get("direct_llm_validation") or direct_llm.get("aggregate"),
        "pipeline_llm_aggregate": comparison_summary.get("pipeline_llm_validation") or pipeline_llm.get("aggregate"),
        "samples": samples,
    }


def document_row(sample_id: str, runs: list[dict]) -> dict:
    base = None
    for run in runs:
        sample = run["samples"].get(sample_id)
        if sample:
            base = sample
            break
    return {
        "sample_id": sample_id,
        "document_id": base.get("document_id") if base else sample_id.replace("pure_", "", 1),
        "source_requirement_count": base.get("source_requirement_count") if base else None,
    }


def build_summary(run_a: dict, run_b: dict) -> dict:
    sample_ids = sorted(set(run_a["samples"]) | set(run_b["samples"]))
    documents = []
    for sample_id in sample_ids:
        base = document_row(sample_id, [run_a, run_b])
        documents.append(
            {
                **base,
                "runs": {
                    run_a["label"]: run_a["samples"].get(sample_id),
                    run_b["label"]: run_b["samples"].get(sample_id),
                },
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slice": "paper_report_4doc",
        "documents": documents,
        "runs": {
            run_a["label"]: {
                "provider": run_a["provider"],
                "model": run_a["model"],
                "validator_enabled": run_a["validator_enabled"],
                "validator_model": run_a["validator_model"],
                "run_id": run_a["run_id"],
                "run_dir": run_a["run_dir"],
                "dialogue_aggregate": run_a["dialogue_aggregate"],
                "direct_aggregate": run_a["direct_aggregate"],
                "pipeline_aggregate": run_a["pipeline_aggregate"],
                "direct_llm_aggregate": run_a["direct_llm_aggregate"],
                "pipeline_llm_aggregate": run_a["pipeline_llm_aggregate"],
            },
            run_b["label"]: {
                "provider": run_b["provider"],
                "model": run_b["model"],
                "validator_enabled": run_b["validator_enabled"],
                "validator_model": run_b["validator_model"],
                "run_id": run_b["run_id"],
                "run_dir": run_b["run_dir"],
                "dialogue_aggregate": run_b["dialogue_aggregate"],
                "direct_aggregate": run_b["direct_aggregate"],
                "pipeline_aggregate": run_b["pipeline_aggregate"],
                "direct_llm_aggregate": run_b["direct_llm_aggregate"],
                "pipeline_llm_aggregate": run_b["pipeline_llm_aggregate"],
            },
        },
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def emphasize_headers(headers: list[str], indices: list[int]) -> list[str]:
    index_set = set(indices)
    return [bold(header) if idx in index_set else header for idx, header in enumerate(headers)]


def build_markdown(summary: dict, run_a: dict, run_b: dict) -> str:
    labels = [run_a["label"], run_b["label"]]

    doc_rows = []
    for doc in summary["documents"]:
        doc_rows.append(
            [
                doc["sample_id"],
                doc["document_id"],
                int_fmt(doc["source_requirement_count"]),
            ]
        )

    aggregate_rows = []
    for run in [run_a, run_b]:
        direct = run["direct_aggregate"] or {}
        pipeline = run["pipeline_aggregate"] or {}
        dialogue = run["dialogue_aggregate"] or {}
        llm = run["pipeline_llm_aggregate"] or {}
        aggregate_rows.append(
            [
                run["label"],
                run["model"],
                metric_fmt(dialogue.get("micro_coverage_recall")),
                metric_fmt(direct.get("micro_precision")),
                metric_fmt(direct.get("micro_coverage_recall")),
                metric_fmt(direct.get("micro_f1")),
                metric_fmt(pipeline.get("micro_precision")),
                metric_fmt(pipeline.get("micro_coverage_recall")),
                metric_fmt(pipeline.get("micro_f1")),
                metric_fmt(llm.get("micro_weighted_coverage_recall")),
                metric_fmt(llm.get("micro_weighted_f1")),
            ]
        )

    dialogue_rows = []
    direct_rows = []
    pipeline_rows = []
    validator_rows = []
    diagnostics_rows = []
    for doc in summary["documents"]:
        a = doc["runs"].get(labels[0]) or {}
        b = doc["runs"].get(labels[1]) or {}
        a_dialogue_recall = (a.get("dialogue") or {}).get("coverage_recall")
        b_dialogue_recall = (b.get("dialogue") or {}).get("coverage_recall")
        a_direct_p = (a.get("direct_semantic") or {}).get("precision")
        b_direct_p = (b.get("direct_semantic") or {}).get("precision")
        a_direct_r = (a.get("direct_semantic") or {}).get("recall")
        b_direct_r = (b.get("direct_semantic") or {}).get("recall")
        a_direct_f1 = (a.get("direct_semantic") or {}).get("f1")
        b_direct_f1 = (b.get("direct_semantic") or {}).get("f1")
        a_pipeline_p = (a.get("pipeline_semantic") or {}).get("precision")
        b_pipeline_p = (b.get("pipeline_semantic") or {}).get("precision")
        a_pipeline_r = (a.get("pipeline_semantic") or {}).get("recall")
        b_pipeline_r = (b.get("pipeline_semantic") or {}).get("recall")
        a_pipeline_f1 = (a.get("pipeline_semantic") or {}).get("f1")
        b_pipeline_f1 = (b.get("pipeline_semantic") or {}).get("f1")
        a_weighted_r = (a.get("pipeline_llm") or {}).get("weighted_recall")
        b_weighted_r = (b.get("pipeline_llm") or {}).get("weighted_recall")
        a_weighted_f1 = (a.get("pipeline_llm") or {}).get("weighted_f1")
        b_weighted_f1 = (b.get("pipeline_llm") or {}).get("weighted_f1")
        dialogue_pair = highlight_pair_markdown(a_dialogue_recall, b_dialogue_recall)
        direct_p_pair = highlight_pair_markdown(a_direct_p, b_direct_p)
        direct_r_pair = highlight_pair_markdown(a_direct_r, b_direct_r)
        direct_f1_pair = highlight_pair_markdown(a_direct_f1, b_direct_f1)
        pipeline_p_pair = highlight_pair_markdown(a_pipeline_p, b_pipeline_p)
        pipeline_r_pair = highlight_pair_markdown(a_pipeline_r, b_pipeline_r)
        pipeline_f1_pair = highlight_pair_markdown(a_pipeline_f1, b_pipeline_f1)
        weighted_r_pair = highlight_pair_markdown(a_weighted_r, b_weighted_r)
        weighted_f1_pair = highlight_pair_markdown(a_weighted_f1, b_weighted_f1)
        dialogue_rows.append(
            [
                doc["sample_id"],
                int_fmt(doc["source_requirement_count"]),
                dialogue_pair[0],
                int_fmt((a.get("dialogue") or {}).get("support_unit_count")),
                dialogue_pair[1],
                int_fmt((b.get("dialogue") or {}).get("support_unit_count")),
            ]
        )
        direct_rows.append(
            [
                doc["sample_id"],
                metric_fmt(a_direct_p),
                direct_r_pair[0],
                metric_fmt(a_direct_f1),
                metric_fmt(b_direct_p),
                direct_r_pair[1],
                metric_fmt(b_direct_f1),
            ]
        )
        pipeline_rows.append(
            [
                doc["sample_id"],
                metric_fmt(a_pipeline_p),
                pipeline_r_pair[0],
                metric_fmt(a_pipeline_f1),
                metric_fmt(b_pipeline_p),
                pipeline_r_pair[1],
                metric_fmt(b_pipeline_f1),
            ]
        )
        validator_rows.append(
            [
                doc["sample_id"],
                weighted_r_pair[0],
                metric_fmt(a_weighted_f1),
                weighted_r_pair[1],
                metric_fmt(b_weighted_f1),
            ]
        )
        diagnostics_rows.append(
            [
                doc["sample_id"],
                int_fmt((a.get("pipeline_diagnostics") or {}).get("evidence_bank_count")),
                int_fmt((a.get("pipeline_diagnostics") or {}).get("proposition_count")),
                int_fmt((a.get("pipeline_validation") or {}).get("grounded")),
                int_fmt((b.get("pipeline_diagnostics") or {}).get("evidence_bank_count")),
                int_fmt((b.get("pipeline_diagnostics") or {}).get("proposition_count")),
                int_fmt((b.get("pipeline_validation") or {}).get("grounded")),
            ]
        )

    dialogue_pair = highlight_pair_markdown(
        (run_a["dialogue_aggregate"] or {}).get("micro_coverage_recall"),
        (run_b["dialogue_aggregate"] or {}).get("micro_coverage_recall"),
    )
    direct_f1_pair = highlight_pair_markdown(
        (run_a["direct_aggregate"] or {}).get("micro_f1"),
        (run_b["direct_aggregate"] or {}).get("micro_f1"),
    )
    pipeline_r_pair = highlight_pair_markdown(
        (run_a["pipeline_aggregate"] or {}).get("micro_coverage_recall"),
        (run_b["pipeline_aggregate"] or {}).get("micro_coverage_recall"),
    )
    pipeline_f1_pair = highlight_pair_markdown(
        (run_a["pipeline_aggregate"] or {}).get("micro_f1"),
        (run_b["pipeline_aggregate"] or {}).get("micro_f1"),
    )
    weighted_r_pair = highlight_pair_markdown(
        (run_a["pipeline_llm_aggregate"] or {}).get("micro_weighted_coverage_recall"),
        (run_b["pipeline_llm_aggregate"] or {}).get("micro_weighted_coverage_recall"),
    )

    parts = [
        "# PURE 4-Document Cross-Model Paper Report",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Headline Findings",
        "",
        f"- Higher dialogue recall: {labels[0]} {dialogue_pair[0]} vs {labels[1]} {dialogue_pair[1]}",
        f"- Stronger direct baseline F1: {labels[0]} {direct_f1_pair[0]} vs {labels[1]} {direct_f1_pair[1]}",
        f"- Stronger conversational semantic recall: {labels[0]} {pipeline_r_pair[0]} vs {labels[1]} {pipeline_r_pair[1]}",
        f"- Stronger conversational semantic F1: {labels[0]} {pipeline_f1_pair[0]} vs {labels[1]} {pipeline_f1_pair[1]}",
        f"- Stronger weighted conversational validator recall: {labels[0]} {weighted_r_pair[0]} vs {labels[1]} {weighted_r_pair[1]}",
        "",
        "## Included Documents",
        "",
        md_table(["Sample ID", "Document ID", "Source Requirements"], doc_rows),
        "",
        "## Aggregate Headline Metrics",
        "",
        md_table(
            emphasize_headers(
                [
                    "Model",
                    "Generation Model",
                    "Dialogue Recall",
                    "Direct P",
                    "Direct R",
                    "Direct F1",
                    "Pipeline P",
                    "Pipeline R",
                    "Pipeline F1",
                    "LLM Weighted R",
                    "LLM Weighted F1",
                ],
                [2, 4, 7, 9],
            ),
            aggregate_rows,
        ),
        "",
        "## Per-Document Dialogue Coverage",
        "",
        md_table(
            emphasize_headers(
                ["Sample ID", "Reqs", f"{labels[0]} Recall", f"{labels[0]} Units", f"{labels[1]} Recall", f"{labels[1]} Units"],
                [2, 4],
            ),
            dialogue_rows,
        ),
        "",
        "## Per-Document Direct Semantic Metrics",
        "",
        md_table(
            emphasize_headers(
                ["Sample ID", f"{labels[0]} P", f"{labels[0]} R", f"{labels[0]} F1", f"{labels[1]} P", f"{labels[1]} R", f"{labels[1]} F1"],
                [2, 5],
            ),
            direct_rows,
        ),
        "",
        "## Per-Document Conversational Semantic Metrics",
        "",
        md_table(
            emphasize_headers(
                ["Sample ID", f"{labels[0]} P", f"{labels[0]} R", f"{labels[0]} F1", f"{labels[1]} P", f"{labels[1]} R", f"{labels[1]} F1"],
                [2, 5],
            ),
            pipeline_rows,
        ),
        "",
        "## Per-Document Conversational Weighted Validator Metrics",
        "",
        md_table(
            emphasize_headers(
                ["Sample ID", f"{labels[0]} Weighted R", f"{labels[0]} Weighted F1", f"{labels[1]} Weighted R", f"{labels[1]} Weighted F1"],
                [1, 3],
            ),
            validator_rows,
        ),
        "",
        "## Per-Document Conversational Diagnostics",
        "",
        md_table(
            ["Sample ID", f"{labels[0]} Evidence", f"{labels[0]} Props", f"{labels[0]} Grounded", f"{labels[1]} Evidence", f"{labels[1]} Props", f"{labels[1]} Grounded"],
            diagnostics_rows,
        ),
        "",
        "## Run Paths",
        "",
        f"- `{run_a['label']}`: `{safe_rel(Path(run_a['run_dir']))}`",
        f"- `{run_b['label']}`: `{safe_rel(Path(run_b['run_dir']))}`",
    ]
    return "\n".join(parts) + "\n"


def build_pdf(summary: dict, run_a: dict, run_b: dict, output_path: Path) -> None:
    if _REPORTLAB_IMPORT_ERROR is not None:
        raise SystemExit(
            "reportlab is required for scripts/build_cross_model_paper_report.py. "
            "Install it with `python3 -m pip install reportlab`."
        ) from _REPORTLAB_IMPORT_ERROR

    labels = [run_a["label"], run_b["label"]]
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        rightMargin=0.4 * inch,
        leftMargin=0.4 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    elements = []

    def add_title(text: str) -> None:
        elements.append(Paragraph(text, styles["Title"]))
        elements.append(Spacer(1, 0.15 * inch))

    def add_heading(text: str) -> None:
        elements.append(Paragraph(text, styles["Heading2"]))
        elements.append(Spacer(1, 0.08 * inch))

    def add_body(text: str) -> None:
        elements.append(Paragraph(text, styles["BodyText"]))
        elements.append(Spacer(1, 0.06 * inch))

    def add_table(
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[float] | None = None,
        highlight_columns: list[int] | None = None,
    ) -> None:
        data = [headers, *rows]
        table = Table(data, repeatRows=1, colWidths=col_widths)
        table_styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAFE")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#888888")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for column in highlight_columns or []:
            table_styles.extend(
                [
                    ("BACKGROUND", (column, 0), (column, 0), colors.HexColor("#BCD7FF")),
                    ("BACKGROUND", (column, 1), (column, -1), colors.HexColor("#EEF5FF")),
                ]
            )
        table.setStyle(TableStyle(table_styles))
        elements.append(table)
        elements.append(Spacer(1, 0.16 * inch))

    add_title("PURE 4-Document Cross-Model Paper Report")
    elements.append(Paragraph(f"Generated: {summary['generated_at_utc']}", styles["Normal"]))
    elements.append(Spacer(1, 0.12 * inch))
    elements.append(Paragraph(f"{labels[0]}: {run_a['model']} ({run_a['provider']})", styles["Normal"]))
    elements.append(Paragraph(f"{labels[1]}: {run_b['model']} ({run_b['provider']})", styles["Normal"]))
    elements.append(Spacer(1, 0.16 * inch))

    add_heading("Included Documents")
    add_table(
        ["Sample ID", "Document ID", "Source Requirements"],
        [
            [doc["sample_id"], doc["document_id"], int_fmt(doc["source_requirement_count"])]
            for doc in summary["documents"]
        ],
        col_widths=[2.2 * inch, 1.8 * inch, 1.2 * inch],
    )

    dialogue_a = metric_fmt((run_a["dialogue_aggregate"] or {}).get("micro_coverage_recall"))
    dialogue_b = metric_fmt((run_b["dialogue_aggregate"] or {}).get("micro_coverage_recall"))
    direct_f1_a = metric_fmt((run_a["direct_aggregate"] or {}).get("micro_f1"))
    direct_f1_b = metric_fmt((run_b["direct_aggregate"] or {}).get("micro_f1"))
    pipeline_r_a = metric_fmt((run_a["pipeline_aggregate"] or {}).get("micro_coverage_recall"))
    pipeline_r_b = metric_fmt((run_b["pipeline_aggregate"] or {}).get("micro_coverage_recall"))
    pipeline_f1_a = metric_fmt((run_a["pipeline_aggregate"] or {}).get("micro_f1"))
    pipeline_f1_b = metric_fmt((run_b["pipeline_aggregate"] or {}).get("micro_f1"))
    weighted_r_a = metric_fmt((run_a["pipeline_llm_aggregate"] or {}).get("micro_weighted_coverage_recall"))
    weighted_r_b = metric_fmt((run_b["pipeline_llm_aggregate"] or {}).get("micro_weighted_coverage_recall"))

    add_heading("Headline Findings")
    add_body(
        f"Higher dialogue recall: {labels[0]} {dialogue_a} vs {labels[1]} {dialogue_b}<br/>"
        f"Stronger direct baseline F1: {labels[0]} {direct_f1_a} vs {labels[1]} {direct_f1_b}<br/>"
        f"Stronger conversational semantic recall: {labels[0]} {pipeline_r_a} vs {labels[1]} {pipeline_r_b}<br/>"
        f"Stronger conversational semantic F1: {labels[0]} {pipeline_f1_a} vs {labels[1]} {pipeline_f1_b}<br/>"
        f"Stronger weighted conversational validator recall: {labels[0]} {weighted_r_a} vs {labels[1]} {weighted_r_b}"
    )

    add_heading("Aggregate Headline Metrics")
    add_table(
        ["Model", "Dialogue R", "Direct P", "Direct R", "Direct F1", "Pipeline P", "Pipeline R", "Pipeline F1", "LLM Weighted R", "LLM Weighted F1"],
        [
            [
                run["label"],
                metric_fmt((run["dialogue_aggregate"] or {}).get("micro_coverage_recall")),
                metric_fmt((run["direct_aggregate"] or {}).get("micro_precision")),
                metric_fmt((run["direct_aggregate"] or {}).get("micro_coverage_recall")),
                metric_fmt((run["direct_aggregate"] or {}).get("micro_f1")),
                metric_fmt((run["pipeline_aggregate"] or {}).get("micro_precision")),
                metric_fmt((run["pipeline_aggregate"] or {}).get("micro_coverage_recall")),
                metric_fmt((run["pipeline_aggregate"] or {}).get("micro_f1")),
                metric_fmt((run["pipeline_llm_aggregate"] or {}).get("micro_weighted_coverage_recall")),
                metric_fmt((run["pipeline_llm_aggregate"] or {}).get("micro_weighted_f1")),
            ]
            for run in [run_a, run_b]
        ],
        highlight_columns=[1, 3, 6, 8],
    )

    add_heading("Per-Document Dialogue Coverage")
    add_table(
        ["Sample ID", "Reqs", f"{labels[0]} Recall", f"{labels[0]} Units", f"{labels[1]} Recall", f"{labels[1]} Units"],
        [
            [
                doc["sample_id"],
                int_fmt(doc["source_requirement_count"]),
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("dialogue", {}).get("coverage_recall")),
                int_fmt((doc["runs"].get(labels[0]) or {}).get("dialogue", {}).get("support_unit_count")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("dialogue", {}).get("coverage_recall")),
                int_fmt((doc["runs"].get(labels[1]) or {}).get("dialogue", {}).get("support_unit_count")),
            ]
            for doc in summary["documents"]
        ],
        highlight_columns=[2, 4],
    )

    add_heading("Per-Document Direct Semantic Metrics")
    add_table(
        ["Sample ID", f"{labels[0]} P", f"{labels[0]} R", f"{labels[0]} F1", f"{labels[1]} P", f"{labels[1]} R", f"{labels[1]} F1"],
        [
            [
                doc["sample_id"],
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("direct_semantic", {}).get("precision")),
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("direct_semantic", {}).get("recall")),
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("direct_semantic", {}).get("f1")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("direct_semantic", {}).get("precision")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("direct_semantic", {}).get("recall")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("direct_semantic", {}).get("f1")),
            ]
            for doc in summary["documents"]
        ],
        highlight_columns=[2, 5],
    )

    add_heading("Per-Document Conversational Semantic Metrics")
    add_table(
        ["Sample ID", f"{labels[0]} P", f"{labels[0]} R", f"{labels[0]} F1", f"{labels[1]} P", f"{labels[1]} R", f"{labels[1]} F1"],
        [
            [
                doc["sample_id"],
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_semantic", {}).get("precision")),
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_semantic", {}).get("recall")),
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_semantic", {}).get("f1")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_semantic", {}).get("precision")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_semantic", {}).get("recall")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_semantic", {}).get("f1")),
            ]
            for doc in summary["documents"]
        ],
        highlight_columns=[2, 5],
    )

    elements.append(PageBreak())
    add_heading("Per-Document Conversational Weighted Validator Metrics")
    add_table(
        ["Sample ID", f"{labels[0]} Weighted R", f"{labels[0]} Weighted F1", f"{labels[1]} Weighted R", f"{labels[1]} Weighted F1"],
        [
            [
                doc["sample_id"],
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_llm", {}).get("weighted_recall")),
                metric_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_llm", {}).get("weighted_f1")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_llm", {}).get("weighted_recall")),
                metric_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_llm", {}).get("weighted_f1")),
            ]
            for doc in summary["documents"]
        ],
        highlight_columns=[1, 3],
    )

    add_heading("Per-Document Conversational Diagnostics")
    add_table(
        ["Sample ID", f"{labels[0]} Evidence", f"{labels[0]} Props", f"{labels[0]} Grounded", f"{labels[1]} Evidence", f"{labels[1]} Props", f"{labels[1]} Grounded"],
        [
            [
                doc["sample_id"],
                int_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_diagnostics", {}).get("evidence_bank_count")),
                int_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_diagnostics", {}).get("proposition_count")),
                int_fmt((doc["runs"].get(labels[0]) or {}).get("pipeline_validation", {}).get("grounded")),
                int_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_diagnostics", {}).get("evidence_bank_count")),
                int_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_diagnostics", {}).get("proposition_count")),
                int_fmt((doc["runs"].get(labels[1]) or {}).get("pipeline_validation", {}).get("grounded")),
            ]
            for doc in summary["documents"]
        ],
    )

    add_heading("Run Paths")
    elements.append(Paragraph(f"{labels[0]}: {safe_rel(Path(run_a['run_dir']))}", styles["Normal"]))
    elements.append(Paragraph(f"{labels[1]}: {safe_rel(Path(run_b['run_dir']))}", styles["Normal"]))

    doc.build(elements)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_a = load_run(args.run_dir_a, args.label_a)
    run_b = load_run(args.run_dir_b, args.label_b)
    summary = build_summary(run_a, run_b)

    summary_path = args.output_dir / "cross_model_comparison_summary.json"
    markdown_path = args.output_dir / "cross_model_comparison_report.md"
    pdf_path = args.output_dir / "cross_model_comparison_report.pdf"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown(summary, run_a, run_b), encoding="utf-8")
    build_pdf(summary, run_a, run_b, pdf_path)

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "markdown_path": str(markdown_path),
                "pdf_path": str(pdf_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
