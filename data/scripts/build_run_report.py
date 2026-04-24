#!/usr/bin/env python3
"""Build PDF reports for a PURE benchmark run."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "reportlab is required for scripts/build_run_report.py. "
        "Install it with `python3 -m pip install reportlab`."
    ) from exc

import evaluate_pure_requirements_coverage as coverage_eval


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen2.5:7b-instruct"
MATCH_THRESHOLD = 0.55
CATEGORY_ORDER = ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]
CATEGORY_LABELS = {
    "functional": "Functional",
    "non_functional": "Non-Functional",
    "data": "Data",
    "business_rules": "Business Rules",
    "interfaces": "Interfaces",
    "constraints": "Constraints",
}

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_UNICODE = "Helvetica"

LIGHT_BLUE = colors.HexColor("#D9EAFE")
LIGHT_GREEN = colors.HexColor("#E7F6E7")
LIGHT_RED = colors.HexColor("#FDECEC")
LIGHT_AMBER = colors.HexColor("#FFF4D6")
VERY_LIGHT_GRAY = colors.HexColor("#F5F5F5")
BAR_GREEN = colors.HexColor("#66BB6A")
BAR_AMBER = colors.HexColor("#FFB74D")
BAR_RED = colors.HexColor("#EF5350")
BAR_BG = colors.HexColor("#E0E0E0")


def register_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD, FONT_UNICODE

    unicode_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    regular_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]

    try:
        for path in unicode_candidates:
            if path.exists():
                pdfmetrics.registerFont(TTFont("ReportUnicode", str(path)))
                FONT_UNICODE = "ReportUnicode"
                FONT_REGULAR = FONT_UNICODE
                break
        for path in regular_candidates:
            if path.exists():
                pdfmetrics.registerFont(TTFont("ReportRegular", str(path)))
                FONT_REGULAR = "ReportRegular"
                break
        for path in bold_candidates:
            if path.exists():
                pdfmetrics.registerFont(TTFont("ReportBold", str(path)))
                FONT_BOLD = "ReportBold"
                break
    except Exception:
        FONT_REGULAR = FONT_UNICODE if FONT_UNICODE != "Helvetica" else "Helvetica"
        FONT_BOLD = "Helvetica-Bold"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PDF reports for a PURE benchmark run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under outputs/pure_full_runs/<run_id>.")
    return parser.parse_args()


def find_first_existing(run_dir: Path, candidates: list[str], kind: str) -> Path:
    for rel in candidates:
        path = run_dir / rel
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {kind} in {run_dir}: tried {candidates}")


def load_model_name(run_dir: Path) -> str:
    run_config = run_dir / "run_config.json"
    if run_config.exists():
        payload = load_json(run_config)
        if isinstance(payload, dict):
            for key in ["model", "llm_model"]:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            llm_config = payload.get("llm")
            if isinstance(llm_config, dict):
                value = llm_config.get("model")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return DEFAULT_MODEL


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def normalize_role(role: str) -> str:
    lowered = (role or "").strip().lower()
    if lowered == "user":
        return "USER"
    return "SYSTEM"


def extract_dialogue_turns(payload: dict) -> list[dict]:
    turns = []
    for index, item in enumerate(payload.get("dialogue", []), start=1):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            content = item.get("text", "")
        text = " ".join(str(content).split())
        if not text:
            continue
        turns.append(
            {
                "turn_id": item.get("turn_id", index),
                "role": normalize_role(str(item.get("role", "system"))),
                "content": text,
            }
        )
    return turns


def extract_clarification_info(payload: dict) -> dict:
    dialogue_generation = payload.get("dialogue_generation", {})
    if not isinstance(dialogue_generation, dict):
        dialogue_generation = {}
    coverage_summary = dialogue_generation.get("coverage_summary", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    return {
        "clarification_rounds_requested": coverage_summary.get(
            "clarification_rounds_requested",
            dialogue_generation.get("clarification_rounds"),
        ),
        "clarification_rounds_used": coverage_summary.get("clarification_rounds_used"),
        "clarification_chunk_count": dialogue_generation.get("clarification_chunk_count"),
        "initial_uncovered_requirement_count": coverage_summary.get("initial_uncovered_requirement_count"),
        "final_uncovered_requirement_count": coverage_summary.get("final_uncovered_requirement_count"),
    }


def extract_source_requirements(payload: dict) -> list[dict]:
    items = []
    for index, item in enumerate(payload.get("ground_truth_requirements", []), start=1):
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        items.append(
            {
                "id": item.get("req_id", f"SRC-{index:03d}"),
                "text": text,
                "category": item.get("category") or item.get("source_type") or "source",
            }
        )
    return items


def extract_generated_requirements(payload: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    reqs = payload.get("requirements", {})
    flat: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    if isinstance(reqs, dict):
        for category in CATEGORY_ORDER:
            items = reqs.get(category, [])
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                text = " ".join(str(item.get("text", "")).split())
                if not text:
                    continue
                entry = {
                    "id": item.get("id", f"{category.upper()}-{index:03d}"),
                    "text": text,
                    "category": category,
                }
                flat.append(entry)
                grouped[category].append(entry)
    return flat, grouped


def best_generated_match(source_text: str, generated: list[dict]) -> tuple[int | None, float]:
    best_index = None
    best_score = 0.0
    for index, item in enumerate(generated):
        score = coverage_eval.token_f1(source_text, item["text"])
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, best_score


def build_requirement_match_rows(source_requirements: list[dict], generated_requirements: list[dict]) -> list[dict]:
    gold_texts = [item["text"] for item in source_requirements]
    pred_texts = [item["text"] for item in generated_requirements]
    matches, _, _ = coverage_eval.greedy_match(gold_texts, pred_texts, MATCH_THRESHOLD)
    matched_by_gold = {item["gold_index"]: item for item in matches}

    rows = []
    for gold_index, source_item in enumerate(source_requirements):
        matched = matched_by_gold.get(gold_index)
        if matched is not None:
            pred_item = generated_requirements[matched["pred_index"]]
            rows.append(
                {
                    "source_text": source_item["text"],
                    "matched": True,
                    "closest_text": pred_item["text"],
                    "score": matched["score"],
                }
            )
            continue

        closest_index, closest_score = best_generated_match(source_item["text"], generated_requirements)
        rows.append(
            {
                "source_text": source_item["text"],
                "matched": False,
                "closest_text": "— not recovered",
                "best_generated_text": generated_requirements[closest_index]["text"] if closest_index is not None else "",
                "score": closest_score,
            }
        )
    return rows


def make_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "subheading": ParagraphStyle(
            "ReportSubheading",
            parent=styles["Heading3"],
            fontName=FONT_BOLD,
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=13,
            spaceAfter=2,
        ),
        "body_small": ParagraphStyle(
            "ReportBodySmall",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9,
            leading=11,
            spaceAfter=1,
        ),
        "body_small_bold": ParagraphStyle(
            "ReportBodySmallBold",
            parent=styles["BodyText"],
            fontName=FONT_BOLD,
            fontSize=9,
            leading=11,
            spaceAfter=1,
        ),
        "unicode_center": ParagraphStyle(
            "ReportUnicodeCenter",
            parent=styles["BodyText"],
            fontName=FONT_UNICODE,
            fontSize=11,
            leading=12,
            alignment=1,
        ),
    }


def build_summary_pdf(context: dict, output_path: Path) -> None:
    run_id = context["run_id"]
    model_name = context["model_name"]
    direct = context["direct_metrics"]["aggregate"]
    pipeline = context["pipeline_metrics"]["aggregate"]
    dialogue = context["dialogue_metrics"]["aggregate"]
    per_sample = context["pipeline_metrics"]["per_sample"]
    clarification_summary = context["clarification_summary"]

    page_width, page_height = letter
    margin = 40
    bar_width = 400

    doc_count = max(1, len(per_sample))
    bar_row_gap = 22 if doc_count <= 6 else max(14, 22 - (doc_count - 6))
    doc_font_size = 9 if doc_count <= 8 else 8

    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    pdf.setTitle(f"Benchmark run report - {run_id}")

    y = page_height - margin

    pdf.setFont(FONT_BOLD, 18)
    pdf.drawString(margin, y, "Benchmark run report")
    y -= 24

    pdf.setFont(FONT_REGULAR, 10)
    header_lines = [
        f"Run ID: {run_id}",
        f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model used: {model_name}",
        f"Clarification rounds: requested {clarification_summary['requested_label']} | used {clarification_summary['used_label']}",
    ]
    for line in header_lines:
        pdf.drawString(margin, y, line)
        y -= 14

    y -= 8
    table_data = [
        ["Condition", "Precision", "Recall", "F1", "Hallucination rate"],
        [
            "Direct baseline",
            format_metric(direct.get("micro_precision")),
            format_metric(direct.get("micro_coverage_recall")),
            format_metric(direct.get("micro_f1")),
            format_metric(direct.get("macro_hallucination_rate")),
        ],
        [
            "Conversational pipeline",
            format_metric(pipeline.get("micro_precision")),
            format_metric(pipeline.get("micro_coverage_recall")),
            format_metric(pipeline.get("micro_f1")),
            format_metric(pipeline.get("macro_hallucination_rate")),
        ],
        ["Oracle", "1.0000", "1.0000", "1.0000", "0.0000"],
    ]
    metrics_table = Table(table_data, colWidths=[185, 70, 70, 70, 110])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
                ("BACKGROUND", (0, 2), (-1, 2), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    _, table_height = metrics_table.wrapOn(pdf, page_width - 2 * margin, y)
    metrics_table.drawOn(pdf, margin, y - table_height)
    y -= table_height + 18

    pdf.setFont(FONT_BOLD, 11)
    pdf.drawString(margin, y, "Per-document coverage")
    y -= 14

    for item in per_sample:
        doc_id = item.get("document_id") or item.get("sample_id")
        recall = float(item.get("coverage_recall", 0.0))
        matched = int(item.get("matched_count", 0))
        source_count = int(item.get("source_requirement_count", 0))
        label = f"{doc_id}: {matched} of {source_count} requirements recovered ({format_percent(recall)})"
        pdf.setFont(FONT_REGULAR, doc_font_size)
        pdf.drawString(margin, y, label)
        y -= 10

        if recall >= 0.80:
            bar_color = BAR_GREEN
        elif recall >= 0.60:
            bar_color = BAR_AMBER
        else:
            bar_color = BAR_RED
        pdf.setFillColor(BAR_BG)
        pdf.rect(margin, y - 2, bar_width, 8, stroke=0, fill=1)
        pdf.setFillColor(bar_color)
        pdf.rect(margin, y - 2, bar_width * recall, 8, stroke=0, fill=1)
        pdf.setFillColor(colors.black)
        y -= bar_row_gap

    y -= 4
    pdf.setFont(FONT_REGULAR, 10)
    pdf.drawString(
        margin,
        y,
        f"Dialogue upper bound: {dialogue.get('micro_coverage_recall', 0.0) * 100:.1f}% of source requirements present in user turns",
    )
    y -= 18

    direct_f1 = float(direct.get("micro_f1", 0.0))
    pipeline_f1 = float(pipeline.get("micro_f1", 0.0))
    delta = pipeline_f1 - direct_f1
    if delta > 0.01:
        finding = f"Conversational pipeline outperforms direct baseline by +{delta:.4f} F1."
    elif delta < -0.01:
        finding = "Direct baseline outperforms conversational pipeline. Dialogue stage may be lossy."
    else:
        finding = "Conversational and direct pipelines performed equivalently this run."

    pdf.setFont(FONT_BOLD, 10)
    pdf.drawString(margin, y, "Key finding:")
    pdf.setFont(FONT_REGULAR, 10)
    pdf.drawString(margin + 62, y, finding)

    pdf.showPage()
    pdf.save()


def page_footer(pdf_canvas, doc) -> None:  # pragma: no cover - rendering callback
    pdf_canvas.saveState()
    pdf_canvas.setFont(FONT_REGULAR, 8)
    pdf_canvas.setFillColor(colors.grey)
    pdf_canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 18, f"Page {doc.page}")
    pdf_canvas.restoreState()


def build_document_header_table(doc_metrics: dict, styles: dict[str, ParagraphStyle]) -> Table:
    clarification_info = doc_metrics.get("clarification_info", {})
    clarification_requested = clarification_info.get("clarification_rounds_requested")
    clarification_used = clarification_info.get("clarification_rounds_used")
    clarification_chunks = clarification_info.get("clarification_chunk_count")
    rows = [
        ["Document ID", str(doc_metrics.get("document_id") or doc_metrics.get("sample_id", ""))],
        ["Source requirement count", str(doc_metrics.get("source_requirement_count", 0))],
        ["Generated requirement count", str(doc_metrics.get("generated_requirement_count", 0))],
        ["Matched count", str(doc_metrics.get("matched_count", 0))],
        [
            "Clarification rounds",
            f"requested {clarification_requested if clarification_requested is not None else 'N/A'} / used {clarification_used if clarification_used is not None else 'N/A'}",
        ],
        [
            "Clarification chunks",
            str(clarification_chunks if clarification_chunks is not None else "N/A"),
        ],
        [
            "Precision / Recall / F1",
            (
                f"{format_metric(doc_metrics.get('precision'))} / "
                f"{format_metric(doc_metrics.get('coverage_recall'))} / "
                f"{format_metric(doc_metrics.get('f1'))}"
            ),
        ],
    ]
    table = Table(rows, colWidths=[170, 330])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F2F2")),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), FONT_BOLD),
                ("FONTNAME", (1, 0), (1, -1), FONT_REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_transcript_table(dialogue_turns: list[dict], doc_width: float, styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    table_style = [
        ("BOX", (0, 0), (-1, -1), 0.35, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]

    for row_index, turn in enumerate(dialogue_turns):
        label = turn["role"]
        paragraph = Paragraph(
            f"<font name='{FONT_BOLD}'>[{escape(label)}]</font> {escape(turn['content'])}",
            styles["body_small"],
        )
        rows.append([paragraph])
        background = VERY_LIGHT_GRAY if label == "USER" else colors.white
        table_style.append(("BACKGROUND", (0, row_index), (-1, row_index), background))

    if not rows:
        rows = [[make_paragraph("No dialogue found.", styles["body_small"])]]

    table = Table(rows, colWidths=[doc_width], repeatRows=0)
    table.setStyle(TableStyle(table_style))
    return table


def build_requirement_category_tables(
    grouped_requirements: dict[str, list[dict]],
    doc_width: float,
    styles: dict[str, ParagraphStyle],
) -> list:
    flowables: list = []
    for category in CATEGORY_ORDER:
        items = grouped_requirements.get(category, [])
        if not items:
            continue
        flowables.append(Paragraph(CATEGORY_LABELS.get(category, category.title()), styles["subheading"]))
        rows = []
        for item in items:
            req_id = str(item.get("id", "REQ"))
            req_text = str(item.get("text", ""))
            rows.append(
                [
                    Paragraph(f"<font name='{FONT_BOLD}'>{escape(req_id)}</font>", styles["body"]),
                    make_paragraph(req_text, styles["body"]),
                ]
            )
        table = Table(rows, colWidths=[72, doc_width - 72])
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        flowables.append(table)
        flowables.append(Spacer(1, 6))
    if not flowables:
        flowables.append(make_paragraph("No generated requirements found.", styles["body"]))
    return flowables


def status_paragraph(matched: bool, styles: dict[str, ParagraphStyle]) -> Paragraph:
    symbol = "✓" if matched else "✗"
    color = "#2E7D32" if matched else "#C62828"
    return Paragraph(
        f"<font name='{FONT_UNICODE}' color='{color}'>{symbol}</font>",
        styles["unicode_center"],
    )


def build_match_table(match_rows: list[dict], doc_width: float, styles: dict[str, ParagraphStyle]) -> LongTable:
    rows = [
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Source requirement text</font>", styles["body_small"]),
            Paragraph(f"<font name='{FONT_BOLD}'>Matched?</font>", styles["body_small"]),
            Paragraph(f"<font name='{FONT_BOLD}'>Closest generated requirement text</font>", styles["body_small"]),
        ]
    ]

    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ("BOX", (0, 0), (-1, -1), 0.35, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]

    for row_index, item in enumerate(match_rows, start=1):
        rows.append(
            [
                make_paragraph(item["source_text"], styles["body_small"]),
                status_paragraph(item["matched"], styles),
                make_paragraph(item["closest_text"], styles["body_small"]),
            ]
        )
        background = LIGHT_GREEN if item["matched"] else LIGHT_RED
        table_style.append(("BACKGROUND", (0, row_index), (-1, row_index), background))

    table = LongTable(rows, colWidths=[doc_width * 0.43, doc_width * 0.10, doc_width * 0.47], repeatRows=1)
    table.setStyle(TableStyle(table_style))
    return table


def build_full_detail_pdf(context: dict, output_path: Path) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=28,
        title=f"Benchmark run full detail - {context['run_id']}",
    )
    doc_width = doc.width

    story = [
        Paragraph("Benchmark run full detail", styles["title"]),
        Paragraph(f"Run ID: {context['run_id']}", styles["body"]),
        Paragraph(f"Model used: {context['model_name']}", styles["body"]),
        Spacer(1, 10),
    ]

    pipeline_by_sample = {item["sample_id"]: item for item in context["pipeline_metrics"]["per_sample"]}

    for doc_index, sample in enumerate(context["documents"]):
        sample_id = sample["sample_id"]
        metrics = dict(pipeline_by_sample.get(sample_id, {}))
        metrics["clarification_info"] = sample.get("clarification_info", {})

        if doc_index > 0:
            story.append(PageBreak())

        story.append(Paragraph(f"Document: {metrics.get('document_id') or sample_id}", styles["heading"]))
        story.append(Paragraph("Part A - Document header", styles["subheading"]))
        story.append(build_document_header_table(metrics, styles))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Part B - Generated dialogue transcript", styles["subheading"]))
        story.append(build_transcript_table(sample["dialogue_turns"], doc_width, styles))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Part C - Extracted requirements by category", styles["subheading"]))
        story.extend(build_requirement_category_tables(sample["generated_grouped"], doc_width, styles))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Part D - Matched vs missed table", styles["subheading"]))
        story.append(build_match_table(sample["match_rows"], doc_width, styles))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def collect_documents(run_dir: Path, source_dir: Path, pred_dir: Path, dialogue_dir: Path) -> list[dict]:
    documents = []
    for source_path in sorted(source_dir.glob("*.json")):
        if source_path.name in {"summary.json", "evaluation.json"}:
            continue
        source_payload = load_json(source_path)
        sample_id = source_payload.get("sample_id")
        if not sample_id:
            continue

        pred_path = pred_dir / f"{sample_id}.json"
        dialogue_path = dialogue_dir / f"{sample_id}.json"
        if not pred_path.exists() or not dialogue_path.exists():
            continue

        pred_payload = load_json(pred_path)
        dialogue_payload = load_json(dialogue_path)
        source_requirements = extract_source_requirements(source_payload)
        generated_requirements, generated_grouped = extract_generated_requirements(pred_payload)
        dialogue_turns = extract_dialogue_turns(dialogue_payload)
        clarification_info = extract_clarification_info(dialogue_payload)
        match_rows = build_requirement_match_rows(source_requirements, generated_requirements)

        documents.append(
            {
                "sample_id": sample_id,
                "document_id": source_payload.get("source", {}).get("document_id", sample_id),
                "source_payload": source_payload,
                "pred_payload": pred_payload,
                "dialogue_payload": dialogue_payload,
                "source_requirements": source_requirements,
                "generated_requirements": generated_requirements,
                "generated_grouped": generated_grouped,
                "dialogue_turns": dialogue_turns,
                "clarification_info": clarification_info,
                "match_rows": match_rows,
            }
        )
    return documents


def summarize_clarification(documents: list[dict]) -> dict:
    requested_values = []
    used_values = []
    chunk_values = []
    for item in documents:
        info = item.get("clarification_info", {})
        requested = info.get("clarification_rounds_requested")
        used = info.get("clarification_rounds_used")
        chunks = info.get("clarification_chunk_count")
        if isinstance(requested, int):
            requested_values.append(requested)
        if isinstance(used, int):
            used_values.append(used)
        if isinstance(chunks, int):
            chunk_values.append(chunks)

    def label(values: list[int]) -> str:
        if not values:
            return "N/A"
        unique = sorted(set(values))
        if len(unique) == 1:
            return str(unique[0])
        return f"{unique[0]}-{unique[-1]}"

    return {
        "requested_label": label(requested_values),
        "used_label": label(used_values),
        "chunk_label": label(chunk_values),
    }


def build_context(run_dir: Path) -> dict:
    metrics_dir = run_dir / "metrics"
    pipeline_metrics_path = find_first_existing(
        run_dir,
        [
            "metrics/pipeline_coverage_postprocessed_v3.json",
            "metrics/pipeline_coverage_postprocessed_v2.json",
            "metrics/pipeline_coverage.json",
        ],
        "pipeline metrics",
    )
    direct_metrics_path = find_first_existing(run_dir, ["metrics/direct_coverage.json"], "direct metrics")
    dialogue_metrics_path = find_first_existing(
        run_dir,
        ["metrics/dialogue_coverage_user_only.json"],
        "dialogue coverage metrics",
    )
    source_dir = find_first_existing(run_dir, ["source_requirements"], "source requirements directory")
    pred_dir = find_first_existing(
        run_dir,
        [
            "generated_requirements_postprocessed_v3",
            "generated_requirements_postprocessed_v2",
            "generated_requirements_postprocessed",
            "generated_requirements",
        ],
        "generated requirements directory",
    )
    dialogue_dir = find_first_existing(run_dir, ["expanded_dialogues"], "expanded dialogues directory")

    pipeline_metrics = load_json(pipeline_metrics_path)
    direct_metrics = load_json(direct_metrics_path)
    dialogue_metrics = load_json(dialogue_metrics_path)

    documents = collect_documents(run_dir, source_dir, pred_dir, dialogue_dir)
    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "model_name": load_model_name(run_dir),
        "metrics_dir": metrics_dir,
        "pipeline_metrics": pipeline_metrics,
        "direct_metrics": direct_metrics,
        "dialogue_metrics": dialogue_metrics,
        "source_dir": source_dir,
        "pred_dir": pred_dir,
        "dialogue_dir": dialogue_dir,
        "documents": documents,
        "clarification_summary": summarize_clarification(documents),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    register_fonts()
    context = build_context(run_dir)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_pdf = reports_dir / "summary.pdf"
    detail_pdf = reports_dir / "full_detail.pdf"

    build_summary_pdf(context, summary_pdf)
    build_full_detail_pdf(context, detail_pdf)

    print(f"Wrote {safe_rel(summary_pdf)}")
    print(f"Wrote {safe_rel(detail_pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
