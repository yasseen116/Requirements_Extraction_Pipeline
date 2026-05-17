#!/usr/bin/env python3
"""Render repository markdown documents to a clean A4 PDF using reportlab.

This renderer is intentionally conservative. It supports the markdown structures
used by the project docs: headings, paragraphs, bullet/numbered lists, code
fences, block quotes, and pipe tables.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HeadingBlock:
    level: int
    text: str


@dataclass
class ParagraphBlock:
    text: str


@dataclass
class ListBlock:
    ordered: bool
    items: list[str]


@dataclass
class CodeBlock:
    language: str
    text: str


@dataclass
class QuoteBlock:
    text: str


@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]


Block = HeadingBlock | ParagraphBlock | ListBlock | CodeBlock | QuoteBlock | TableBlock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown files to a clean PDF.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input markdown files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory")
    return parser.parse_args()


def strip_frontmatter(lines: list[str]) -> list[str]:
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return lines[idx + 1 :]
    return lines


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|").replace(":", "").replace("-", "").replace(" ", "")
    return stripped == ""


def is_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+", line))


def is_ordered_item(line: str) -> bool:
    return bool(re.match(r"^\d+\.\s+", line))


def is_unordered_item(line: str) -> bool:
    return line.startswith("- ")


def parse_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def collect_paragraph(lines: list[str], start: int) -> tuple[str, int]:
    collected: list[str] = []
    idx = start
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            break
        if (
            is_heading(line)
            or line.startswith("```")
            or is_table_line(line)
            or line.startswith("> ")
            or is_ordered_item(line)
            or is_unordered_item(line)
        ):
            break
        collected.append(line.strip())
        idx += 1
    return " ".join(collected).strip(), idx


def parse_markdown(text: str) -> list[Block]:
    lines = strip_frontmatter(text.splitlines())
    blocks: list[Block] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if not stripped:
            idx += 1
            continue

        if is_heading(line):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(HeadingBlock(level=level, text=line[level:].strip()))
            idx += 1
            continue

        if line.startswith("```"):
            language = line[3:].strip()
            idx += 1
            code_lines: list[str] = []
            while idx < len(lines) and not lines[idx].startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            if idx < len(lines):
                idx += 1
            blocks.append(CodeBlock(language=language, text="\n".join(code_lines).rstrip()))
            continue

        if is_table_line(line):
            table_lines = []
            while idx < len(lines) and is_table_line(lines[idx]):
                table_lines.append(lines[idx])
                idx += 1
            if len(table_lines) >= 2 and is_table_separator(table_lines[1]):
                headers = parse_pipe_row(table_lines[0])
                rows = [parse_pipe_row(row) for row in table_lines[2:]]
                blocks.append(TableBlock(headers=headers, rows=rows))
            else:
                paragraph_text = " ".join(item.strip() for item in table_lines)
                blocks.append(ParagraphBlock(text=paragraph_text))
            continue

        if line.startswith("> "):
            quote_lines = []
            while idx < len(lines) and lines[idx].startswith("> "):
                quote_lines.append(lines[idx][2:].strip())
                idx += 1
            blocks.append(QuoteBlock(text=" ".join(quote_lines)))
            continue

        if is_ordered_item(line) or is_unordered_item(line):
            ordered = is_ordered_item(line)
            items: list[str] = []
            while idx < len(lines):
                current = lines[idx]
                if ordered and is_ordered_item(current):
                    items.append(re.sub(r"^\d+\.\s+", "", current).strip())
                    idx += 1
                    continue
                if (not ordered) and is_unordered_item(current):
                    items.append(current[2:].strip())
                    idx += 1
                    continue
                if not current.strip():
                    idx += 1
                    break
                break
            blocks.append(ListBlock(ordered=ordered, items=items))
            continue

        paragraph_text, idx = collect_paragraph(lines, idx)
        if paragraph_text:
            blocks.append(ParagraphBlock(text=paragraph_text))
        else:
            idx += 1

    return blocks


def render_inline(text: str) -> str:
    text = escape(text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<link href="{escape(m.group(2))}">{m.group(1)}</link>',
        text,
    )
    text = re.sub(r"`([^`]+)`", lambda m: f'<font face="Courier">{escape(m.group(1))}</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip()
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=19,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=17,
            leading=21,
            spaceBefore=14,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=14,
            leading=17,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "h4": ParagraphStyle(
            "H4",
            parent=base["Heading4"],
            fontName="Times-Bold",
            fontSize=10.5,
            leading=13,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14,
            spaceAfter=7,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#444444"),
            leftIndent=18,
            rightIndent=12,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "list": ParagraphStyle(
            "List",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=8,
            backColor=colors.HexColor("#F4F4F4"),
            borderColor=colors.HexColor("#D7D7D7"),
            borderWidth=0.5,
            borderPadding=6,
            borderRadius=2,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
        ),
    }


def build_story(path: Path, text: str) -> tuple[str, list]:
    doc_styles = styles()
    doc_title = extract_title(path, text)
    blocks = parse_markdown(text)

    story: list = [
        Paragraph(escape(doc_title), doc_styles["title"]),
        Paragraph(escape(path.name), doc_styles["meta"]),
    ]

    first_h1_consumed = False
    for block in blocks:
        if isinstance(block, HeadingBlock):
            if block.level == 1 and not first_h1_consumed and block.text == doc_title:
                first_h1_consumed = True
                continue
            first_h1_consumed = first_h1_consumed or block.level == 1
            style_key = {1: "h1", 2: "h2", 3: "h3"}.get(block.level, "h4")
            story.append(Paragraph(render_inline(block.text), doc_styles[style_key]))
            continue

        if isinstance(block, ParagraphBlock):
            story.append(Paragraph(render_inline(block.text), doc_styles["body"]))
            continue

        if isinstance(block, QuoteBlock):
            story.append(Paragraph(render_inline(block.text), doc_styles["quote"]))
            continue

        if isinstance(block, ListBlock):
            flowables = [
                ListItem(Paragraph(render_inline(item), doc_styles["list"]), leftIndent=12)
                for item in block.items
            ]
            story.append(
                ListFlowable(
                    flowables,
                    bulletType="1" if block.ordered else "bullet",
                    start="1",
                    leftIndent=18,
                    spaceBefore=2,
                    spaceAfter=8,
                )
            )
            continue

        if isinstance(block, CodeBlock):
            label = "Diagram source (Mermaid)" if block.language == "mermaid" else f"Code block ({block.language})" if block.language else "Code block"
            story.append(Paragraph(escape(label), doc_styles["h4"]))
            story.append(Preformatted(block.text or " ", doc_styles["code"]))
            continue

        if isinstance(block, TableBlock):
            rows = [[Paragraph(render_inline(cell), doc_styles["table"]) for cell in block.headers]]
            for row in block.rows:
                rows.append([Paragraph(render_inline(cell), doc_styles["table"]) for cell in row])
            col_width = 7.0 * inch / max(1, len(block.headers))
            table = LongTable(rows, repeatRows=1, colWidths=[col_width] * len(block.headers))
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FB")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(KeepTogether([table, Spacer(1, 0.12 * inch)]))
            continue

    return doc_title, story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    page_number = canvas.getPageNumber()
    canvas.drawString(doc.leftMargin, 0.45 * inch, doc.title)
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, f"Page {page_number}")
    canvas.restoreState()


def render_file(path: Path, output_dir: Path | None) -> Path:
    text = path.read_text(encoding="utf-8")
    title, story = build_story(path, text)
    destination = (output_dir or path.parent) / f"{path.stem}.pdf"
    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        title=title,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.7 * inch,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for input_path in args.inputs:
        path = input_path.resolve()
        outputs.append(str(render_file(path, output_dir)))

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
