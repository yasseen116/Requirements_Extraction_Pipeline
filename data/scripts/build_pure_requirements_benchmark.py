#!/usr/bin/env python3
"""Build a PURE XML requirement benchmark for full-requirement validation."""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_INDEX = ROOT / "outputs" / "public_source_index.json"
DEFAULT_OUTPUT_DIR = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_SUMMARY_PATH = ROOT / "raw_sources" / "pure_benchmark" / "summary.json"


MODAL_TERMS = (
    "shall",
    "must",
    "should",
    "required",
    "cannot",
    "must not",
    "should not",
    "never",
    "only if",
    "at least",
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_text(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?;])\s+", text)
    return [clean_text(part) for part in parts if clean_text(part)]


def has_modal_requirement_signal(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in MODAL_TERMS)


def extract_text_from_xml(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    chunks = [clean_text(chunk) for chunk in root.itertext()]
    chunks = [chunk for chunk in chunks if chunk]
    return "\n".join(chunks)


def extract_explicit_requirements(document_text: str) -> list[dict]:
    compact = re.sub(r"\s+", " ", document_text)
    pattern = re.compile(
        r"\bREQ[\s\-]*(\d+)\s*:\s*(.+?)(?=\bREQ[\s\-]*\d+\s*:|$)",
        re.IGNORECASE,
    )
    requirements = []
    for match in pattern.finditer(compact):
        req_id = f"REQ-{int(match.group(1))}"
        text = clean_text(match.group(2))
        text = re.sub(r"^[\-\:\.]+", "", text).strip()
        if len(text) < 8:
            continue
        requirements.append(
            {
                "id": req_id,
                "text": text,
                "source_type": "explicit_req_tag",
            }
        )
    return requirements


def extract_inferred_requirements(document_text: str, explicit_texts: set[str]) -> list[dict]:
    inferred = []
    for sentence in sentence_split(document_text):
        if len(sentence.split()) < 6:
            continue
        if len(sentence.split()) > 45:
            continue
        if sentence.lower().startswith("chapter "):
            continue
        if not has_modal_requirement_signal(sentence):
            continue

        normalized = normalize_text(sentence)
        if normalized in explicit_texts:
            continue

        inferred.append(
            {
                "id": "",
                "text": sentence,
                "source_type": "inferred_modal_sentence",
            }
        )
    return inferred


def dedupe_requirements(requirements: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    auto_index = 1
    for item in requirements:
        normalized = normalize_text(item["text"])
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        req_id = item["id"] or f"AUTO-{auto_index:04d}"
        if not item["id"]:
            auto_index += 1
        deduped.append(
            {
                "req_id": req_id,
                "text": item["text"],
                "source_type": item["source_type"],
                "normalized_text": normalized,
            }
        )
    return deduped


def load_pure_xml_documents(source_index_path: Path) -> list[dict]:
    payload = json.loads(source_index_path.read_text(encoding="utf-8"))
    pure_docs = payload["datasets"]["pure"]["documents"]
    return [doc for doc in pure_docs if doc.get("has_xml") and doc.get("xml_path")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, default=SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-requirements", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic document sampling (0 keeps stable default ordering).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = load_pure_xml_documents(args.source_index)
    if args.seed != 0:
        rng = random.Random(args.seed)
        rng.shuffle(documents)
    if args.max_docs is not None:
        documents = documents[: args.max_docs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = []
    total_requirements = 0
    skipped = 0

    for doc in documents:
        xml_path = ROOT / doc["xml_path"]
        source_path = ROOT / doc["source_path"] if doc.get("source_path") else None
        text = extract_text_from_xml(xml_path)

        explicit = extract_explicit_requirements(text)
        explicit_norm = {normalize_text(item["text"]) for item in explicit}
        inferred = extract_inferred_requirements(text, explicit_norm)
        requirements = dedupe_requirements(explicit + inferred)

        if len(requirements) < args.min_requirements:
            skipped += 1
            summary.append(
                {
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "status": "skipped_too_few_requirements",
                    "requirement_count": len(requirements),
                }
            )
            continue

        sample_id = f"pure_{doc['document_id']}"
        output_path = args.output_dir / f"{sample_id}.json"
        payload = {
            "sample_id": sample_id,
            "source": {
                "dataset": "pure",
                "dataset_doi": "10.5281/zenodo.7118517",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "xml_path": str(xml_path.relative_to(ROOT)),
                "source_path": str(source_path.relative_to(ROOT)) if source_path else None,
            },
            "ground_truth_requirements": requirements,
            "statistics": {
                "explicit_count": sum(item["source_type"] == "explicit_req_tag" for item in requirements),
                "inferred_count": sum(item["source_type"] == "inferred_modal_sentence" for item in requirements),
                "total_count": len(requirements),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        total_requirements += len(requirements)
        summary.append(
            {
                "sample_id": sample_id,
                "document_id": doc["document_id"],
                "title": doc["title"],
                "path": str(output_path.relative_to(ROOT)),
                "requirement_count": len(requirements),
                "explicit_count": payload["statistics"]["explicit_count"],
                "inferred_count": payload["statistics"]["inferred_count"],
                "status": "ok",
            }
        )

    summary_payload = {
        "dataset": "pure",
        "seed": args.seed,
        "input_documents": len(documents),
        "generated_samples": sum(item["status"] == "ok" for item in summary),
        "skipped_samples": skipped,
        "total_ground_truth_requirements": total_requirements,
        "samples": summary,
    }
    args.summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
