#!/usr/bin/env python3
"""Recommend source documents for manual gold annotation bootstrap."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "outputs" / "public_source_index.json"
OUTPUT_PATH = ROOT / "outputs" / "seed_candidate_shortlist.json"


def score_document(document: dict) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    if document["has_raw_text"]:
        score += 3
        reasons.append("has raw text")
    if document["has_functional_extract"]:
        score += 3
        reasons.append("has functional requirement extract")
    if document["has_all_relevant"]:
        score += 2
        reasons.append("has curated relevant text")
    if document["has_use_cases"]:
        score += 1
        reasons.append("has use cases")
    if document["has_raw_markdown"]:
        score += 1
        reasons.append("has markdown conversion")

    family = document["family_prefix"]
    if family == "P":
        score += 2
        reasons.append("PURE-family derivative")
    elif family == "C":
        score += 1
        reasons.append("custom public source")

    return score, reasons


def main() -> int:
    payload = json.loads(INDEX_PATH.read_text())
    documents = payload["datasets"]["software_requirements_dataset"]["documents"]

    shortlisted = []
    for document in documents:
        if not document["has_raw_text"]:
            continue
        if not (
            document["has_functional_extract"]
            or document["has_all_relevant"]
            or document["has_use_cases"]
        ):
            continue

        score, reasons = score_document(document)
        shortlisted.append(
            {
                "document_id": document["document_id"],
                "folder_name": document["folder_name"],
                "title": document["title"],
                "family_prefix": document["family_prefix"],
                "score": score,
                "selection_reasons": reasons,
                "paths": document["paths"],
            }
        )

    shortlisted.sort(
        key=lambda item: (
            -item["score"],
            item["family_prefix"],
            item["title"].lower(),
        )
    )

    payload = {
        "selection_policy": {
            "requires_raw_text": True,
            "requires_curated_requirement_subset": True,
            "scoring": "raw text + curated extracts + PURE-family preference",
        },
        "recommended_candidates": shortlisted[:20],
        "candidate_count": len(shortlisted),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote shortlist to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
