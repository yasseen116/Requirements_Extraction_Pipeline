#!/usr/bin/env python3
"""Build a document-level index of downloaded public requirements assets."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = ROOT / "raw_sources" / "public"
OUTPUT = ROOT / "outputs" / "public_source_index.json"


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def index_pure() -> tuple[list[dict], dict]:
    xml_root = PUBLIC_ROOT / "pure" / "extracted" / "requirements-xml" / "XMLZIPFile"
    req_root = PUBLIC_ROOT / "pure" / "extracted" / "requirements" / "req"

    xml_map = {path.stem.lower(): path for path in xml_root.glob("*.xml")}
    req_map = {path.stem.lower(): path for path in req_root.iterdir() if path.is_file()}
    keys = sorted(set(xml_map) | set(req_map))

    documents = []
    for key in keys:
        xml_path = xml_map.get(key)
        source_path = req_map.get(key)
        documents.append(
            {
                "dataset_id": "pure",
                "document_id": slugify(key),
                "title": key,
                "has_xml": xml_path is not None,
                "has_source_file": source_path is not None,
                "xml_path": relative(xml_path) if xml_path else None,
                "source_path": relative(source_path) if source_path else None,
                "source_extension": source_path.suffix.lower() if source_path else None,
            }
        )

    summary = {
        "document_count": len(documents),
        "xml_count": sum(doc["has_xml"] for doc in documents),
        "source_file_count": sum(doc["has_source_file"] for doc in documents),
        "source_extensions": Counter(
            doc["source_extension"] for doc in documents if doc["source_extension"]
        ),
    }
    summary["source_extensions"] = dict(summary["source_extensions"])
    return documents, summary


def index_supporting_dataset() -> tuple[list[dict], dict]:
    root = PUBLIC_ROOT / "software_requirements_dataset" / "extracted" / "Dataset"
    documents = []
    family_counts = Counter()

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        prefix = folder.name.split(".", 1)[0]
        family = prefix[:1]
        family_counts[family] += 1
        files = sorted(path for path in folder.iterdir() if path.is_file())
        documents.append(
            {
                "dataset_id": "software_requirements_dataset",
                "document_id": slugify(folder.name),
                "folder_name": folder.name,
                "family_prefix": family,
                "title": folder.name.split(". ", 1)[-1] if ". " in folder.name else folder.name,
                "paths": [relative(path) for path in files],
                "has_raw_text": any(path.name == "_Raw.txt" for path in files),
                "has_raw_markdown": any(path.name == "_Raw.md" for path in files),
                "has_functional_extract": any(
                    path.name.startswith("FunctionalRequirements") for path in files
                ),
                "has_all_relevant": any(path.name.startswith("AllRelevant") for path in files),
                "has_use_cases": any(path.name.startswith("UseCases") for path in files),
                "source_extensions": sorted({path.suffix.lower() for path in files if path.suffix}),
            }
        )

    summary = {
        "document_count": len(documents),
        "family_counts": dict(family_counts),
        "with_raw_text": sum(doc["has_raw_text"] for doc in documents),
        "with_functional_extract": sum(doc["has_functional_extract"] for doc in documents),
        "with_all_relevant": sum(doc["has_all_relevant"] for doc in documents),
    }
    return documents, summary


def index_single_file_dataset(dataset_id: str) -> tuple[list[dict], dict]:
    root = PUBLIC_ROOT / dataset_id
    files = sorted(path for path in root.iterdir() if path.is_file())
    entries = [
        {
            "dataset_id": dataset_id,
            "document_id": slugify(path.name),
            "title": path.name,
            "path": relative(path),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return entries, {"file_count": len(entries)}


def main() -> int:
    pure_docs, pure_summary = index_pure()
    srd_docs, srd_summary = index_supporting_dataset()
    promise_docs, promise_summary = index_single_file_dataset("promise_plus")
    nice_docs, nice_summary = index_single_file_dataset("nice")

    payload = {
        "datasets": {
            "pure": {
                "summary": pure_summary,
                "documents": pure_docs,
            },
            "software_requirements_dataset": {
                "summary": srd_summary,
                "documents": srd_docs,
            },
            "promise_plus": {
                "summary": promise_summary,
                "documents": promise_docs,
            },
            "nice": {
                "summary": nice_summary,
                "documents": nice_docs,
            },
        }
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote source index to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
