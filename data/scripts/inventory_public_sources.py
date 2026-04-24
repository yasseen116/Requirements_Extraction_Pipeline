#!/usr/bin/env python3
"""Create a lightweight inventory of downloaded public source datasets."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = ROOT / "raw_sources" / "public"
OUTPUT = ROOT / "outputs" / "public_source_inventory.json"


def describe_file(path: Path) -> dict:
    entry = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
    }

    if path.suffix.lower() == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                entry["zip_members"] = archive.namelist()[:25]
                entry["zip_member_count"] = len(archive.namelist())
        except Exception as exc:  # pragma: no cover
            entry["zip_error"] = str(exc)

    return entry


def main() -> int:
    inventory = []
    for dataset_dir in sorted(path for path in PUBLIC_ROOT.iterdir() if path.is_dir()):
        files = [describe_file(path) for path in sorted(dataset_dir.rglob("*")) if path.is_file()]
        inventory.append(
            {
                "dataset_id": dataset_dir.name,
                "file_count": len(files),
                "files": files,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(inventory, indent=2))
    print(f"Wrote inventory to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
