#!/usr/bin/env python3
"""Populate or refresh frame annotations from slot structures in sample files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frame_slot_utils import derive_frames_from_slots


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Files or directories containing sample json files.")
    args = parser.parse_args()

    updated = []
    for raw_path in args.paths:
        path = Path(raw_path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file_path in files:
            if file_path.name in {"summary.json", "evaluation.json"} or file_path.name.startswith("template_"):
                continue
            payload = load_json(file_path)
            if not isinstance(payload, dict) or "slots" not in payload:
                continue
            payload["frames"] = derive_frames_from_slots(payload["slots"])
            file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            updated.append(str(file_path))

    print(json.dumps({"updated": updated}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
