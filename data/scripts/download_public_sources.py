#!/usr/bin/env python3
"""Download DOI-backed public requirements datasets with checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "raw_sources" / "public_manifest.json"
DEFAULT_DEST = ROOT / "raw_sources" / "public"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(destination)


def extract_zip(archive_path: Path, extract_root: Path) -> Path:
    target_dir = extract_root / archive_path.stem
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help="Restrict downloads to one or more dataset ids from the manifest.",
    )
    parser.add_argument(
        "--extract-zips",
        action="store_true",
        help="Extract any downloaded .zip archives into an extracted/ subdirectory.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selected_ids = set(args.dataset_ids or [])
    datasets = [
        dataset
        for dataset in manifest["datasets"]
        if not selected_ids or dataset["id"] in selected_ids
    ]

    if not datasets:
        print("No datasets matched the requested ids.", file=sys.stderr)
        return 1

    summary = []

    for dataset in datasets:
        dataset_dir = args.dest / dataset["id"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] {dataset['id']} -> {dataset_dir}")

        dataset_entry = {
            "id": dataset["id"],
            "title": dataset["title"],
            "doi": dataset["doi"],
            "record_url": dataset["record_url"],
            "files": [],
        }

        for file_info in dataset["files"]:
            path = dataset_dir / file_info["name"]
            if path.exists() and md5sum(path) == file_info["md5"]:
                print(f"  [skip] {file_info['name']} (checksum ok)")
            else:
                print(f"  [download] {file_info['name']}")
                download(file_info["download_url"], path)
                actual_md5 = md5sum(path)
                if actual_md5 != file_info["md5"]:
                    raise RuntimeError(
                        f"Checksum mismatch for {path.name}: {actual_md5} != {file_info['md5']}"
                    )

            extracted_to = None
            if args.extract_zips and path.suffix.lower() == ".zip":
                extract_root = dataset_dir / "extracted"
                extracted_to = str(extract_zip(path, extract_root).relative_to(ROOT))
                print(f"  [extract] {file_info['name']} -> {extracted_to}")

            dataset_entry["files"].append(
                {
                    "name": file_info["name"],
                    "path": str(path.relative_to(ROOT)),
                    "md5": file_info["md5"],
                    "size_bytes": path.stat().st_size,
                    "extracted_to": extracted_to,
                }
            )

        summary.append(dataset_entry)

    output_path = ROOT / "outputs" / "downloaded_public_sources.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote summary -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
