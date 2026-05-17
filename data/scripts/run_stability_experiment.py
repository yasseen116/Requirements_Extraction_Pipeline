#!/usr/bin/env python3
"""Orchestrate stability + ablation experiment for the paper.

Steps:
  1. Backup Gemini reference 4-doc run
  2. Three Qwen stability runs (seeds 1-3, full_context, pipeline-only)
  3. One ablation run (no anchor preservation)
  4-minute total break budget (2 min between each of the 4 runs)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / "venv/bin/python3"
RUNNER = ROOT / "scripts/run_pure_full_benchmark.py"
REFERENCE_RUN_DIR = ROOT / "outputs/pure_full_runs/20260429T002305Z_pure_full"
GEMINI_RUN_DIR = ROOT / "outputs/pure_full_runs/20260429T001722Z_pure_full"
BACKUP_DIR = ROOT / "outputs/backups/gemini_reference_4doc"
RESULTS_FILE = ROOT / "outputs/paper_reports/stability_ablation_run_ids.json"
LATEST_POINTER = ROOT / "outputs/pure_full_latest_run.json"

BREAK_SECONDS = 120
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_benchmark(seed: int, *, disable_anchor: bool = False) -> str:
    cmd = [
        PYTHON, str(RUNNER),
        "--llm-provider", "ollama",
        "--pipeline-preset", "full_context",
        "--reuse-source-dir", str(REFERENCE_RUN_DIR / "source_requirements"),
        "--reuse-dialogue-dir", str(REFERENCE_RUN_DIR / "expanded_dialogues"),
        "--pipeline-only",
        "--skip-report-build",
        "--python-bin", PYTHON,
        "--seed", str(seed),
    ]
    if disable_anchor:
        cmd.append("--disable-anchor-preservation")
    label = f"seed={seed}" + (" no-anchor" if disable_anchor else "")
    print(f"\n[run] {label}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    run_id = json.loads(LATEST_POINTER.read_text())["run_id"]
    print(f"  completed: {run_id}", flush=True)
    return run_id


def main() -> int:
    # 1. Backup Gemini reference run
    print("[step 1] Backing up Gemini reference 4-doc run...", flush=True)
    if not GEMINI_RUN_DIR.exists():
        print(f"  WARNING: Gemini run dir not found: {GEMINI_RUN_DIR}", flush=True)
    else:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / GEMINI_RUN_DIR.name
        if dest.exists():
            print(f"  already exists: {dest}", flush=True)
        else:
            shutil.copytree(GEMINI_RUN_DIR, dest)
            print(f"  backed up to {dest}", flush=True)

    # 2. Three Qwen stability runs
    stability_run_ids: list[str] = []
    for i, seed in enumerate([1, 2, 3], start=1):
        if i > 1:
            print(f"\n[break] {BREAK_SECONDS}s before stability run {i}...", flush=True)
            time.sleep(BREAK_SECONDS)
        print(f"\n[step 2.{i}/3] Qwen stability run seed={seed}", flush=True)
        run_id = run_benchmark(seed)
        stability_run_ids.append(run_id)

    # 3. Ablation: no anchor preservation
    print(f"\n[break] {BREAK_SECONDS}s before ablation run...", flush=True)
    time.sleep(BREAK_SECONDS)
    print("\n[step 3] Ablation: no anchor preservation", flush=True)
    ablation_run_id = run_benchmark(1, disable_anchor=True)

    # 4. Save results manifest
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "reference_qwen_run_id": REFERENCE_RUN_DIR.name,
        "gemini_backup": str(BACKUP_DIR / GEMINI_RUN_DIR.name),
        "stability_run_ids": stability_run_ids,
        "ablation_run_id": ablation_run_id,
        "stability_run_dirs": [
            str(ROOT / "outputs/pure_full_runs" / rid) for rid in stability_run_ids
        ],
        "ablation_run_dir": str(ROOT / "outputs/pure_full_runs" / ablation_run_id),
    }
    RESULTS_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[done] Manifest saved: {RESULTS_FILE}", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
