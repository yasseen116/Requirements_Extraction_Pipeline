#!/usr/bin/env python3
"""Run the PURE full benchmark across multiple seeds and summarize variance.

This is a lightweight experiment driver to support paper reporting.
It repeatedly calls `run_pure_full_benchmark.py` with different `--seed` values and
aggregates the resulting micro metrics (precision/recall/F1) for direct and gemini tracks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_pure_full_benchmark.py"
LATEST_POINTER = ROOT / "outputs" / "pure_full_latest_run.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


@dataclass(frozen=True)
class TrackStats:
    micro_precision: float
    micro_recall: float
    micro_f1: float


def extract_track(summary: dict | None) -> TrackStats | None:
    if summary is None:
        return None
    return TrackStats(
        micro_precision=float(summary["micro_precision"]),
        micro_recall=float(summary["micro_coverage_recall"]),
        micro_f1=float(summary["micro_f1"]),
    )


def run(seed: int, args: argparse.Namespace) -> dict:
    cmd = [
        "python3",
        str(RUNNER),
        "--seed",
        str(seed),
        "--max-samples",
        str(args.max_samples),
        "--min-requirements",
        str(args.min_requirements),
        "--max-source-requirements",
        str(args.max_source_requirements),
        "--match-threshold",
        str(args.match_threshold),
        "--max-reqs-per-answer",
        str(args.max_reqs_per_answer),
        "--max-chars-per-answer",
        str(args.max_chars_per_answer),
        "--dialogue-coverage-threshold",
        str(args.dialogue_coverage_threshold),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    latest = load_json(LATEST_POINTER)
    comparison = load_json(Path(latest["run_dir"]) / "comparison_summary.json")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--min-requirements", type=int, default=10)
    parser.add_argument("--max-source-requirements", type=int, default=200)
    parser.add_argument("--match-threshold", type=float, default=0.55)
    parser.add_argument("--max-reqs-per-answer", type=int, default=3)
    parser.add_argument("--max-chars-per-answer", type=int, default=650)
    parser.add_argument("--dialogue-coverage-threshold", type=float, default=0.55)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "pure_full_multi_seed_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    per_seed = []
    for seed in seeds:
        comparison = run(seed, args)
        per_seed.append(
            {
                "seed": seed,
                "run_id": comparison["run_id"],
                "run_dir": comparison["run_dir"],
                "direct": comparison.get("direct"),
                "gemini": comparison.get("gemini"),
                "dialogue_upper_bound": comparison.get("dialogue_upper_bound"),
            }
        )

    direct_stats = [extract_track(item["direct"]) for item in per_seed if item.get("direct")]
    direct_stats = [item for item in direct_stats if item is not None]
    gemini_stats = [extract_track(item["gemini"]) for item in per_seed if item.get("gemini")]
    gemini_stats = [item for item in gemini_stats if item is not None]

    payload = {
        "settings": {
            "seeds": seeds,
            "max_samples": args.max_samples,
            "min_requirements": args.min_requirements,
            "max_source_requirements": args.max_source_requirements,
            "match_threshold": args.match_threshold,
            "max_reqs_per_answer": args.max_reqs_per_answer,
            "max_chars_per_answer": args.max_chars_per_answer,
            "dialogue_coverage_threshold": args.dialogue_coverage_threshold,
        },
        "aggregate": {
            "direct": (
                {
                    "micro_precision_mean": mean([s.micro_precision for s in direct_stats]),
                    "micro_precision_std": std([s.micro_precision for s in direct_stats]),
                    "micro_recall_mean": mean([s.micro_recall for s in direct_stats]),
                    "micro_recall_std": std([s.micro_recall for s in direct_stats]),
                    "micro_f1_mean": mean([s.micro_f1 for s in direct_stats]),
                    "micro_f1_std": std([s.micro_f1 for s in direct_stats]),
                    "n": len(direct_stats),
                }
                if direct_stats
                else None
            ),
            "gemini": (
                {
                    "micro_precision_mean": mean([s.micro_precision for s in gemini_stats]),
                    "micro_precision_std": std([s.micro_precision for s in gemini_stats]),
                    "micro_recall_mean": mean([s.micro_recall for s in gemini_stats]),
                    "micro_recall_std": std([s.micro_recall for s in gemini_stats]),
                    "micro_f1_mean": mean([s.micro_f1 for s in gemini_stats]),
                    "micro_f1_std": std([s.micro_f1 for s in gemini_stats]),
                    "n": len(gemini_stats),
                }
                if gemini_stats
                else None
            ),
        },
        "per_seed": per_seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

