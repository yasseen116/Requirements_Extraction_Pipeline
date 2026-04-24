#!/usr/bin/env python3
"""Measure how much of the PURE gold requirements are recoverable from the dialogue.

This is an *upper bound* diagnostic metric:
- If a gold requirement isn't mentioned anywhere in the elicitation dialogue, no dialogue->requirements
  system can recover it.

We compute coverage by checking whether each gold requirement has at least one dialogue turn that
token-matches above a threshold (same token-F1 as the main coverage evaluator).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_pure_requirements_coverage as ev


ROOT = Path(__file__).resolve().parent.parent


def load_gold(gold_dir: Path) -> dict[str, dict]:
    return ev.load_gold(gold_dir)


def load_dialogues(dialogue_dir: Path) -> dict[str, dict]:
    payload = {}
    for path in sorted(dialogue_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        sample = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in sample or "dialogue" not in sample:
            continue
        payload[sample["sample_id"]] = sample
    return payload


def extract_turn_texts(dialogue_sample: dict, *, user_only: bool) -> list[str]:
    turns = dialogue_sample.get("dialogue", [])
    if not isinstance(turns, list):
        return []
    texts = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if user_only and turn.get("role") != "user":
            continue
        text = str(turn.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def evaluate_dialogue_coverage(gold_sample: dict, dialogue_sample: dict, threshold: float, *, user_only: bool) -> dict:
    gold_texts = [item["text"] for item in gold_sample["ground_truth_requirements"]]
    turn_texts = extract_turn_texts(dialogue_sample, user_only=user_only)

    covered = 0
    best_scores = []
    uncovered_examples = []
    for text in gold_texts:
        if not turn_texts:
            best = 0.0
        else:
            best = max(ev.token_f1(text, candidate) for candidate in turn_texts)
        best_scores.append(best)
        if best >= threshold:
            covered += 1
        else:
            if len(uncovered_examples) < 15:
                uncovered_examples.append(text)

    total = len(gold_texts)
    recall = covered / total if total else 0.0
    return {
        "sample_id": gold_sample["sample_id"],
        "document_id": gold_sample["source"]["document_id"],
        "source_requirement_count": total,
        "covered_count": covered,
        "coverage_recall": recall,
        "average_best_turn_score": ev.average(best_scores),
        "uncovered_source_examples": uncovered_examples,
        "turn_count": len(dialogue_sample.get("dialogue", [])) if isinstance(dialogue_sample.get("dialogue", []), list) else 0,
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--dialogue-dir", type=Path, required=True)
    parser.add_argument("--match-threshold", type=float, default=0.55)
    parser.add_argument("--user-only", action="store_true", help="only match against user turns (recommended)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold = load_gold(args.gold_dir)
    dialogues = load_dialogues(args.dialogue_dir)

    per_sample = []
    for sample_id, gold_sample in gold.items():
        if sample_id not in dialogues:
            continue
        per_sample.append(
            evaluate_dialogue_coverage(gold_sample, dialogues[sample_id], args.match_threshold, user_only=args.user_only)
        )

    total_source = sum(item["source_requirement_count"] for item in per_sample)
    total_covered = sum(item["covered_count"] for item in per_sample)
    micro_recall = total_covered / total_source if total_source else 0.0

    payload = {
        "gold_dir": safe_rel(args.gold_dir),
        "dialogue_dir": safe_rel(args.dialogue_dir),
        "match_threshold": args.match_threshold,
        "user_only": bool(args.user_only),
        "aggregate": {
            "sample_count": len(per_sample),
            "macro_coverage_recall": average([item["coverage_recall"] for item in per_sample]),
            "micro_coverage_recall": micro_recall,
            "macro_average_best_turn_score": average([item["average_best_turn_score"] for item in per_sample]),
            "total_source_requirements": total_source,
            "total_covered_requirements": total_covered,
        },
        "per_sample": per_sample,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

