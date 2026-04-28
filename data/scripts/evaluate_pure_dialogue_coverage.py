#!/usr/bin/env python3
"""Measure how much of the PURE gold requirements are recoverable from the dialogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coverage_scorer import CoverageScorer, average
import evaluate_pure_requirements_coverage as ev


ROOT = Path(__file__).resolve().parent.parent
LATEST_POINTER = ROOT / "outputs" / "pure_full_latest_run.json"
DEFAULT_OUTPUT_NAME = "dialogue_coverage_user_only_fixed.json"
_SCORER = CoverageScorer()


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


def evaluate_dialogue_coverage(gold_sample: dict, dialogue_sample: dict, threshold: float, *, user_only: bool) -> dict:
    gold_texts = [item["text"] for item in gold_sample["ground_truth_requirements"]]
    support_units = _SCORER.build_dialogue_support_units(dialogue_sample, user_only=user_only)
    coverage = _SCORER.coverage_against_units(gold_texts, support_units, threshold=threshold, contextualized=True)

    uncovered_examples = []
    covered = 0
    best_scores = []
    for item in coverage:
        best_scores.append(float(item["best_score"]))
        if item["covered"]:
            covered += 1
            continue
        if len(uncovered_examples) < 15:
            uncovered_examples.append(item["query_text"])

    total = len(gold_texts)
    recall = covered / total if total else 0.0
    return {
        "sample_id": gold_sample["sample_id"],
        "document_id": gold_sample["source"]["document_id"],
        "source_requirement_count": total,
        "covered_count": covered,
        "coverage_recall": recall,
        "average_best_turn_score": average(best_scores),
        "uncovered_source_examples": uncovered_examples,
        "turn_count": len(dialogue_sample.get("dialogue", [])) if isinstance(dialogue_sample.get("dialogue", []), list) else 0,
        "support_unit_count": len(support_units),
    }


def load_latest_run_dir() -> Path | None:
    if not LATEST_POINTER.exists():
        return None
    try:
        payload = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    run_dir = payload.get("run_dir")
    if not run_dir:
        return None
    path = Path(run_dir)
    return path if path.exists() else None


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing PURE benchmark run directory. If omitted, uses outputs/pure_full_latest_run.json when needed.",
    )
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--dialogue-dir", type=Path, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.55)
    parser.add_argument("--user-only", action="store_true", help="only match against user turns (recommended)")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    if run_dir is None and (args.gold_dir is None or args.dialogue_dir is None or args.output is None):
        run_dir = load_latest_run_dir()

    gold_dir = args.gold_dir.resolve() if args.gold_dir is not None else None
    dialogue_dir = args.dialogue_dir.resolve() if args.dialogue_dir is not None else None
    output = args.output.resolve() if args.output is not None else None

    if gold_dir is None and run_dir is not None:
        gold_dir = run_dir / "source_requirements"
    if dialogue_dir is None and run_dir is not None:
        dialogue_dir = run_dir / "expanded_dialogues"
    if output is None and run_dir is not None:
        output_name = DEFAULT_OUTPUT_NAME if args.user_only else "dialogue_coverage_all_turns_fixed.json"
        output = run_dir / "metrics" / output_name

    if gold_dir is None or dialogue_dir is None or output is None:
        raise ValueError(
            "Unable to resolve inputs. Provide --gold-dir/--dialogue-dir/--output, "
            "or pass --run-dir, or populate outputs/pure_full_latest_run.json."
        )

    return gold_dir, dialogue_dir, output


def main() -> int:
    args = parse_args()
    gold_dir, dialogue_dir, output_path = resolve_inputs(args)
    gold = load_gold(gold_dir)
    dialogues = load_dialogues(dialogue_dir)

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
        "gold_dir": safe_rel(gold_dir),
        "dialogue_dir": safe_rel(dialogue_dir),
        "match_threshold": args.match_threshold,
        "user_only": bool(args.user_only),
        "support_unit": "user_sentence_or_clause_context",
        "similarity_method": _SCORER.similarity_method,
        "aggregate": {
            "sample_count": len(per_sample),
            "macro_coverage_recall": average(item["coverage_recall"] for item in per_sample),
            "micro_coverage_recall": micro_recall,
            "macro_average_best_turn_score": average(item["average_best_turn_score"] for item in per_sample),
            "total_source_requirements": total_source,
            "total_covered_requirements": total_covered,
            "total_support_units": sum(item.get("support_unit_count", 0) for item in per_sample),
        },
        "per_sample": per_sample,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
