#!/usr/bin/env python3
"""Analyze PURE coverage errors into reproducible heuristic categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_pure_requirements_coverage as ev


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD_DIR = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_PRED_DIR = ROOT / "outputs" / "pure_full" / "generated_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "error_analysis.json"


GENERIC_DOMAIN_TERMS = {
    "cart",
    "checkout",
    "browse",
    "catalog",
    "login",
    "sign in",
    "sign up",
    "payment",
    "invoice",
    "dashboard",
    "notification",
    "profile",
}

META_TERMS = {
    "source requirements",
    "not mentioned",
    "not specified",
    "not explicitly",
    "to be determined",
    "tbd",
    "unknown",
}


def best_score_against(text: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    sim_matrix = ev.calculate_similarity_matrix([text], candidates).cpu().numpy()
    return float(max(sim_matrix[0]))


def classify_unmatched_source(text: str, pred_texts: list[str], threshold: float) -> str:
    score = best_score_against(text, pred_texts)
    if score < 0.20:
        return "omission_full"
    if score < threshold:
        return "omission_partial"
    return "matched_above_threshold"


def classify_unmatched_prediction(text: str, gold_texts: list[str], threshold: float) -> str:
    lowered = ev.normalize_text(text)
    if any(marker in lowered for marker in META_TERMS):
        return "hallucination_meta"
    if any(term in lowered for term in GENERIC_DOMAIN_TERMS):
        return "hallucination_generic_domain"

    best = best_score_against(text, gold_texts)
    if best >= max(0.01, threshold * 0.70):
        return "near_match_paraphrase"
    return "hallucination_unsupported"


def empty_counts() -> dict[str, int]:
    return {
        "omission_full": 0,
        "omission_partial": 0,
        "hallucination_meta": 0,
        "hallucination_generic_domain": 0,
        "hallucination_unsupported": 0,
        "near_match_paraphrase": 0,
    }


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", "--match-threshold", dest="threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")

    gold = ev.load_gold(args.gold_dir)
    predictions = ev.load_predictions(args.pred_dir)

    per_sample = []
    aggregate_counts = empty_counts()

    for sample_id, gold_sample in gold.items():
        pred_sample = predictions.get(sample_id)
        if pred_sample is None:
            continue

        gold_texts = [item["text"] for item in gold_sample["ground_truth_requirements"]]
        pred_texts = ev.extract_predicted_texts(pred_sample)
        _, used_gold, used_pred = ev.greedy_match(gold_texts, pred_texts, args.threshold)

        unmatched_gold = [gold_texts[i] for i in range(len(gold_texts)) if i not in used_gold]
        unmatched_pred = [pred_texts[i] for i in range(len(pred_texts)) if i not in used_pred]

        sample_counts = empty_counts()

        omitted_examples = []
        for text in unmatched_gold:
            category = classify_unmatched_source(text, pred_texts, args.threshold)
            if category == "matched_above_threshold":
                continue
            sample_counts[category] += 1
            if len(omitted_examples) < 12:
                omitted_examples.append({"category": category, "text": text})

        hallucinated_examples = []
        for text in unmatched_pred:
            category = classify_unmatched_prediction(text, gold_texts, args.threshold)
            sample_counts[category] += 1
            if len(hallucinated_examples) < 12:
                hallucinated_examples.append({"category": category, "text": text})

        for key, value in sample_counts.items():
            aggregate_counts[key] += value

        per_sample.append(
            {
                "sample_id": sample_id,
                "document_id": gold_sample.get("source", {}).get("document_id"),
                "source_requirement_count": len(gold_texts),
                "generated_requirement_count": len(pred_texts),
                "unmatched_source_count": len(unmatched_gold),
                "unmatched_generated_count": len(unmatched_pred),
                "counts": sample_counts,
                "examples": {
                    "omitted": omitted_examples,
                    "hallucinated": hallucinated_examples,
                },
            }
        )

    total_unmatched_source = sum(item["unmatched_source_count"] for item in per_sample)
    total_unmatched_generated = sum(item["unmatched_generated_count"] for item in per_sample)

    payload = {
        "gold_dir": safe_rel(args.gold_dir),
        "pred_dir": safe_rel(args.pred_dir),
        "threshold": args.threshold,
        "aggregate": {
            "sample_count": len(per_sample),
            "total_unmatched_source": total_unmatched_source,
            "total_unmatched_generated": total_unmatched_generated,
            "counts": aggregate_counts,
            "rates": {
                "omission_full_rate": aggregate_counts["omission_full"] / total_unmatched_source
                if total_unmatched_source
                else 0.0,
                "omission_partial_rate": aggregate_counts["omission_partial"] / total_unmatched_source
                if total_unmatched_source
                else 0.0,
                "hallucination_meta_rate": aggregate_counts["hallucination_meta"] / total_unmatched_generated
                if total_unmatched_generated
                else 0.0,
                "hallucination_generic_domain_rate": aggregate_counts["hallucination_generic_domain"] / total_unmatched_generated
                if total_unmatched_generated
                else 0.0,
                "hallucination_unsupported_rate": aggregate_counts["hallucination_unsupported"] / total_unmatched_generated
                if total_unmatched_generated
                else 0.0,
                "near_match_paraphrase_rate": aggregate_counts["near_match_paraphrase"] / total_unmatched_generated
                if total_unmatched_generated
                else 0.0,
            },
        },
        "per_sample": per_sample,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
