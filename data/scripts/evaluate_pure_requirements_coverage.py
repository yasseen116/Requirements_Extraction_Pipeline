#!/usr/bin/env python3
"""Evaluate generated full requirements against PURE source requirements."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        # Load standard small semantic embedding model
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def calculate_similarity_matrix(texts_a: list[str], texts_b: list[str]) -> torch.Tensor:
    if not texts_a or not texts_b:
        return torch.zeros((len(texts_a), len(texts_b)))
    model = get_model()
    embeddings_a = model.encode(texts_a, convert_to_tensor=True)
    embeddings_b = model.encode(texts_b, convert_to_tensor=True)
    return util.cos_sim(embeddings_a, embeddings_b)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD_DIR = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_PRED_DIR = ROOT / "outputs" / "pure_full" / "generated_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "coverage_evaluation.json"

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "for",
    "with",
    "by",
    "is",
    "are",
    "be",
    "as",
    "that",
    "this",
    "it",
    "from",
    "at",
    "must",
    "shall",
    "should",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    tokens = normalize_text(text).split()
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def token_f1(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_predicted_texts(payload: dict) -> list[str]:
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return []
    texts = []
    for key in ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]:
        items = reqs.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            normalized = normalize_text(text)
            if normalized:
                texts.append(text)
    return texts


def load_gold(gold_dir: Path) -> dict[str, dict]:
    payload = {}
    for path in sorted(gold_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        sample = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in sample or "ground_truth_requirements" not in sample:
            continue
        payload[sample["sample_id"]] = sample
    return payload


def load_predictions(pred_dir: Path) -> dict[str, dict]:
    payload = {}
    for path in sorted(pred_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        sample = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in sample:
            continue
        payload[sample["sample_id"]] = sample
    return payload


def greedy_match(gold_texts: list[str], pred_texts: list[str], threshold: float) -> tuple[list[dict], set[int], set[int]]:
    candidates = []
    if gold_texts and pred_texts:
        sim_matrix = calculate_similarity_matrix(gold_texts, pred_texts).cpu().numpy()
        for g_idx in range(len(gold_texts)):
            for p_idx in range(len(pred_texts)):
                score = float(sim_matrix[g_idx][p_idx])
                if score >= threshold:
                    candidates.append((score, g_idx, p_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_gold = set()
    used_pred = set()
    matches = []
    for score, g_idx, p_idx in candidates:
        if g_idx in used_gold or p_idx in used_pred:
            continue
        used_gold.add(g_idx)
        used_pred.add(p_idx)
        matches.append(
            {
                "gold_index": g_idx,
                "pred_index": p_idx,
                "score": score,
            }
        )
    return matches, used_gold, used_pred


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_sample(gold_sample: dict, pred_sample: dict, threshold: float) -> dict:
    gold_texts = [item["text"] for item in gold_sample["ground_truth_requirements"]]
    pred_texts = extract_predicted_texts(pred_sample)
    gold_count = len(gold_texts)
    pred_count = len(pred_texts)

    matches, used_gold, used_pred = greedy_match(gold_texts, pred_texts, threshold)
    matched = len(matches)
    precision = matched / pred_count if pred_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    unmatched_gold = [gold_texts[index] for index in range(gold_count) if index not in used_gold]
    unmatched_pred = [pred_texts[index] for index in range(pred_count) if index not in used_pred]

    best_source_scores = []
    if gold_texts and pred_texts:
        sim_matrix = calculate_similarity_matrix(gold_texts, pred_texts).cpu().numpy()
    else:
        sim_matrix = None

    for g_idx in range(gold_count):
        if sim_matrix is None:
            best_source_scores.append(0.0)
        else:
            best_source_scores.append(float(max(sim_matrix[g_idx])))

    return {
        "sample_id": gold_sample["sample_id"],
        "document_id": gold_sample["source"]["document_id"],
        "source_requirement_count": gold_count,
        "generated_requirement_count": pred_count,
        "matched_count": matched,
        "precision": precision,
        "coverage_recall": recall,
        "f1": f1,
        "hallucination_rate": len(unmatched_pred) / pred_count if pred_count else 0.0,
        "average_match_score": average([item["score"] for item in matches]),
        "average_best_source_score": average(best_source_scores),
        "unmatched_source_examples": unmatched_gold[:15],
        "unmatched_generated_examples": unmatched_pred[:15],
        "match_examples": [
            {
                "score": item["score"],
                "source_requirement": gold_texts[item["gold_index"]],
                "generated_requirement": pred_texts[item["pred_index"]],
            }
            for item in matches[:15]
        ],
    }


def build_aggregate(per_sample: list[dict]) -> dict:
    total_source = sum(item["source_requirement_count"] for item in per_sample)
    total_pred = sum(item["generated_requirement_count"] for item in per_sample)
    total_matched = sum(item["matched_count"] for item in per_sample)
    micro_precision = total_matched / total_pred if total_pred else 0.0
    micro_recall = total_matched / total_source if total_source else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    return {
        "sample_count": len(per_sample),
        "macro_precision": average([item["precision"] for item in per_sample]),
        "macro_coverage_recall": average([item["coverage_recall"] for item in per_sample]),
        "macro_f1": average([item["f1"] for item in per_sample]),
        "macro_hallucination_rate": average([item["hallucination_rate"] for item in per_sample]),
        "macro_average_match_score": average([item["average_match_score"] for item in per_sample]),
        "macro_average_best_source_score": average([item["average_best_source_score"] for item in per_sample]),
        "micro_precision": micro_precision,
        "micro_coverage_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_source_requirements": total_source,
        "total_generated_requirements": total_pred,
        "total_matched_requirements": total_matched,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Matching threshold (alias; overrides --match-threshold when set).",
    )
    parser.add_argument("--match-threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold = args.threshold if args.threshold is not None else args.match_threshold
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")

    gold = load_gold(args.gold_dir)
    predictions = load_predictions(args.pred_dir)

    per_sample = []
    for sample_id, gold_sample in gold.items():
        if sample_id not in predictions:
            continue
        per_sample.append(evaluate_sample(gold_sample, predictions[sample_id], threshold))

    def safe_rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except Exception:  # noqa: BLE001
            return str(path)

    payload = {
        "gold_dir": safe_rel(args.gold_dir),
        "pred_dir": safe_rel(args.pred_dir),
        "match_threshold": threshold,
        "aggregate": build_aggregate(per_sample),
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
