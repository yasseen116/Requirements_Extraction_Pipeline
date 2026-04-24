#!/usr/bin/env python3
"""Evaluate generated requirement predictions against gold samples.

This evaluator is used for the small manual pilot set (source-grounded JSON samples).

For paper-quality reporting, evaluation should be consistent with the PURE benchmark
coverage evaluator. We therefore default to a greedy token-F1 matching scheme with a
threshold (default 0.55), rather than strict normalized exact-string matching.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "raw_sources" / "manual_gold"
DEFAULT_PRED = ROOT / "outputs" / "b0_template_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "b0_template_requirements" / "evaluation.json"

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


def greedy_match(
    gold_texts: list[str], pred_texts: list[str], threshold: float
) -> tuple[list[dict], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for g_idx, g_text in enumerate(gold_texts):
        for p_idx, p_text in enumerate(pred_texts):
            score = token_f1(g_text, p_text)
            if score >= threshold:
                candidates.append((score, g_idx, p_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict] = []
    for score, g_idx, p_idx in candidates:
        if g_idx in used_gold or p_idx in used_pred:
            continue
        used_gold.add(g_idx)
        used_pred.add(p_idx)
        matches.append({"gold_index": g_idx, "pred_index": p_idx, "score": score})
    return matches, used_gold, used_pred


def prf(pred_items: set[str], gold_items: set[str]) -> dict:
    tp = len(pred_items & gold_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_gold(gold_dir: Path) -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for path in sorted(gold_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        sample = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(sample, dict) or "sample_id" not in sample or "requirements" not in sample:
            continue
        gold[sample["sample_id"]] = sample
    return gold


def load_predictions(pred_dir: Path) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    for path in sorted(pred_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in prediction or "requirements" not in prediction:
            continue
        predictions[prediction["sample_id"]] = prediction
    return predictions


def extract_texts(requirements: list[dict]) -> list[str]:
    texts: list[str] = []
    for req in requirements:
        if not isinstance(req, dict):
            continue
        text = str(req.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def extract_source_slots(requirements: list[dict]) -> set[str]:
    slots: set[str] = set()
    for req in requirements:
        if isinstance(req, dict):
            slots.update(req.get("source_slots", []))
    return slots


def compliance_score(prediction: dict) -> float:
    requirements = prediction["requirements"]
    checks: list[bool] = []
    for req in requirements["functional"]:
        checks.append(bool(req.get("id") and req.get("text") and req.get("source_slots")))
    for req in requirements["non_functional"]:
        checks.append(
            bool(
                req.get("id")
                and req.get("text")
                and req.get("source_slots")
                and req.get("category") in {"performance", "security"}
            )
        )
    return sum(checks) / len(checks) if checks else 0.0


def evaluate_sample(gold: dict, prediction: dict, *, match_threshold: float, exact: bool) -> dict:
    gold_func = gold["requirements"]["functional"]
    gold_nfr = gold["requirements"]["non_functional"]
    pred_func = prediction["requirements"]["functional"]
    pred_nfr = prediction["requirements"]["non_functional"]

    gold_func_texts = extract_texts(gold_func)
    gold_nfr_texts = extract_texts(gold_nfr)
    pred_func_texts = extract_texts(pred_func)
    pred_nfr_texts = extract_texts(pred_nfr)

    if exact:
        gold_func_norm = {normalize_text(text) for text in gold_func_texts}
        gold_nfr_norm = {normalize_text(text) for text in gold_nfr_texts}
        pred_func_norm = {normalize_text(text) for text in pred_func_texts}
        pred_nfr_norm = {normalize_text(text) for text in pred_nfr_texts}

        func_scores = prf(pred_func_norm, gold_func_norm)
        nfr_scores = prf(pred_nfr_norm, gold_nfr_norm)
        overall_scores = prf(
            pred_func_norm | pred_nfr_norm, gold_func_norm | gold_nfr_norm
        )
        total_pred = len(pred_func_norm | pred_nfr_norm)
        match_examples: list[dict] = []
    else:
        func_matches, _, _ = greedy_match(gold_func_texts, pred_func_texts, match_threshold)
        nfr_matches, _, _ = greedy_match(gold_nfr_texts, pred_nfr_texts, match_threshold)

        func_matched = len(func_matches)
        nfr_matched = len(nfr_matches)

        func_precision = func_matched / len(pred_func_texts) if pred_func_texts else 0.0
        func_recall = func_matched / len(gold_func_texts) if gold_func_texts else 0.0
        func_f1 = (
            2 * func_precision * func_recall / (func_precision + func_recall)
            if (func_precision + func_recall)
            else 0.0
        )
        nfr_precision = nfr_matched / len(pred_nfr_texts) if pred_nfr_texts else 0.0
        nfr_recall = nfr_matched / len(gold_nfr_texts) if gold_nfr_texts else 0.0
        nfr_f1 = (
            2 * nfr_precision * nfr_recall / (nfr_precision + nfr_recall)
            if (nfr_precision + nfr_recall)
            else 0.0
        )

        func_scores = {
            "precision": func_precision,
            "recall": func_recall,
            "f1": func_f1,
            "tp": func_matched,
            "fp": len(pred_func_texts) - func_matched,
            "fn": len(gold_func_texts) - func_matched,
        }
        nfr_scores = {
            "precision": nfr_precision,
            "recall": nfr_recall,
            "f1": nfr_f1,
            "tp": nfr_matched,
            "fp": len(pred_nfr_texts) - nfr_matched,
            "fn": len(gold_nfr_texts) - nfr_matched,
        }

        overall_gold = gold_func_texts + gold_nfr_texts
        overall_pred = pred_func_texts + pred_nfr_texts
        overall_matches, _, _ = greedy_match(overall_gold, overall_pred, match_threshold)
        matched = len(overall_matches)
        total_pred = len(overall_pred)
        total_gold = len(overall_gold)

        precision = matched / total_pred if total_pred else 0.0
        recall = matched / total_gold if total_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        overall_scores = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": matched,
            "fp": total_pred - matched,
            "fn": total_gold - matched,
        }

        match_examples = [
            {
                "score": item["score"],
                "gold_requirement": overall_gold[item["gold_index"]],
                "pred_requirement": overall_pred[item["pred_index"]],
            }
            for item in overall_matches[:15]
        ]

    gold_slot_refs = extract_source_slots(gold_func + gold_nfr)
    pred_slot_refs = extract_source_slots(pred_func + pred_nfr)
    matched_slot_refs = pred_slot_refs & gold_slot_refs

    hallucination_count = overall_scores["fp"]

    return {
        "sample_id": gold["sample_id"],
        "functional": func_scores,
        "non_functional": nfr_scores,
        "overall": {
            **overall_scores,
            "coverage": overall_scores["recall"],
            "hallucination_rate": hallucination_count / total_pred if total_pred else 0.0,
        },
        "source_slot_coverage": len(matched_slot_refs) / len(gold_slot_refs) if gold_slot_refs else 0.0,
        "compliance": compliance_score(prediction),
        "match_threshold": None if exact else match_threshold,
        "match_examples": match_examples,
    }


def average(metrics: list[float]) -> float:
    return sum(metrics) / len(metrics) if metrics else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.55,
        help="Token-F1 greedy match threshold (ignored with --exact).",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use strict normalized exact-string matching (legacy behavior).",
    )
    args = parser.parse_args()

    gold = load_gold(args.gold_dir)
    predictions = load_predictions(args.pred_dir)

    per_sample = []
    for sample_id, gold_sample in gold.items():
        if sample_id not in predictions:
            raise SystemExit(f"Missing prediction for {sample_id}")
        per_sample.append(
            evaluate_sample(
                gold_sample,
                predictions[sample_id],
                match_threshold=args.match_threshold,
                exact=bool(args.exact),
            )
        )

    payload = {
        "sample_count": len(per_sample),
        "match_threshold": None if args.exact else args.match_threshold,
        "matching_method": "exact_normalized_string" if args.exact else "greedy_token_f1",
        "aggregate": {
            "functional_f1_macro": average([item["functional"]["f1"] for item in per_sample]),
            "non_functional_f1_macro": average([item["non_functional"]["f1"] for item in per_sample]),
            "overall_f1_macro": average([item["overall"]["f1"] for item in per_sample]),
            "coverage_macro": average([item["overall"]["coverage"] for item in per_sample]),
            "hallucination_rate_macro": average([item["overall"]["hallucination_rate"] for item in per_sample]),
            "source_slot_coverage_macro": average([item["source_slot_coverage"] for item in per_sample]),
            "compliance_macro": average([item["compliance"] for item in per_sample]),
        },
        "per_sample": per_sample,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

