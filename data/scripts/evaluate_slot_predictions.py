#!/usr/bin/env python3
"""Evaluate predicted slot structures against gold samples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "raw_sources" / "manual_gold"
DEFAULT_PRED = ROOT / "outputs" / "b1_rule_based_slots"
DEFAULT_OUTPUT = ROOT / "outputs" / "b1_rule_based_slots" / "evaluation.json"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


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
    gold = {}
    for path in sorted(gold_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        sample = json.loads(path.read_text())
        if not isinstance(sample, dict) or "sample_id" not in sample or "slots" not in sample:
            continue
        gold[sample["sample_id"]] = sample
    return gold


def load_predictions(pred_dir: Path) -> dict[str, dict]:
    predictions = {}
    for path in sorted(pred_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        sample = json.loads(path.read_text())
        if "sample_id" not in sample or "slots" not in sample:
            continue
        predictions[sample["sample_id"]] = sample
    return predictions


def flatten_slots(slots: dict) -> dict[str, set[str]]:
    flattened = {
        "system_type": {normalize_text(slots["system_type"]["value"])},
        "user_roles": {normalize_text(item["value"]) for item in slots["user_roles"]},
        "functional_capabilities": {
            f"{normalize_text(item['actor'])}::{normalize_text(item['action'])}"
            for item in slots["functional_capabilities"]
        },
        "authentication_required": {
            f"required::{str(slots['authentication']['required']).lower()}"
        },
        "authentication_methods": {
            normalize_text(item) for item in slots["authentication"]["methods"]
        },
        "performance_constraints": {
            normalize_text(item["text"]) for item in slots["performance_constraints"]
        },
        "security_constraints": {
            normalize_text(item["text"]) for item in slots["security_constraints"]
        },
    }
    return flattened


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_sample(gold: dict, prediction: dict) -> dict:
    gold_flat = flatten_slots(gold["slots"])
    pred_flat = flatten_slots(prediction["slots"])

    slot_scores = {}
    all_gold = set()
    all_pred = set()
    for slot_name in gold_flat:
        score = prf(pred_flat[slot_name], gold_flat[slot_name])
        slot_scores[slot_name] = score
        all_gold.update({f"{slot_name}::{item}" for item in gold_flat[slot_name]})
        all_pred.update({f"{slot_name}::{item}" for item in pred_flat[slot_name]})

    overall = prf(all_pred, all_gold)
    return {
        "sample_id": gold["sample_id"],
        "slots": slot_scores,
        "overall": overall,
    }


def aggregate(per_sample: list[dict]) -> dict:
    slot_names = list(per_sample[0]["slots"].keys()) if per_sample else []
    aggregate_slots = {}
    for slot_name in slot_names:
        aggregate_slots[slot_name] = {
            "f1_macro": average([item["slots"][slot_name]["f1"] for item in per_sample]),
            "precision_macro": average(
                [item["slots"][slot_name]["precision"] for item in per_sample]
            ),
            "recall_macro": average([item["slots"][slot_name]["recall"] for item in per_sample]),
        }

    return {
        "overall_f1_macro": average([item["overall"]["f1"] for item in per_sample]),
        "overall_precision_macro": average(
            [item["overall"]["precision"] for item in per_sample]
        ),
        "overall_recall_macro": average([item["overall"]["recall"] for item in per_sample]),
        "per_slot": aggregate_slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    gold = load_gold(args.gold_dir)
    predictions = load_predictions(args.pred_dir)
    per_sample = []
    split_counts = Counter()

    for sample_id, gold_sample in gold.items():
        if sample_id not in predictions:
            raise SystemExit(f"Missing slot prediction for {sample_id}")
        per_sample.append(evaluate_sample(gold_sample, predictions[sample_id]))
        split_counts.update([gold_sample["metadata"]["split"]])

    payload = {
        "sample_count": len(per_sample),
        "split_counts": dict(split_counts),
        "aggregate": aggregate(per_sample),
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
