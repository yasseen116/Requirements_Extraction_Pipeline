import json
from pathlib import Path

def compute_prf(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return precision, recall, f1

gold = {
    "core_features": ["browse products", "add to cart", "purchase products"]
}

pred = {
    "core_features": ["browse products", "purchase products"]
}

p, r, f1 = compute_prf(pred["core_features"], gold["core_features"])
print("Precision:", p)
print("Recall:", r)
print("F1:", f1)
