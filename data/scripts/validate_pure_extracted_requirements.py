#!/usr/bin/env python3
"""Semantic deduplication and grounding validation for generated requirements."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from coverage_scorer import CoverageScorer, clean_text


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_ORDER = ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]
_SCORER = CoverageScorer()


def dedup_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def flatten_requirement_items(payload: dict) -> list[dict]:
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return []
    flat = []
    for category in CATEGORY_ORDER:
        items = reqs.get(category, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("text", ""))
            if not text:
                continue
            flat.append(
                {
                    "category": category,
                    "id": clean_text(item.get("id", "")),
                    "text": text,
                    "priority": clean_text(item.get("priority", "medium")).lower() or "medium",
                    "evidence_turns": [int(turn) for turn in item.get("evidence_turns", []) if isinstance(turn, int) and turn >= 1],
                    "nfr_category": clean_text(item.get("category", "")).lower() if category == "non_functional" else None,
                }
            )
    return flat


def merge_semantic_duplicates(items: list[dict], *, threshold: float) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    removed = []
    for item in items:
        duplicate_index = None
        duplicate_score = 0.0
        for index, kept_item in enumerate(kept):
            score = _SCORER.similarity_row(item["text"], [kept_item["text"]])[0]
            if score >= threshold:
                duplicate_index = index
                duplicate_score = float(score)
                break
        if duplicate_index is None:
            kept.append(dict(item))
            continue
        target = kept[duplicate_index]
        target["evidence_turns"] = sorted(set(target["evidence_turns"]) | set(item["evidence_turns"]))
        if len(item["text"]) > len(target["text"]):
            target["text"] = item["text"]
            target["category"] = item["category"]
            target["id"] = item["id"] or target["id"]
            target["priority"] = item["priority"] or target["priority"]
            if item["nfr_category"]:
                target["nfr_category"] = item["nfr_category"]
        removed.append(
            {
                "removed": item["text"],
                "kept": target["text"],
                "similarity": round(duplicate_score, 3),
                "method": _SCORER.similarity_method,
            }
        )
    return kept, removed


def build_requirement_payload(template_payload: dict, items: list[dict]) -> dict:
    grouped = {category: [] for category in CATEGORY_ORDER}
    for item in items:
        payload_item = {
            "id": item["id"],
            "text": item["text"],
            "priority": item["priority"],
            "evidence_turns": item["evidence_turns"],
        }
        if item["category"] == "non_functional":
            payload_item["category"] = item.get("nfr_category") or "performance"
        grouped[item["category"]].append(payload_item)

    result = dict(template_payload)
    result["requirements"] = grouped
    diagnostics = dict(result.get("diagnostics", {})) if isinstance(result.get("diagnostics"), dict) else {}
    diagnostics["grounded_after_rewrite_count"] = len(items)
    result["diagnostics"] = diagnostics
    return result


def validate_items(dialogue_payload: dict, items: list[dict], *, threshold: float) -> dict:
    support_units = _SCORER.build_dialogue_support_units(dialogue_payload, user_only=True)
    if not support_units or not items:
        return {
            "total": len(items),
            "grounded": len(items),
            "hallucinated": 0,
            "hallucination_rate": 0.0,
            "similarity_method": _SCORER.similarity_method,
            "support_unit": "user_sentence_or_clause_context",
            "grounded_items": items,
            "hallucinated_items": [],
        }

    grounded = []
    hallucinated = []
    for item in items:
        candidate_turn_ids = set(item["evidence_turns"]) if item["evidence_turns"] else None
        best_score, best_unit = _SCORER.best_unit_for_query(
            item["text"],
            support_units,
            candidate_turn_ids=candidate_turn_ids,
            contextualized=True,
        )
        matched_claimed_turns = bool(best_unit) and (not candidate_turn_ids or best_unit.get("turn_id") in candidate_turn_ids)
        if best_score < threshold and candidate_turn_ids:
            fallback_score, fallback_unit = _SCORER.best_unit_for_query(
                item["text"],
                support_units,
                candidate_turn_ids=None,
                contextualized=True,
            )
            if fallback_score > best_score:
                best_score = fallback_score
                best_unit = fallback_unit
                matched_claimed_turns = False
        if best_score >= threshold and best_unit is not None:
            grounded.append(
                {
                    "item": item,
                    "best_supporting_turn_id": best_unit.get("turn_id"),
                    "best_supporting_span": best_unit.get("text"),
                    "best_supporting_context": best_unit.get("context_text"),
                    "best_supporting_sentence_index": best_unit.get("sentence_index"),
                    "matched_claimed_turns": matched_claimed_turns,
                    "similarity": round(best_score, 3),
                }
            )
        else:
            hallucinated.append(
                {
                    "item": item,
                    "best_score": round(best_score, 3),
                    "closest_turn_id": best_unit.get("turn_id") if best_unit else None,
                    "closest_span": best_unit.get("text") if best_unit else None,
                    "closest_context": best_unit.get("context_text") if best_unit else None,
                }
            )

    return {
        "total": len(items),
        "grounded": len(grounded),
        "hallucinated": len(hallucinated),
        "hallucination_rate": round(len(hallucinated) / len(items), 4) if items else 0.0,
        "similarity_method": _SCORER.similarity_method,
        "support_unit": "user_sentence_or_clause_context",
        "grounded_items": grounded,
        "hallucinated_items": hallucinated,
    }


def process_sample(pred_payload: dict, dialogue_payload: dict, *, dedup_threshold: float, grounding_threshold: float) -> dict:
    flat_items = flatten_requirement_items(pred_payload)
    deduped_items, removed_dups = merge_semantic_duplicates(flat_items, threshold=dedup_threshold)
    validation = validate_items(dialogue_payload, deduped_items, threshold=grounding_threshold)
    grounded_items = [entry["item"] for entry in validation["grounded_items"]]
    filtered_payload = build_requirement_payload(pred_payload, grounded_items)
    return {
        "sample_id": pred_payload.get("sample_id", ""),
        "filtered_payload": filtered_payload,
        "deduplication": {
            "input_count": len(flat_items),
            "after_dedup_count": len(deduped_items),
            "removed_count": len(removed_dups),
            "removed_details": removed_dups,
        },
        "validation": validation,
        "output_requirement_count": len(grounded_items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic dedup + grounding validation for generated requirements.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Generated requirements directory.")
    parser.add_argument("--dialogue-dir", type=Path, required=True, help="Expanded dialogues directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for filtered requirements.")
    parser.add_argument("--report-path", type=Path, default=None, help="Path to write validation report JSON.")
    parser.add_argument("--dedup-threshold", type=float, default=0.90, help="Similarity threshold for deduplication.")
    parser.add_argument("--grounding-threshold", type=float, default=0.25, help="Minimum similarity to count as grounded.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(args.pred_dir.glob("*.json"))
    all_results = []
    for pred_path in pred_paths:
        if pred_path.name in {"summary.json", "evaluation.json"} or pred_path.name.endswith(".raw_response.json"):
            continue
        pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
        if "sample_id" not in pred_payload or "requirements" not in pred_payload:
            continue

        sample_id = pred_payload["sample_id"]
        dialogue_path = args.dialogue_dir / f"{sample_id}.json"
        if dialogue_path.exists():
            dialogue_payload = json.loads(dialogue_path.read_text(encoding="utf-8"))
        else:
            print(f"[warn] No dialogue found for {sample_id}, validating against empty dialogue.")
            dialogue_payload = {"dialogue": []}

        result = process_sample(
            pred_payload,
            dialogue_payload,
            dedup_threshold=args.dedup_threshold,
            grounding_threshold=args.grounding_threshold,
        )

        out_path = args.output_dir / f"{sample_id}.json"
        out_path.write_text(json.dumps(result["filtered_payload"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        all_results.append({key: value for key, value in result.items() if key != "filtered_payload"})
        print(
            f"[{sample_id}] {result['deduplication']['input_count']} reqs -> "
            f"{result['deduplication']['after_dedup_count']} after dedup -> "
            f"{result['output_requirement_count']} grounded (method: {_SCORER.similarity_method})"
        )

    report_path = args.report_path or (args.output_dir / "validation_report.json")
    aggregate = {
        "similarity_method": _SCORER.similarity_method,
        "support_unit": "user_sentence_or_clause_context",
        "samples": len(all_results),
        "total_input": sum(item["deduplication"]["input_count"] for item in all_results),
        "total_after_dedup": sum(item["deduplication"]["after_dedup_count"] for item in all_results),
        "total_grounded": sum(item["output_requirement_count"] for item in all_results),
        "total_hallucinated": sum(item["validation"]["hallucinated"] for item in all_results),
        "mean_hallucination_rate": (
            round(sum(item["validation"]["hallucination_rate"] for item in all_results) / len(all_results), 4)
            if all_results
            else 0.0
        ),
    }
    report = {"aggregate": aggregate, "per_sample": all_results}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
