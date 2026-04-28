#!/usr/bin/env python3
"""Second-layer LLM judge for PURE requirement coverage.

This script does not replace the semantic sentence-transformer coverage metric.
It adds a standardized Gemini-based verification layer that judges shortlisted
gold/prediction pairs with a checklist-style rubric.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from coverage_scorer import CoverageScorer, average, clean_text, token_f1
from gemini_native_client import GeminiConfig, GeminiNativeClient


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD_DIR = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_PRED_DIR = ROOT / "outputs" / "pure_full" / "generated_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "coverage_evaluation_llm.json"
DEFAULT_VALIDATOR_MODEL = "gemini-2.5-flash"
DEFAULT_SEMANTIC_TOP_K = 2
DEFAULT_LEXICAL_TOP_K = 1
DEFAULT_BATCH_SIZE = 24
STRICT_SCORE_BY_VERDICT = {"full": 1.0, "partial": 0.0, "none": 0.0}
WEIGHTED_SCORE_BY_VERDICT = {"full": 1.0, "partial": 0.5, "none": 0.0}
VALIDATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evaluations"],
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "pair_id",
                    "verdict",
                    "same_core_need",
                    "preserves_critical_details",
                    "no_material_additions",
                    "brief_reason",
                ],
                "properties": {
                    "pair_id": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": ["full", "partial", "none"]},
                    "same_core_need": {"type": "boolean"},
                    "preserves_critical_details": {"type": "boolean"},
                    "no_material_additions": {"type": "boolean"},
                    "brief_reason": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}
_SCORER = CoverageScorer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--semantic-top-k", type=int, default=DEFAULT_SEMANTIC_TOP_K)
    parser.add_argument("--lexical-top-k", type=int, default=DEFAULT_LEXICAL_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def validator_config_from_env() -> GeminiConfig:
    api_key = (
        os.environ.get("REQ_VALIDATOR_GEMINI_API_KEY", "").strip()
        or os.environ.get("REQ_GEMINI_API_KEY", "").strip()
    )
    model = (
        os.environ.get("REQ_VALIDATOR_GEMINI_MODEL", "").strip()
        or os.environ.get("REQ_GEMINI_VALIDATION_MODEL", "").strip()
        or DEFAULT_VALIDATOR_MODEL
    )
    base_url = (
        os.environ.get("REQ_VALIDATOR_GEMINI_BASE_URL", "").strip()
        or os.environ.get("REQ_GEMINI_BASE_URL", "").strip()
        or "https://generativelanguage.googleapis.com/v1beta"
    )
    timeout_seconds = int(
        os.environ.get("REQ_VALIDATOR_GEMINI_TIMEOUT_SECONDS", "").strip()
        or os.environ.get("REQ_GEMINI_TIMEOUT_SECONDS", "").strip()
        or "90"
    )
    max_retries = int(
        os.environ.get("REQ_VALIDATOR_GEMINI_MAX_RETRIES", "").strip()
        or os.environ.get("REQ_GEMINI_MAX_RETRIES", "").strip()
        or "6"
    )
    retry_backoff_seconds = float(
        os.environ.get("REQ_VALIDATOR_GEMINI_RETRY_BACKOFF_SECONDS", "").strip()
        or os.environ.get("REQ_GEMINI_RETRY_BACKOFF_SECONDS", "").strip()
        or "10.0"
    )
    cache_ttl_seconds = int(
        os.environ.get("REQ_VALIDATOR_GEMINI_CACHE_TTL_SECONDS", "").strip()
        or os.environ.get("REQ_GEMINI_CACHE_TTL_SECONDS", "").strip()
        or "3600"
    )
    if not api_key:
        raise ValueError("Missing REQ_VALIDATOR_GEMINI_API_KEY or REQ_GEMINI_API_KEY")
    return GeminiConfig(
        api_key=api_key,
        model=model,
        base_url=base_url.rstrip("/"),
        timeout_seconds=timeout_seconds,
        temperature=0.0,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        cache_ttl_seconds=cache_ttl_seconds,
    )


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


def extract_gold_items(payload: dict) -> list[dict]:
    items = []
    for index, item in enumerate(payload.get("ground_truth_requirements", []), start=1):
        if not isinstance(item, dict):
            continue
        text = clean_text(item.get("text", ""))
        if not text:
            continue
        items.append(
            {
                "id": clean_text(item.get("req_id", "")) or f"SRC-{index:03d}",
                "text": text,
                "category": clean_text(item.get("category", "")) or "source",
            }
        )
    return items


def extract_predicted_items(payload: dict) -> list[dict]:
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return []
    items = []
    for category in ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]:
        category_items = reqs.get(category, [])
        if not isinstance(category_items, list):
            continue
        for index, item in enumerate(category_items, start=1):
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("text", ""))
            if not text:
                continue
            items.append(
                {
                    "id": clean_text(item.get("id", "")) or f"{category.upper()}-{index:03d}",
                    "text": text,
                    "category": category,
                }
            )
    return items


def rank_top_indices(values: list[float], top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    pairs = [(index, float(value)) for index, value in enumerate(values)]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _value in pairs[:top_k]]


def build_candidate_pairs(
    gold_items: list[dict],
    pred_items: list[dict],
    *,
    semantic_top_k: int,
    lexical_top_k: int,
) -> list[dict]:
    if not gold_items or not pred_items:
        return []

    gold_texts = [item["text"] for item in gold_items]
    pred_texts = [item["text"] for item in pred_items]
    semantic_matrix = _SCORER.similarity_matrix(gold_texts, pred_texts)
    lexical_matrix = [[token_f1(gold_text, pred_text) for pred_text in pred_texts] for gold_text in gold_texts]
    pair_map: dict[tuple[int, int], dict] = {}

    def register_pair(gold_index: int, pred_index: int) -> None:
        pair_map[(gold_index, pred_index)] = {
            "pair_id": f"g{gold_index}_p{pred_index}",
            "gold_index": gold_index,
            "pred_index": pred_index,
            "gold": gold_items[gold_index],
            "pred": pred_items[pred_index],
            "semantic_score": float(semantic_matrix[gold_index][pred_index]),
            "lexical_score": float(lexical_matrix[gold_index][pred_index]),
        }

    for gold_index, row in enumerate(semantic_matrix):
        for pred_index in rank_top_indices(row, semantic_top_k):
            register_pair(gold_index, pred_index)
        for pred_index in rank_top_indices(lexical_matrix[gold_index], lexical_top_k):
            register_pair(gold_index, pred_index)

    for pred_index in range(len(pred_items)):
        semantic_column = [float(semantic_matrix[gold_index][pred_index]) for gold_index in range(len(gold_items))]
        lexical_column = [float(lexical_matrix[gold_index][pred_index]) for gold_index in range(len(gold_items))]
        for gold_index in rank_top_indices(semantic_column, semantic_top_k):
            register_pair(gold_index, pred_index)
        for gold_index in rank_top_indices(lexical_column, lexical_top_k):
            register_pair(gold_index, pred_index)

    pairs = list(pair_map.values())
    pairs.sort(key=lambda item: (item["gold_index"], -item["semantic_score"], -item["lexical_score"], item["pred_index"]))
    return pairs


def build_batch_prompt(pairs: list[dict]) -> str:
    lines = [build_prompt_prefix(), ""]
    for item in pairs:
        lines.extend(
            [
                f"PAIR_ID: {item['pair_id']}",
                f"SOURCE_ID: {item['gold']['id']}",
                f"SOURCE_CATEGORY: {item['gold']['category']}",
                f"SOURCE_TEXT: {item['gold']['text']}",
                f"GENERATED_ID: {item['pred']['id']}",
                f"GENERATED_CATEGORY: {item['pred']['category']}",
                f"GENERATED_TEXT: {item['pred']['text']}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_prompt_prefix() -> str:
    lines = [
        "You are validating requirement recovery against a trusted source requirement.",
        "Judge each source/generated pair independently.",
        "Use this rubric:",
        "- verdict=full: the generated requirement faithfully covers the same requirement, preserving critical details such as roles, numbers, negation, persistence, conditions, interfaces, compliance constraints, and modality strength.",
        "- verdict=partial: the generated requirement is about the same core need but loses, weakens, generalizes, or alters at least one material detail, or adds a material extra condition.",
        "- verdict=none: the generated requirement does not cover the source requirement.",
        "Important: fluency is irrelevant. Prefer faithfulness to the source requirement over stylistic similarity.",
        "Return every pair_id exactly once.",
    ]
    return "\n".join(lines).strip()


def batched(items: list[dict], size: int) -> list[list[dict]]:
    chunk_size = max(1, int(size))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def judge_pairs(client: GeminiNativeClient, pairs: list[dict], *, batch_size: int) -> dict[str, dict]:
    if not pairs:
        return {}

    verdicts: dict[str, dict] = {}
    prompt_prefix = build_prompt_prefix()
    cache_prefix = prompt_prefix if len(prompt_prefix) >= 16000 else None

    batches = batched(pairs, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        print(f"[llm-judge] Evaluating batch {batch_index}/{len(batches)} with {len(batch)} candidate pairs...")
        response = client.generate_json(
            build_batch_prompt(batch),
            VALIDATOR_SCHEMA,
            temperature=0.0,
            cache_prefix=cache_prefix,
            cache_namespace="req-coverage-judge",
        )
        parsed = json.loads(response["text"])
        evaluations = parsed.get("evaluations", [])
        if len(evaluations) != len(batch):
            raise ValueError(
                f"Validator returned {len(evaluations)} evaluations for {len(batch)} candidate pairs"
            )
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                raise ValueError("Validator returned a non-object evaluation entry")
            pair_id = clean_text(evaluation.get("pair_id", ""))
            if not pair_id:
                raise ValueError("Validator returned an evaluation without pair_id")
            verdicts[pair_id] = evaluation
    return verdicts


def select_matches(candidate_pairs: list[dict], verdicts: dict[str, dict], score_by_verdict: dict[str, float]) -> tuple[list[dict], set[int], set[int], float]:
    ranked = []
    for pair in candidate_pairs:
        verdict = verdicts.get(pair["pair_id"], {}).get("verdict", "none")
        score = float(score_by_verdict.get(str(verdict), 0.0))
        if score <= 0.0:
            continue
        ranked.append((score, float(pair["semantic_score"]), float(pair["lexical_score"]), pair))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    score_total = 0.0
    for score, _semantic, _lexical, pair in ranked:
        gold_index = pair["gold_index"]
        pred_index = pair["pred_index"]
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        verdict = str(verdicts[pair["pair_id"]]["verdict"])
        score_total += score
        matches.append(
            {
                **pair,
                "verdict": verdict,
                "score": score,
                "brief_reason": verdicts[pair["pair_id"]].get("brief_reason", ""),
                "same_core_need": verdicts[pair["pair_id"]].get("same_core_need"),
                "preserves_critical_details": verdicts[pair["pair_id"]].get("preserves_critical_details"),
                "no_material_additions": verdicts[pair["pair_id"]].get("no_material_additions"),
            }
        )
    return matches, used_gold, used_pred, score_total


def evaluate_sample(gold_sample: dict, pred_sample: dict, client: GeminiNativeClient, args: argparse.Namespace) -> dict:
    gold_items = extract_gold_items(gold_sample)
    pred_items = extract_predicted_items(pred_sample)
    candidate_pairs = build_candidate_pairs(
        gold_items,
        pred_items,
        semantic_top_k=args.semantic_top_k,
        lexical_top_k=args.lexical_top_k,
    )
    verdicts = judge_pairs(client, candidate_pairs, batch_size=args.batch_size)

    strict_matches, strict_used_gold, strict_used_pred, strict_score_total = select_matches(
        candidate_pairs,
        verdicts,
        STRICT_SCORE_BY_VERDICT,
    )
    weighted_matches, weighted_used_gold, weighted_used_pred, weighted_score_total = select_matches(
        candidate_pairs,
        verdicts,
        WEIGHTED_SCORE_BY_VERDICT,
    )

    gold_count = len(gold_items)
    pred_count = len(pred_items)
    strict_precision = strict_score_total / pred_count if pred_count else 0.0
    strict_recall = strict_score_total / gold_count if gold_count else 0.0
    strict_f1 = (
        2 * strict_precision * strict_recall / (strict_precision + strict_recall)
        if (strict_precision + strict_recall)
        else 0.0
    )
    weighted_precision = weighted_score_total / pred_count if pred_count else 0.0
    weighted_recall = weighted_score_total / gold_count if gold_count else 0.0
    weighted_f1 = (
        2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        if (weighted_precision + weighted_recall)
        else 0.0
    )
    full_count = sum(1 for item in weighted_matches if item["verdict"] == "full")
    partial_count = sum(1 for item in weighted_matches if item["verdict"] == "partial")

    unmatched_gold = [gold_items[index]["text"] for index in range(gold_count) if index not in weighted_used_gold]
    unmatched_pred = [pred_items[index]["text"] for index in range(pred_count) if index not in weighted_used_pred]
    selected_pairs = {item["pair_id"] for item in weighted_matches}
    partial_examples = [
        {
            "source_requirement": pair["gold"]["text"],
            "generated_requirement": pair["pred"]["text"],
            "reason": verdicts[pair["pair_id"]].get("brief_reason", ""),
        }
        for pair in candidate_pairs
        if verdicts.get(pair["pair_id"], {}).get("verdict") == "partial" and pair["pair_id"] not in selected_pairs
    ][:10]

    return {
        "sample_id": gold_sample["sample_id"],
        "document_id": gold_sample["source"]["document_id"],
        "source_requirement_count": gold_count,
        "generated_requirement_count": pred_count,
        "strict_match_count": len(strict_matches),
        "weighted_match_score": weighted_score_total,
        "full_match_count": full_count,
        "partial_match_count": partial_count,
        "precision": strict_precision,
        "coverage_recall": strict_recall,
        "f1": strict_f1,
        "hallucination_rate": (pred_count - len(strict_used_pred)) / pred_count if pred_count else 0.0,
        "weighted_precision": weighted_precision,
        "weighted_coverage_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "candidate_pair_count": len(candidate_pairs),
        "validator_match_examples": [
            {
                "verdict": item["verdict"],
                "score": item["score"],
                "source_requirement": item["gold"]["text"],
                "generated_requirement": item["pred"]["text"],
                "reason": item["brief_reason"],
            }
            for item in weighted_matches[:15]
        ],
        "unmatched_source_examples": unmatched_gold[:15],
        "unmatched_generated_examples": unmatched_pred[:15],
        "partial_only_examples": partial_examples,
    }


def build_aggregate(per_sample: list[dict]) -> dict:
    total_source = sum(item["source_requirement_count"] for item in per_sample)
    total_pred = sum(item["generated_requirement_count"] for item in per_sample)
    total_strict = sum(item["strict_match_count"] for item in per_sample)
    total_weighted = sum(float(item["weighted_match_score"]) for item in per_sample)
    micro_precision = total_strict / total_pred if total_pred else 0.0
    micro_recall = total_strict / total_source if total_source else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    weighted_micro_precision = total_weighted / total_pred if total_pred else 0.0
    weighted_micro_recall = total_weighted / total_source if total_source else 0.0
    weighted_micro_f1 = (
        2 * weighted_micro_precision * weighted_micro_recall / (weighted_micro_precision + weighted_micro_recall)
        if (weighted_micro_precision + weighted_micro_recall)
        else 0.0
    )
    return {
        "sample_count": len(per_sample),
        "macro_precision": average(item["precision"] for item in per_sample),
        "macro_coverage_recall": average(item["coverage_recall"] for item in per_sample),
        "macro_f1": average(item["f1"] for item in per_sample),
        "macro_hallucination_rate": average(item["hallucination_rate"] for item in per_sample),
        "macro_weighted_precision": average(item["weighted_precision"] for item in per_sample),
        "macro_weighted_coverage_recall": average(item["weighted_coverage_recall"] for item in per_sample),
        "macro_weighted_f1": average(item["weighted_f1"] for item in per_sample),
        "micro_precision": micro_precision,
        "micro_coverage_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_weighted_precision": weighted_micro_precision,
        "micro_weighted_coverage_recall": weighted_micro_recall,
        "micro_weighted_f1": weighted_micro_f1,
        "total_source_requirements": total_source,
        "total_generated_requirements": total_pred,
        "total_strict_matches": total_strict,
        "total_weighted_match_score": total_weighted,
        "total_full_matches": sum(item["full_match_count"] for item in per_sample),
        "total_partial_matches": sum(item["partial_match_count"] for item in per_sample),
    }


def main() -> int:
    args = parse_args()
    gold = load_gold(args.gold_dir)
    predictions = load_predictions(args.pred_dir)
    validator_config = validator_config_from_env()
    client = GeminiNativeClient(validator_config)

    per_sample = []
    for sample_id, gold_sample in gold.items():
        pred_sample = predictions.get(sample_id)
        if pred_sample is None:
            continue
        per_sample.append(evaluate_sample(gold_sample, pred_sample, client, args))

    payload = {
        "gold_dir": safe_rel(args.gold_dir),
        "pred_dir": safe_rel(args.pred_dir),
        "validator": {
            "provider": "gemini",
            "model": validator_config.model,
            "method": "gemini_checklist_judge_v1",
            "semantic_candidate_top_k": args.semantic_top_k,
            "lexical_candidate_top_k": args.lexical_top_k,
            "batch_size": args.batch_size,
            "strict_scoring": STRICT_SCORE_BY_VERDICT,
            "weighted_scoring": WEIGHTED_SCORE_BY_VERDICT,
        },
        "aggregate": build_aggregate(per_sample),
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
