#!/usr/bin/env python3
"""
Validate generated PURE requirements against the source dialogue.

For each generated requirement, check whether it is grounded in at least one
user turn in the dialogue using token-F1 (no external ML deps needed).

A requirement is GROUNDED  if its best token-F1 against any user turn >= threshold.
A requirement is HALLUCINATED if no user turn supports it.

Semantic similarity via sentence-transformers is used when available;
otherwise falls back to token-F1 (always available, no deps).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ── optional sentence-transformer support ─────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore

    _ST_MODEL: SentenceTransformer | None = None

    def _get_st_model() -> SentenceTransformer:
        global _ST_MODEL
        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _ST_MODEL

    def _semantic_sim(req: str, user_turns: list[str]) -> tuple[float, int]:
        model = _get_st_model()
        req_emb = model.encode([req], convert_to_tensor=True)
        turn_embs = model.encode(user_turns, convert_to_tensor=True)
        sims = st_util.cos_sim(req_emb, turn_embs)[0].tolist()
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        return sims[best_idx], best_idx

    _HAS_ST = True
    SIMILARITY_METHOD = "cosine_sentence_transformer"

except ImportError:
    _HAS_ST = False
    SIMILARITY_METHOD = "token_f1"

    def _semantic_sim(req: str, user_turns: list[str]) -> tuple[float, int]:  # type: ignore
        return _best_token_f1(req, user_turns)


ROOT = Path(__file__).resolve().parent.parent

CATEGORY_ORDER = ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]


# ── Token-F1 (zero-dep fallback) ───────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "of", "to", "in", "on", "and", "or", "for",
        "with", "by", "is", "are", "be", "as", "that", "this", "it",
        "from", "at", "must", "shall", "should",
    }
    return {t for t in _normalize(text).split() if t not in stopwords and len(t) > 1}


def _token_f1(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r) if p + r else 0.0


def _best_token_f1(req: str, user_turns: list[str]) -> tuple[float, int]:
    scores = [_token_f1(req, t) for t in user_turns]
    if not scores:
        return 0.0, 0
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return scores[best_idx], best_idx


# ── Semantic deduplication (cross-category) ────────────────────────────────────

def semantic_deduplicate(
    requirements: list[str],
    threshold: float = 0.85,
) -> tuple[list[str], list[dict]]:
    """
    Remove semantically duplicate requirements globally across all categories.
    Keeps the first occurrence, removes later near-duplicates.
    threshold=0.85 is conservative — only removes near-identical meaning.

    Uses sentence-transformers when available, falls back to token-F1.
    """
    if not requirements:
        return [], []

    if _HAS_ST:
        model = _get_st_model()
        embeddings = model.encode(requirements, convert_to_tensor=True)

        def _sim(i: int, j: int) -> float:
            return st_util.cos_sim(embeddings[i], embeddings[j]).item()  # type: ignore
    else:
        # Token-F1 fallback: acts as a soft similarity measure
        def _sim(i: int, j: int) -> float:
            return _token_f1(requirements[i], requirements[j])

    kept_indices: list[int] = []
    removed: list[dict] = []

    for i, req in enumerate(requirements):
        is_dup = False
        for j in kept_indices:
            sim = _sim(i, j)
            if sim >= threshold:
                is_dup = True
                removed.append({
                    "removed": req,
                    "kept": requirements[j],
                    "similarity": round(sim, 3),
                    "method": SIMILARITY_METHOD,
                })
                break
        if not is_dup:
            kept_indices.append(i)

    return [requirements[i] for i in kept_indices], removed


# ── Grounding validator ────────────────────────────────────────────────────────

def validate_requirements(
    dialogue: list[dict],
    requirements: list[str],
    threshold: float = 0.25,
) -> dict:
    """
    For each generated requirement, check whether it is supported by at least
    one user turn in the dialogue.

    A requirement is GROUNDED    if best similarity >= threshold.
    A requirement is HALLUCINATED if no user turn supports it.

    threshold=0.40 is intentionally lower than the match threshold (0.55).
    We are asking: "could this requirement have been motivated by something
    the user said?" — a softer test than "does this match a source req?"
    """
    # Extract user turns only
    user_turns = [
        t.get("content") or t.get("text", "")
        for t in dialogue
        if (t.get("role") or "").lower() in {"user", "USER"}
    ]
    user_turns = [t for t in user_turns if t.strip()]

    if not user_turns or not requirements:
        return {
            "total": len(requirements),
            "grounded": len(requirements),
            "hallucinated": 0,
            "hallucination_rate": 0.0,
            "similarity_method": SIMILARITY_METHOD,
            "grounded_requirements": requirements,
            "hallucinated_requirements": [],
        }

    grounded = []
    hallucinated = []

    for req in requirements:
        best_score, best_idx = _semantic_sim(req, user_turns)
        if best_score >= threshold:
            grounded.append({
                "requirement": req,
                "best_supporting_turn": user_turns[best_idx],
                "similarity": round(best_score, 3),
            })
        else:
            hallucinated.append({
                "requirement": req,
                "best_score": round(best_score, 3),
                "closest_turn": user_turns[best_idx] if user_turns else "",
            })

    return {
        "total": len(requirements),
        "grounded": len(grounded),
        "hallucinated": len(hallucinated),
        "hallucination_rate": round(len(hallucinated) / len(requirements), 4) if requirements else 0.0,
        "similarity_method": SIMILARITY_METHOD,
        "grounded_requirements": [g["requirement"] for g in grounded],
        "hallucinated_requirements": hallucinated,
        "grounded_detail": grounded,
    }


# ── File-level processing ──────────────────────────────────────────────────────

def flatten_requirements(payload: dict) -> list[str]:
    """Collect all requirement texts across all categories."""
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return []
    flat = []
    for cat in CATEGORY_ORDER:
        for item in reqs.get(cat, []):
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)
            text = " ".join(text.split())
            if text:
                flat.append(text)
    return flat


def rebuild_payload_with_grounded_only(payload: dict, grounded_texts: set[str]) -> dict:
    """Return a copy of payload keeping only grounded requirements."""
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return payload
    filtered = {}
    for cat in CATEGORY_ORDER:
        filtered[cat] = [
            item for item in reqs.get(cat, [])
            if (item.get("text", "") if isinstance(item, dict) else str(item)).strip() in grounded_texts
        ]
    return {**payload, "requirements": filtered}


def process_sample(
    pred_payload: dict,
    dialogue_payload: dict,
    dedup_threshold: float,
    grounding_threshold: float,
) -> dict:
    """Full Layer 2 + Layer 3 processing for one sample."""
    flat_reqs = flatten_requirements(pred_payload)

    # Layer 2: cross-category semantic deduplication
    deduped, removed_dups = semantic_deduplicate(flat_reqs, threshold=dedup_threshold)

    # Layer 3: grounding validation
    dialogue_turns = dialogue_payload.get("dialogue", [])
    validation = validate_requirements(dialogue_turns, deduped, threshold=grounding_threshold)

    grounded_set = set(validation["grounded_requirements"])
    filtered_payload = rebuild_payload_with_grounded_only(pred_payload, grounded_set)

    return {
        "sample_id": pred_payload.get("sample_id", ""),
        "filtered_payload": filtered_payload,
        "deduplication": {
            "input_count": len(flat_reqs),
            "after_dedup_count": len(deduped),
            "removed_count": len(removed_dups),
            "removed_details": removed_dups,
        },
        "validation": validation,
        "output_requirement_count": len(grounded_set),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic dedup + grounding validation for generated requirements."
    )
    parser.add_argument("--pred-dir", type=Path, required=True, help="Generated requirements directory.")
    parser.add_argument("--dialogue-dir", type=Path, required=True, help="Expanded dialogues directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for filtered requirements.")
    parser.add_argument("--report-path", type=Path, default=None, help="Path to write validation report JSON.")
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="Similarity threshold for deduplication (default 0.85).")
    parser.add_argument("--grounding-threshold", type=float, default=0.25,
                        help="Minimum similarity to a user turn to count as grounded (default 0.25).")
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
        if not dialogue_path.exists():
            print(f"[warn] No dialogue found for {sample_id}, skipping validation.")
            dialogue_payload = {"dialogue": []}
        else:
            dialogue_payload = json.loads(dialogue_path.read_text(encoding="utf-8"))

        result = process_sample(
            pred_payload,
            dialogue_payload,
            dedup_threshold=args.dedup_threshold,
            grounding_threshold=args.grounding_threshold,
        )

        # Write filtered requirements payload
        out_path = args.output_dir / f"{sample_id}.json"
        out_path.write_text(
            json.dumps(result["filtered_payload"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        all_results.append({k: v for k, v in result.items() if k != "filtered_payload"})
        print(
            f"[{sample_id}] {result['deduplication']['input_count']} reqs → "
            f"{result['deduplication']['after_dedup_count']} after dedup → "
            f"{result['output_requirement_count']} grounded (method: {SIMILARITY_METHOD})"
        )

    # Write validation report
    report_path = args.report_path or (args.output_dir / "validation_report.json")
    aggregate = {
        "similarity_method": SIMILARITY_METHOD,
        "samples": len(all_results),
        "total_input": sum(r["deduplication"]["input_count"] for r in all_results),
        "total_after_dedup": sum(r["deduplication"]["after_dedup_count"] for r in all_results),
        "total_grounded": sum(r["output_requirement_count"] for r in all_results),
        "total_hallucinated": sum(r["validation"]["hallucinated"] for r in all_results),
        "mean_hallucination_rate": (
            round(sum(r["validation"]["hallucination_rate"] for r in all_results) / len(all_results), 4)
            if all_results else 0.0
        ),
    }
    report = {"aggregate": aggregate, "per_sample": all_results}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
