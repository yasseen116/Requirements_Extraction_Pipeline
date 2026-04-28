#!/usr/bin/env python3
"""Shared semantic coverage and support-unit utilities for the PURE benchmark."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

try:
    from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore

    _HAS_ST = True
except ImportError:  # pragma: no cover - runtime fallback
    SentenceTransformer = None  # type: ignore[assignment]
    st_util = None  # type: ignore[assignment]
    _HAS_ST = False


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MATCH_THRESHOLD = 0.55
SUPPORT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|;\s+")
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


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token not in STOPWORDS and len(token) > 1}


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


def average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def split_support_text(text: str) -> list[str]:
    compact = clean_text(text)
    if not compact:
        return []
    parts = [clean_text(part).strip(" -•\t") for part in SUPPORT_SPLIT_RE.split(compact)]
    return [part for part in parts if part]


def _trace_theme_by_user_turn(dialogue_payload: dict) -> dict[int, str]:
    trace = dialogue_payload.get("dialogue_generation", {}).get("trace", [])
    theme_by_turn: dict[int, str] = {}
    if not isinstance(trace, list):
        return theme_by_turn
    for item in trace:
        if not isinstance(item, dict):
            continue
        turn_id = item.get("user_turn_id")
        theme = clean_text(item.get("theme", ""))
        if isinstance(turn_id, int) and theme:
            theme_by_turn[turn_id] = theme
    return theme_by_turn


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    if not _HAS_ST:  # pragma: no cover - guarded by caller
        raise RuntimeError("sentence-transformers is not available")
    return SentenceTransformer(model_name)


class CoverageScorer:
    """Sentence-transformer-backed coverage helper with support-unit utilities."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        context_window: int = 1,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.context_window = max(0, int(context_window))
        self.has_sentence_transformer = _HAS_ST
        self.similarity_method = "cosine_sentence_transformer" if _HAS_ST else "token_f1"

    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> list[list[float]]:
        if not texts_a or not texts_b:
            return [[0.0 for _ in texts_b] for _ in texts_a]
        if self.has_sentence_transformer:
            model = _load_model(self.model_name)
            embeddings_a = model.encode(texts_a, convert_to_tensor=True)
            embeddings_b = model.encode(texts_b, convert_to_tensor=True)
            return st_util.cos_sim(embeddings_a, embeddings_b).cpu().tolist()  # type: ignore[union-attr]
        return [[token_f1(a, b) for b in texts_b] for a in texts_a]

    def similarity_row(self, query: str, candidates: list[str]) -> list[float]:
        matrix = self.similarity_matrix([query], candidates)
        return matrix[0] if matrix else []

    def greedy_match(
        self,
        gold_texts: list[str],
        pred_texts: list[str],
        threshold: float | None = None,
    ) -> tuple[list[dict], set[int], set[int]]:
        cutoff = self.threshold if threshold is None else threshold
        candidates = []
        matrix = self.similarity_matrix(gold_texts, pred_texts)
        for g_idx, row in enumerate(matrix):
            for p_idx, score in enumerate(row):
                if score >= cutoff:
                    candidates.append((float(score), g_idx, p_idx))
        candidates.sort(reverse=True, key=lambda item: item[0])

        used_gold: set[int] = set()
        used_pred: set[int] = set()
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
                    "score": float(score),
                }
            )
        return matches, used_gold, used_pred

    def build_dialogue_support_units(self, dialogue_payload: dict, *, user_only: bool = True) -> list[dict]:
        dialogue = dialogue_payload.get("dialogue", [])
        if not isinstance(dialogue, list):
            return []
        theme_by_turn = _trace_theme_by_user_turn(dialogue_payload)
        units = []
        for turn in dialogue:
            if not isinstance(turn, dict):
                continue
            role = clean_text(turn.get("role", "")).lower()
            if user_only and role != "user":
                continue
            turn_id = turn.get("turn_id")
            turn_text = clean_text(turn.get("content") or turn.get("text", ""))
            if not turn_text:
                continue
            parts = split_support_text(turn_text)
            if not parts:
                continue
            for index, part in enumerate(parts, start=1):
                start = max(0, index - 1 - self.context_window)
                end = min(len(parts), index + self.context_window)
                context_text = " ".join(parts[start:end])
                units.append(
                    {
                        "unit_id": f"{turn_id}:{index}",
                        "turn_id": turn_id,
                        "sentence_index": index,
                        "text": part,
                        "context_text": context_text,
                        "turn_text": turn_text,
                        "role": role,
                        "trace_theme": theme_by_turn.get(turn_id),
                    }
                )
        return units

    def unit_texts(self, units: list[dict], *, contextualized: bool = True) -> list[str]:
        key = "context_text" if contextualized else "text"
        return [clean_text(unit.get(key) or unit.get("text", "")) for unit in units]

    def coverage_against_units(
        self,
        query_texts: list[str],
        units: list[dict],
        *,
        threshold: float | None = None,
        contextualized: bool = True,
    ) -> list[dict]:
        support_texts = self.unit_texts(units, contextualized=False)
        cutoff = self.threshold if threshold is None else threshold
        matrix = self.similarity_matrix(query_texts, support_texts)
        context_matrix = None
        if contextualized:
            context_texts = self.unit_texts(units, contextualized=True)
            context_matrix = self.similarity_matrix(query_texts, context_texts)
        results = []
        for q_idx, query in enumerate(query_texts):
            row = matrix[q_idx] if q_idx < len(matrix) else []
            context_row = context_matrix[q_idx] if context_matrix and q_idx < len(context_matrix) else []
            if context_row and row:
                row = [max(float(base_score), float(context_score)) for base_score, context_score in zip(row, context_row)]
            if not row:
                best_idx = 0
                best_score = 0.0
            else:
                best_idx = max(range(len(row)), key=lambda idx: row[idx])
                best_score = float(row[best_idx])
            best_unit = units[best_idx] if row else None
            results.append(
                {
                    "query_index": q_idx,
                    "query_text": query,
                    "best_score": best_score,
                    "covered": best_score >= cutoff,
                    "best_unit": best_unit,
                }
            )
        return results

    def best_unit_for_query(
        self,
        query_text: str,
        units: list[dict],
        *,
        candidate_turn_ids: set[int] | None = None,
        contextualized: bool = True,
    ) -> tuple[float, dict | None]:
        filtered_units = units
        if candidate_turn_ids:
            filtered_units = [unit for unit in units if unit.get("turn_id") in candidate_turn_ids]
        if not filtered_units:
            return 0.0, None
        results = self.coverage_against_units(
            [query_text],
            filtered_units,
            threshold=0.0,
            contextualized=contextualized,
        )
        best = results[0]
        return float(best["best_score"]), best.get("best_unit")

    def rank_units(
        self,
        query_text: str,
        units: list[dict],
        *,
        top_k: int | None = None,
        contextualized: bool = True,
    ) -> list[dict]:
        base_scores = self.similarity_row(query_text, self.unit_texts(units, contextualized=False))
        if contextualized:
            context_scores = self.similarity_row(query_text, self.unit_texts(units, contextualized=True))
            scores = [max(float(base_score), float(context_score)) for base_score, context_score in zip(base_scores, context_scores)]
        else:
            scores = base_scores
        ranked = []
        for index, score in enumerate(scores):
            ranked.append(
                {
                    "score": float(score),
                    "unit": units[index],
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        if top_k is not None and top_k >= 0:
            return ranked[:top_k]
        return ranked

    def semantic_deduplicate(self, items: list[str], *, threshold: float) -> tuple[list[str], list[dict]]:
        if not items:
            return [], []

        kept_indices: list[int] = []
        removed = []
        matrix = self.similarity_matrix(items, items)
        for index, text in enumerate(items):
            is_duplicate = False
            for kept_index in kept_indices:
                score = float(matrix[index][kept_index])
                if score >= threshold:
                    is_duplicate = True
                    removed.append(
                        {
                            "removed": text,
                            "kept": items[kept_index],
                            "similarity": round(score, 3),
                            "method": self.similarity_method,
                        }
                    )
                    break
            if not is_duplicate:
                kept_indices.append(index)
        return [items[index] for index in kept_indices], removed
