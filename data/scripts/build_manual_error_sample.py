#!/usr/bin/env python3
"""Build a draft manual error-analysis sample from a PURE benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from coverage_scorer import CoverageScorer, clean_text


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_reports" / "20260517_qwen_stability_ablation"
LABELS = [
    "FULL_MATCH",
    "PARTIAL_ACTOR_DROPPED",
    "PARTIAL_CONDITION_DROPPED",
    "PARTIAL_NUMERIC_ANCHOR_DROPPED",
    "PARTIAL_SCOPE_GENERALIZED",
    "UNSUPPORTED_ADDITION",
    "DUPLICATE_OR_MERGED",
    "NO_MATCH",
]
ACTOR_TERMS = {
    "administrator",
    "administrators",
    "auditor",
    "auditors",
    "citizen",
    "citizens",
    "customer",
    "customers",
    "manager",
    "managers",
    "police",
    "staff",
    "super-user",
    "system",
    "user",
    "users",
}
CONDITION_TERMS = {
    "after",
    "before",
    "if",
    "only",
    "unless",
    "when",
    "whenever",
    "where",
    "without",
}
NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "minimum",
    "maximum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manual error-analysis sample with draft labels.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.55)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def resolve_saved_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for base in (ROOT, ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def load_gold_items(gold_dir: Path) -> dict[str, list[dict]]:
    samples = {}
    for path in sorted(gold_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        payload = load_json(path)
        sample_id = payload.get("sample_id")
        if not sample_id:
            continue
        items = []
        for index, item in enumerate(payload.get("ground_truth_requirements", []), start=1):
            text = clean_text(item.get("text", ""))
            if not text:
                continue
            items.append(
                {
                    "id": clean_text(item.get("req_id", "")) or f"SRC-{index:03d}",
                    "text": text,
                    "category": clean_text(item.get("category", "")) or "source",
                    "document_id": payload.get("source", {}).get("document_id"),
                }
            )
        samples[sample_id] = items
    return samples


def load_pred_items(pred_dir: Path) -> dict[str, list[dict]]:
    samples = {}
    for path in sorted(pred_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = load_json(path)
        sample_id = payload.get("sample_id")
        if not sample_id:
            continue
        reqs = payload.get("requirements", {})
        items = []
        if isinstance(reqs, dict):
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
        samples[sample_id] = items
    return samples


def word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9\-']*", clean_text(text).lower()))


def has_number_anchor(text: str) -> bool:
    words = word_set(text)
    return any(ch.isdigit() for ch in text) or bool(words & NUMBER_WORDS)


def has_condition(text: str) -> bool:
    words = word_set(text)
    return bool(words & CONDITION_TERMS) or "in case" in clean_text(text).lower()


def missing_source_terms(source: str, generated: str, terms: set[str]) -> set[str]:
    source_terms = word_set(source) & terms
    generated_terms = word_set(generated) & terms
    return source_terms - generated_terms


def has_unsupported_addition(source: str, generated: str) -> bool:
    source_words = word_set(source)
    generated_words = word_set(generated)
    added = generated_words - source_words
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "in",
        "is",
        "it",
        "of",
        "or",
        "shall",
        "should",
        "system",
        "the",
        "to",
        "we",
        "with",
    }
    material = [word for word in added if len(word) >= 6 and word not in stop]
    return len(material) >= 4


def draft_label(source: str, generated: str, score: float, pair_kind: str) -> str:
    if pair_kind == "unmatched_generated":
        return "UNSUPPORTED_ADDITION"
    if pair_kind == "unmatched_source":
        if score >= 0.50:
            return "DUPLICATE_OR_MERGED"
        return "NO_MATCH"
    if has_number_anchor(source) and not has_number_anchor(generated):
        return "PARTIAL_NUMERIC_ANCHOR_DROPPED"
    if missing_source_terms(source, generated, ACTOR_TERMS):
        return "PARTIAL_ACTOR_DROPPED"
    if has_condition(source) and not has_condition(generated):
        return "PARTIAL_CONDITION_DROPPED"
    if score >= 0.82 and not has_unsupported_addition(source, generated):
        return "FULL_MATCH"
    if has_unsupported_addition(source, generated) and score < 0.68:
        return "UNSUPPORTED_ADDITION"
    return "PARTIAL_SCOPE_GENERALIZED"


def best_index(row: list[float]) -> tuple[int | None, float]:
    if not row:
        return None, 0.0
    index, value = max(enumerate(row), key=lambda item: item[1])
    return int(index), float(value)


def build_candidates(run_dir: Path, threshold: float) -> list[dict]:
    comparison = load_json(run_dir / "comparison_summary.json")
    source_dir = resolve_saved_path(comparison["paths"]["source_summary"])
    source_dir = source_dir.parent / "source_requirements"
    pred_rel = comparison["paths"].get("pipeline_coverage")
    if not pred_rel:
        raise ValueError(f"Run does not have pipeline coverage: {run_dir}")
    pred_path = resolve_saved_path(comparison["paths"]["pipeline_coverage"])
    pred_payload = load_json(pred_path)
    pred_dir = resolve_saved_path(pred_payload["pred_dir"])

    gold_by_sample = load_gold_items(source_dir)
    pred_by_sample = load_pred_items(pred_dir)
    scorer = CoverageScorer()
    rows = []

    for sample_id, gold_items in gold_by_sample.items():
        pred_items = pred_by_sample.get(sample_id, [])
        gold_texts = [item["text"] for item in gold_items]
        pred_texts = [item["text"] for item in pred_items]
        matrix = scorer.similarity_matrix(gold_texts, pred_texts) if gold_texts and pred_texts else []
        matches, used_gold, used_pred = scorer.greedy_match(gold_texts, pred_texts, threshold)

        for match in matches:
            gold = gold_items[match["gold_index"]]
            pred = pred_items[match["pred_index"]]
            score = float(match["score"])
            label = draft_label(gold["text"], pred["text"], score, "matched")
            rows.append(
                {
                    "run_id": comparison.get("run_id") or run_dir.name,
                    "sample_id": sample_id,
                    "document_id": gold.get("document_id"),
                    "pair_kind": "matched",
                    "source_id": gold["id"],
                    "generated_id": pred["id"],
                    "score": score,
                    "draft_label": label,
                    "source_requirement": gold["text"],
                    "generated_requirement": pred["text"],
                }
            )

        for gold_index, gold in enumerate(gold_items):
            if gold_index in used_gold:
                continue
            pred_index, score = best_index(matrix[gold_index] if matrix else [])
            pred = pred_items[pred_index] if pred_index is not None and pred_items else {"id": "", "text": ""}
            label = draft_label(gold["text"], pred["text"], score, "unmatched_source")
            rows.append(
                {
                    "run_id": comparison.get("run_id") or run_dir.name,
                    "sample_id": sample_id,
                    "document_id": gold.get("document_id"),
                    "pair_kind": "unmatched_source",
                    "source_id": gold["id"],
                    "generated_id": pred["id"],
                    "score": score,
                    "draft_label": label,
                    "source_requirement": gold["text"],
                    "generated_requirement": pred["text"],
                }
            )

        for pred_index, pred in enumerate(pred_items):
            if pred_index in used_pred:
                continue
            column = [float(matrix[gold_index][pred_index]) for gold_index in range(len(gold_items))] if matrix else []
            gold_index, score = best_index(column)
            gold = gold_items[gold_index] if gold_index is not None and gold_items else {"id": "", "text": "", "document_id": ""}
            label = draft_label(gold["text"], pred["text"], score, "unmatched_generated")
            rows.append(
                {
                    "run_id": comparison.get("run_id") or run_dir.name,
                    "sample_id": sample_id,
                    "document_id": gold.get("document_id"),
                    "pair_kind": "unmatched_generated",
                    "source_id": gold["id"],
                    "generated_id": pred["id"],
                    "score": score,
                    "draft_label": label,
                    "source_requirement": gold["text"],
                    "generated_requirement": pred["text"],
                }
            )
    return rows


def stratified_sample(rows: list[dict], max_examples: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["draft_label"]].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item["sample_id"], item["pair_kind"], -float(item["score"]), item["source_id"]))

    selected = []
    positions = {label: 0 for label in LABELS}
    while len(selected) < max_examples:
        added = False
        for label in LABELS:
            bucket = buckets.get(label, [])
            position = positions[label]
            if position >= len(bucket):
                continue
            selected.append(bucket[position])
            positions[label] += 1
            added = True
            if len(selected) >= max_examples:
                break
        if not added:
            break
    return selected


def counts_table(rows: list[dict]) -> list[dict]:
    counts = Counter(row["draft_label"] for row in rows)
    total = sum(counts.values())
    return [
        {
            "error_type": label,
            "count": counts.get(label, 0),
            "percent": (counts.get(label, 0) / total * 100.0) if total else 0.0,
        }
        for label in LABELS
    ]


def render_markdown(sample: list[dict], counts: list[dict], run_dir: Path) -> str:
    lines = [
        "# Manual Error Analysis Draft Sample",
        "",
        f"Run: `{run_dir.name}`",
        "",
        "Draft labels are heuristic Codex labels for rapid human spot-checking.",
        "",
        "## Counts",
        "",
        "| Error type | Count | Percent |",
        "| --- | ---: | ---: |",
    ]
    for row in counts:
        lines.append(f"| {row['error_type']} | {row['count']} | {row['percent']:.1f}% |")
    lines.extend(
        [
            "",
            "## Sample",
            "",
            "| # | Sample | Pair | Score | Draft label | Source requirement | Generated requirement |",
            "| ---: | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for index, row in enumerate(sample, start=1):
        source = clean_text(row["source_requirement"]).replace("|", "\\|")
        generated = clean_text(row["generated_requirement"]).replace("|", "\\|")
        if len(source) > 180:
            source = source[:177].rstrip() + "..."
        if len(generated) > 180:
            generated = generated[:177].rstrip() + "..."
        lines.append(
            f"| {index} | `{row['sample_id']}` | {row['pair_kind']} | "
            f"{float(row['score']):.4f} | {row['draft_label']} | {source} | {generated} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_id",
        "sample_id",
        "document_id",
        "pair_kind",
        "source_id",
        "generated_id",
        "score",
        "draft_label",
        "source_requirement",
        "generated_requirement",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = build_candidates(args.run_dir.resolve(), args.threshold)
    sample = stratified_sample(rows, max(1, args.max_examples))
    counts = counts_table(sample)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": safe_rel(args.run_dir.resolve()),
        "threshold": args.threshold,
        "label_policy": "heuristic_codex_draft",
        "counts": counts,
        "examples": sample,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "manual_error_sample.json"
    md_path = args.output_dir / "manual_error_sample.md"
    csv_path = args.output_dir / "manual_error_sample.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(sample, counts, args.run_dir.resolve()), encoding="utf-8")
    write_csv(csv_path, sample)
    print(json.dumps({"json": safe_rel(json_path), "markdown": safe_rel(md_path), "csv": safe_rel(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
