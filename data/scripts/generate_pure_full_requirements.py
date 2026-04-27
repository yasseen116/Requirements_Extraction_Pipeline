#!/usr/bin/env python3
"""Generate full structured requirements files from expanded dialogues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import llm_router as llm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "pure_full" / "expanded_dialogues"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "generated_requirements"
DEFAULT_PROMPT = ROOT / "prompts" / "dialogue_to_full_requirements_gemini.txt"
DEFAULT_SCHEMA = ROOT / "schemas" / "gemini_full_requirements_response.schema.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


MODAL_RE = re.compile(r"\b(shall|must|should|can|may|will)\b", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?;])\s+")
REQUIREMENT_KEYS = ["functional", "non_functional", "data", "business_rules", "interfaces", "constraints"]
CATEGORY_PREFIXES = {
    "functional": "FR",
    "non_functional": "NFR",
    "data": "DATA",
    "business_rules": "BR",
    "interfaces": "IF",
    "constraints": "C",
}
MEMORY_CATEGORY_LABELS = {
    "functional": "Functional",
    "non_functional": "Non-functional",
    "data": "Data",
    "business_rules": "Business rules",
    "interfaces": "Interfaces",
    "constraints": "Constraints",
}


def ensure_period(text: str) -> str:
    text = clean_text(text).strip()
    if not text:
        return ""
    text = text.rstrip(".").strip()
    return text + "."


def is_meta_disclaimer(text: str) -> bool:
    lowered = clean_text(text).lower()
    if not lowered:
        return True
    # These are not requirements; they are dataset/prompt meta or "absence statements".
    meta_markers = [
        "source requirements",
        "the source requirements",
        "provided source requirements",
        "requirements do not mention",
        "requirements don't mention",
        "requirements do not specify",
        "requirements don't specify",
        "not specified in the requirements",
        "not mentioned in the requirements",
        "not explicitly mentioned",
        "not explicitly stated",
        "tbd",
        "to be determined",
        "not decided yet",
        "not sure yet",
        "unknown at this time",
        "this project is based on the",
        "capture the main requirements and constraints",
        "capture the main requirements",
    ]
    return any(marker in lowered for marker in meta_markers)


def normalize_req_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    # Undo a common normalization failure: "The system shall X shall ..." where X is already a requirement.
    m = re.match(r"^\s*the system shall\s+(?P<rest>.+)$", text, flags=re.IGNORECASE)
    if m:
        rest = clean_text(m.group("rest"))
        # If the remainder already contains a modal verb, keep the original subject (do not prefix).
        if MODAL_RE.search(rest) and not rest.lower().startswith("the system shall"):
            text = rest

    # Keep stakeholder wording where possible; only add "The system shall" for bare verb-phrases.
    if MODAL_RE.search(text):
        return ensure_period(text)

    if re.match(
        r"^(allow|provide|support|enable|prevent|log|store|send|receive|export|import|generate|manage|configure|display|calculate|compute|compress|operate|run|perform)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ensure_period(f"The system shall {text}")

    return ensure_period(text)


def render_dialogue(sample: dict) -> str:
    return "\n".join(f"{turn['turn_id']}. {turn['role']}: {turn['text']}" for turn in sample["dialogue"])


def build_prompt(
    template: str,
    sample: dict,
    memory_text: str = "",
    *,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> str:
    prompt = (
        template.replace("{{SAMPLE_ID}}", sample["sample_id"])
        .replace("{{DOMAIN_HINT}}", sample["metadata"]["domain"])
        .replace("{{DIALOGUE}}", render_dialogue(sample))
    )
    replacements = {
        "{{PREVIOUS_REQUIREMENTS}}": memory_text if memory_text else "None yet.",
        "{{CHUNK_INDEX}}": str(chunk_index),
        "{{CHUNK_COUNT}}": str(chunk_count),
    }
    for marker, value in replacements.items():
        if marker in prompt:
            prompt = prompt.replace(marker, value)
    return prompt


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "dialogue" not in payload:
            continue
        samples.append(payload)
    return samples


def normalize_requirement_list(requirements: list[dict], *, nfr: bool) -> list[dict]:
    normalized = []
    seen_texts = set()
    for index, item in enumerate(requirements, start=1):
        if not isinstance(item, dict):
            continue
        req_id = clean_text(item.get("id", "")) or ("NFR" if nfr else "REQ") + f"-{index:03d}"
        raw_text = clean_text(item.get("text", ""))
        if is_meta_disclaimer(raw_text):
            continue
        text = normalize_req_text(raw_text)
        if not text:
            continue
        dedup_key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if dedup_key in seen_texts:
            continue
        seen_texts.add(dedup_key)
        priority = clean_text(item.get("priority", "medium")).lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        evidence_turns = item.get("evidence_turns", [])
        if not isinstance(evidence_turns, list):
            evidence_turns = []
        evidence_turns = sorted({turn for turn in evidence_turns if isinstance(turn, int) and turn >= 1})
        if not evidence_turns:
            evidence_turns = [2]
        normalized_item = {
            "id": req_id,
            "text": text,
            "priority": priority,
            "evidence_turns": evidence_turns,
        }
        if nfr:
            category = clean_text(item.get("category", "performance")).lower()
            if category not in {
                "performance",
                "security",
                "reliability",
                "availability",
                "usability",
                "maintainability",
                "portability",
                "compliance",
            }:
                category = "performance"
            normalized_item["category"] = category
        normalized.append(normalized_item)
    return normalized


def dedup_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def finalize_requirements(reqs: dict) -> dict:
    if not isinstance(reqs, dict):
        reqs = {}

    normalized = {
        "functional": normalize_requirement_list(reqs.get("functional", []), nfr=False),
        "non_functional": normalize_requirement_list(reqs.get("non_functional", []), nfr=True),
        "data": normalize_requirement_list(reqs.get("data", []), nfr=False),
        "business_rules": normalize_requirement_list(reqs.get("business_rules", []), nfr=False),
        "interfaces": normalize_requirement_list(reqs.get("interfaces", []), nfr=False),
        "constraints": normalize_requirement_list(reqs.get("constraints", []), nfr=False),
    }

    deduped = {key: [] for key in REQUIREMENT_KEYS}
    seen_texts: set[str] = set()
    for key in REQUIREMENT_KEYS:
        for item in normalized[key]:
            item_text = item.get("text", "")
            text_key = dedup_text_key(item_text)
            if not text_key or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            deduped[key].append(item)

    return {
        key: renumber_requirement_ids(deduped[key], CATEGORY_PREFIXES[key])
        for key in REQUIREMENT_KEYS
    }


def normalize_payload(payload: dict) -> dict:
    reqs = payload.get("requirements", {})
    if not isinstance(reqs, dict):
        raise ValueError("Missing requirements object")

    normalized = {
        "project_summary": clean_text(payload.get("project_summary", "")),
        "requirements": finalize_requirements(reqs),
    }
    if not normalized["project_summary"]:
        normalized["project_summary"] = "Generated from elicitation dialogue."
    return normalized


def renumber_requirement_ids(items: list[dict], prefix: str) -> list[dict]:
    renumbered = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        updated["id"] = f"{prefix}-{index:03d}"
        renumbered.append(updated)
    return renumbered


def merge_normalized_payloads(payloads: list[dict]) -> dict:
    merged = {
        "project_summary": "",
        "requirements": {key: [] for key in REQUIREMENT_KEYS},
    }
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        summary = clean_text(payload.get("project_summary", ""))
        if summary and not merged["project_summary"]:
            merged["project_summary"] = summary
        reqs = payload.get("requirements", {})
        if not isinstance(reqs, dict):
            continue
        for key in REQUIREMENT_KEYS:
            merged["requirements"][key].extend(reqs.get(key, []))

    merged["requirements"] = finalize_requirements(merged["requirements"])
    if not merged["project_summary"]:
        merged["project_summary"] = "Generated from elicitation dialogue."
    return merged


def requirement_count(normalized_payload: dict) -> int:
    reqs = normalized_payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return 0
    return sum(len(reqs.get(key, [])) for key in REQUIREMENT_KEYS)


def update_memory_entries(memory_entries: list[dict], normalized_payload: dict) -> None:
    reqs = normalized_payload.get("requirements", {})
    if not isinstance(reqs, dict):
        return

    seen = {dedup_text_key(entry.get("text", "")) for entry in memory_entries}
    for key in REQUIREMENT_KEYS:
        items = reqs.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("text", ""))
            text_key = dedup_text_key(text)
            if not text_key or text_key in seen:
                continue
            seen.add(text_key)
            memory_entries.append(
                {
                    "category": key,
                    "text": text,
                }
            )


def format_memory_entries(memory_entries: list[dict]) -> str:
    if not memory_entries:
        return "None yet."

    lines = []
    for key in REQUIREMENT_KEYS:
        category_entries = [entry for entry in memory_entries if entry.get("category") == key]
        if not category_entries:
            continue
        lines.append(f"{MEMORY_CATEGORY_LABELS[key]}:")
        for entry in category_entries:
            lines.append(f"- {entry.get('text', '')}")
    return "\n".join(lines) if lines else "None yet."


def render_memory_text(memory_entries: list[dict], *, max_items: int, max_chars: int) -> str:
    if not memory_entries or max_items == 0:
        return "None yet."

    selected = memory_entries[-max_items:] if max_items > 0 else list(memory_entries)
    while selected:
        rendered = format_memory_entries(selected)
        if max_chars <= 0 or len(rendered) <= max_chars:
            return rendered
        selected = selected[1:]
    return "None yet."


def heuristic_payload_from_dialogue(sample: dict) -> dict:
    dialogue = sample.get("dialogue", [])
    functional = []
    seen = set()

    for turn in dialogue:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        turn_id = turn.get("turn_id") if isinstance(turn.get("turn_id"), int) and turn.get("turn_id") >= 1 else 2
        text = clean_text(turn.get("text", ""))
        if not text:
            continue

        for sentence in SENTENCE_SPLIT_RE.split(text):
            sentence = clean_text(sentence).strip(" -•\t")
            if not sentence:
                continue
            word_count = len(sentence.split())
            if word_count < 6 or word_count > 70:
                continue

            lowered = sentence.lower()
            if not MODAL_RE.search(sentence) and not re.match(
                r"^(allow|provide|support|enable|prevent|log|store|send|receive|export|import|generate|manage|configure|display|calculate|compute|compress|operate|run|perform)\b",
                lowered,
            ):
                continue

            normalized_text = normalize_req_text(sentence)
            if not normalized_text or is_meta_disclaimer(normalized_text):
                continue

            key = re.sub(r"[^a-z0-9]+", " ", normalized_text.lower()).strip()
            if key in seen:
                continue
            seen.add(key)

            functional.append(
                {
                    "id": f"FR-{len(functional) + 1:03d}",
                    "text": normalized_text,
                    "priority": "medium",
                    "evidence_turns": [turn_id],
                }
            )

    return {
        "project_summary": "Heuristic fallback generated from elicitation dialogue due empty/failed model output.",
        "requirements": {
            "functional": functional,
            "non_functional": [],
            "data": [],
            "business_rules": [],
            "interfaces": [],
            "constraints": [],
        },
    }


def build_repair_prompt(schema_text: str, invalid_text: str) -> str:
    return (
        "Repair the following invalid JSON so it matches the schema exactly.\n"
        "Return JSON only.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Invalid JSON:\n{invalid_text}\n"
    )


def select_dialogue_turns(sample: dict, turn_ids: list[int]) -> list[dict]:
    wanted = set(turn_ids)
    return [turn for turn in sample.get("dialogue", []) if isinstance(turn, dict) and turn.get("turn_id") in wanted]


def merge_turn_lists(*turn_groups: list[dict]) -> list[dict]:
    merged = []
    seen_turn_ids = set()
    for turn_group in turn_groups:
        for turn in turn_group:
            if not isinstance(turn, dict):
                continue
            turn_id = turn.get("turn_id")
            key = turn_id if isinstance(turn_id, int) else sha256_text(json.dumps(turn, sort_keys=True, ensure_ascii=False))
            if key in seen_turn_ids:
                continue
            seen_turn_ids.add(key)
            merged.append(turn)
    return merged


def segment_char_count(turns: list[dict]) -> int:
    return sum(len(clean_text(turn.get("text", ""))) for turn in turns if isinstance(turn, dict))


def segment_turn_ids(turns: list[dict]) -> list[int]:
    return [
        turn.get("turn_id")
        for turn in turns
        if isinstance(turn, dict) and isinstance(turn.get("turn_id"), int)
    ]


def fits_segment(turns: list[dict], *, max_turns: int, max_chars: int) -> bool:
    return len(turns) <= max_turns and segment_char_count(turns) <= max_chars


def build_seed_segment(
    scope_turns: list[dict],
    overlap_context: list[dict],
    pair_segment: list[dict],
    *,
    max_turns: int,
    max_chars: int,
) -> list[dict]:
    candidates = [
        merge_turn_lists(scope_turns, overlap_context, pair_segment),
        merge_turn_lists(overlap_context, pair_segment),
        merge_turn_lists(scope_turns, pair_segment),
        merge_turn_lists(pair_segment),
    ]
    for candidate in candidates:
        if fits_segment(candidate, max_turns=max_turns, max_chars=max_chars):
            return candidate
    return candidates[-1]


def build_dialogue_segments(
    sample: dict,
    *,
    max_turns: int,
    max_chars: int,
    overlap_turns: int,
) -> list[list[dict]]:
    dialogue = sample.get("dialogue", [])
    if not isinstance(dialogue, list) or not dialogue:
        return []

    overlap_turns = max(0, overlap_turns)
    segments = []
    trace = sample.get("dialogue_generation", {}).get("trace", [])
    if isinstance(trace, list) and trace:
        scope_turns: list[dict] = []
        trace_pairs: list[list[dict]] = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            bot_turn_id = item.get("bot_turn_id")
            user_turn_id = item.get("user_turn_id")
            if not isinstance(bot_turn_id, int) or not isinstance(user_turn_id, int):
                continue
            pair_segment = select_dialogue_turns(sample, [bot_turn_id, user_turn_id])
            if not pair_segment:
                continue
            if item.get("theme") == "goal_scope":
                scope_turns = pair_segment
                continue
            if not item.get("req_ids"):
                continue
            trace_pairs.append(pair_segment)

        if trace_pairs:
            current = list(scope_turns)
            for pair_segment in trace_pairs:
                candidate = merge_turn_lists(current, pair_segment)
                if current and not fits_segment(candidate, max_turns=max_turns, max_chars=max_chars):
                    segments.append(current)
                    overlap_context = current[-overlap_turns:] if overlap_turns > 0 else []
                    current = build_seed_segment(
                        scope_turns,
                        overlap_context,
                        pair_segment,
                        max_turns=max_turns,
                        max_chars=max_chars,
                    )
                    continue
                current = candidate

            if current and (not segments or segment_turn_ids(current) != segment_turn_ids(segments[-1])):
                segments.append(current)
            if segments:
                return segments

    current = []
    current_chars = 0

    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        turn_text = clean_text(turn.get("text", ""))
        turn_chars = len(turn_text)
        if current and (len(current) >= max_turns or current_chars + turn_chars > max_chars):
            segments.append(current)
            current = current[-overlap_turns:] if len(current) > overlap_turns else []
            current_chars = sum(len(clean_text(t.get("text", ""))) for t in current)
        current.append(turn)
        current_chars += turn_chars

    if current and (not segments or segment_turn_ids(current) != segment_turn_ids(segments[-1])):
        segments.append(current)
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=1,
        help="Number of independent generations to run per sample (merged by dedup).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Keep this at 0.0 for deterministic extraction.",
    )
    parser.add_argument("--chunk-dialogues", action="store_true")
    parser.add_argument("--dialogue-chunk-max-turns", type=int, default=8)
    parser.add_argument("--dialogue-chunk-max-chars", type=int, default=2600)
    parser.add_argument("--dialogue-chunk-overlap-turns", type=int, default=2)
    parser.add_argument("--memory-max-items", type=int, default=24)
    parser.add_argument("--memory-max-chars", type=int, default=3500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    prompt_hash = sha256_text(prompt_template)
    schema_hash = sha256_text(schema_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        if not samples:
            print("No samples available for dry run.")
            return 0
        preview = {
            "sample_id": samples[0]["sample_id"],
            "prompt": build_prompt(prompt_template, samples[0]),
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "schema": schema,
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    model_name = os.environ.get("REQ_OLLAMA_MODEL") if llm.provider() == "ollama" else os.environ.get("REQ_GEMINI_MODEL")
    summary = []

    for sample in samples:
        repair_attempted = False
        error_messages = []
        segment_payloads = []
        segment_statuses = []
        raw_responses = []
        usages = []

        segments = [sample.get("dialogue", [])]
        if args.chunk_dialogues:
            segments = build_dialogue_segments(
                sample,
                max_turns=args.dialogue_chunk_max_turns,
                max_chars=args.dialogue_chunk_max_chars,
                overlap_turns=args.dialogue_chunk_overlap_turns,
            )
            if not segments:
                segments = [sample.get("dialogue", [])]

        memory_entries: list[dict] = []
        for segment_index, segment in enumerate(segments, start=1):
            segment_sample = dict(sample)
            segment_sample["dialogue"] = segment
            prompt = build_prompt(
                prompt_template,
                segment_sample,
                render_memory_text(
                    memory_entries,
                    max_items=max(0, args.memory_max_items),
                    max_chars=max(0, args.memory_max_chars),
                ),
                chunk_index=segment_index,
                chunk_count=len(segments),
            )
            invalid_text = ""
            try:
                runs = max(1, int(args.self_consistency))
                candidates = []
                segment_usages = []
                segment_raw_responses = []
                for _run_idx in range(runs):
                    response = llm.generate_json(prompt, schema, temperature=args.temperature)
                    invalid_text = response.text
                    parsed = llm.parse_first_json_object(response.text)
                    candidates.append(normalize_payload(parsed))
                    segment_usages.append(response.usage)
                    segment_raw_responses.append(response.raw_response)

                merged_segment = merge_normalized_payloads(candidates)
                if requirement_count(merged_segment) == 0:
                    merged_segment = heuristic_payload_from_dialogue(segment_sample)
                    segment_statuses.append(f"segment_{segment_index}:fallback_heuristic_empty_model_output")
                else:
                    status = "ok" if runs == 1 else f"ok_self_consistency_{runs}"
                    segment_statuses.append(f"segment_{segment_index}:{status}")
                segment_payloads.append(merged_segment)
                usages.append({"segment_index": segment_index, "runs": runs, "per_run": segment_usages})
                raw_responses.append({"segment_index": segment_index, "responses": segment_raw_responses})
                update_memory_entries(memory_entries, merged_segment)
            except Exception as first_error:  # noqa: BLE001
                repair_attempted = True
                try:
                    repair_prompt = build_repair_prompt(schema_text, invalid_text or str(first_error))
                    response = llm.generate_json(repair_prompt, schema, temperature=0.0)
                    parsed = llm.parse_first_json_object(response.text)
                    repaired_payload = normalize_payload(parsed)
                    if requirement_count(repaired_payload) == 0:
                        repaired_payload = heuristic_payload_from_dialogue(segment_sample)
                        segment_statuses.append(f"segment_{segment_index}:fallback_heuristic_empty_after_repair")
                    else:
                        segment_statuses.append(f"segment_{segment_index}:repaired")
                    segment_payloads.append(repaired_payload)
                    usages.append({"segment_index": segment_index, "repair": response.usage})
                    raw_responses.append({"segment_index": segment_index, "responses": [response.raw_response]})
                    update_memory_entries(memory_entries, repaired_payload)
                except Exception as second_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"segment_{segment_index}: {type(first_error).__name__}: {first_error}; "
                        f"repair failed with {type(second_error).__name__}: {second_error}"
                    ) from second_error

        normalized_payload = merge_normalized_payloads(segment_payloads)
        if requirement_count(normalized_payload) == 0:
            normalized_payload = heuristic_payload_from_dialogue(sample)
            segment_statuses.append("global:fallback_heuristic_empty_merge")

        if args.chunk_dialogues:
            parse_status = f"chunked_{len(segments)}_segments"
        else:
            parse_status = segment_statuses[0].split(":", 1)[-1] if segment_statuses else "failed"
        error = " | ".join(error_messages) if error_messages else None
        raw_response = {"segments": raw_responses} if raw_responses else None
        usage = {"segments": usages}

        output_payload = {
            "sample_id": sample["sample_id"],
            "source": sample.get("source"),
            "method": "g_full_dialogue_to_full_requirements_v1",
            "model": model_name,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "parse_status": parse_status,
            "repair_attempted": repair_attempted,
            "usage": usage,
            "error": error,
            "chunking": {
                "enabled": bool(args.chunk_dialogues),
                "segment_count": len(segments),
                "segment_statuses": segment_statuses,
                "dialogue_chunk_max_turns": args.dialogue_chunk_max_turns if args.chunk_dialogues else None,
                "dialogue_chunk_max_chars": args.dialogue_chunk_max_chars if args.chunk_dialogues else None,
                "dialogue_chunk_overlap_turns": args.dialogue_chunk_overlap_turns if args.chunk_dialogues else None,
            },
            "memory": {
                "enabled": bool(args.chunk_dialogues),
                "max_items": args.memory_max_items,
                "max_chars": args.memory_max_chars,
                "stored_unique_requirements": len(memory_entries),
            },
            "project_summary": normalized_payload["project_summary"] if normalized_payload else "",
            "requirements": normalized_payload["requirements"] if normalized_payload else {
                "functional": [],
                "non_functional": [],
                "data": [],
                "business_rules": [],
                "interfaces": [],
                "constraints": [],
            },
        }
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if raw_response is not None:
            raw_path = args.output_dir / f"{sample['sample_id']}.raw_response.json"
            raw_path.write_text(json.dumps(raw_response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        req_count = requirement_count(normalized_payload) if normalized_payload else 0
        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(output_path),
                "parse_status": parse_status,
                "segment_count": len(segments),
                "requirement_count": req_count,
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
