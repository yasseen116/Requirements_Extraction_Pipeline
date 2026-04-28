#!/usr/bin/env python3
"""Generate full structured requirements files from expanded dialogues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from coverage_scorer import CoverageScorer, clean_text
import llm_router as llm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "pure_full" / "expanded_dialogues"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "generated_requirements"
DEFAULT_PROMPT = ROOT / "prompts" / "dialogue_to_full_requirements_gemini.txt"
DEFAULT_SCHEMA = ROOT / "schemas" / "gemini_full_requirements_response.schema.json"
SCOPED_DIALOGUE_MAX_TURNS = 12
SCOPED_DIALOGUE_MAX_CHARS = 5000
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
NFR_CATEGORIES = {
    "performance",
    "security",
    "reliability",
    "availability",
    "usability",
    "maintainability",
    "portability",
    "compliance",
}
MODAL_RE = re.compile(r"\b(shall|must|should|can|may|will)\b", flags=re.IGNORECASE)
GENERIC_MARKERS = {
    "quickly",
    "efficiently",
    "robust",
    "scalable",
    "secure",
    "good performance",
    "user friendly",
    "easy to use",
    "a lot of data",
    "a lot of users",
}
PROPOSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["project_summary", "propositions"],
    "properties": {
        "project_summary": {"type": "string", "minLength": 1},
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "category", "text", "priority", "evidence_turns", "source_unit_ids"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "enum": REQUIREMENT_KEYS},
                    "nfr_category": {"type": "string", "enum": sorted(NFR_CATEGORIES)},
                    "text": {"type": "string", "minLength": 1},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence_turns": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                    },
                    "source_unit_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                },
            },
        },
    },
}
REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["propositions"],
    "properties": {
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_period(text: str) -> str:
    text = clean_text(text).strip()
    if not text:
        return ""
    text = text.rstrip(".").strip()
    return text + "."


def dedup_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def is_meta_disclaimer(text: str) -> bool:
    lowered = clean_text(text).lower()
    if not lowered:
        return True
    markers = [
        "source requirements",
        "provided source requirements",
        "requirements do not mention",
        "not specified in the requirements",
        "not mentioned in the requirements",
        "not explicitly mentioned",
        "not explicitly stated",
        "this project is based on the",
        "capture the main requirements and constraints",
    ]
    return any(marker in lowered for marker in markers)


def normalize_req_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    match = re.match(r"^\s*the system shall\s+(?P<rest>.+)$", text, flags=re.IGNORECASE)
    if match:
        rest = clean_text(match.group("rest"))
        if MODAL_RE.search(rest) and not rest.lower().startswith("the system shall"):
            text = rest
    if MODAL_RE.search(text):
        return ensure_period(text)
    if re.match(
        r"^(allow|provide|support|enable|prevent|log|store|send|receive|export|import|generate|manage|configure|display|calculate|compute|compress|operate|run|perform)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ensure_period(f"The system shall {text}")
    return ensure_period(text)


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
        text_key = dedup_text_key(text)
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        priority = clean_text(item.get("priority", "medium")).lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        evidence_turns = item.get("evidence_turns", [])
        if not isinstance(evidence_turns, list):
            evidence_turns = []
        evidence_turns = sorted({int(turn) for turn in evidence_turns if isinstance(turn, int) and turn >= 1})
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
            normalized_item["category"] = category if category in NFR_CATEGORIES else "performance"
        normalized.append(normalized_item)
    return normalized


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
    seen = set()
    for category in REQUIREMENT_KEYS:
        for item in normalized[category]:
            key = dedup_text_key(item.get("text", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            deduped[category].append(item)
    for category in REQUIREMENT_KEYS:
        for index, item in enumerate(deduped[category], start=1):
            item["id"] = f"{CATEGORY_PREFIXES[category]}-{index:03d}"
    return deduped


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


def render_turns(turns: list[dict]) -> str:
    return "\n".join(f"{turn['turn_id']}. {turn['role']}: {turn['text']}" for turn in turns)


def render_dialogue(sample: dict) -> str:
    return render_turns(sample.get("dialogue", []))


def relevant_dialogue_turns(
    sample: dict,
    evidence_units: list[dict],
    *,
    max_turns: int = SCOPED_DIALOGUE_MAX_TURNS,
    max_chars: int = SCOPED_DIALOGUE_MAX_CHARS,
) -> list[dict]:
    dialogue = [turn for turn in sample.get("dialogue", []) if isinstance(turn, dict)]
    if not dialogue:
        return []

    trace = sample.get("dialogue_generation", {}).get("trace", [])
    bot_turn_by_user_turn = {}
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            user_turn_id = item.get("user_turn_id")
            bot_turn_id = item.get("bot_turn_id")
            if isinstance(user_turn_id, int) and isinstance(bot_turn_id, int):
                bot_turn_by_user_turn[user_turn_id] = bot_turn_id

    selected_turn_ids = set()
    for unit in evidence_units:
        turn_id = unit.get("turn_id")
        if not isinstance(turn_id, int):
            continue
        selected_turn_ids.add(turn_id)
        bot_turn_id = bot_turn_by_user_turn.get(turn_id)
        if isinstance(bot_turn_id, int):
            selected_turn_ids.add(bot_turn_id)
        elif turn_id > 1:
            selected_turn_ids.add(turn_id - 1)

    selected = [turn for turn in dialogue if turn.get("turn_id") in selected_turn_ids]
    selected.sort(key=lambda turn: turn.get("turn_id", 0))
    if max_turns > 0 and len(selected) > max_turns:
        selected = selected[-max_turns:]

    while selected:
        rendered = render_turns(selected)
        if max_chars <= 0 or len(rendered) <= max_chars:
            return selected
        selected = selected[1:]
    return []


def render_dialogue_context(sample: dict, evidence_units: list[dict], *, extraction_mode: str) -> str:
    if extraction_mode == "full_context":
        return render_dialogue(sample)
    scoped_turns = relevant_dialogue_turns(sample, evidence_units)
    if scoped_turns:
        return render_turns(scoped_turns)
    return render_dialogue(sample)


def render_evidence_units(units: list[dict]) -> str:
    lines = []
    for unit in units:
        theme = unit.get("trace_theme") or "unknown"
        lines.append(
            f"- {unit['unit_id']} | turn {unit.get('turn_id')} | theme {theme} | "
            f"span: {unit.get('text', '')} | context: {unit.get('context_text', '')}"
        )
    return "\n".join(lines)


def build_prompt(
    template: str,
    sample: dict,
    evidence_units: list[dict],
    memory_text: str,
    *,
    batch_index: int,
    batch_count: int,
    extraction_mode: str,
) -> str:
    prompt = template
    replacements = {
        "{{SAMPLE_ID}}": sample["sample_id"],
        "{{DOMAIN_HINT}}": sample.get("metadata", {}).get("domain", "pure_document"),
        "{{PREVIOUS_PROPOSITIONS}}": memory_text if memory_text else "None yet.",
        "{{CHUNK_INDEX}}": str(batch_index),
        "{{CHUNK_COUNT}}": str(batch_count),
        "{{EXTRACTION_MODE}}": extraction_mode,
        "{{EVIDENCE_BANK}}": render_evidence_units(evidence_units),
        "{{DIALOGUE}}": render_dialogue_context(sample, evidence_units, extraction_mode=extraction_mode),
    }
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def shared_cache_prefix(prompt: str) -> str | None:
    marker = "Previously Extracted Propositions:"
    index = prompt.find(marker)
    if index <= 0:
        return None
    return prompt[:index]


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


def load_final_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_bank_from_sample(sample: dict, scorer: CoverageScorer) -> list[dict]:
    units = scorer.build_dialogue_support_units(sample, user_only=True)
    for unit in units:
        unit["category_hint"] = unit.get("trace_theme") or "other_constraints"
    return units


def build_theme_batches(units: list[dict], *, top_k: int, overlap: int, chunked: bool) -> list[dict]:
    if not units:
        return []
    if not chunked:
        return [{"batch_id": "full", "theme": "all", "units": units}]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        grouped[unit.get("trace_theme") or "other_constraints"].append(unit)
    if len(grouped) > 1 and "goal_scope" in grouped:
        grouped.pop("goal_scope", None)

    batches = []
    for theme, theme_units in sorted(
        grouped.items(),
        key=lambda item: (
            min((unit.get("turn_id") or 0) for unit in item[1]),
            item[0],
        ),
    ):
        ordered = sorted(theme_units, key=lambda unit: ((unit.get("turn_id") or 0), unit.get("sentence_index") or 0))
        if len(ordered) <= top_k:
            batches.append({"batch_id": f"{theme}:1", "theme": theme, "units": ordered})
            continue
        step = max(1, top_k - max(0, overlap))
        start = 0
        batch_num = 1
        while start < len(ordered):
            window = ordered[start:start + top_k]
            if not window:
                break
            batches.append({"batch_id": f"{theme}:{batch_num}", "theme": theme, "units": window})
            if start + top_k >= len(ordered):
                break
            start += step
            batch_num += 1
    return batches


def infer_bucket_from_theme(theme: str | None) -> tuple[str, str | None]:
    theme = clean_text(theme or "").lower()
    if theme == "security_audit":
        return "non_functional", "security"
    if theme == "performance_capacity":
        return "non_functional", "performance"
    if theme == "availability_reliability":
        return "non_functional", "reliability"
    if theme == "usability_help_accessibility":
        return "non_functional", "usability"
    if theme == "maintainability_portability_testability":
        return "non_functional", "maintainability"
    if theme == "data_validation":
        return "data", None
    if theme == "interfaces_integrations":
        return "interfaces", None
    if theme == "workflows_business_rules":
        return "business_rules", None
    if theme == "deployment_environment_constraints":
        return "constraints", None
    if theme == "reporting_documentation":
        return "constraints", None
    return "functional", None


def heuristic_propositions_from_units(units: list[dict]) -> dict:
    propositions = []
    seen = set()
    for unit in units:
        text = clean_text(unit.get("text", ""))
        if not text or len(text.split()) < 5:
            continue
        if is_meta_disclaimer(text):
            continue
        if not MODAL_RE.search(text) and not re.search(
            r"\b(allow|provide|support|enable|view|search|register|log|track|display|manage|send|export|store|configure|browser|accessible|defaults|profile)\b",
            text,
            flags=re.IGNORECASE,
        ):
            continue
        normalized = normalize_req_text(text)
        key = dedup_text_key(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        bucket, nfr_category = infer_bucket_from_theme(unit.get("trace_theme"))
        proposition = {
            "id": f"P-{len(propositions) + 1:03d}",
            "category": bucket,
            "text": normalized,
            "priority": "medium",
            "evidence_turns": [int(unit.get("turn_id") or 1)],
            "source_unit_ids": [unit["unit_id"]],
        }
        if bucket == "non_functional":
            proposition["nfr_category"] = nfr_category or "performance"
        propositions.append(proposition)
    return {
        "project_summary": "Heuristic proposition fallback generated from evidence units.",
        "propositions": propositions,
    }


def normalize_proposition(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    category = clean_text(item.get("category", "functional")).lower()
    if category not in REQUIREMENT_KEYS:
        category = "functional"
    raw_text = clean_text(item.get("text", ""))
    if is_meta_disclaimer(raw_text):
        return None
    text = normalize_req_text(raw_text)
    if not text:
        return None
    priority = clean_text(item.get("priority", "medium")).lower()
    if priority not in {"high", "medium", "low"}:
        priority = "medium"
    evidence_turns = item.get("evidence_turns", [])
    if not isinstance(evidence_turns, list):
        evidence_turns = []
    evidence_turns = sorted({int(turn) for turn in evidence_turns if isinstance(turn, int) and turn >= 1})
    if not evidence_turns:
        evidence_turns = [2]
    source_unit_ids = item.get("source_unit_ids", [])
    if not isinstance(source_unit_ids, list):
        source_unit_ids = []
    source_unit_ids = [clean_text(unit_id) for unit_id in source_unit_ids if clean_text(unit_id)]
    if not source_unit_ids:
        source_unit_ids = [f"{evidence_turns[0]}:1"]
    proposition = {
        "id": clean_text(item.get("id", "")) or f"P-{sha256_text(text)[:8]}",
        "category": category,
        "text": text,
        "priority": priority,
        "evidence_turns": evidence_turns,
        "source_unit_ids": source_unit_ids,
    }
    if category == "non_functional":
        nfr_category = clean_text(item.get("nfr_category", "performance")).lower()
        proposition["nfr_category"] = nfr_category if nfr_category in NFR_CATEGORIES else "performance"
    return proposition


def normalize_proposition_payload(payload: dict) -> dict:
    propositions = []
    for item in payload.get("propositions", []):
        normalized = normalize_proposition(item)
        if normalized is not None:
            propositions.append(normalized)
    project_summary = clean_text(payload.get("project_summary", "")) or "Generated from elicitation dialogue."
    return {"project_summary": project_summary, "propositions": propositions}


def specificity_score(text: str) -> tuple[int, int]:
    digit_count = sum(ch.isdigit() for ch in text)
    uppercase_count = len(re.findall(r"\b[A-Z][A-Za-z0-9\-\.]+\b", text))
    return digit_count + uppercase_count, len(text)


def merge_proposition_entries(
    propositions: list[dict],
    scorer: CoverageScorer,
    *,
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    merged: list[dict] = []
    removed = []
    for proposition in propositions:
        text = proposition["text"]
        matched_index = None
        matched_score = 0.0
        for index, kept in enumerate(merged):
            score = scorer.similarity_row(text, [kept["text"]])[0]
            if score >= threshold:
                matched_index = index
                matched_score = float(score)
                break
        if matched_index is None:
            merged.append(dict(proposition))
            continue
        kept = merged[matched_index]
        kept["evidence_turns"] = sorted(set(kept["evidence_turns"]) | set(proposition["evidence_turns"]))
        kept["source_unit_ids"] = sorted(set(kept["source_unit_ids"]) | set(proposition["source_unit_ids"]))
        if specificity_score(proposition["text"]) > specificity_score(kept["text"]):
            kept["text"] = proposition["text"]
            kept["category"] = proposition["category"]
            if proposition.get("nfr_category"):
                kept["nfr_category"] = proposition["nfr_category"]
        removed.append(
            {
                "removed": proposition["text"],
                "kept": kept["text"],
                "similarity": round(matched_score, 3),
                "method": scorer.similarity_method,
            }
        )
    return merged, removed


def update_memory_entries(memory_entries: list[dict], propositions: list[dict]) -> None:
    seen = {dedup_text_key(entry.get("text", "")) for entry in memory_entries}
    for proposition in propositions:
        key = dedup_text_key(proposition.get("text", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        memory_entries.append(
            {
                "category": proposition["category"],
                "text": proposition["text"],
            }
        )


def format_memory_entries(memory_entries: list[dict]) -> str:
    if not memory_entries:
        return "None yet."
    lines = []
    for category in REQUIREMENT_KEYS:
        category_entries = [entry for entry in memory_entries if entry.get("category") == category]
        if not category_entries:
            continue
        lines.append(f"{MEMORY_CATEGORY_LABELS[category]}:")
        for entry in category_entries:
            lines.append(f"- {entry['text']}")
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


def build_repair_prompt(schema_text: str, invalid_text: str) -> str:
    return (
        "Repair the following invalid JSON so it matches the schema exactly.\n"
        "Return JSON only.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Invalid JSON:\n{invalid_text}\n"
    )


def proposition_output_cap(evidence_units: list[dict]) -> int:
    # Bound local JSON generations so one batch cannot monopolize the entire run.
    return max(512, min(1400, 128 + (96 * max(1, len(evidence_units)))))


def proposition_timeout_seconds(evidence_units: list[dict]) -> int:
    return min(240, max(90, 45 + (12 * max(1, len(evidence_units)))))


def extract_batch_propositions(
    prompt: str,
    *,
    runs: int,
    temperature: float,
    schema_text: str,
    evidence_units: list[dict],
    scorer: CoverageScorer,
    proposition_dedup_threshold: float,
    cache_namespace: str | None = None,
) -> tuple[dict, list[dict], list[dict], str]:
    candidates = []
    usages = []
    raw_responses = []
    invalid_text = ""
    cache_prefix = shared_cache_prefix(prompt) if runs >= 2 else None
    output_cap = proposition_output_cap(evidence_units)
    timeout_seconds = proposition_timeout_seconds(evidence_units)
    try:
        for _run_idx in range(runs):
            response = llm.generate_json(
                prompt,
                PROPOSITION_SCHEMA,
                temperature=temperature,
                cache_prefix=cache_prefix,
                cache_namespace=cache_namespace,
                max_output_tokens=output_cap,
                timeout_seconds=timeout_seconds,
            )
            invalid_text = response.text
            parsed = llm.parse_first_json_object(response.text)
            candidates.append(normalize_proposition_payload(parsed))
            usages.append(response.usage)
            raw_responses.append(response.raw_response)
    except Exception as first_error:  # noqa: BLE001
        try:
            repair_prompt = build_repair_prompt(schema_text, invalid_text or str(first_error))
            response = llm.generate_json(
                repair_prompt,
                PROPOSITION_SCHEMA,
                temperature=0.0,
                cache_namespace=cache_namespace,
                max_output_tokens=output_cap,
                timeout_seconds=timeout_seconds,
            )
            parsed = llm.parse_first_json_object(response.text)
            candidates = [normalize_proposition_payload(parsed)]
            usages.append(response.usage)
            raw_responses.append(response.raw_response)
            status = "repaired"
        except Exception:
            fallback = heuristic_propositions_from_units(evidence_units)
            return fallback, usages, raw_responses, "fallback_heuristic"
        else:
            propositions, _ = merge_proposition_entries(
                [item for candidate in candidates for item in candidate["propositions"]],
                scorer,
                threshold=proposition_dedup_threshold,
            )
            return {"project_summary": candidates[0]["project_summary"], "propositions": propositions}, usages, raw_responses, status

    propositions, _ = merge_proposition_entries(
        [item for candidate in candidates for item in candidate["propositions"]],
        scorer,
        threshold=proposition_dedup_threshold,
    )
    project_summary = next((candidate["project_summary"] for candidate in candidates if candidate["project_summary"]), "Generated from elicitation dialogue.")
    return {"project_summary": project_summary, "propositions": propositions}, usages, raw_responses, f"ok_self_consistency_{runs}" if runs > 1 else "ok"


def is_generic_proposition(proposition: dict, unit_lookup: dict[str, dict]) -> bool:
    text = clean_text(proposition.get("text", "")).lower()
    if any(marker in text for marker in GENERIC_MARKERS):
        return True
    source_units = [unit_lookup.get(unit_id) for unit_id in proposition.get("source_unit_ids", [])]
    source_context = " ".join(clean_text(unit.get("context_text", "")) for unit in source_units if unit)
    if any(ch.isdigit() for ch in source_context) and not any(ch.isdigit() for ch in text):
        return True
    return False


def rewrite_generic_propositions(
    propositions: list[dict],
    unit_lookup: dict[str, dict],
    *,
    cache_namespace: str | None = None,
) -> tuple[list[dict], int, list[dict]]:
    flagged = [item for item in propositions if is_generic_proposition(item, unit_lookup)]
    if not flagged:
        return propositions, 0, []

    evidence_lines = []
    for item in flagged:
        units = [unit_lookup.get(unit_id) for unit_id in item["source_unit_ids"]]
        context = " | ".join(clean_text(unit.get("context_text", "")) for unit in units if unit)
        evidence_lines.append(f"- {item['id']}: {item['text']} || evidence: {context}")

    prompt = (
        "Rewrite the following propositions so they preserve exact technical details from the evidence. "
        "Do not invent new details and do not add or remove propositions.\n\n"
        "Return JSON only.\n\n"
        + "\n".join(evidence_lines)
    )
    rewritten = []
    try:
        response = llm.generate_json(
            prompt,
            REWRITE_SCHEMA,
            temperature=0.0,
            cache_namespace=cache_namespace,
            max_output_tokens=1024,
            timeout_seconds=120,
        )
        parsed = llm.parse_first_json_object(response.text)
        by_id = {clean_text(item.get("id", "")): clean_text(item.get("text", "")) for item in parsed.get("propositions", []) if isinstance(item, dict)}
        for proposition in propositions:
            updated = dict(proposition)
            candidate = by_id.get(proposition["id"])
            if candidate:
                updated["text"] = normalize_req_text(candidate)
            rewritten.append(updated)
        return rewritten, len(flagged), [response.raw_response]
    except Exception:
        return propositions, 0, []


def novelty_gap_units(units: list[dict], propositions: list[dict], scorer: CoverageScorer, *, top_k: int) -> list[dict]:
    if not units:
        return []
    if not propositions:
        return units[:top_k]
    proposition_texts = [item["text"] for item in propositions]
    ranked = []
    for unit in units:
        scores = scorer.similarity_row(unit["context_text"], proposition_texts)
        best_score = max(scores) if scores else 0.0
        ranked.append((best_score, unit))
    ranked.sort(key=lambda item: item[0])
    return [unit for _, unit in ranked[:top_k]]


def propositions_to_requirements(propositions: list[dict]) -> dict:
    grouped = {key: [] for key in REQUIREMENT_KEYS}
    seen = set()
    for proposition in propositions:
        text = proposition["text"]
        key = dedup_text_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        category = proposition["category"]
        item = {
            "id": "",
            "text": text,
            "priority": proposition["priority"],
            "evidence_turns": proposition["evidence_turns"],
        }
        if category == "non_functional":
            item["category"] = proposition.get("nfr_category", "performance")
        grouped[category].append(item)
    for category in REQUIREMENT_KEYS:
        for index, item in enumerate(grouped[category], start=1):
            item["id"] = f"{CATEGORY_PREFIXES[category]}-{index:03d}"
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--self-consistency", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--chunk-dialogues", action="store_true")
    parser.add_argument("--dialogue-chunk-overlap-turns", type=int, default=2)
    parser.add_argument("--memory-max-items", type=int, default=24)
    parser.add_argument("--memory-max-chars", type=int, default=3500)
    parser.add_argument("--retrieval-top-k", type=int, default=14)
    parser.add_argument("--proposition-dedup-threshold", type=float, default=0.90)
    parser.add_argument("--gap-pass-top-k", type=int, default=10)
    parser.add_argument("--extraction-mode", choices=["evidence_bank", "full_context"], default="evidence_bank")
    parser.add_argument("--enable-gap-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    final_schema = load_final_schema(args.schema_path)
    prompt_hash = sha256_text(prompt_template)
    final_schema_hash = sha256_text(json.dumps(final_schema, indent=2, ensure_ascii=False))
    proposition_schema_text = json.dumps(PROPOSITION_SCHEMA, indent=2, ensure_ascii=False)
    scorer = CoverageScorer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.output_dir / "_artifacts"
    evidence_dir = artifacts_dir / "evidence_bank"
    proposition_dir = artifacts_dir / "propositions"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    proposition_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "sample_count": len(samples),
                    "prompt_hash": prompt_hash,
                    "final_schema_hash": final_schema_hash,
                    "proposition_schema_hash": sha256_text(proposition_schema_text),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    model_name = llm.active_model_name()
    summary = []
    for sample in samples:
        evidence_bank = evidence_bank_from_sample(sample, scorer)
        evidence_lookup = {item["unit_id"]: item for item in evidence_bank}
        evidence_path = evidence_dir / f"{sample['sample_id']}.json"
        evidence_path.write_text(json.dumps(evidence_bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if args.extraction_mode == "full_context":
            batches = [{"batch_id": "full", "theme": "all", "units": evidence_bank}]
        else:
            batches = build_theme_batches(
                evidence_bank,
                top_k=args.retrieval_top_k,
                overlap=args.dialogue_chunk_overlap_turns,
                chunked=bool(args.chunk_dialogues),
            )
        if not batches:
            batches = [{"batch_id": "full", "theme": "all", "units": evidence_bank}]

        memory_entries: list[dict] = []
        batch_statuses = []
        raw_responses = []
        usages = []
        all_propositions: list[dict] = []
        project_summary = "Generated from elicitation dialogue."

        for batch_index, batch in enumerate(batches, start=1):
            batch_sample = dict(sample)
            batch_sample["dialogue"] = sample.get("dialogue", [])
            prompt = build_prompt(
                prompt_template,
                batch_sample,
                batch["units"],
                render_memory_text(
                    memory_entries,
                    max_items=max(0, args.memory_max_items),
                    max_chars=max(0, args.memory_max_chars),
                ),
                batch_index=batch_index,
                batch_count=len(batches),
                extraction_mode=args.extraction_mode,
            )
            payload, batch_usages, batch_raw_responses, status = extract_batch_propositions(
                prompt,
                runs=max(1, int(args.self_consistency)),
                temperature=args.temperature,
                schema_text=proposition_schema_text,
                evidence_units=batch["units"],
                scorer=scorer,
                proposition_dedup_threshold=args.proposition_dedup_threshold,
                cache_namespace=f"{sample['sample_id']}-{args.extraction_mode}",
            )
            if payload["project_summary"] and project_summary == "Generated from elicitation dialogue.":
                project_summary = payload["project_summary"]
            all_propositions.extend(payload["propositions"])
            merged_now, _ = merge_proposition_entries(
                all_propositions,
                scorer,
                threshold=args.proposition_dedup_threshold,
            )
            all_propositions = merged_now
            update_memory_entries(memory_entries, payload["propositions"])
            batch_statuses.append(f"{batch['batch_id']}:{status}")
            usages.append({"batch_id": batch["batch_id"], "usage": batch_usages})
            raw_responses.append({"batch_id": batch["batch_id"], "responses": batch_raw_responses})

        all_propositions, removed_duplicates = merge_proposition_entries(
            all_propositions,
            scorer,
            threshold=args.proposition_dedup_threshold,
        )

        rewritten_propositions, rewrite_count, rewrite_raw = rewrite_generic_propositions(
            all_propositions,
            evidence_lookup,
            cache_namespace=f"{sample['sample_id']}-rewrite",
        )
        all_propositions = rewritten_propositions
        if rewrite_raw:
            raw_responses.append({"batch_id": "rewrite", "responses": rewrite_raw})

        gap_pass_added_count = 0
        if args.enable_gap_pass and evidence_bank:
            gap_units = novelty_gap_units(evidence_bank, all_propositions, scorer, top_k=args.gap_pass_top_k)
            if gap_units:
                gap_prompt = build_prompt(
                    prompt_template,
                    sample,
                    gap_units,
                    render_memory_text(memory_entries, max_items=args.memory_max_items, max_chars=args.memory_max_chars),
                    batch_index=len(batches) + 1,
                    batch_count=len(batches) + 1,
                    extraction_mode=f"{args.extraction_mode}_gap_pass",
                )
                gap_payload, gap_usages, gap_raw, gap_status = extract_batch_propositions(
                    gap_prompt,
                    runs=max(1, int(args.self_consistency)),
                    temperature=args.temperature,
                    schema_text=proposition_schema_text,
                    evidence_units=gap_units,
                    scorer=scorer,
                    proposition_dedup_threshold=args.proposition_dedup_threshold,
                    cache_namespace=f"{sample['sample_id']}-{args.extraction_mode}",
                )
                before = len(all_propositions)
                all_propositions.extend(gap_payload["propositions"])
                all_propositions, _ = merge_proposition_entries(
                    all_propositions,
                    scorer,
                    threshold=args.proposition_dedup_threshold,
                )
                gap_pass_added_count = max(0, len(all_propositions) - before)
                batch_statuses.append(f"gap_pass:{gap_status}")
                usages.append({"batch_id": "gap_pass", "usage": gap_usages})
                raw_responses.append({"batch_id": "gap_pass", "responses": gap_raw})

        final_requirements = propositions_to_requirements(all_propositions)
        proposition_payload = {
            "sample_id": sample["sample_id"],
            "project_summary": project_summary,
            "propositions": all_propositions,
            "removed_duplicates": removed_duplicates,
        }
        proposition_path = proposition_dir / f"{sample['sample_id']}.json"
        proposition_path.write_text(json.dumps(proposition_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        output_payload = {
            "sample_id": sample["sample_id"],
            "source": sample.get("source"),
            "method": "g_evidence_bank_proposition_pipeline_v2",
            "model": model_name,
            "prompt_hash": prompt_hash,
            "schema_hash": final_schema_hash,
            "parse_status": "ok" if batch_statuses else "failed",
            "usage": {"batches": usages},
            "chunking": {
                "enabled": bool(args.chunk_dialogues),
                "segment_count": len(batches),
                "segment_statuses": batch_statuses,
                "retrieval_top_k": args.retrieval_top_k,
                "overlap": args.dialogue_chunk_overlap_turns,
                "extraction_mode": args.extraction_mode,
            },
            "memory": {
                "enabled": True,
                "max_items": args.memory_max_items,
                "max_chars": args.memory_max_chars,
                "stored_unique_requirements": len(memory_entries),
            },
            "diagnostics": {
                "evidence_bank_count": len(evidence_bank),
                "proposition_count": len(all_propositions),
                "removed_duplicate_propositions": len(removed_duplicates),
                "rewrite_candidate_count": rewrite_count,
                "gap_pass_added_count": gap_pass_added_count,
                "grounded_after_rewrite_count": None,
            },
            "artifacts": {
                "evidence_bank_path": str(evidence_path.relative_to(args.output_dir)),
                "proposition_path": str(proposition_path.relative_to(args.output_dir)),
            },
            "project_summary": project_summary,
            "requirements": final_requirements,
        }
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        raw_path = args.output_dir / f"{sample['sample_id']}.raw_response.json"
        raw_path.write_text(json.dumps({"batches": raw_responses}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(output_path),
                "parse_status": output_payload["parse_status"],
                "segment_count": len(batches),
                "requirement_count": sum(len(final_requirements[key]) for key in REQUIREMENT_KEYS),
                "evidence_bank_count": len(evidence_bank),
                "proposition_count": len(all_propositions),
                "gap_pass_added_count": gap_pass_added_count,
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
