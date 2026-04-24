#!/usr/bin/env python3
"""Generate controlled elicitation dialogues from PURE requirements.

This replaces the single free-form "whole document -> whole dialogue" prompt with:
1. heuristic theme grouping
2. fixed bot questions
3. Gemini paraphrase on small requirement chunks
4. coverage guardrail with deterministic fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import llm_router as llm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "expanded_dialogues"
DEFAULT_PROMPT = ROOT / "prompts" / "pure_requirement_group_to_answer.txt"

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": {
            "type": "string",
            "minLength": 1,
        }
    },
}

THEME_ORDER = [
    "goal_scope",
    "user_roles_permissions",
    "functional_capabilities",
    "workflows_business_rules",
    "data_validation",
    "interfaces_integrations",
    "performance_capacity",
    "availability_reliability",
    "security_audit",
    "usability_help_accessibility",
    "deployment_environment_constraints",
    "maintainability_portability_testability",
    "reporting_documentation",
    "other_constraints",
]

QUESTION_MAP = {
    "goal_scope": "To start, what is the overall purpose and scope of this system?",
    "user_roles_permissions": "Who are the main users or roles, and what access responsibilities do they have?",
    "functional_capabilities": "What main actions and services does the system need to support?",
    "workflows_business_rules": "Are there any workflow rules or business rules we need to capture?",
    "data_validation": "What data, storage, or validation requirements do you have?",
    "interfaces_integrations": "Does the system need any external interfaces, communications, or integration features?",
    "performance_capacity": "What performance, timing, or capacity targets do you have?",
    "availability_reliability": "What availability, recovery, or reliability expectations should the system meet?",
    "security_audit": "What security, authorization, or audit requirements do you have?",
    "usability_help_accessibility": "What usability, help, accessibility, or interface quality expectations do you have?",
    "deployment_environment_constraints": "What operating environment, platform, deployment, or technical constraints apply?",
    "maintainability_portability_testability": "Are there maintainability, portability, upgrade, or testability constraints?",
    "reporting_documentation": "What reporting or documentation requirements should we capture?",
    "other_constraints": "Are there any additional technical or operational constraints we have not covered yet?",
}

FOLLOWUP_QUESTION_MAP = {
    theme: f"What else should I capture about {theme.replace('_', ' ')}?"
    for theme in QUESTION_MAP
}

CLARIFICATION_QUESTION_MAP = {
    "goal_scope": "Before we finish, is there any additional scope or purpose detail that still needs to be captured?",
    "user_roles_permissions": "Before we finish, are there any user roles, permissions, or access rules we have not captured yet?",
    "functional_capabilities": "Before we finish, what other system functions or user actions still need to be captured?",
    "workflows_business_rules": "Before we finish, are there any remaining workflow steps or business rules we have not covered yet?",
    "data_validation": "Before we finish, are there any remaining data, storage, or validation rules we still need to capture?",
    "interfaces_integrations": "Before we finish, are there any remaining interfaces, communications, or integrations we still need to capture?",
    "performance_capacity": "Before we finish, are there any remaining performance, timing, or capacity targets we have not captured yet?",
    "availability_reliability": "Before we finish, are there any remaining availability, recovery, or reliability expectations we have not captured yet?",
    "security_audit": "Before we finish, are there any remaining security, authorization, or audit requirements we have not captured yet?",
    "usability_help_accessibility": "Before we finish, are there any remaining usability, help, or accessibility expectations we have not captured yet?",
    "deployment_environment_constraints": "Before we finish, are there any remaining deployment, platform, or technical environment constraints we have not captured yet?",
    "maintainability_portability_testability": "Before we finish, are there any remaining maintainability, portability, upgrade, or testability constraints we have not captured yet?",
    "reporting_documentation": "Before we finish, are there any remaining reporting or documentation requirements we have not captured yet?",
    "other_constraints": "Before we finish, are there any remaining technical or operational constraints we have not captured yet?",
}

THEME_KEYWORDS = {
    "performance_capacity": [
        "less than",
        "under ",
        "per second",
        "ms",
        "milliseconds",
        "seconds",
        "concurrent",
        "concurrently",
        "response time",
        "throughput",
        "performance",
        "latency",
    ],
    "availability_reliability": [
        "availability",
        "uptime",
        "reliability",
        "reliable",
        "restore",
        "recovery",
        "recover",
        "backup",
        "backed up",
        "failover",
        "operational in less than",
    ],
    "security_audit": [
        "security",
        "secure",
        "encrypt",
        "https",
        "password",
        "login",
        "log in",
        "authentication",
        "authorization",
        "authorisation",
        "audit",
        "audit trail",
        "fraud",
        "unauthorised",
        "unauthorized",
        "firewall",
        "access rights",
        "privileges",
        "super-user",
        "security attributes",
    ],
    "usability_help_accessibility": [
        "help material",
        "online help",
        "usability",
        "error messages",
        "special needs",
        "look and feel",
        "horizontal scrolling",
        "input devices",
        "navigation",
        "customizable",
        "configurable",
        "user interface",
        "interface rules",
        "accessibility",
        "meaningful",
        "controls",
    ],
    "deployment_environment_constraints": [
        "operating environment",
        "operate on",
        "browser",
        "internet explorer",
        "netscape",
        "slackware",
        "apache",
        "intel based",
        "hardware",
        "usb",
        "deployment",
        "technical constraints",
        "shortest path algorithm",
    ],
    "maintainability_portability_testability": [
        "maintainability",
        "portable",
        "portability",
        "migrate",
        "upgrade",
        "updatable",
        "patches",
        "debug mode",
        "plugins",
        "interchangeable",
        "testability",
        "easy to upgrade",
        "easy to migrate",
    ],
    "reporting_documentation": [
        "report",
        "reports",
        "manual",
        "guide book",
        "guide",
        "documentation",
        "installation instructions",
        "operations and maintenance",
        "users guide",
    ],
    "interfaces_integrations": [
        "interface",
        "interfaces",
        "email",
        "sms",
        "plug-ins",
        "plugin interface",
        "web interface",
        "communicat",
        "tracking number",
        "technical queries",
    ],
    "data_validation": [
        "database",
        "data",
        "store",
        "stored",
        "capture",
        "records",
        "inventory",
        "validate",
        "validation",
        "entities",
        "suspect",
        "property",
        "credit card",
        "email address",
    ],
    "workflows_business_rules": [
        "if ",
        "when ",
        "after ",
        "before ",
        "based on",
        "thereafter",
        "returning customers",
        "chooses to",
        "selectable",
        "must allow only",
        "shall take",
        "workflow",
        "business rule",
    ],
    "user_roles_permissions": [
        "citizens",
        "citizen",
        "police",
        "help-desk",
        "admin-users",
        "administrator",
        "administrators",
        "sales people",
        "sales person",
        "customer",
        "customers",
        "user groups",
        "groups",
        "user profiles",
        "role-based",
        "member of more than one group",
    ],
    "goal_scope": [
        "main goal",
        "overall scope",
        "overall purpose",
        "solution",
        "system goal",
        "scope",
    ],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    stopwords = {
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
    return {token for token in normalize_text(text).split() if token not in stopwords and len(token) > 1}


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


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "sample_id" not in payload or "ground_truth_requirements" not in payload:
            continue
        samples.append(payload)
    return samples


def score_theme(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def classify_requirement(text: str) -> str:
    scores = {theme: score_theme(text, keywords) for theme, keywords in THEME_KEYWORDS.items()}
    best_theme = max(scores, key=scores.get)
    if scores[best_theme] > 0:
        return best_theme

    lowered = text.lower()
    if re.search(r"\b(allow|enable|provide|support|view|search|register|log|track|display|manage|send|export)\b", lowered):
        return "functional_capabilities"
    return "other_constraints"


def group_requirements(sample: dict) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {theme: [] for theme in THEME_ORDER}
    for item in sample["ground_truth_requirements"]:
        theme = classify_requirement(item["text"])
        grouped[theme].append(item)

    ordered = [(theme, grouped[theme]) for theme in THEME_ORDER if grouped[theme]]

    # If we have no clear scope group, start with the first functional chunk as scope context.
    if ordered and ordered[0][0] != "goal_scope":
        ordered.insert(0, ("goal_scope", []))
    return ordered


def group_requirement_items(items: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {theme: [] for theme in THEME_ORDER}
    for item in items:
        theme = classify_requirement(item["text"])
        grouped[theme].append(item)
    return [(theme, grouped[theme]) for theme in THEME_ORDER if grouped[theme]]


def chunk_items(items: list[dict], max_items: int, max_chars: int) -> list[list[dict]]:
    if not items:
        return [[]]
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for item in items:
        item_chars = len(item["text"])
        if current and (len(current) >= max_items or current_chars + item_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        chunks.append(current)
    return chunks


def build_scope_answer(sample: dict) -> str:
    title = clean_text(sample["source"]["title"])
    doc_id = clean_text(sample["source"]["document_id"])
    return f"This project is based on the {title} system, and the goal is to capture the main requirements and constraints for document {doc_id}."


def build_requirements_block(items: list[dict]) -> str:
    return "\n".join(f"- {item['req_id']}: {clean_text(item['text'])}" for item in items)


def build_prompt(template: str, theme: str, question: str, items: list[dict]) -> str:
    return (
        template.replace("{{THEME}}", theme)
        .replace("{{QUESTION}}", question)
        .replace("{{REQUIREMENTS}}", build_requirements_block(items))
    )


def deterministic_answer(items: list[dict]) -> str:
    sentences = []
    for item in items:
        text = clean_text(item["text"]).rstrip(".")
        if not text:
            continue
        sentences.append(text + ".")
    return " ".join(sentences)


def answer_covers_items(answer: str, items: list[dict], threshold: float) -> tuple[bool, float]:
    if not answer.strip():
        return False, 0.0
    scores = [token_f1(item["text"], answer) for item in items]
    covered = sum(score >= threshold for score in scores)
    ratio = covered / len(items) if items else 1.0
    return ratio >= 1.0, ratio


def best_requirement_coverage(text: str, candidate_texts: list[str]) -> float:
    if not candidate_texts:
        return 0.0
    return max(token_f1(text, candidate) for candidate in candidate_texts)


def collect_user_turn_texts(dialogue: list[dict]) -> list[str]:
    return [clean_text(turn.get("text", "")) for turn in dialogue if turn.get("role") == "user" and turn.get("text")]


def find_uncovered_requirements(
    sample: dict,
    dialogue: list[dict],
    threshold: float,
) -> list[dict]:
    user_turn_texts = collect_user_turn_texts(dialogue)
    uncovered = []
    for item in sample["ground_truth_requirements"]:
        best = best_requirement_coverage(item["text"], user_turn_texts)
        if best < threshold:
            uncovered.append({**item, "best_dialogue_coverage": best})
    return uncovered


def generate_answer(
    template: str,
    theme: str,
    question: str,
    items: list[dict],
    threshold: float,
) -> tuple[str, str, float]:
    fallback = deterministic_answer(items)

    prompt = build_prompt(template, theme, question, items)
    try:
        response = llm.generate_json(prompt, ANSWER_SCHEMA, temperature=0.0)
        parsed = llm.parse_first_json_object(response.text)
        answer = clean_text(parsed.get("answer", ""))
        if answer:
            # Always use the LLM answer when it is non-empty.
            # The coverage guardrail (token-F1 threshold) is intentionally
            # disabled here: naturalistic paraphrased speech will always score
            # low against formal requirement text, so the old guardrail was
            # silently replacing every naturalistic answer with a verbatim copy.
            _, ratio = answer_covers_items(answer, items, threshold)
            return answer, llm.provider(), ratio
    except Exception:
        pass
    return fallback, "deterministic_error_fallback", 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-reqs-per-answer", type=int, default=3)
    parser.add_argument("--max-chars-per-answer", type=int, default=650)
    parser.add_argument("--coverage-threshold", type=float, default=0.55)
    parser.add_argument("--clarification-rounds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_dialogue(
    sample: dict,
    template: str,
    max_reqs_per_answer: int,
    max_chars_per_answer: int,
    coverage_threshold: float,
    clarification_rounds: int,
) -> tuple[list[dict], list[dict], dict]:
    dialogue = []
    trace = []
    turn_id = 1

    # Always start with a scope-setting question.
    dialogue.append({"turn_id": turn_id, "role": "bot", "text": QUESTION_MAP["goal_scope"]})
    turn_id += 1
    scope_answer = build_scope_answer(sample)
    dialogue.append({"turn_id": turn_id, "role": "user", "text": scope_answer})
    trace.append(
        {
            "theme": "goal_scope",
            "req_ids": [],
            "bot_turn_id": turn_id - 1,
            "user_turn_id": turn_id,
            "generation_mode": "deterministic_scope",
            "coverage_ratio": 1.0,
            "stage": "initial_scope",
        }
    )
    turn_id += 1

    grouped = group_requirements(sample)
    for theme, items in grouped:
        if theme == "goal_scope":
            continue
        chunks = chunk_items(items, max_reqs_per_answer, max_chars_per_answer)
        for chunk_index, chunk in enumerate(chunks):
            question = QUESTION_MAP[theme] if chunk_index == 0 else FOLLOWUP_QUESTION_MAP[theme]
            dialogue.append({"turn_id": turn_id, "role": "bot", "text": question})
            turn_id += 1
            answer, mode, ratio = generate_answer(template, theme, question, chunk, coverage_threshold)
            dialogue.append({"turn_id": turn_id, "role": "user", "text": answer})
            trace.append(
                {
                    "theme": theme,
                    "req_ids": [item["req_id"] for item in chunk],
                    "bot_turn_id": turn_id - 1,
                    "user_turn_id": turn_id,
                    "generation_mode": mode,
                    "coverage_ratio": ratio,
                    "stage": "initial",
                }
            )
            turn_id += 1

    initial_uncovered = find_uncovered_requirements(sample, dialogue, coverage_threshold)
    clarification_rounds_used = 0

    for round_index in range(clarification_rounds):
        uncovered = find_uncovered_requirements(sample, dialogue, coverage_threshold)
        if not uncovered:
            break
        clarification_rounds_used += 1
        for theme, items in group_requirement_items(uncovered):
            chunks = chunk_items(items, max_reqs_per_answer, max_chars_per_answer)
            for chunk in chunks:
                question = CLARIFICATION_QUESTION_MAP.get(theme, CLARIFICATION_QUESTION_MAP["other_constraints"])
                dialogue.append({"turn_id": turn_id, "role": "bot", "text": question})
                turn_id += 1
                answer, mode, ratio = generate_answer(template, theme, question, chunk, coverage_threshold)
                dialogue.append({"turn_id": turn_id, "role": "user", "text": answer})
                trace.append(
                    {
                        "theme": theme,
                        "req_ids": [item["req_id"] for item in chunk],
                        "bot_turn_id": turn_id - 1,
                        "user_turn_id": turn_id,
                        "generation_mode": mode,
                        "coverage_ratio": ratio,
                        "stage": "clarification",
                        "clarification_round": round_index + 1,
                    }
                )
                turn_id += 1

    final_uncovered = find_uncovered_requirements(sample, dialogue, coverage_threshold)
    coverage_summary = {
        "clarification_rounds_requested": clarification_rounds,
        "clarification_rounds_used": clarification_rounds_used,
        "initial_uncovered_requirement_count": len(initial_uncovered),
        "final_uncovered_requirement_count": len(final_uncovered),
        "final_uncovered_req_ids": [item["req_id"] for item in final_uncovered],
    }
    return dialogue, trace, coverage_summary


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    template = args.prompt_path.read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    schema_hash = sha256_text(json.dumps(ANSWER_SCHEMA, sort_keys=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        preview = {
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "sample_count": len(samples),
            "max_reqs_per_answer": args.max_reqs_per_answer,
            "max_chars_per_answer": args.max_chars_per_answer,
            "coverage_threshold": args.coverage_threshold,
            "clarification_rounds": args.clarification_rounds,
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    model_name = os.environ.get("REQ_OLLAMA_MODEL") if llm.provider() == "ollama" else os.environ.get("REQ_GEMINI_MODEL")

    summary = []
    for sample in samples:
        dialogue, trace, coverage_summary = build_dialogue(
            sample,
            template,
            args.max_reqs_per_answer,
            args.max_chars_per_answer,
            args.coverage_threshold,
            args.clarification_rounds,
        )
        fallback_count = sum(item["generation_mode"] != llm.provider() for item in trace)
        clarification_chunk_count = sum(1 for item in trace if item.get("stage") == "clarification")
        payload = {
            "sample_id": sample["sample_id"],
            "metadata": {
                "domain": "pure_document",
                "source_type": "synthetic",
                "parent_id": sample["sample_id"],
                "split": "benchmark",
                "dialogue_style": "controlled_grouped",
            },
            "source": sample["source"],
            "dialogue": dialogue,
            "dialogue_generation": {
                "method": "g_controlled_grouped_source_to_dialogue_v3",
                "model": model_name,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "max_reqs_per_answer": args.max_reqs_per_answer,
                "max_chars_per_answer": args.max_chars_per_answer,
                "coverage_threshold": args.coverage_threshold,
                "clarification_rounds": args.clarification_rounds,
                "trace": trace,
                "fallback_chunk_count": fallback_count,
                "clarification_chunk_count": clarification_chunk_count,
                "coverage_summary": coverage_summary,
            },
        }
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(output_path),
                "turn_count": len(dialogue),
                "chunk_count": len(trace),
                "fallback_chunk_count": fallback_count,
                "clarification_chunk_count": clarification_chunk_count,
                "final_uncovered_requirement_count": coverage_summary["final_uncovered_requirement_count"],
            }
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
