#!/usr/bin/env python3
"""Generate adaptive elicitation dialogues from PURE requirements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from coverage_scorer import CoverageScorer, average, clean_text
import llm_router as llm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "pure_benchmark" / "source_requirements"
DEFAULT_OUTPUT = ROOT / "outputs" / "pure_full" / "expanded_dialogues"
DEFAULT_PROMPT = ROOT / "prompts" / "pure_requirement_group_to_answer.txt"
DEFAULT_MAX_TURNS = 28
DEFAULT_TARGET_DIALOGUE_RECALL = 0.82
DEFAULT_THEME_MAX_EXCHANGES = 3
THEME_MARGIN = 0.05
QUESTION_ALGORITHM_VERSION = "semantic_gap_llm_v2"
CRITICAL_THEME_RECALL_FLOORS = {
    "user_roles_permissions": 0.60,
    "availability_reliability": 0.60,
    "security_audit": 0.70,
    "interfaces_integrations": 0.65,
    "data_validation": 0.65,
}

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

GAP_QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question"],
    "properties": {
        "question": {
            "type": "string",
            "minLength": 10,
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

PRIORITY_ORDER = [
    "functional_capabilities",
    "usability_help_accessibility",
    "security_audit",
    "data_validation",
    "interfaces_integrations",
    "performance_capacity",
    "workflows_business_rules",
    "user_roles_permissions",
    "availability_reliability",
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

THEME_DESCRIPTIONS = {
    "goal_scope": "Overall purpose, product scope, and the system's main business objective.",
    "user_roles_permissions": "Users, actors, roles, groups, permissions, privileges, and access responsibilities.",
    "functional_capabilities": "Core actions, services, workflows users perform, and system behaviors that deliver business value.",
    "workflows_business_rules": "Decision logic, process rules, sequencing, conditional behavior, and business policies.",
    "data_validation": "Data storage, database needs, records, input validation, user profiles, defaults, and data quality rules.",
    "interfaces_integrations": "Browser access, web interfaces, external systems, APIs, email, SMS, plugins, and communication interfaces.",
    "performance_capacity": "Response times, throughput, concurrency, timing, latency, and capacity requirements.",
    "availability_reliability": "Availability, uptime, recovery, backup, resiliency, and failover expectations.",
    "security_audit": "Authentication, authorization, encryption, audit trails, immutability, login security, and firewall/security controls.",
    "usability_help_accessibility": "Navigation, accessibility, readability, help content, interface consistency, scrolling, text resizing, input independence, and user experience.",
    "deployment_environment_constraints": "Operating systems, browsers, hardware, servers, deployment platforms, and technical environment constraints.",
    "maintainability_portability_testability": "Portability, upgradeability, debugging, maintainability, migration, configurability, and testability constraints.",
    "reporting_documentation": "Reports, manuals, documentation, user guides, and operational instructions.",
    "other_constraints": "Remaining technical, legal, or operational constraints that do not fit another theme.",
}

FOCUS_HINTS = {
    "usability_accessibility": "Focus on usability, accessibility, navigation clarity, text handling, and browser presentation expectations.",
    "audit_immutability": "Focus on whether audit trail data can ever be modified, deleted, or altered by any user.",
    "persistent_defaults": "Focus on persistent defaults, customizable entry values, and whether users can keep their own saved defaults.",
    "user_profile_storage": "Focus on whether personal configuration settings must be stored in the user profile.",
    "browser_interface_support": "Focus on support both inside the application and outside it through a browser interface.",
}

FOCUS_QUESTION_MAP = {
    "usability_accessibility": "What expectations do you have around accessibility, navigation, readability, and how easy the interface should be to use?",
    "audit_immutability": "On the audit side, should that history ever be editable or removable by users, or does it need to stay locked down?",
    "persistent_defaults": "Do users need their own saved defaults or repeated entry values so they do not have to set them every time?",
    "user_profile_storage": "If users customize settings, do those preferences need to be stored in their profile and kept between sessions?",
    "browser_interface_support": "Should people be able to reach the support features only inside the application, or also through a regular browser interface?",
}

QUESTION_ALGORITHM_SUMMARY = (
    "Select the next question by semantically scoring uncovered requirements against dialogue evidence, "
    "ranking uncovered themes by priority and uncovered count, constructing one focused prompt from the "
    "top uncovered requirement snippets plus recent dialogue context, and asking the active LLM for exactly "
    "one natural follow-up question. If global recall is reached but critical themes remain under-covered, "
    "use clarification rounds to force targeted follow-up on those weak clusters. Fixed fallback templates "
    "are used only if JSON generation fails."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def build_scope_answer(sample: dict) -> str:
    title = clean_text(sample["source"]["title"])
    doc_id = clean_text(sample["source"]["document_id"])
    return (
        f"This project is based on the {title} system, and "
        f"the goal is to capture the main requirements and constraints for document {doc_id}."
    )


def build_requirements_block(items: list[dict]) -> str:
    lines = []
    for item in items:
        labels = [item["req_id"]]
        if item.get("primary_theme"):
            labels.append(item["primary_theme"])
        lines.append(f"- {' | '.join(labels)}: {clean_text(item['text'])}")
    return "\n".join(lines)


def build_answer_prompt(template: str, theme: str, question: str, items: list[dict]) -> str:
    return (
        template.replace("{{THEME}}", theme)
        .replace("{{QUESTION}}", question)
        .replace("{{REQUIREMENTS}}", build_requirements_block(items))
    )


def deterministic_answer(items: list[dict]) -> str:
    return " ".join(clean_text(item["text"]).rstrip(".") + "." for item in items if clean_text(item["text"]))


def build_dialogue_payload(dialogue: list[dict], trace: list[dict]) -> dict:
    return {"dialogue": dialogue, "dialogue_generation": {"trace": trace}}


def annotate_requirements(requirements: list[dict], scorer: CoverageScorer) -> list[dict]:
    theme_names = [theme for theme in THEME_ORDER if theme != "goal_scope"]
    theme_texts = [f"{theme}: {THEME_DESCRIPTIONS[theme]}" for theme in theme_names]
    req_texts = [req["text"] for req in requirements]
    matrix = scorer.similarity_matrix(req_texts, theme_texts)

    annotated = []
    for index, req in enumerate(requirements):
        row = matrix[index] if index < len(matrix) else []
        ranked = sorted(
            [
                {"theme": theme_names[theme_index], "score": float(score)}
                for theme_index, score in enumerate(row)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        primary = ranked[0]["theme"] if ranked else "other_constraints"
        primary_score = ranked[0]["score"] if ranked else 0.0
        secondary = None
        secondary_score = 0.0
        if len(ranked) > 1 and primary_score - ranked[1]["score"] < THEME_MARGIN:
            secondary = ranked[1]["theme"]
            secondary_score = ranked[1]["score"]
        annotated.append(
            {
                **req,
                "primary_theme": primary,
                "primary_theme_score": primary_score,
                "secondary_theme": secondary,
                "secondary_theme_score": secondary_score,
            }
        )
    return annotated


def evaluate_requirement_coverage(
    annotated_requirements: list[dict],
    dialogue: list[dict],
    trace: list[dict],
    scorer: CoverageScorer,
    threshold: float,
) -> tuple[list[dict], list[dict], float]:
    payload = build_dialogue_payload(dialogue, trace)
    support_units = scorer.build_dialogue_support_units(payload, user_only=True)
    results = scorer.coverage_against_units(
        [item["text"] for item in annotated_requirements],
        support_units,
        threshold=threshold,
        contextualized=True,
    )

    coverage_results = []
    for req, result in zip(annotated_requirements, results):
        coverage_results.append(
            {
                **req,
                "best_score": float(result["best_score"]),
                "covered": bool(result["covered"]),
                "best_unit": result.get("best_unit"),
            }
        )
    uncovered = [item for item in coverage_results if not item["covered"]]
    recall = 1.0 - (len(uncovered) / len(coverage_results) if coverage_results else 0.0)
    return coverage_results, uncovered, recall


def compute_theme_coverage(coverage_results: list[dict]) -> dict[str, dict]:
    summary = {theme: {"total": 0, "covered": 0} for theme in THEME_ORDER if theme != "goal_scope"}
    for item in coverage_results:
        themes = [item.get("primary_theme")]
        if item.get("secondary_theme"):
            themes.append(item["secondary_theme"])
        seen = set()
        for theme in themes:
            if not theme or theme in seen or theme == "goal_scope":
                continue
            seen.add(theme)
            summary.setdefault(theme, {"total": 0, "covered": 0})
            summary[theme]["total"] += 1
            if item["covered"]:
                summary[theme]["covered"] += 1
    for theme, stats in summary.items():
        total = stats["total"]
        covered = stats["covered"]
        stats["uncovered"] = max(0, total - covered)
        stats["recall"] = covered / total if total else 1.0
    return summary


def focus_label_for_requirement(item: dict) -> str:
    text = clean_text(item.get("text", "")).lower()
    if "audit trail" in text or "unalterable" in text or "cannot be modified" in text or "deleted by any user" in text:
        return "audit_immutability"
    if "user profile" in text:
        return "user_profile_storage"
    if "persistent defaults" in text or "defaults should include" in text or "user-definable values" in text:
        return "persistent_defaults"
    if "browser interface" in text or "outside the application" in text:
        return "browser_interface_support"
    if item.get("primary_theme") == "usability_help_accessibility":
        return "usability_accessibility"
    return item.get("primary_theme") or "other_constraints"


def theme_priority(theme: str) -> int:
    try:
        return PRIORITY_ORDER.index(theme)
    except ValueError:
        return len(PRIORITY_ORDER)


def choose_target_theme(
    uncovered_results: list[dict],
    theme_exchange_counts: Counter,
    theme_max_exchanges: int,
) -> str | None:
    theme_buckets: dict[str, list[dict]] = defaultdict(list)
    for item in uncovered_results:
        primary = item.get("primary_theme")
        secondary = item.get("secondary_theme")
        if primary:
            theme_buckets[primary].append(item)
        if secondary:
            theme_buckets[secondary].append(item)

    ranked = []
    for theme, items in theme_buckets.items():
        uncovered_count = len({item["req_id"] for item in items})
        if uncovered_count == 0:
            continue
        if int(theme_exchange_counts.get(theme, 0)) >= theme_max_exchanges and uncovered_count < 2:
            continue
        avg_gap = average((1.0 - item["best_score"]) for item in items)
        ranked.append((theme_priority(theme), -uncovered_count, -avg_gap, theme))

    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def select_candidate_requirements(
    uncovered_results: list[dict],
    target_theme: str,
    max_reqs_per_answer: int,
    max_chars_per_answer: int,
) -> list[dict]:
    candidates = []
    for item in uncovered_results:
        primary_match = item.get("primary_theme") == target_theme
        secondary_match = item.get("secondary_theme") == target_theme
        if not (primary_match or secondary_match):
            continue
        candidates.append(
            (
                0 if primary_match else 1,
                item["best_score"],
                len(clean_text(item["text"])),
                item,
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = []
    char_budget = max_chars_per_answer
    for _, _, _, item in candidates:
        req_len = len(clean_text(item["text"]))
        if len(selected) >= max_reqs_per_answer:
            break
        if selected and char_budget - req_len < 0:
            break
        selected.append(item)
        char_budget -= req_len
    return selected


def generate_gap_question(
    *,
    target_theme: str,
    covered_theme_names: list[str],
    candidate_requirements: list[dict],
    dialogue_history: list[dict],
    stage: str,
    focus_label: str | None = None,
) -> tuple[str, str]:
    target_label = target_theme.replace("_", " ")
    covered_labels = [theme.replace("_", " ") for theme in covered_theme_names]
    recent = dialogue_history[-8:]
    history_text = "\n".join(f"[{turn['role'].upper()}]: {turn['text']}" for turn in recent)
    examples = "\n".join(f"- {clean_text(item['text'])}" for item in candidate_requirements[:3])
    extra_focus = FOCUS_HINTS.get(focus_label or "", "")

    prompt = (
        "You are a skilled requirements analyst conducting a stakeholder interview.\n\n"
        f"Requirement area to target next: {target_label}.\n"
        f"Topics already covered reasonably well: {covered_labels or ['none yet']}.\n"
        f"Stage: {stage}.\n"
        f"{extra_focus}\n\n"
        "These are the specific requirement ideas that still seem under-covered. "
        "Do not quote them; use them only to ask one focused follow-up question.\n"
        f"{examples}\n\n"
        "Rules:\n"
        "- Ask exactly one natural, conversational question.\n"
        "- Do not use specification language like 'shall' or 'must'.\n"
        "- Build on the recent conversation instead of starting over.\n"
        "- Keep the question focused on one area.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        "Return JSON with a single 'question' field."
    )
    try:
        response = llm.generate_json(prompt, GAP_QUESTION_SCHEMA, temperature=0.4)
        parsed = llm.parse_first_json_object(response.text)
        question = clean_text(parsed.get("question", ""))
        if question:
            return question, llm.provider()
    except Exception:
        pass
    if focus_label and focus_label in FOCUS_QUESTION_MAP:
        return FOCUS_QUESTION_MAP[focus_label], "fallback_question"
    return QUESTION_MAP.get(target_theme, f"What else should I know about {target_label}?"), "fallback_question"


def generate_answer(
    template: str,
    theme: str,
    question: str,
    items: list[dict],
    threshold: float,
) -> tuple[str, str, float]:
    fallback = deterministic_answer(items)
    prompt = build_answer_prompt(template, theme, question, items)
    try:
        response = llm.generate_json(prompt, ANSWER_SCHEMA, temperature=0.0)
        parsed = llm.parse_first_json_object(response.text)
        answer = clean_text(parsed.get("answer", ""))
        if answer:
            ratios = []
            for item in items:
                overlap = 1.0 if clean_text(item["text"]).lower() in answer.lower() else 0.0
                ratios.append(overlap)
            ratio = average(ratios) if ratios else 1.0
            return answer, llm.provider(), ratio
    except Exception:
        pass
    return fallback, "fallback_answer", 1.0 if items else threshold


def build_stopped_reason(exchange_count: int, max_turns: int, recall: float, target_recall: float, uncovered_results: list[dict]) -> str:
    if not uncovered_results:
        return "all_covered"
    if recall >= target_recall:
        return "target_dialogue_recall_reached"
    if exchange_count >= max_turns:
        return "max_turns"
    return "no_viable_theme"


def critical_theme_gaps(theme_coverage: dict[str, dict]) -> list[dict]:
    gaps = []
    for theme, min_recall in CRITICAL_THEME_RECALL_FLOORS.items():
        stats = theme_coverage.get(theme, {})
        total = int(stats.get("total", 0) or 0)
        uncovered = int(stats.get("uncovered", 0) or 0)
        recall = float(stats.get("recall", 1.0) or 0.0)
        if total > 0 and uncovered > 0 and recall < min_recall:
            gaps.append(
                {
                    "theme": theme,
                    "recall": recall,
                    "min_recall": min_recall,
                    "uncovered": uncovered,
                    "total": total,
                }
            )
    gaps.sort(key=lambda item: (item["recall"], -item["uncovered"], theme_priority(item["theme"])))
    return gaps


def should_force_clarification(
    *,
    recall: float,
    target_dialogue_recall: float,
    theme_coverage: dict[str, dict],
    uncovered_results: list[dict],
) -> bool:
    if not uncovered_results:
        return False
    if recall < target_dialogue_recall:
        return True
    return bool(critical_theme_gaps(theme_coverage))


def build_dialogue(
    sample: dict,
    template: str,
    max_reqs_per_answer: int,
    max_chars_per_answer: int,
    coverage_threshold: float,
    clarification_rounds: int,
    *,
    max_turns: int,
    target_dialogue_recall: float,
    theme_max_exchanges: int,
) -> tuple[list[dict], list[dict], dict]:
    scorer = CoverageScorer(threshold=coverage_threshold)
    annotated_requirements = annotate_requirements(sample["ground_truth_requirements"], scorer)

    dialogue: list[dict] = []
    trace: list[dict] = []
    turn_id = 1
    exchange_count = 0
    theme_exchange_counts: Counter = Counter()
    coverage_history: list[float] = []

    opening_q = QUESTION_MAP["goal_scope"]
    dialogue.append({"turn_id": turn_id, "role": "bot", "text": opening_q})
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
            "stage": "adaptive_opening",
        }
    )
    turn_id += 1
    exchange_count += 1

    coverage_results, uncovered_results, recall = evaluate_requirement_coverage(
        annotated_requirements,
        dialogue,
        trace,
        scorer,
        coverage_threshold,
    )
    current_theme_coverage = compute_theme_coverage(coverage_results)
    initial_uncovered_count = len(uncovered_results)
    coverage_history.append(recall)

    while exchange_count < max_turns and recall < target_dialogue_recall and uncovered_results:
        target_theme = choose_target_theme(uncovered_results, theme_exchange_counts, theme_max_exchanges)
        if not target_theme:
            break
        candidates = select_candidate_requirements(
            uncovered_results,
            target_theme,
            max_reqs_per_answer=max_reqs_per_answer,
            max_chars_per_answer=max_chars_per_answer,
        )
        if not candidates:
            break

        covered_themes = [
            theme
            for theme, stats in compute_theme_coverage(coverage_results).items()
            if stats["recall"] >= target_dialogue_recall or stats["uncovered"] == 0
        ]
        question, question_mode = generate_gap_question(
            target_theme=target_theme,
            covered_theme_names=covered_themes,
            candidate_requirements=candidates,
            dialogue_history=dialogue,
            stage="adaptive",
        )
        dialogue.append({"turn_id": turn_id, "role": "bot", "text": question})
        turn_id += 1

        answer, answer_mode, ratio = generate_answer(template, target_theme, question, candidates, coverage_threshold)
        dialogue.append({"turn_id": turn_id, "role": "user", "text": answer})
        trace.append(
            {
                "theme": target_theme,
                "req_ids": [item["req_id"] for item in candidates],
                "bot_turn_id": turn_id - 1,
                "user_turn_id": turn_id,
                "generation_mode": f"{question_mode}|{answer_mode}",
                "coverage_ratio": ratio,
                "stage": "adaptive",
                "focus_label": None,
                "uncovered_requirement_count_at_start": len(uncovered_results),
            }
        )
        turn_id += 1
        exchange_count += 1
        theme_exchange_counts[target_theme] += 1

        coverage_results, uncovered_results, recall = evaluate_requirement_coverage(
            annotated_requirements,
            dialogue,
            trace,
            scorer,
            coverage_threshold,
        )
        current_theme_coverage = compute_theme_coverage(coverage_results)
        coverage_history.append(recall)

    clarification_used = 0
    for _round_index in range(clarification_rounds):
        if exchange_count >= max_turns or not should_force_clarification(
            recall=recall,
            target_dialogue_recall=target_dialogue_recall,
            theme_coverage=current_theme_coverage,
            uncovered_results=uncovered_results,
        ):
            break

        clusters: dict[str, list[dict]] = defaultdict(list)
        for item in uncovered_results:
            clusters[focus_label_for_requirement(item)].append(item)
        ranked_clusters = []
        gap_by_theme = {item["theme"]: item for item in critical_theme_gaps(current_theme_coverage)}
        for label, items in clusters.items():
            primary_theme = items[0].get("primary_theme") or "other_constraints"
            gap = gap_by_theme.get(primary_theme)
            ranked_clusters.append(
                (
                    0 if gap else 1,
                    gap["recall"] if gap else 1.0,
                    -len(items),
                    theme_priority(primary_theme),
                    label,
                    items,
                )
            )
        ranked_clusters.sort()
        if not ranked_clusters:
            break
        _, _, _, _, focus_label, focus_items = ranked_clusters[0]
        target_theme = focus_items[0].get("primary_theme") or "other_constraints"
        candidates = select_candidate_requirements(
            focus_items,
            target_theme,
            max_reqs_per_answer=max_reqs_per_answer,
            max_chars_per_answer=max_chars_per_answer,
        )
        if not candidates:
            candidates = focus_items[:max_reqs_per_answer]
        covered_themes = [
            theme
            for theme, stats in compute_theme_coverage(coverage_results).items()
            if stats["uncovered"] == 0
        ]
        question, question_mode = generate_gap_question(
            target_theme=target_theme,
            covered_theme_names=covered_themes,
            candidate_requirements=candidates,
            dialogue_history=dialogue,
            stage="clarification",
            focus_label=focus_label,
        )
        dialogue.append({"turn_id": turn_id, "role": "bot", "text": question})
        turn_id += 1

        answer, answer_mode, ratio = generate_answer(template, target_theme, question, candidates, coverage_threshold)
        dialogue.append({"turn_id": turn_id, "role": "user", "text": answer})
        trace.append(
            {
                "theme": target_theme,
                "req_ids": [item["req_id"] for item in candidates],
                "bot_turn_id": turn_id - 1,
                "user_turn_id": turn_id,
                "generation_mode": f"{question_mode}|{answer_mode}",
                "coverage_ratio": ratio,
                "stage": "clarification",
                "focus_label": focus_label,
                "uncovered_requirement_count_at_start": len(uncovered_results),
            }
        )
        turn_id += 1
        exchange_count += 1
        clarification_used += 1
        theme_exchange_counts[target_theme] += 1

        coverage_results, uncovered_results, recall = evaluate_requirement_coverage(
            annotated_requirements,
            dialogue,
            trace,
            scorer,
            coverage_threshold,
        )
        current_theme_coverage = compute_theme_coverage(coverage_results)
        coverage_history.append(recall)

    final_theme_coverage = current_theme_coverage
    coverage_summary = {
        "clarification_rounds_requested": clarification_rounds,
        "clarification_rounds_used": clarification_used,
        "initial_uncovered_requirement_count": initial_uncovered_count,
        "final_uncovered_requirement_count": len(uncovered_results),
        "final_uncovered_req_ids": [item["req_id"] for item in uncovered_results],
        "coverage_history": coverage_history,
        "exchanges": exchange_count,
        "target_dialogue_recall": target_dialogue_recall,
        "final_dialogue_recall": recall,
        "theme_exchange_counts": dict(theme_exchange_counts),
        "theme_coverage": final_theme_coverage,
        "critical_theme_gaps": critical_theme_gaps(final_theme_coverage),
        "stopped_reason": build_stopped_reason(exchange_count, max_turns, recall, target_dialogue_recall, uncovered_results),
    }
    return dialogue, trace, coverage_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-reqs-per-answer", type=int, default=4)
    parser.add_argument("--max-chars-per-answer", type=int, default=900)
    parser.add_argument("--coverage-threshold", type=float, default=0.55)
    parser.add_argument("--clarification-rounds", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--target-dialogue-recall", type=float, default=DEFAULT_TARGET_DIALOGUE_RECALL)
    parser.add_argument("--theme-max-exchanges", type=int, default=DEFAULT_THEME_MAX_EXCHANGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
        print(
            json.dumps(
                {
                    "prompt_hash": prompt_hash,
                    "schema_hash": schema_hash,
                    "sample_count": len(samples),
                    "max_reqs_per_answer": args.max_reqs_per_answer,
                    "max_chars_per_answer": args.max_chars_per_answer,
                    "coverage_threshold": args.coverage_threshold,
                    "max_turns": args.max_turns,
                    "target_dialogue_recall": args.target_dialogue_recall,
                    "theme_max_exchanges": args.theme_max_exchanges,
                    "clarification_rounds": args.clarification_rounds,
                    "question_algorithm_version": QUESTION_ALGORITHM_VERSION,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    model_name = llm.active_model_name()
    summary = []
    for sample in samples:
        dialogue, trace, coverage_summary = build_dialogue(
            sample,
            template,
            args.max_reqs_per_answer,
            args.max_chars_per_answer,
            args.coverage_threshold,
            args.clarification_rounds,
            max_turns=args.max_turns,
            target_dialogue_recall=args.target_dialogue_recall,
            theme_max_exchanges=args.theme_max_exchanges,
        )
        fallback_count = sum(
            1
            for item in trace
            if "fallback" in item.get("generation_mode", "")
        )
        clarification_chunk_count = sum(1 for item in trace if item.get("stage") == "clarification")
        payload = {
            "sample_id": sample["sample_id"],
            "metadata": {
                "domain": "pure_document",
                "source_type": "synthetic",
                "parent_id": sample["sample_id"],
                "split": "benchmark",
                "dialogue_style": "adaptive_semantic_coverage",
            },
            "source": sample["source"],
            "dialogue": dialogue,
            "dialogue_generation": {
                "method": "g_adaptive_semantic_coverage_v2",
                "model": model_name,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "max_reqs_per_answer": args.max_reqs_per_answer,
                "max_chars_per_answer": args.max_chars_per_answer,
                "coverage_threshold": args.coverage_threshold,
                "max_turns": args.max_turns,
                "target_dialogue_recall": args.target_dialogue_recall,
                "theme_max_exchanges": args.theme_max_exchanges,
                "question_algorithm_version": QUESTION_ALGORITHM_VERSION,
                "question_algorithm_summary": QUESTION_ALGORITHM_SUMMARY,
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
                "stopped_reason": coverage_summary.get("stopped_reason"),
                "exchanges": coverage_summary.get("exchanges"),
                "final_dialogue_recall": coverage_summary.get("final_dialogue_recall"),
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
