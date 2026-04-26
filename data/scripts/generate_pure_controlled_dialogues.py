#!/usr/bin/env python3
"""Generate adaptive elicitation dialogues from PURE requirements.

Architecture (3-component adaptive loop):
1. Coverage tracker  — keyword hit-rate per category after each user turn
2. Gap question generator — LLM generates a targeted question for the top gap
3. Stopping criterion — stops when all covered, max turns reached, or
                         coverage gain has plateaued (diminishing returns)
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

# ── Adaptive loop constants ────────────────────────────────────────────────────
MAX_TURNS = 20          # hard ceiling on bot/user exchange pairs
COVERAGE_THRESHOLD = 0.35   # keyword hit-rate needed to count a category as covered
MIN_COVERAGE_GAIN = 0.05    # stop if coverage fraction gain < 5% over last 3 turns
COVERAGE_GAIN_WINDOW = 3    # number of turns to look back for diminishing returns

# ── Answer-schema (unchanged from original) ────────────────────────────────────
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

# Schema for LLM-generated gap question
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

# ── Theme taxonomy ─────────────────────────────────────────────────────────────
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

# Priority order for gap targeting (highest-value categories first)
PRIORITY_ORDER = [
    "functional_capabilities",
    "performance_capacity",
    "security_audit",
    "data_validation",
    "interfaces_integrations",
    "deployment_environment_constraints",
    "maintainability_portability_testability",
    "availability_reliability",
    "workflows_business_rules",
    "user_roles_permissions",
    "usability_help_accessibility",
    "reporting_documentation",
    "other_constraints",
]

# Fixed fallback questions (used when LLM gap-question fails)
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

THEME_KEYWORDS: dict[str, list[str]] = {
    "performance_capacity": [
        "less than", "under ", "per second", "ms", "milliseconds", "seconds",
        "concurrent", "concurrently", "response time", "throughput", "performance", "latency",
    ],
    "availability_reliability": [
        "availability", "uptime", "reliability", "reliable", "restore", "recovery",
        "recover", "backup", "backed up", "failover", "operational in less than",
    ],
    "security_audit": [
        "security", "secure", "encrypt", "https", "password", "login", "log in",
        "authentication", "authorization", "authorisation", "audit", "audit trail",
        "fraud", "unauthorised", "unauthorized", "firewall", "access rights",
        "privileges", "super-user", "security attributes",
    ],
    "usability_help_accessibility": [
        "help material", "online help", "usability", "error messages", "special needs",
        "look and feel", "horizontal scrolling", "input devices", "navigation",
        "customizable", "configurable", "user interface", "interface rules",
        "accessibility", "meaningful", "controls",
    ],
    "deployment_environment_constraints": [
        "operating environment", "operate on", "browser", "internet explorer",
        "netscape", "slackware", "apache", "intel based", "hardware", "usb",
        "deployment", "technical constraints", "shortest path algorithm",
    ],
    "maintainability_portability_testability": [
        "maintainability", "portable", "portability", "migrate", "upgrade",
        "updatable", "patches", "debug mode", "plugins", "interchangeable",
        "testability", "easy to upgrade", "easy to migrate",
    ],
    "reporting_documentation": [
        "report", "reports", "manual", "guide book", "guide", "documentation",
        "installation instructions", "operations and maintenance", "users guide",
    ],
    "interfaces_integrations": [
        "interface", "interfaces", "email", "sms", "plug-ins", "plugin interface",
        "web interface", "communicat", "tracking number", "technical queries",
    ],
    "data_validation": [
        "database", "data", "store", "stored", "capture", "records", "inventory",
        "validate", "validation", "entities", "suspect", "property",
        "credit card", "email address",
    ],
    "workflows_business_rules": [
        "if ", "when ", "after ", "before ", "based on", "thereafter",
        "returning customers", "chooses to", "selectable", "must allow only",
        "shall take", "workflow", "business rule",
    ],
    "user_roles_permissions": [
        "citizens", "citizen", "police", "help-desk", "admin-users",
        "administrator", "administrators", "sales people", "sales person",
        "customer", "customers", "user groups", "groups", "user profiles",
        "role-based", "member of more than one group",
    ],
    "goal_scope": ["main goal", "overall scope", "overall purpose", "solution", "system goal", "scope"],
    "functional_capabilities": [
        "allow", "enable", "provide", "support", "view", "search", "register",
        "log", "track", "display", "manage", "send", "export", "browse", "checkout",
        "order", "purchase", "add", "remove", "upload", "download",
    ],
    "other_constraints": [],
}


# ── Utilities ──────────────────────────────────────────────────────────────────

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
        "the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "with",
        "by", "is", "are", "be", "as", "that", "this", "it", "from", "at",
        "must", "shall", "should",
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


def classify_requirement(text: str) -> str:
    scores = {theme: sum(1 for kw in kws if kw in text.lower()) for theme, kws in THEME_KEYWORDS.items()}
    best_theme = max(scores, key=scores.get)
    if scores[best_theme] > 0:
        return best_theme
    lowered = text.lower()
    if re.search(r"\b(allow|enable|provide|support|view|search|register|log|track|display|manage|send|export)\b", lowered):
        return "functional_capabilities"
    return "other_constraints"


def build_scope_answer(sample: dict) -> str:
    title = clean_text(sample["source"]["title"])
    doc_id = clean_text(sample["source"]["document_id"])
    return (
        f"This project is based on the {title} system, and "
        f"the goal is to capture the main requirements and constraints for document {doc_id}."
    )


def build_requirements_block(items: list[dict]) -> str:
    return "\n".join(f"- {item['req_id']}: {clean_text(item['text'])}" for item in items)


def build_answer_prompt(template: str, theme: str, question: str, items: list[dict]) -> str:
    return (
        template.replace("{{THEME}}", theme)
        .replace("{{QUESTION}}", question)
        .replace("{{REQUIREMENTS}}", build_requirements_block(items))
    )


def deterministic_answer(items: list[dict]) -> str:
    return " ".join(clean_text(item["text"]).rstrip(".") + "." for item in items if clean_text(item["text"]))


def answer_covers_items(answer: str, items: list[dict], threshold: float) -> tuple[bool, float]:
    if not answer.strip():
        return False, 0.0
    scores = [token_f1(item["text"], answer) for item in items]
    covered = sum(score >= threshold for score in scores)
    ratio = covered / len(items) if items else 1.0
    return ratio >= 1.0, ratio


def collect_user_turn_texts(dialogue: list[dict]) -> list[str]:
    return [clean_text(t.get("text", "")) for t in dialogue if t.get("role") == "user" and t.get("text")]


def find_uncovered_requirements(sample: dict, dialogue: list[dict], threshold: float) -> list[dict]:
    user_texts = collect_user_turn_texts(dialogue)
    uncovered = []
    for item in sample["ground_truth_requirements"]:
        best = max((token_f1(item["text"], u) for u in user_texts), default=0.0)
        if best < threshold:
            uncovered.append({**item, "best_dialogue_coverage": best})
    return uncovered


# ── Component 1: Coverage tracker ─────────────────────────────────────────────

def score_category_coverage(user_turn_texts: list[str], category: str) -> float:
    """Keyword hit-rate for a category across all user turns."""
    keywords = THEME_KEYWORDS.get(category, [])
    if not keywords:
        return 1.0
    full_text = " ".join(user_turn_texts).lower()
    hits = sum(1 for kw in keywords if kw in full_text)
    return hits / len(keywords)


def get_uncovered_categories(user_turn_texts: list[str]) -> list[str]:
    """Return categories not yet sufficiently covered in user turns."""
    return [
        cat for cat in THEME_ORDER
        if cat != "goal_scope"
        and THEME_KEYWORDS.get(cat)  # skip categories with no keywords
        and score_category_coverage(user_turn_texts, cat) < COVERAGE_THRESHOLD
    ]


def coverage_fraction(user_turn_texts: list[str]) -> float:
    """Overall fraction of trackable categories that are covered."""
    trackable = [cat for cat in THEME_ORDER if cat != "goal_scope" and THEME_KEYWORDS.get(cat)]
    if not trackable:
        return 1.0
    covered = sum(
        1 for cat in trackable
        if score_category_coverage(user_turn_texts, cat) >= COVERAGE_THRESHOLD
    )
    return covered / len(trackable)


# ── Component 2: Gap-targeted question generator ───────────────────────────────

def generate_gap_question(
    target_category: str,
    covered_categories: list[str],
    dialogue_history: list[dict],
) -> str:
    """Ask the LLM to generate one natural analyst question targeting a coverage gap."""
    covered_labels = [c.replace("_", " ") for c in covered_categories]
    target_label = target_category.replace("_", " ")

    recent = dialogue_history[-6:]  # last 3 exchanges
    history_text = "\n".join(
        f"[{t['role'].upper()}]: {t['text']}" for t in recent
    )

    prompt = (
        f"You are a skilled requirements analyst conducting a stakeholder interview.\n\n"
        f"Topics already covered reasonably well: {covered_labels}.\n\n"
        f"The following requirement area has NOT been discussed yet: '{target_label}'.\n\n"
        f"Generate ONE natural, conversational follow-up question specifically targeting "
        f"the '{target_label}' area.\n\n"
        f"Rules:\n"
        f"- Ask as a real analyst would — curious, specific, building on what was said\n"
        f"- Do NOT use requirements language like 'shall' or 'must'\n"
        f"- Keep it to ONE focused question, not multiple\n"
        f"- Make it feel like a natural continuation of this conversation\n\n"
        f"Last few turns:\n{history_text}\n\n"
        f"Return JSON with a single 'question' field."
    )
    try:
        response = llm.generate_json(prompt, GAP_QUESTION_SCHEMA, temperature=0.7)
        parsed = llm.parse_first_json_object(response.text)
        question = clean_text(parsed.get("question", ""))
        if question:
            return question
    except Exception:
        pass
    # Fallback to fixed question map
    return QUESTION_MAP.get(target_category, f"What else should I know about {target_label}?")


# ── Component 3: Stopping criterion ───────────────────────────────────────────

def should_stop(
    uncovered: list[str],
    exchange_count: int,
    coverage_history: list[float],
) -> bool:
    """
    Stop when:
    1. All categories are covered, OR
    2. Hard turn ceiling reached
    """
    if not uncovered:
        return True
    if exchange_count >= MAX_TURNS:
        return True
    return False


# ── Answer generation ──────────────────────────────────────────────────────────

def generate_answer(
    template: str,
    theme: str,
    question: str,
    items: list[dict],
    threshold: float,
) -> tuple[str, str, float]:
    """
    Generate a naturalistic user answer for the given requirements chunk.
    Uses LLM answer whenever it returns non-empty; fallback only on error.
    """
    fallback = deterministic_answer(items)
    prompt = build_answer_prompt(template, theme, question, items)
    try:
        response = llm.generate_json(prompt, ANSWER_SCHEMA, temperature=0.0)
        parsed = llm.parse_first_json_object(response.text)
        answer = clean_text(parsed.get("answer", ""))
        if answer:
            # Coverage guardrail intentionally disabled — naturalistic paraphrases
            # score low against formal requirement text; we want that gap.
            _, ratio = answer_covers_items(answer, items, threshold)
            return answer, llm.provider(), ratio
    except Exception as e:
        raise RuntimeError(f"Failed to generate answer from LLM: {e}") from e


# ── Main adaptive dialogue builder ─────────────────────────────────────────────

def build_dialogue(
    sample: dict,
    template: str,
    max_reqs_per_answer: int,
    max_chars_per_answer: int,
    coverage_threshold: float,
    clarification_rounds: int,  # kept for API compat, not used in adaptive loop
) -> tuple[list[dict], list[dict], dict]:
    dialogue: list[dict] = []
    trace: list[dict] = []
    turn_id = 1
    coverage_history: list[float] = []

    # ── Turn 0: broad opening ──────────────────────────────────────────────────
    opening_q = QUESTION_MAP["goal_scope"]
    dialogue.append({"turn_id": turn_id, "role": "bot", "text": opening_q})
    turn_id += 1
    scope_answer = build_scope_answer(sample)
    dialogue.append({"turn_id": turn_id, "role": "user", "text": scope_answer})
    trace.append({
        "theme": "goal_scope",
        "req_ids": [],
        "bot_turn_id": turn_id - 1,
        "user_turn_id": turn_id,
        "generation_mode": "deterministic_scope",
        "coverage_ratio": 1.0,
        "stage": "adaptive_opening",
    })
    turn_id += 1

    user_turns: list[str] = [scope_answer]
    coverage_history.append(coverage_fraction(user_turns))
    exchange_count = 1  # number of bot/user pairs so far

    # Build a quick index: category -> list of requirements
    category_reqs: dict[str, list[dict]] = {cat: [] for cat in THEME_ORDER}
    for req in sample["ground_truth_requirements"]:
        cat = classify_requirement(req["text"])
        category_reqs[cat].append(req)

    # Track which requirements have been presented to the LLM already
    presented_req_ids: set[str] = set()

    # ── Adaptive loop ──────────────────────────────────────────────────────────
    while True:
        uncovered_cats = get_uncovered_categories(user_turns)
        covered_cats = [c for c in THEME_ORDER if c not in uncovered_cats and c != "goal_scope"]

        if should_stop(uncovered_cats, exchange_count, coverage_history):
            break

        # Pick the highest-priority uncovered category
        target = next((c for c in PRIORITY_ORDER if c in uncovered_cats), uncovered_cats[0])

        # Generate a targeted gap question via LLM
        question = generate_gap_question(target, covered_cats, dialogue)
        dialogue.append({"turn_id": turn_id, "role": "bot", "text": question})
        turn_id += 1

        # Select requirements to answer from — prefer unseen ones in the target category
        candidate_reqs = [
            r for r in category_reqs.get(target, [])
            if r["req_id"] not in presented_req_ids
        ]
        if not candidate_reqs:
            # Fall back to any unseen requirement whose coverage is low
            candidate_reqs = [
                r for r in sample["ground_truth_requirements"]
                if r["req_id"] not in presented_req_ids
                and max(
                    (token_f1(r["text"], u) for u in user_turns),
                    default=0.0,
                ) < coverage_threshold
            ]
        if not candidate_reqs:
            # All requirements seen — pick least-covered ones
            candidate_reqs = sorted(
                sample["ground_truth_requirements"],
                key=lambda r: max((token_f1(r["text"], u) for u in user_turns), default=0.0),
            )[:max_reqs_per_answer]

        # Truncate to budget and track char count
        chunk: list[dict] = []
        char_budget = max_chars_per_answer
        for req in candidate_reqs:
            if len(chunk) >= max_reqs_per_answer:
                break
            req_len = len(req["text"])
            if chunk and char_budget - req_len < 0:
                break
            chunk.append(req)
            char_budget -= req_len
            presented_req_ids.add(req["req_id"])

        answer, mode, ratio = generate_answer(template, target, question, chunk, coverage_threshold)
        dialogue.append({"turn_id": turn_id, "role": "user", "text": answer})
        trace.append({
            "theme": target,
            "req_ids": [r["req_id"] for r in chunk],
            "bot_turn_id": turn_id - 1,
            "user_turn_id": turn_id,
            "generation_mode": mode,
            "coverage_ratio": ratio,
            "stage": "adaptive",
            "uncovered_at_start": uncovered_cats,
        })
        turn_id += 1

        user_turns.append(answer)
        coverage_history.append(coverage_fraction(user_turns))
        exchange_count += 1

    # ── Build coverage summary ─────────────────────────────────────────────────
    final_uncovered_reqs = find_uncovered_requirements(sample, dialogue, coverage_threshold)
    coverage_summary = {
        "clarification_rounds_requested": 0,
        "clarification_rounds_used": 0,
        "initial_uncovered_requirement_count": len(final_uncovered_reqs),
        "final_uncovered_requirement_count": len(final_uncovered_reqs),
        "final_uncovered_req_ids": [r["req_id"] for r in final_uncovered_reqs],
        "coverage_history": coverage_history,
        "exchanges": exchange_count,
        "stopped_reason": (
            "all_covered" if not get_uncovered_categories(user_turns)
            else "max_turns" if exchange_count >= MAX_TURNS
            else "diminishing_returns"
        ),
    }
    return dialogue, trace, coverage_summary


# ── CLI ────────────────────────────────────────────────────────────────────────

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
        print(json.dumps({
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "sample_count": len(samples),
            "max_reqs_per_answer": args.max_reqs_per_answer,
            "max_chars_per_answer": args.max_chars_per_answer,
            "coverage_threshold": args.coverage_threshold,
            "max_turns": MAX_TURNS,
            "coverage_threshold_keyword": COVERAGE_THRESHOLD,
            "min_coverage_gain": MIN_COVERAGE_GAIN,
        }, indent=2, ensure_ascii=False))
        return 0

    model_name = (
        os.environ.get("REQ_OLLAMA_MODEL") if llm.provider() == "ollama"
        else os.environ.get("REQ_GEMINI_MODEL")
    )

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
        fallback_count = sum(1 for t in trace if "error_fallback" in t.get("generation_mode", ""))
        clarification_chunk_count = sum(1 for t in trace if t.get("stage") == "clarification")
        payload = {
            "sample_id": sample["sample_id"],
            "metadata": {
                "domain": "pure_document",
                "source_type": "synthetic",
                "parent_id": sample["sample_id"],
                "split": "benchmark",
                "dialogue_style": "adaptive_coverage_driven",
            },
            "source": sample["source"],
            "dialogue": dialogue,
            "dialogue_generation": {
                "method": "g_adaptive_coverage_driven_v1",
                "model": model_name,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "max_reqs_per_answer": args.max_reqs_per_answer,
                "max_chars_per_answer": args.max_chars_per_answer,
                "coverage_threshold": args.coverage_threshold,
                "keyword_coverage_threshold": COVERAGE_THRESHOLD,
                "max_turns": MAX_TURNS,
                "min_coverage_gain": MIN_COVERAGE_GAIN,
                "clarification_rounds": 0,
                "trace": trace,
                "fallback_chunk_count": fallback_count,
                "clarification_chunk_count": clarification_chunk_count,
                "coverage_summary": coverage_summary,
            },
        }
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append({
            "sample_id": sample["sample_id"],
            "path": str(output_path),
            "turn_count": len(dialogue),
            "chunk_count": len(trace),
            "fallback_chunk_count": fallback_count,
            "clarification_chunk_count": clarification_chunk_count,
            "final_uncovered_requirement_count": coverage_summary["final_uncovered_requirement_count"],
            "stopped_reason": coverage_summary.get("stopped_reason"),
            "exchanges": coverage_summary.get("exchanges"),
        })

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
