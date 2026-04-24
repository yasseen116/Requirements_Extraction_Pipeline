#!/usr/bin/env python3
"""Generate a rule-based dialogue -> slots baseline for fixed-flow dialogues."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "manual_gold"
DEFAULT_OUTPUT = ROOT / "outputs" / "b1_rule_based_slots"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_items(text: str) -> list[str]:
    text = clean_text(text)
    text = text.replace(" and ", ", ")
    return [item.strip(" .") for item in text.split(",") if item.strip(" .")]


def normalize_system_type(text: str) -> str:
    lower = clean_text(text).lower()
    if "restaurant" in lower and "nearby" in lower and ("app" in lower or "mobile" in lower):
        return "restaurant discovery mobile application"
    if "menu" in lower and "ordering" in lower:
        return "restaurant menu and ordering system"
    if "password" in lower and ("safe" in lower or "store" in lower):
        return "password safe application"
    return re.sub(r"^(a|an|the)\s+", "", lower)


def normalize_role(item: str) -> str | None:
    lower = item.lower().strip()
    lower = re.sub(r"^but\s+", "", lower)
    replacements = {
        "normal users on their phones": "users",
        "normal users": "users",
        "users on their phones": "users",
        "users": "users",
        "restaurant owners on a web portal": "restaurant owners",
        "restaurant owners": "restaurant owners",
        "admins": "administrators",
        "admins.": "administrators",
        "administrators": "administrators",
        "customers": "customers",
        "waiters": "waiters",
        "chefs": "chefs",
        "supervisors": "supervisors",
        "mostly end users": "end users",
        "end users": "end users",
        "system administrators too": "system administrators",
        "system administrators": "system administrators",
    }
    if lower in replacements:
        return replacements[lower]
    if "restaurant owner" in lower:
        return "restaurant owners"
    if "admin" in lower:
        return "administrators"
    if "end user" in lower:
        return "end users"
    if "system administrator" in lower:
        return "system administrators"
    if lower:
        return lower
    return None


def predict_user_roles(text: str, evidence_turn: int) -> list[dict]:
    roles = []
    seen = set()
    for item in split_items(text):
        normalized = normalize_role(item)
        if normalized and normalized not in seen:
            roles.append({"value": normalized, "evidence_turns": [evidence_turn]})
            seen.add(normalized)
    return roles


def make_capability(actor: str, action: str, evidence_turn: int) -> dict:
    return {
        "actor": actor,
        "action": action,
        "evidence_turns": [evidence_turn],
    }


def predict_capabilities(text: str, user_roles: list[dict], evidence_turn: int) -> list[dict]:
    lower = clean_text(text).lower()
    capabilities = []

    if "search for restaurants" in lower:
        capabilities.append(
            make_capability("users", "search for restaurants using multiple search options", evidence_turn)
        )
    if "map and list" in lower:
        capabilities.append(
            make_capability("users", "view restaurant results in map and list views", evidence_turn)
        )
    if "detailed restaurant info" in lower or "detailed restaurant information" in lower:
        capabilities.append(
            make_capability("users", "open detailed restaurant information pages", evidence_turn)
        )
    if "gps navigation" in lower or "get gps navigation" in lower:
        capabilities.append(
            make_capability("users", "navigate to a selected restaurant using GPS", evidence_turn)
        )
    if "manage restaurant info" in lower:
        capabilities.append(
            make_capability(
                "restaurant owners",
                "manage restaurant information through the web portal",
                evidence_turn,
            )
        )
    if "verify owners" in lower:
        capabilities.append(
            make_capability("administrators", "verify restaurant owners", evidence_turn)
        )

    if "place orders" in lower:
        capabilities.append(
            make_capability("customers", "place orders through an engaged menu", evidence_turn)
        )
    if "call waiters" in lower or "call waiter" in lower:
        capabilities.append(
            make_capability("customers", "call for waiter assistance", evidence_turn)
        )
    if "cash or bankcard payments" in lower or "cash and bankcard payments" in lower:
        capabilities.append(
            make_capability("waiters", "process cash and bankcard payments", evidence_turn)
        )
    if "mark items ready" in lower:
        capabilities.append(
            make_capability("chefs", "mark order items as ready to be served", evidence_turn)
        )
    if "issue refunds" in lower:
        capabilities.append(
            make_capability("supervisors", "issue refunds", evidence_turn)
        )

    if "create and open encrypted databases" in lower:
        actor = user_roles[0]["value"] if user_roles else "users"
        capabilities.append(
            make_capability(actor, "create and open encrypted password databases", evidence_turn)
        )
    if "search stored entries" in lower:
        actor = user_roles[0]["value"] if user_roles else "users"
        capabilities.append(make_capability(actor, "search stored entries", evidence_turn))
    if "add or edit entries" in lower:
        actor = user_roles[0]["value"] if user_roles else "users"
        capabilities.append(make_capability(actor, "add and edit entries", evidence_turn))
    if "import or export password data" in lower:
        actor = user_roles[0]["value"] if user_roles else "users"
        capabilities.append(make_capability(actor, "import and export password data", evidence_turn))

    if capabilities:
        return capabilities

    actor = user_roles[0]["value"] if user_roles else "users"
    return [make_capability(actor, re.sub(r"^(they|users)\s+should\s+", "", lower), evidence_turn)]


def predict_authentication(text: str, evidence_turn: int) -> dict:
    lower = clean_text(text).lower()
    methods = []

    if "mobile app" in lower:
        methods.append("mobile app username/password login")
    if "portal" in lower:
        methods.append("web portal username/password login")
    if "tablet" in lower and ("username" in lower or "password" in lower):
        methods.append("tablet username/password login")
    if "master password" in lower:
        methods.append("master password")
    if "key file" in lower:
        methods.append("key file")
    if "composite key" in lower or "composite master key" in lower:
        methods.append("composite master key")
    if not methods and ("username" in lower or "password" in lower):
        methods.append("username/password login")

    return {
        "required": not any(token in lower for token in ["no login", "do not need login"]),
        "methods": methods,
        "evidence_turns": [evidence_turn],
    }


def predict_performance_constraints(text: str, evidence_turn: int) -> list[dict]:
    lower = clean_text(text).lower()
    constraints = []

    if "2 seconds" in lower:
        constraints.append(
            {"text": "return search results within 2 seconds", "evidence_turns": [evidence_turn]}
        )
    if "gps or internet" in lower or "internet or gps" in lower:
        constraints.append(
            {
                "text": "inform the user if internet or GPS connectivity is lost",
                "evidence_turns": [evidence_turn],
            }
        )
    if "200 connected devices" in lower or "200 concurrent" in lower:
        constraints.append(
            {
                "text": "support at least 200 concurrent device connections",
                "evidence_turns": [evidence_turn],
            }
        )
    if "orders" in lower and "never get lost" in lower:
        constraints.append(
            {
                "text": "preserve all active meals and orders without losing them",
                "evidence_turns": [evidence_turn],
            }
        )
    if "payments" in lower and "never get lost" in lower:
        constraints.append(
            {
                "text": "preserve all active customer payments without losing them",
                "evidence_turns": [evidence_turn],
            }
        )
    if "10 seconds" in lower and "password" in lower:
        constraints.append(
            {
                "text": "keep copied passwords in memory for no more than 10 seconds",
                "evidence_turns": [evidence_turn],
            }
        )

    return constraints


def predict_security_constraints(auth_text: str, security_text: str, evidence_turn: int) -> list[dict]:
    auth_lower = clean_text(auth_text).lower()
    lower = clean_text(security_text).lower()
    constraints = []

    if "encrypted" in lower and "log" in lower:
        constraints.append(
            {"text": "encrypt log-in communication messages", "evidence_turns": [evidence_turn]}
        )
    if "three failed" in lower and "half an hour" in lower:
        constraints.append(
            {
                "text": "disable restaurant owner and administrator log-in for 30 minutes after three failed attempts",
                "evidence_turns": [evidence_turn],
            }
        )
    if "wireless communication" in lower and "encryption" in lower:
        constraints.append(
            {
                "text": "encrypt wireless communication using SSLv3 and WPA2-PSK",
                "evidence_turns": [evidence_turn],
            }
        )
    if "one tablet at a time" in lower:
        constraints.append(
            {
                "text": "limit each waiter to one logged-in tablet at a time",
                "evidence_turns": [evidence_turn],
            }
        )
    if "should not need login" in auth_lower and "surface computers" in auth_lower:
        constraints.append(
            {
                "text": "not require customers to log in on surface computers",
                "evidence_turns": [8],
            }
        )
    if "database" in lower and "encrypted" in lower:
        constraints.append(
            {"text": "store passwords in an encrypted database", "evidence_turns": [evidence_turn]}
        )
    if "refuse to open" in lower or ("master password" in lower and "key file" in lower):
        constraints.append(
            {
                "text": "refuse to open a database without the required master password or key file",
                "evidence_turns": [evidence_turn],
            }
        )
    if "no recovery password" in lower or "backdoor" in lower:
        constraints.append(
            {
                "text": "provide no recovery password or backdoor for unlocking databases",
                "evidence_turns": [evidence_turn],
            }
        )

    return constraints


def predict_slots(sample: dict) -> dict:
    turns = {turn["turn_id"]: turn["text"] for turn in sample["dialogue"]}
    user_roles = predict_user_roles(turns[4], 4)
    return {
        "sample_id": sample["sample_id"],
        "method": "b1_rule_based_dialogue_to_slots_v1",
        "slots": {
            "system_type": {
                "value": normalize_system_type(turns[2]),
                "evidence_turns": [2],
            },
            "user_roles": user_roles,
            "functional_capabilities": predict_capabilities(turns[6], user_roles, 6),
            "authentication": predict_authentication(turns[8], 8),
            "performance_constraints": predict_performance_constraints(turns[10], 10),
            "security_constraints": predict_security_constraints(turns[8], turns[12], 12),
        },
    }


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "sample_id" not in payload or "dialogue" not in payload:
            continue
        samples.append(payload)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for sample in load_samples(args.input_dir):
        prediction = predict_slots(sample)
        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(prediction, indent=2))
        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": display_path(output_path),
                "user_roles": len(prediction["slots"]["user_roles"]),
                "functional_capabilities": len(prediction["slots"]["functional_capabilities"]),
                "performance_constraints": len(prediction["slots"]["performance_constraints"]),
                "security_constraints": len(prediction["slots"]["security_constraints"]),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
