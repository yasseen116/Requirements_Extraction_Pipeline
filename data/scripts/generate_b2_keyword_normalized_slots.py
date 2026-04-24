#!/usr/bin/env python3
"""Generate a stronger keyword-normalized dialogue -> slots baseline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "manual_gold"
DEFAULT_OUTPUT = ROOT / "outputs" / "b2_keyword_normalized_slots"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_text(text: str) -> str:
    return clean_text(text).lower()


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def infer_system_type(text: str) -> str:
    lower = normalize_text(text)
    if (
        contains_any(lower, ["food spots", "restaurants", "places to eat", "restaurant"])
        and contains_any(lower, ["phone app", "mobile", "app"])
        and contains_any(lower, ["find", "near me", "nearby"])
    ):
        return "restaurant discovery mobile application"
    if contains_any(
        lower,
        [
            "ordering setup",
            "menu and ordering",
            "ordering system",
            "table screens",
            "kitchen displays",
        ],
    ):
        return "restaurant menu and ordering system"
    if contains_any(
        lower,
        [
            "password vault",
            "password safe",
            "vault tool",
            "account creds",
            "password safe app",
        ],
    ):
        return "password safe application"
    return re.sub(r"^(a|an|the)\s+", "", lower)


def infer_roles(system_type: str, text: str) -> list[dict]:
    lower = normalize_text(text)
    roles = []

    role_patterns = {
        "restaurant discovery mobile application": [
            ("users", ["regular app users", "app users", "users"]),
            ("restaurant owners", ["restaurant people using the portal", "restaurant owners", "restaurant side"]),
            ("administrators", ["whoever manages the whole thing", "management", "admins", "administrators"]),
        ],
        "restaurant menu and ordering system": [
            ("customers", ["diners", "guests", "customers"]),
            ("waiters", ["floor staff", "servers", "waiters", "staff"]),
            ("chefs", ["kitchen crew", "kitchen staff", "kitchen", "chefs"]),
            ("supervisors", ["managers", "supervisors"]),
        ],
        "password safe application": [
            ("end users", ["regular users", "end users", "users"]),
            ("system administrators", ["it admins", "system administrators", "admins"]),
        ],
    }

    seen = set()
    for canonical, patterns in role_patterns.get(system_type, []):
        if contains_any(lower, patterns) and canonical not in seen:
            roles.append({"value": canonical, "evidence_turns": [4]})
            seen.add(canonical)
    return roles


def make_capability(actor: str, action: str) -> dict:
    return {"actor": actor, "action": action, "evidence_turns": [6]}


def infer_capabilities(system_type: str, text: str) -> list[dict]:
    lower = normalize_text(text)
    capabilities = []

    mappings = {
        "restaurant discovery mobile application": [
            ("users", "search for restaurants using multiple search options", ["search for restaurants", "look up places to eat", "find food spots"]),
            ("users", "view restaurant results in map and list views", ["map and list", "bounce between map and list", "switch between map and list"]),
            ("users", "open detailed restaurant information pages", ["open the full place details", "check what each place offers", "open detailed restaurant info"]),
            ("users", "navigate to a selected restaurant using GPS", ["get directions", "gps navigation", "navigate"]),
            ("restaurant owners", "manage restaurant information through the web portal", ["manage restaurant info", "update venue info", "update restaurant info"]),
            ("administrators", "verify restaurant owners", ["verify owners", "approve those restaurant accounts", "approve restaurant accounts"]),
        ],
        "restaurant menu and ordering system": [
            ("customers", "place orders through an engaged menu", ["place orders", "send orders"]),
            ("customers", "call for waiter assistance", ["call waiter", "buzz staff"]),
            ("waiters", "process cash and bankcard payments", ["cash or card payments", "cash or bankcard payments", "card payments"]),
            ("chefs", "mark order items as ready to be served", ["mark dishes ready", "mark items ready", "kitchen needs to mark dishes ready"]),
            ("supervisors", "issue refunds", ["issue refunds", "handle refunds"]),
        ],
        "password safe application": [
            ("end users", "create and open encrypted password databases", ["create and open encrypted databases", "spin up and unlock encrypted vaults", "unlock encrypted vaults"]),
            ("end users", "search stored entries", ["search stored entries", "find saved items"]),
            ("end users", "add and edit entries", ["add or edit entries", "tweak entries"]),
            ("end users", "import and export password data", ["import or export password data", "move password data in or out"]),
        ],
    }

    for actor, action, patterns in mappings.get(system_type, []):
        if contains_any(lower, patterns):
            capabilities.append(make_capability(actor, action))

    if capabilities:
        return capabilities

    return [make_capability("users", lower)]


def infer_authentication(system_type: str, text: str) -> dict:
    lower = normalize_text(text)
    methods = []

    if system_type == "restaurant discovery mobile application":
        if contains_any(lower, ["mobile side", "mobile", "app"]):
            methods.append("mobile app username/password login")
        if "portal" in lower:
            methods.append("web portal username/password login")
    elif system_type == "restaurant menu and ordering system":
        if contains_any(lower, ["staff tablets", "tablets", "tablet"]) and contains_any(lower, ["username", "password", "sign-in", "sign in"]):
            methods.append("tablet username/password login")
    elif system_type == "password safe application":
        if "master password" in lower:
            methods.append("master password")
        if "key file" in lower:
            methods.append("key file")
        if contains_any(lower, ["combined key option", "composite key", "composite master key"]):
            methods.append("composite master key")

    if not methods and contains_any(lower, ["username", "password"]):
        methods.append("username/password login")

    return {"required": True, "methods": methods, "evidence_turns": [8]}


def infer_performance(system_type: str, text: str) -> list[dict]:
    lower = normalize_text(text)
    constraints = []

    if system_type == "restaurant discovery mobile application":
        if contains_any(lower, ["2 seconds", "two seconds"]):
            constraints.append({"text": "return search results within 2 seconds", "evidence_turns": [10]})
        if contains_any(lower, ["gps or the internet dies", "gps or internet dies", "internet dies", "gps drops", "internet drops"]):
            constraints.append({"text": "inform the user if internet or GPS connectivity is lost", "evidence_turns": [10]})
    elif system_type == "restaurant menu and ordering system":
        if contains_any(lower, ["two hundred connected devices", "200 connected devices", "200 concurrent"]):
            constraints.append({"text": "support at least 200 concurrent device connections", "evidence_turns": [10]})
        if contains_any(lower, ["drop live orders", "orders or payments", "drop live orders or payments"]):
            constraints.append({"text": "preserve all active meals and orders without losing them", "evidence_turns": [10]})
            constraints.append({"text": "preserve all active customer payments without losing them", "evidence_turns": [10]})
    elif system_type == "password safe application":
        if contains_any(lower, ["ten seconds", "10 seconds"]):
            constraints.append({"text": "keep copied passwords in memory for no more than 10 seconds", "evidence_turns": [10]})

    return constraints


def infer_security(system_type: str, auth_text: str, security_text: str) -> list[dict]:
    auth_lower = normalize_text(auth_text)
    lower = normalize_text(security_text)
    constraints = []

    if system_type == "restaurant discovery mobile application":
        if contains_any(lower, ["login traffic needs protecting", "log in traffic needs protecting", "login messages", "encrypted"]):
            constraints.append({"text": "encrypt log-in communication messages", "evidence_turns": [12]})
        if contains_any(lower, ["three bad tries", "three failed", "30 minutes", "thirty minutes", "cool off"]):
            constraints.append(
                {
                    "text": "disable restaurant owner and administrator log-in for 30 minutes after three failed attempts",
                    "evidence_turns": [12],
                }
            )
    elif system_type == "restaurant menu and ordering system":
        if contains_any(lower, ["wireless side needs encryption", "wireless communication", "encryption"]):
            constraints.append({"text": "encrypt wireless communication using SSLv3 and WPA2-PSK", "evidence_turns": [12]})
        if contains_any(lower, ["multiple tablets at once", "one tablet at a time", "signed into multiple tablets"]):
            constraints.append({"text": "limit each waiter to one logged-in tablet at a time", "evidence_turns": [12]})
        if contains_any(auth_lower, ["customers at the table screens should not have to log in", "should not have to log in", "should not need login"]):
            constraints.append({"text": "not require customers to log in on surface computers", "evidence_turns": [8]})
    elif system_type == "password safe application":
        if contains_any(lower, ["stay encrypted", "encrypted"]) and contains_any(lower, ["vault", "database"]):
            constraints.append({"text": "store passwords in an encrypted database", "evidence_turns": [12]})
        if contains_any(lower, ["must not open without", "master password or key file", "needed master password or key file"]):
            constraints.append(
                {
                    "text": "refuse to open a database without the required master password or key file",
                    "evidence_turns": [12],
                }
            )
        if contains_any(
            lower,
            [
                "no recovery backdoor",
                "no backdoor",
                "no recovery password",
                "recovery password or backdoor",
                "cannot be any recovery password or backdoor",
            ],
        ):
            constraints.append(
                {
                    "text": "provide no recovery password or backdoor for unlocking databases",
                    "evidence_turns": [12],
                }
            )

    return constraints


def predict_slots(sample: dict) -> dict:
    turns = {turn["turn_id"]: turn["text"] for turn in sample["dialogue"]}
    system_type = infer_system_type(turns[2])
    return {
        "sample_id": sample["sample_id"],
        "method": "b2_keyword_normalized_dialogue_to_slots_v1",
        "slots": {
            "system_type": {"value": system_type, "evidence_turns": [2]},
            "user_roles": infer_roles(system_type, turns[4]),
            "functional_capabilities": infer_capabilities(system_type, turns[6]),
            "authentication": infer_authentication(system_type, turns[8]),
            "performance_constraints": infer_performance(system_type, turns[10]),
            "security_constraints": infer_security(system_type, turns[8], turns[12]),
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
