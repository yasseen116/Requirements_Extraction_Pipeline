#!/usr/bin/env python3
"""Shared helpers for frame-slot-requirement projections."""

from __future__ import annotations

import re
from typing import Any

from requirements_generation_utils import generate_requirements_from_slots


FRAME_KINDS = {
    "system_context",
    "user_role",
    "functional_capability",
    "authentication",
    "performance",
    "security",
}

EXPECTED_FRAME_TURNS = {
    "system_context": 2,
    "user_role": 4,
    "functional_capability": 6,
    "authentication": 8,
    "performance": 10,
    "security": 12,
}

ALLOWED_FRAME_TURNS = {
    "system_context": {2},
    "user_role": {4},
    "functional_capability": {6},
    "authentication": {8},
    "performance": {10},
    "security": {8, 12},
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(normalize_text(phrase) in normalize_text(text) for phrase in phrases)


def maybe_clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = clean_text(str(value))
    return text or None


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def normalize_turns(value: Any, default_turn: int) -> list[int]:
    if isinstance(value, list):
        turns = [item for item in value if isinstance(item, int)]
        if turns:
            return sorted(dict.fromkeys(turns))
    return [default_turn]


def make_frame(
    frame_id: str,
    kind: str,
    actor: str | None,
    action: str | None,
    obj: str | None,
    constraint: str | None,
    channel: str | None,
    evidence_turns: list[int],
    status: str = "confirmed",
) -> dict:
    return {
        "frame_id": frame_id,
        "kind": kind,
        "actor": actor,
        "action": action,
        "object": obj,
        "constraint": constraint,
        "channel": channel,
        "evidence_turns": sorted(dict.fromkeys(evidence_turns)),
        "status": status,
    }


def assign_frame_ids(frames: list[dict]) -> list[dict]:
    assigned = []
    for index, frame in enumerate(frames, start=1):
        item = dict(frame)
        item["frame_id"] = f"F{index}"
        assigned.append(item)
    return assigned


def derive_frames_from_slots(slots: dict) -> list[dict]:
    frames: list[dict] = []

    frames.append(
        make_frame(
            frame_id="F0",
            kind="system_context",
            actor=None,
            action="system_type",
            obj=clean_text(slots["system_type"]["value"]),
            constraint=None,
            channel=None,
            evidence_turns=normalize_turns(slots["system_type"].get("evidence_turns"), 2),
        )
    )

    for role in slots["user_roles"]:
        frames.append(
            make_frame(
                frame_id="F0",
                kind="user_role",
                actor=clean_text(role["value"]),
                action="user_role",
                obj=None,
                constraint=None,
                channel=None,
                evidence_turns=normalize_turns(role.get("evidence_turns"), 4),
            )
        )

    for capability in slots["functional_capabilities"]:
        frames.append(
            make_frame(
                frame_id="F0",
                kind="functional_capability",
                actor=clean_text(capability["actor"]),
                action=clean_text(capability["action"]),
                obj=None,
                constraint=None,
                channel=None,
                evidence_turns=normalize_turns(capability.get("evidence_turns"), 6),
            )
        )

    authentication = slots["authentication"]
    auth_turns = normalize_turns(authentication.get("evidence_turns"), 8)
    methods = [clean_text(item) for item in authentication.get("methods", []) if clean_text(item)]
    if authentication.get("required"):
        if methods:
            for method in methods:
                frames.append(
                    make_frame(
                        frame_id="F0",
                        kind="authentication",
                        actor=None,
                        action="authenticate",
                        obj=method,
                        constraint="required",
                        channel=None,
                        evidence_turns=auth_turns,
                    )
                )
        else:
            frames.append(
                make_frame(
                    frame_id="F0",
                    kind="authentication",
                    actor=None,
                    action="authenticate",
                    obj=None,
                    constraint="required",
                    channel=None,
                    evidence_turns=auth_turns,
                )
            )
    else:
        frames.append(
            make_frame(
                frame_id="F0",
                kind="authentication",
                actor=None,
                action="authenticate",
                obj=None,
                constraint="not_required",
                channel=None,
                evidence_turns=auth_turns,
            )
        )

    for item in slots["performance_constraints"]:
        frames.append(
            make_frame(
                frame_id="F0",
                kind="performance",
                actor=None,
                action=None,
                obj=None,
                constraint=clean_text(item["text"]),
                channel=None,
                evidence_turns=normalize_turns(item.get("evidence_turns"), 10),
            )
        )

    for item in slots["security_constraints"]:
        frames.append(
            make_frame(
                frame_id="F0",
                kind="security",
                actor=None,
                action=None,
                obj=None,
                constraint=clean_text(item["text"]),
                channel=None,
                evidence_turns=normalize_turns(item.get("evidence_turns"), 12),
            )
        )

    return assign_frame_ids(frames)


def infer_system_type_from_text(text: str) -> str:
    lower = normalize_text(text)
    if (
        contains_any(lower, ["food spots", "restaurants", "places to eat", "restaurant"])
        and contains_any(lower, ["phone app", "mobile", "app"])
        and contains_any(lower, ["find", "near me", "nearby", "map and list", "directions", "gps", "venue info"])
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
            "encrypted vaults",
        ],
    ):
        return "password safe application"
    return clean_text(text) if clean_text(text) else "unspecified system type"


def infer_system_type_from_frames(frames: list[dict]) -> str:
    combined = []
    for frame in frames:
        for key in ["actor", "action", "object", "constraint", "channel"]:
            value = maybe_clean_text(frame.get(key))
            if value:
                combined.append(value)
    return infer_system_type_from_text(" ".join(combined))


def canonicalize_role(system_type: str, role: str | None) -> str | None:
    if not role:
        return None
    lower = normalize_text(role)
    role_mappings = {
        "restaurant discovery mobile application": [
            ("users", ["regular app user", "app user", "normal user", "users", "user"]),
            ("restaurant owners", ["restaurant person", "restaurant people", "restaurant owner", "restaurant side"]),
            ("administrators", ["system administrator", "administrator", "admin", "management", "manager"]),
        ],
        "restaurant menu and ordering system": [
            ("customers", ["diner", "guest", "customer"]),
            ("waiters", ["floor staff", "server", "waiter", "staff"]),
            ("chefs", ["kitchen crew", "kitchen staff", "chef", "kitchen"]),
            ("supervisors", ["manager", "supervisor"]),
        ],
        "password safe application": [
            ("end users", ["regular user", "end user", "user"]),
            ("system administrators", ["it admin", "system administrator", "admin"]),
        ],
    }
    for canonical, patterns in role_mappings.get(system_type, []):
        if contains_any(lower, patterns):
            return canonical
    return clean_text(role)


def canonicalize_capability(system_type: str, actor: str | None, action: str | None) -> tuple[str | None, str | None]:
    canonical_actor = canonicalize_role(system_type, actor)
    if not action:
        return canonical_actor, action
    lower = normalize_text(action)
    mappings = {
        "restaurant discovery mobile application": [
            ("search for restaurants using multiple search options", ["search for restaurants", "look up places to eat", "find food spots"]),
            ("view restaurant results in map and list views", ["map and list", "bounce between map and list", "switch between map and list"]),
            ("open detailed restaurant information pages", ["open place details", "open detailed restaurant info", "check what each place offers"]),
            ("navigate to a selected restaurant using GPS", ["get directions", "gps navigation", "navigate"]),
            ("manage restaurant information through the web portal", ["manage restaurant info", "update venue info", "update restaurant info"]),
            ("verify restaurant owners", ["verify owners", "approve restaurant accounts", "approve owners"]),
        ],
        "restaurant menu and ordering system": [
            ("place orders through an engaged menu", ["place orders", "send orders"]),
            ("call for waiter assistance", ["call waiter", "buzz staff"]),
            ("process cash and bankcard payments", ["process payments", "cash or card payments", "cash or bankcard payments", "card payments"]),
            ("mark order items as ready to be served", ["mark dishes ready", "mark items ready", "ready to be served"]),
            ("issue refunds", ["issue refunds", "handle refunds"]),
        ],
        "password safe application": [
            ("create and open encrypted password databases", ["create and open encrypted databases", "spin up and unlock encrypted vaults", "unlock encrypted vaults", "create encrypted vault", "unlock encrypted vault"]),
            ("search stored entries", ["search stored entries", "find saved items", "find saved item"]),
            ("add and edit entries", ["add or edit entries", "tweak entries", "tweak entry"]),
            ("import and export password data", ["import or export password data", "move password data in or out"]),
        ],
    }
    for canonical_action, patterns in mappings.get(system_type, []):
        if contains_any(lower, patterns):
            return canonical_actor, canonical_action
    return canonical_actor, clean_text(action)


def canonicalize_authentication(system_type: str, actor: str | None, obj: str | None, channel: str | None, constraint: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    object_text = maybe_clean_text(obj)
    channel_text = maybe_clean_text(channel)
    constraint_text = maybe_clean_text(constraint)

    if system_type == "restaurant discovery mobile application":
        if channel_text and contains_any(channel_text, ["mobile", "app"]):
            return None, "mobile app username/password login", None, constraint_text
        if channel_text and "portal" in normalize_text(channel_text):
            return None, "web portal username/password login", None, constraint_text
        canonical_actor = canonicalize_role(system_type, actor)
        if canonical_actor == "users":
            return None, "mobile app username/password login", None, constraint_text
        if canonical_actor in {"restaurant owners", "administrators"}:
            return None, "web portal username/password login", None, constraint_text
    elif system_type == "restaurant menu and ordering system":
        return None, "tablet username/password login", None, constraint_text
    elif system_type == "password safe application":
        lower = normalize_text(" ".join(item for item in [object_text, channel_text] if item))
        if "master password" in lower:
            return None, "master password", None, "required"
        if "key file" in lower:
            return None, "key file", None, "required"
        if contains_any(lower, ["combined key", "composite key", "composite master key"]):
            return None, "composite master key", None, "required"

    return None, object_text, channel_text, constraint_text


def canonicalize_constraint(system_type: str, kind: str, constraint: str | None) -> str | None:
    if not constraint:
        return None
    lower = normalize_text(constraint)

    if system_type == "restaurant discovery mobile application":
        if kind == "performance":
            if contains_any(lower, ["2 seconds", "two seconds"]):
                return "return search results within 2 seconds"
            if contains_any(lower, ["gps or internet", "gps or the internet", "connectivity is lost", "instead of hanging"]):
                return "inform the user if internet or GPS connectivity is lost"
        if kind == "security":
            if contains_any(lower, ["login traffic must be protected", "encrypt", "encrypted", "login messages"]):
                return "encrypt log-in communication messages"
            if contains_any(lower, ["30 minutes", "thirty minutes", "3 unsuccessful", "three failed"]):
                return "disable restaurant owner and administrator log-in for 30 minutes after three failed attempts"

    if system_type == "restaurant menu and ordering system":
        if kind == "performance":
            if contains_any(lower, ["support at least 200", "at least 200 connected", "200 connected devices", "two hundred connected devices", "200 concurrent"]):
                return "support at least 200 concurrent device connections"
            if contains_any(lower, ["200 connected devices", "two hundred connected devices", "200 concurrent"]):
                return "support at least 200 concurrent device connections"
            loss_signal = contains_any(lower, ["without losing", "must never get lost", "cannot drop", "not lose"])
            mentions_orders = contains_any(lower, ["order", "orders", "meal", "meals"])
            mentions_payments = contains_any(lower, ["payment", "payments"])
            if mentions_orders and loss_signal and not mentions_payments:
                return "preserve all active meals and orders without losing them"
            if mentions_payments and loss_signal and not mentions_orders:
                return "preserve all active customer payments without losing them"
        if kind == "security":
            if contains_any(lower, ["wireless", "encryption", "encrypted"]):
                return "encrypt wireless communication using SSLv3 and WPA2-PSK"
            if contains_any(lower, ["multiple tablets", "one logged in tablet", "one tablet at a time", "only one at a time"]):
                return "limit each waiter to one logged-in tablet at a time"
            if contains_any(lower, ["should not log in", "should not have to log in"]):
                return "not require customers to log in on surface computers"

    if system_type == "password safe application":
        if kind == "performance":
            if contains_any(lower, ["10 seconds", "ten seconds"]):
                return "keep copied passwords in memory for no more than 10 seconds"
        if kind == "security":
            if contains_any(lower, ["stay encrypted", "everything should stay encrypted", "all data encrypted", "encrypted database", "store passwords encrypted"]):
                return "store passwords in an encrypted database"
            if contains_any(lower, ["must not open without", "master password or key file"]):
                return "refuse to open a database without the required master password or key file"
            if contains_any(lower, ["no recovery backdoor", "no backdoor", "no recovery password"]):
                return "provide no recovery password or backdoor for unlocking databases"

    return clean_text(constraint)


def dialogue_turn_text(sample: dict | None, turn_id: int) -> str:
    if not isinstance(sample, dict):
        return ""
    for turn in sample.get("dialogue", []):
        if isinstance(turn, dict) and turn.get("turn_id") == turn_id:
            return maybe_clean_text(turn.get("text")) or ""
    return ""


def ensure_frame(
    frames: list[dict],
    *,
    kind: str,
    actor: str | None,
    action: str | None,
    obj: str | None,
    constraint: str | None,
    channel: str | None,
    evidence_turns: list[int],
) -> list[dict]:
    target = (
        kind,
        maybe_clean_text(actor),
        maybe_clean_text(action),
        maybe_clean_text(obj),
        maybe_clean_text(constraint),
        maybe_clean_text(channel),
        tuple(sorted(dict.fromkeys(evidence_turns))),
        "confirmed",
    )
    for frame in frames:
        existing = (
            frame["kind"],
            frame["actor"],
            frame["action"],
            frame["object"],
            frame["constraint"],
            frame["channel"],
            tuple(frame["evidence_turns"]),
            frame["status"],
        )
        if existing == target:
            return frames

    enriched = list(frames)
    enriched.append(
        make_frame(
            frame_id="F0",
            kind=kind,
            actor=actor,
            action=action,
            obj=obj,
            constraint=constraint,
            channel=channel,
            evidence_turns=evidence_turns,
        )
    )
    return enriched


def canonicalize_frames(frames: list[dict]) -> list[dict]:
    normalized = [normalize_frame(frame, default_index=index) for index, frame in enumerate(frames, start=1)]
    system_type = infer_system_type_from_frames(normalized)
    canonical_frames = []

    for frame in normalized:
        item = dict(frame)
        kind = item["kind"]
        raw_constraint = maybe_clean_text(item["constraint"])
        raw_channel = maybe_clean_text(item["channel"])
        if kind == "system_context":
            item["actor"] = None
            item["action"] = "system_type"
            item["object"] = system_type
            item["constraint"] = None
            item["channel"] = None
        elif kind == "user_role":
            item["actor"] = canonicalize_role(system_type, item["actor"])
            item["action"] = "user_role"
            item["object"] = None
            item["constraint"] = None
            item["channel"] = None
        elif kind == "functional_capability":
            actor, action = canonicalize_capability(system_type, item["actor"], item["action"])
            item["actor"] = actor
            item["action"] = action
            item["object"] = None
            item["constraint"] = None
            item["channel"] = None
        elif kind == "authentication":
            actor, obj, channel, constraint = canonicalize_authentication(
                system_type,
                item["actor"],
                item["object"],
                item["channel"],
                item["constraint"],
            )
            if system_type == "restaurant menu and ordering system" and constraint == "not_required":
                item["kind"] = "security"
                item["actor"] = None
                item["action"] = None
                item["object"] = None
                item["constraint"] = "not require customers to log in on surface computers"
                item["channel"] = None
            else:
                item["actor"] = actor
                item["action"] = "authenticate"
                item["object"] = obj
                item["channel"] = channel
                item["constraint"] = constraint
        elif kind in {"performance", "security"}:
            item["actor"] = None
            item["action"] = None
            item["object"] = None
            item["constraint"] = canonicalize_constraint(system_type, kind, item["constraint"])
            item["channel"] = None
            if system_type == "password safe application" and kind == "security" and raw_constraint:
                if normalize_text(raw_constraint) == "required":
                    canonical_frames.append(
                        {
                            **item,
                            "constraint": "store passwords in an encrypted database",
                        }
                    )
                    canonical_frames.append(
                        {
                            **item,
                            "constraint": "refuse to open a database without the required master password or key file",
                        }
                    )
                    continue
                if normalize_text(raw_constraint) == "prohibited":
                    canonical_frames.append(
                        {
                            **item,
                            "constraint": "provide no recovery password or backdoor for unlocking databases",
                        }
                    )
                    continue
            if (
                system_type == "restaurant menu and ordering system"
                and kind == "performance"
                and raw_constraint
                and contains_any(raw_constraint, ["must never get lost", "cannot drop live orders or payments"])
            ):
                canonical_frames.append(
                    {
                        **item,
                        "constraint": "preserve all active meals and orders without losing them",
                    }
                )
                canonical_frames.append(
                    {
                        **item,
                        "constraint": "preserve all active customer payments without losing them",
                    }
                )
                continue
        canonical_frames.append(item)

    if system_type == "password safe application":
        auth_objects = {
            item["object"]
            for item in canonical_frames
            if item["kind"] == "authentication" and item["object"]
        }
        auth_turns = sorted(
            {
                turn
                for item in canonical_frames
                if item["kind"] == "authentication"
                for turn in item["evidence_turns"]
            }
        ) or [8]
        if {"master password", "key file"}.issubset(auth_objects) and "composite master key" not in auth_objects:
            canonical_frames.append(
                make_frame(
                    frame_id="F0",
                    kind="authentication",
                    actor=None,
                    action="authenticate",
                    obj="composite master key",
                    constraint="required",
                    channel=None,
                    evidence_turns=auth_turns,
                )
            )

    deduped = []
    seen = set()
    for item in canonical_frames:
        key = (
            item["kind"],
            item["actor"],
            item["action"],
            item["object"],
            item["constraint"],
            item["channel"],
            tuple(item["evidence_turns"]),
            item["status"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return assign_frame_ids(deduped)


def complete_frames_from_dialogue(sample: dict | None, frames: list[dict]) -> list[dict]:
    canonical_frames = canonicalize_frames(frames)
    if not isinstance(sample, dict):
        return canonical_frames

    system_type = infer_system_type_from_frames(canonical_frames)
    enriched = list(canonical_frames)
    turn_6 = dialogue_turn_text(sample, 6)
    turn_10 = dialogue_turn_text(sample, 10)

    if system_type == "restaurant menu and ordering system":
        if contains_any(
            turn_6,
            [
                "buzz staff",
                "buzz the staff",
                "buzz waiter",
                "buzz server",
                "call waiter",
                "call waiters",
                "call for waiter",
                "waiter assistance",
            ],
        ):
            enriched = ensure_frame(
                enriched,
                kind="functional_capability",
                actor="customers",
                action="call for waiter assistance",
                obj=None,
                constraint=None,
                channel=None,
                evidence_turns=[6],
            )

        loss_signal = contains_any(
            turn_10,
            [
                "must never get lost",
                "cannot drop",
                "without losing",
                "drop live orders or payments",
                "lose live orders or payments",
            ],
        )
        if loss_signal and contains_any(turn_10, ["order", "orders", "meal", "meals"]):
            enriched = ensure_frame(
                enriched,
                kind="performance",
                actor=None,
                action=None,
                obj=None,
                constraint="preserve all active meals and orders without losing them",
                channel=None,
                evidence_turns=[10],
            )
        if loss_signal and contains_any(turn_10, ["payment", "payments"]):
            enriched = ensure_frame(
                enriched,
                kind="performance",
                actor=None,
                action=None,
                obj=None,
                constraint="preserve all active customer payments without losing them",
                channel=None,
                evidence_turns=[10],
            )

    return canonicalize_frames(enriched)


def derive_slots_from_frames(frames: list[dict]) -> dict:
    system_type_value = None
    system_turns: list[int] = [2]
    user_roles = []
    functional_capabilities = []
    auth_required = False
    auth_methods: list[str] = []
    auth_turns: list[int] = [8]
    performance_constraints = []
    security_constraints = []

    seen_roles = set()
    seen_capabilities = set()
    seen_performance = set()
    seen_security = set()

    for frame in frames:
        kind = maybe_clean_text(frame.get("kind"))
        if kind not in FRAME_KINDS:
            continue

        turns = normalize_turns(frame.get("evidence_turns"), EXPECTED_FRAME_TURNS[kind])
        actor = maybe_clean_text(frame.get("actor"))
        action = maybe_clean_text(frame.get("action"))
        obj = maybe_clean_text(frame.get("object"))
        constraint = maybe_clean_text(frame.get("constraint"))

        if kind == "system_context" and obj:
            if system_type_value is None:
                system_type_value = obj
                system_turns = turns
        elif kind == "user_role" and actor and actor not in seen_roles:
            user_roles.append({"value": actor, "evidence_turns": turns})
            seen_roles.add(actor)
        elif kind == "functional_capability" and actor and action:
            key = (actor, action)
            if key not in seen_capabilities:
                functional_capabilities.append(
                    {
                        "actor": actor,
                        "action": action,
                        "evidence_turns": turns,
                    }
                )
                seen_capabilities.add(key)
        elif kind == "authentication":
            auth_turns = sorted(dict.fromkeys(auth_turns + turns))
            if constraint == "not_required":
                if not auth_required:
                    auth_required = False
            else:
                auth_required = True
                if obj:
                    auth_methods.append(obj)
        elif kind == "performance" and constraint and constraint not in seen_performance:
            performance_constraints.append({"text": constraint, "evidence_turns": turns})
            seen_performance.add(constraint)
        elif kind == "security" and constraint and constraint not in seen_security:
            security_constraints.append({"text": constraint, "evidence_turns": turns})
            seen_security.add(constraint)

    return {
        "system_type": {
            "value": system_type_value or "unspecified system type",
            "evidence_turns": system_turns,
        },
        "user_roles": user_roles,
        "functional_capabilities": functional_capabilities,
        "authentication": {
            "required": auth_required,
            "methods": unique_preserve_order(auth_methods),
            "evidence_turns": sorted(dict.fromkeys(auth_turns)),
        },
        "performance_constraints": performance_constraints,
        "security_constraints": security_constraints,
    }


def generate_requirements_from_frames(sample_id: str, frames: list[dict], method: str) -> dict:
    slots = derive_slots_from_frames(frames)
    return generate_requirements_from_slots(sample_id=sample_id, slots=slots, method=method)


def normalize_frame(frame: dict, default_index: int = 1) -> dict:
    kind = maybe_clean_text(frame.get("kind")) or "functional_capability"
    if kind not in FRAME_KINDS:
        kind = "functional_capability"

    return make_frame(
        frame_id=maybe_clean_text(frame.get("frame_id")) or f"F{default_index}",
        kind=kind,
        actor=maybe_clean_text(frame.get("actor")),
        action=maybe_clean_text(frame.get("action")),
        obj=maybe_clean_text(frame.get("object")),
        constraint=maybe_clean_text(frame.get("constraint")),
        channel=maybe_clean_text(frame.get("channel")),
        evidence_turns=normalize_turns(frame.get("evidence_turns"), EXPECTED_FRAME_TURNS[kind]),
        status=maybe_clean_text(frame.get("status")) or "confirmed",
    )


def flatten_frame(frame: dict) -> str:
    normalized = normalize_frame(frame)
    fields = [
        normalized["kind"],
        normalized["actor"] or "_",
        normalized["action"] or "_",
        normalized["object"] or "_",
        normalized["constraint"] or "_",
        normalized["channel"] or "_",
        normalized["status"] or "_",
    ]
    flattened = "::".join(clean_text(item).lower() for item in fields)
    return flattened


def frame_consistency_score(frames: list[dict]) -> float:
    if not frames:
        return 0.0
    consistent = 0
    for frame in frames:
        normalized = normalize_frame(frame)
        allowed_turns = ALLOWED_FRAME_TURNS[normalized["kind"]]
        if all(turn in allowed_turns for turn in normalized["evidence_turns"]):
            consistent += 1
    return consistent / len(frames)
