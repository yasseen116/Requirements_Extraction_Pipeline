#!/usr/bin/env python3
"""Shared helpers for generating templated requirements from slot structures."""

from __future__ import annotations


def join_phrases(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def requirement_sentence(text: str) -> str:
    stripped = text.strip().rstrip(".")
    if stripped.startswith("The system shall "):
        return stripped + "."
    return f"The system shall {stripped}."


def generate_requirements_from_slots(sample_id: str, slots: dict, method: str) -> dict:
    functional = []
    non_functional = []

    fr_index = 1
    if slots["authentication"]["required"]:
        methods = join_phrases(slots["authentication"]["methods"])
        functional.append(
            {
                "id": f"FR{fr_index}",
                "text": requirement_sentence(f"support authentication using {methods}"),
                "source_slots": ["authentication"],
            }
        )
        fr_index += 1

    for idx, capability in enumerate(slots["functional_capabilities"]):
        functional.append(
            {
                "id": f"FR{fr_index}",
                "text": requirement_sentence(
                    f"allow {capability['actor']} to {capability['action']}"
                ),
                "source_slots": [f"functional_capabilities[{idx}]"],
            }
        )
        fr_index += 1

    nfr_index = 1
    for idx, constraint in enumerate(slots["performance_constraints"]):
        non_functional.append(
            {
                "id": f"NFR{nfr_index}",
                "category": "performance",
                "text": requirement_sentence(constraint["text"]),
                "source_slots": [f"performance_constraints[{idx}]"],
            }
        )
        nfr_index += 1

    for idx, constraint in enumerate(slots["security_constraints"]):
        non_functional.append(
            {
                "id": f"NFR{nfr_index}",
                "category": "security",
                "text": requirement_sentence(constraint["text"]),
                "source_slots": [f"security_constraints[{idx}]"],
            }
        )
        nfr_index += 1

    return {
        "sample_id": sample_id,
        "method": method,
        "requirements": {
            "functional": functional,
            "non_functional": non_functional,
        },
    }
