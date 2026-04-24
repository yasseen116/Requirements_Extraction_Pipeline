#!/usr/bin/env python3
"""Validate source-grounded conversational requirements samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from frame_slot_utils import EXPECTED_FRAME_TURNS, derive_slots_from_frames, frame_consistency_score


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "manual_gold"
DEFAULT_OUTPUT = ROOT / "outputs" / "source_grounded_validation_summary.json"

EXPECTED_QUESTIONS = {
    1: "What kind of system do you want?",
    3: "Who will use it?",
    5: "What should users be able to do?",
    7: "Do users need login or authentication?",
    9: "Are there performance requirements?",
    11: "Are there security requirements?",
}


def load_samples(input_dir: Path) -> list[tuple[Path, dict]]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("template_"):
            continue
        samples.append((path, json.loads(path.read_text())))
    return samples


def collect_slot_refs(sample: dict) -> set[str]:
    refs = {"system_type", "authentication"}
    refs.update(f"user_roles[{idx}]" for idx, _ in enumerate(sample["slots"]["user_roles"]))
    refs.update(
        f"functional_capabilities[{idx}]"
        for idx, _ in enumerate(sample["slots"]["functional_capabilities"])
    )
    refs.update(
        f"performance_constraints[{idx}]"
        for idx, _ in enumerate(sample["slots"]["performance_constraints"])
    )
    refs.update(
        f"security_constraints[{idx}]"
        for idx, _ in enumerate(sample["slots"]["security_constraints"])
    )
    return refs


def check_requirement_block(
    sample: dict,
    path: Path,
    block_name: str,
    requirements: list[dict],
    valid_refs: set[str],
) -> list[str]:
    errors = []
    ids = set()
    for requirement in requirements:
        req_id = requirement.get("id")
        if not req_id or req_id in ids:
            errors.append(f"{path.name}: duplicate or missing requirement id in {block_name}")
        ids.add(req_id)

        text = requirement.get("text", "")
        if not text.startswith("The system shall "):
            errors.append(f"{path.name}: requirement {req_id} does not start with 'The system shall'")

        refs = requirement.get("source_slots", [])
        if not refs:
            errors.append(f"{path.name}: requirement {req_id} is missing source_slots")
        for ref in refs:
            if ref not in valid_refs:
                errors.append(f"{path.name}: requirement {req_id} references unknown slot path {ref}")

        if block_name == "non_functional":
            category = requirement.get("category")
            if category not in {"performance", "security"}:
                errors.append(f"{path.name}: non-functional requirement {req_id} has invalid category")
    return errors


def validate_sample(path: Path, sample: dict) -> tuple[list[str], dict]:
    errors = []
    summary = {
        "sample_id": sample.get("sample_id"),
        "split": sample.get("metadata", {}).get("split"),
        "domain": sample.get("metadata", {}).get("domain"),
        "functional_requirements": len(sample.get("requirements", {}).get("functional", [])),
        "non_functional_requirements": len(sample.get("requirements", {}).get("non_functional", [])),
    }

    required_top = {"sample_id", "metadata", "source", "dialogue", "frames", "slots", "requirements"}
    missing_top = sorted(required_top - sample.keys())
    if missing_top:
        errors.append(f"{path.name}: missing top-level fields {missing_top}")
        return errors, summary

    dialogue = sample["dialogue"]
    if len(dialogue) != 12:
        errors.append(f"{path.name}: expected 12 dialogue turns, found {len(dialogue)}")

    user_turn_ids = set()
    for expected_id, turn in enumerate(dialogue, start=1):
        if turn.get("turn_id") != expected_id:
            errors.append(f"{path.name}: turn_id mismatch at position {expected_id}")
        expected_role = "bot" if expected_id % 2 == 1 else "user"
        if turn.get("role") != expected_role:
            errors.append(f"{path.name}: expected role {expected_role} at turn {expected_id}")
        if expected_role == "bot":
            expected_text = EXPECTED_QUESTIONS.get(expected_id)
            if turn.get("text") != expected_text:
                errors.append(f"{path.name}: unexpected bot question at turn {expected_id}")
        else:
            user_turn_ids.add(expected_id)

    if not sample["source"].get("source_sentence_ids"):
        errors.append(f"{path.name}: source_sentence_ids must not be empty")

    frames = sample["frames"]
    if not frames:
        errors.append(f"{path.name}: frames must not be empty")
    frame_ids = set()
    for frame in frames:
        frame_id = frame.get("frame_id")
        if not frame_id or frame_id in frame_ids:
            errors.append(f"{path.name}: duplicate or missing frame_id")
        frame_ids.add(frame_id)

        kind = frame.get("kind")
        if kind not in EXPECTED_FRAME_TURNS:
            errors.append(f"{path.name}: invalid frame kind {kind}")
            continue

        turns = frame.get("evidence_turns", [])
        if not turns:
            errors.append(f"{path.name}: frame {frame_id} missing evidence_turns")
            continue
        for turn_id in turns:
            if turn_id not in user_turn_ids:
                errors.append(f"{path.name}: frame {frame_id} has invalid evidence turn {turn_id}")

        if frame.get("status") != "confirmed":
            errors.append(f"{path.name}: frame {frame_id} must have status=confirmed")

    if frames and frame_consistency_score(frames) < 1.0:
        errors.append(f"{path.name}: at least one frame uses an unexpected dialogue turn")

    slot_groups = sample["slots"]
    if not slot_groups["user_roles"]:
        errors.append(f"{path.name}: user_roles must not be empty")
    if not slot_groups["functional_capabilities"]:
        errors.append(f"{path.name}: functional_capabilities must not be empty")

    evidence_containers = [
        slot_groups["system_type"],
        *slot_groups["user_roles"],
        *slot_groups["functional_capabilities"],
        slot_groups["authentication"],
        *slot_groups["performance_constraints"],
        *slot_groups["security_constraints"],
    ]

    for container in evidence_containers:
        turns = container.get("evidence_turns", [])
        if not turns:
            errors.append(f"{path.name}: missing evidence_turns in slot container")
            continue
        for turn_id in turns:
            if turn_id not in user_turn_ids:
                errors.append(f"{path.name}: invalid evidence turn {turn_id}")

    derived_slots = derive_slots_from_frames(sample["frames"])
    if json.dumps(derived_slots, sort_keys=True) != json.dumps(sample["slots"], sort_keys=True):
        errors.append(f"{path.name}: slots are not a deterministic projection of frames")

    valid_refs = collect_slot_refs(sample)
    errors.extend(
        check_requirement_block(
            sample,
            path,
            "functional",
            sample["requirements"]["functional"],
            valid_refs,
        )
    )
    errors.extend(
        check_requirement_block(
            sample,
            path,
            "non_functional",
            sample["requirements"]["non_functional"],
            valid_refs,
        )
    )

    return errors, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples = load_samples(args.input_dir)
    if not samples:
        print("No samples found.", file=sys.stderr)
        return 1

    all_errors = []
    split_counts = Counter()
    domain_counts = Counter()
    sample_summaries = []

    for path, sample in samples:
        errors, summary = validate_sample(path, sample)
        all_errors.extend(errors)
        sample_summaries.append(summary)
        split_counts.update([summary["split"]])
        domain_counts.update([summary["domain"]])

    payload = {
        "sample_count": len(samples),
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "samples": sample_summaries,
        "errors": all_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    if all_errors:
        print(json.dumps(payload, indent=2))
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
