#!/usr/bin/env python3
"""Generate dialogue -> slots predictions with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    extract_first_json_object,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "synthetic" / "pilot_noisy"
DEFAULT_OUTPUT = ROOT / "outputs" / "b3_openai_compatible_slots_noisy"
DEFAULT_PROMPT = ROOT / "prompts" / "dialogue_to_slots_openai_compatible.txt"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.startswith("template_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "sample_id" not in payload or "dialogue" not in payload:
            continue
        samples.append(payload)
    return samples


def render_dialogue(sample: dict) -> str:
    lines = []
    for turn in sample["dialogue"]:
        lines.append(f"{turn['turn_id']}. {turn['role']}: {turn['text']}")
    return "\n".join(lines)


def build_user_prompt(template_text: str, sample: dict) -> str:
    return (
        template_text.replace("{{SAMPLE_ID}}", sample["sample_id"])
        .replace("{{DOMAIN}}", sample["metadata"]["domain"])
        .replace("{{DIALOGUE}}", render_dialogue(sample))
    )


def ensure_turn_list(value: object, default_turn: int) -> list[int]:
    if isinstance(value, list):
        turns = [item for item in value if isinstance(item, int)]
        if turns:
            return turns
    return [default_turn]


def normalize_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for item in values:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if value and value not in seen:
            cleaned.append(value)
            seen.add(value)
    return cleaned


def normalize_slot_payload(sample: dict, payload: dict) -> dict:
    slots = payload.get("slots", payload)

    system_type_raw = slots.get("system_type", {})
    if isinstance(system_type_raw, dict):
        system_type_value = str(system_type_raw.get("value", "")).strip()
        system_type_turns = ensure_turn_list(system_type_raw.get("evidence_turns"), 2)
    else:
        system_type_value = str(system_type_raw).strip()
        system_type_turns = [2]

    user_roles = []
    for item in slots.get("user_roles", []):
        if isinstance(item, dict):
            value = str(item.get("value", "")).strip()
            if value:
                user_roles.append(
                    {
                        "value": value,
                        "evidence_turns": ensure_turn_list(item.get("evidence_turns"), 4),
                    }
                )

    functional_capabilities = []
    for item in slots.get("functional_capabilities", []):
        if isinstance(item, dict):
            actor = str(item.get("actor", "")).strip()
            action = str(item.get("action", "")).strip()
            if actor and action:
                functional_capabilities.append(
                    {
                        "actor": actor,
                        "action": action,
                        "evidence_turns": ensure_turn_list(item.get("evidence_turns"), 6),
                    }
                )

    authentication_raw = slots.get("authentication", {})
    authentication = {
        "required": bool(authentication_raw.get("required", False)) if isinstance(authentication_raw, dict) else False,
        "methods": normalize_string_list(authentication_raw.get("methods", [])) if isinstance(authentication_raw, dict) else [],
        "evidence_turns": ensure_turn_list(authentication_raw.get("evidence_turns"), 8)
        if isinstance(authentication_raw, dict)
        else [8],
    }

    performance_constraints = []
    for item in slots.get("performance_constraints", []):
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                performance_constraints.append(
                    {
                        "text": text,
                        "evidence_turns": ensure_turn_list(item.get("evidence_turns"), 10),
                    }
                )

    security_constraints = []
    for item in slots.get("security_constraints", []):
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                security_constraints.append(
                    {
                        "text": text,
                        "evidence_turns": ensure_turn_list(item.get("evidence_turns"), 12),
                    }
                )

    return {
        "sample_id": sample["sample_id"],
        "method": "b3_openai_compatible_dialogue_to_slots_v1",
        "slots": {
            "system_type": {
                "value": system_type_value,
                "evidence_turns": system_type_turns,
            },
            "user_roles": user_roles,
            "functional_capabilities": functional_capabilities,
            "authentication": authentication,
            "performance_constraints": performance_constraints,
            "security_constraints": security_constraints,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    prompt_text = args.prompt_path.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        if not samples:
            print("No samples found for dry run.")
            return 0
        sample = samples[0]
        preview = {
            "sample_id": sample["sample_id"],
            "prompt_path": display_path(args.prompt_path),
            "user_prompt": build_user_prompt(prompt_text, sample),
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    config = OpenAICompatibleConfig.from_env()
    client = OpenAICompatibleClient(config)
    summary = []

    for sample in samples:
        user_prompt = build_user_prompt(prompt_text, sample)
        response = client.chat(
            system_prompt=(
                "You extract fixed-schema requirements slots from controlled dialogue. "
                "Return JSON only."
            ),
            user_prompt=user_prompt,
            temperature=0.0,
        )
        parsed = extract_first_json_object(response["text"])
        normalized = normalize_slot_payload(sample, parsed)

        output_path = args.output_dir / f"{sample['sample_id']}.json"
        output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        raw_path = args.output_dir / f"{sample['sample_id']}.raw_response.json"
        raw_path.write_text(
            json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "usage": response["usage"],
                    "text": response["text"],
                    "raw_response": response["raw_response"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        summary.append(
            {
                "sample_id": sample["sample_id"],
                "path": display_path(output_path),
                "raw_response_path": display_path(raw_path),
                "usage": response["usage"],
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
