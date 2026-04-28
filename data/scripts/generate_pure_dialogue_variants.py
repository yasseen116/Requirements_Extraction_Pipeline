#!/usr/bin/env python3
"""Generate more realistic dialogue variants from an existing PURE dialogue directory.

Variants supported:
- transcript_paraphrase: rewrites user turns (Gemini if available) to sound like transcripts.
- partial_information: drops a fraction of user turns to simulate incomplete elicitation.

This helps address the "gold-to-dialogue leakage" criticism by producing tougher conditions
while still keeping the same evaluation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import llm_router as llm


ROOT = Path(__file__).resolve().parent.parent

REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {
        "text": {"type": "string", "minLength": 1},
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def iter_samples(input_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation.json"} or path.name.endswith(".raw_response.json"):
            continue
        paths.append(path)
    return paths


def rewrite_turn(template: str, text: str) -> tuple[str, str]:
    original = clean_text(text)
    if not original:
        return original, "empty"

    prompt = template.replace("{{TURN_TEXT}}", original)
    try:
        response = llm.generate_json(prompt, REWRITE_SCHEMA, temperature=0.2)
        parsed = llm.parse_first_json_object(response.text)
        rewritten = clean_text(parsed.get("text", ""))
        if rewritten:
            return rewritten, f"{llm.provider()}_rewrite"
    except Exception:
        pass
    return original, "rewrite_error_passthrough"


def deterministic_drop(sample_id: str, turn_id: int, *, drop_rate: float, seed: int) -> bool:
    # Stable drop decision (no global RNG), so re-runs are deterministic.
    token = f"{seed}::{sample_id}::{turn_id}".encode("utf-8")
    h = hashlib.sha256(token).digest()
    value = int.from_bytes(h[:4], "big") / (2**32 - 1)
    return value < drop_rate


def build_transcript_paraphrase(
    payload: dict,
    rewrite_template: str,
) -> dict:
    dialogue = payload.get("dialogue", [])
    trace = []
    rewritten_dialogue = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role != "user":
            rewritten_dialogue.append(turn)
            continue
        new_text, mode = rewrite_turn(rewrite_template, turn.get("text", ""))
        rewritten = {**turn, "text": new_text}
        rewritten_dialogue.append(rewritten)
        trace.append({"turn_id": turn.get("turn_id"), "mode": mode})

    out = dict(payload)
    out["dialogue"] = rewritten_dialogue
    out.setdefault("metadata", {})
    out["metadata"]["dialogue_style"] = "transcript_paraphrase"
    out["dialogue_variant_generation"] = {
        "method": "g_dialogue_variant_transcript_paraphrase_v1",
        "prompt_hash": sha256_text(rewrite_template),
        "schema_hash": sha256_text(json.dumps(REWRITE_SCHEMA, sort_keys=True)),
        "trace": trace,
    }
    return out


def build_partial_information(payload: dict, *, drop_rate: float, seed: int) -> dict:
    dialogue = payload.get("dialogue", [])
    kept = []
    drop_trace = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        turn_id = int(turn.get("turn_id", 0) or 0)
        # Keep all bot turns. Drop only some user turns, but always keep the first user scope answer (turn_id=2).
        if role == "user" and turn_id not in {2}:
            drop = deterministic_drop(payload.get("sample_id", ""), turn_id, drop_rate=drop_rate, seed=seed)
            drop_trace.append({"turn_id": turn_id, "dropped": bool(drop)})
            if drop:
                continue
        kept.append(turn)

    out = dict(payload)
    out["dialogue"] = kept
    out.setdefault("metadata", {})
    out["metadata"]["dialogue_style"] = "partial_information"
    out["dialogue_variant_generation"] = {
        "method": "g_dialogue_variant_partial_information_v1",
        "drop_rate": drop_rate,
        "seed": seed,
        "trace": drop_trace,
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["transcript_paraphrase", "partial_information"],
        required=True,
    )
    parser.add_argument("--rewrite-prompt", type=Path, default=ROOT / "prompts" / "pure_dialogue_turn_rewrite.txt")
    parser.add_argument("--use-gemini", action="store_true")
    parser.add_argument("--partial-drop-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = llm.active_model_name()

    rewrite_template = args.rewrite_prompt.read_text(encoding="utf-8")

    summary = []
    for path in iter_samples(args.input_dir):
        payload = load_json(path)
        if args.variant == "transcript_paraphrase":
            out = build_transcript_paraphrase(payload, rewrite_template)
            out["dialogue_variant_generation"]["model"] = model_name
        else:
            out = build_partial_information(payload, drop_rate=args.partial_drop_rate, seed=args.seed)

        out_path = args.output_dir / path.name
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append(
            {
                "sample_id": out.get("sample_id"),
                "path": str(out_path),
                "variant": args.variant,
                "turn_count": len(out.get("dialogue", [])) if isinstance(out.get("dialogue", []), list) else 0,
            }
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
