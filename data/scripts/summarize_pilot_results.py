#!/usr/bin/env python3
"""Summarize clean vs noisy pilot results in JSON and Markdown."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRAME_CLEAN = ROOT / "outputs" / "b1_rule_based_frames" / "evaluation.json"
FRAME_NOISY = ROOT / "outputs" / "b1_rule_based_frames_noisy" / "evaluation.json"
SLOT_CLEAN = ROOT / "outputs" / "b1_rule_based_slots" / "evaluation.json"
SLOT_NOISY = ROOT / "outputs" / "b1_rule_based_slots_noisy" / "evaluation.json"
REQ_CLEAN = ROOT / "outputs" / "b1_rule_based_requirements" / "evaluation.json"
REQ_NOISY = ROOT / "outputs" / "b1_rule_based_requirements_noisy" / "evaluation.json"
OUTPUT_JSON = ROOT / "outputs" / "pilot_results_summary.json"
OUTPUT_MD = ROOT / "outputs" / "pilot_results_summary.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_summary(frame_eval: dict, slot_eval: dict, req_eval: dict) -> dict:
    return {
        "frame_overall_f1": frame_eval["aggregate"]["overall_f1_macro"],
        "frame_consistency": frame_eval["aggregate"]["dialogue_frame_consistency_macro"],
        "slot_overall_f1": slot_eval["aggregate"]["overall_f1_macro"],
        "slot_authentication_methods_f1": slot_eval["aggregate"]["per_slot"]["authentication_methods"]["f1_macro"],
        "slot_functional_capabilities_f1": slot_eval["aggregate"]["per_slot"]["functional_capabilities"]["f1_macro"],
        "slot_security_constraints_f1": slot_eval["aggregate"]["per_slot"]["security_constraints"]["f1_macro"],
        "requirement_overall_f1": req_eval["aggregate"]["overall_f1_macro"],
        "requirement_coverage": req_eval["aggregate"]["coverage_macro"],
        "requirement_hallucination_rate": req_eval["aggregate"]["hallucination_rate_macro"],
        "source_slot_coverage": req_eval["aggregate"]["source_slot_coverage_macro"],
    }


def main() -> int:
    frame_clean = load(FRAME_CLEAN)
    frame_noisy = load(FRAME_NOISY)
    slot_clean = load(SLOT_CLEAN)
    slot_noisy = load(SLOT_NOISY)
    req_clean = load(REQ_CLEAN)
    req_noisy = load(REQ_NOISY)

    clean = extract_summary(frame_clean, slot_clean, req_clean)
    noisy = extract_summary(frame_noisy, slot_noisy, req_noisy)
    deltas = {key: noisy[key] - clean[key] for key in clean}

    payload = {
        "clean": clean,
        "noisy": noisy,
        "delta_noisy_minus_clean": deltas,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Pilot Results Summary",
        "",
        "| Metric | Clean | Noisy | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    label_map = {
        "frame_overall_f1": "Frame Overall F1",
        "frame_consistency": "Dialogue-Frame Consistency",
        "slot_overall_f1": "Slot Overall F1",
        "slot_authentication_methods_f1": "Slot Auth Methods F1",
        "slot_functional_capabilities_f1": "Slot Functional Capabilities F1",
        "slot_security_constraints_f1": "Slot Security Constraints F1",
        "requirement_overall_f1": "Requirement Overall F1",
        "requirement_coverage": "Requirement Coverage",
        "requirement_hallucination_rate": "Requirement Hallucination Rate",
        "source_slot_coverage": "Requirement Source Slot Coverage",
    }

    for key, label in label_map.items():
        lines.append(
            f"| {label} | {clean[key]:.4f} | {noisy[key]:.4f} | {deltas[key]:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The clean pilot is saturated by the current rule baseline because the dataset is tiny and the phrasing remains close to the schema.",
            "- The noisy pilot exposes the real weakness of the rule baseline at both the frame and slot levels: capabilities, roles, and constraints break quickly under paraphrase.",
            "- This makes the next step clear: compare a stronger LLM frame extractor against the same evaluation pipeline rather than changing the metrics.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
