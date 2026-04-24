#!/usr/bin/env python3
"""Compare pilot results across baselines and conditions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


FILES = {
    "b1_clean_frames": ROOT / "outputs" / "b1_rule_based_frames" / "evaluation.json",
    "b1_noisy_frames": ROOT / "outputs" / "b1_rule_based_frames_noisy" / "evaluation.json",
    "b1_clean_slots": ROOT / "outputs" / "b1_rule_based_slots" / "evaluation.json",
    "b1_noisy_slots": ROOT / "outputs" / "b1_rule_based_slots_noisy" / "evaluation.json",
    "b1_clean_req": ROOT / "outputs" / "b1_rule_based_requirements" / "evaluation.json",
    "b1_noisy_req": ROOT / "outputs" / "b1_rule_based_requirements_noisy" / "evaluation.json",
    "b2_clean_frames": ROOT / "outputs" / "b2_keyword_normalized_frames" / "evaluation.json",
    "b2_noisy_frames": ROOT / "outputs" / "b2_keyword_normalized_frames_noisy" / "evaluation.json",
    "b2_clean_slots": ROOT / "outputs" / "b2_keyword_normalized_slots" / "evaluation.json",
    "b2_noisy_slots": ROOT / "outputs" / "b2_keyword_normalized_slots_noisy" / "evaluation.json",
    "b2_clean_req": ROOT / "outputs" / "b2_keyword_normalized_requirements" / "evaluation.json",
    "b2_noisy_req": ROOT / "outputs" / "b2_keyword_normalized_requirements_noisy" / "evaluation.json",
    "b3_clean_req": ROOT / "outputs" / "b3_pilot_direct_requirements" / "evaluation.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize(frame_eval: dict, slot_eval: dict, req_eval: dict) -> dict:
    return {
        "frame_f1": frame_eval["aggregate"]["overall_f1_macro"],
        "frame_consistency": frame_eval["aggregate"]["dialogue_frame_consistency_macro"],
        "slot_f1": slot_eval["aggregate"]["overall_f1_macro"],
        "slot_functional_capability_f1": slot_eval["aggregate"]["per_slot"]["functional_capabilities"]["f1_macro"],
        "slot_security_f1": slot_eval["aggregate"]["per_slot"]["security_constraints"]["f1_macro"],
        "requirement_f1": req_eval["aggregate"]["overall_f1_macro"],
        "requirement_coverage": req_eval["aggregate"]["coverage_macro"],
        "hallucination_rate": req_eval["aggregate"]["hallucination_rate_macro"],
    }


def maybe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load(path)


def main() -> int:
    data = {name: maybe_load(path) for name, path in FILES.items()}
    payload = {
        "b1_clean": summarize(data["b1_clean_frames"], data["b1_clean_slots"], data["b1_clean_req"]),
        "b1_noisy": summarize(data["b1_noisy_frames"], data["b1_noisy_slots"], data["b1_noisy_req"]),
        "b2_clean": summarize(data["b2_clean_frames"], data["b2_clean_slots"], data["b2_clean_req"]),
        "b2_noisy": summarize(data["b2_noisy_frames"], data["b2_noisy_slots"], data["b2_noisy_req"]),
    }
    if data.get("b3_clean_req"):
        # Direct LLM baseline is requirements-only; frame/slot metrics are not applicable.
        req_eval = data["b3_clean_req"]
        payload["b3_clean"] = {
            "frame_f1": None,
            "frame_consistency": None,
            "slot_f1": None,
            "slot_functional_capability_f1": None,
            "slot_security_f1": None,
            "requirement_f1": req_eval["aggregate"]["overall_f1_macro"],
            "requirement_coverage": req_eval["aggregate"]["coverage_macro"],
            "hallucination_rate": req_eval["aggregate"]["hallucination_rate_macro"],
        }

    latest_g1_path = ROOT / "outputs" / "g1_gemini_latest_run.json"
    if latest_g1_path.exists():
        latest = load(latest_g1_path)
        run_dir = Path(latest["run_dir"])
        for condition in ["clean", "noisy"]:
            frame_eval = maybe_load(run_dir / condition / "metrics" / "frame_evaluation.json")
            slot_eval = maybe_load(run_dir / condition / "metrics" / "slot_evaluation.json")
            req_eval = maybe_load(run_dir / condition / "metrics" / "requirement_evaluation.json")
            if frame_eval and slot_eval and req_eval:
                payload[f"g1_{condition}"] = summarize(frame_eval, slot_eval, req_eval)

    output_json = ROOT / "outputs" / "baseline_comparison_summary.json"
    output_md = ROOT / "outputs" / "baseline_comparison_summary.md"
    output_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Baseline Comparison Summary",
        "",
        "| Baseline | Condition | Frame F1 | Frame Consistency | Slot F1 | Capability F1 | Security F1 | Requirement F1 | Coverage | Hallucination |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline in ["b1", "b2", "b3", "g1"]:
        for condition in ["clean", "noisy"]:
            key = f"{baseline}_{condition}"
            if key not in payload:
                continue
            item = payload[key]
            def fmt(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.4f}"
            lines.append(
                f"| {baseline.upper()} | {condition} | {fmt(item['frame_f1'])} | {fmt(item['frame_consistency'])} | {fmt(item['slot_f1'])} | {fmt(item['slot_functional_capability_f1'])} | {fmt(item['slot_security_f1'])} | {item['requirement_f1']:.4f} | {item['requirement_coverage']:.4f} | {item['hallucination_rate']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Read This First",
            "",
            "- `B1` is the brittle lexical baseline.",
            "- `B2` adds normalization and paraphrase-aware keyword mapping for the fixed six-question dialogue flow.",
            "- `G1` is the native Gemini frame-first pipeline when a Gemini run has been executed.",
            "- Frame metrics isolate dialogue understanding quality before slot projection and requirement generation.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
