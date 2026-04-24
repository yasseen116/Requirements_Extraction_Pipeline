#!/usr/bin/env python3
"""Build a readable pilot report from the generated evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MD = ROOT / "outputs" / "pilot_flow_report.md"
OUTPUT_JSON = ROOT / "outputs" / "pilot_flow_report.json"


BASELINES = {
    "b1_clean": {
        "label": "B1",
        "condition": "clean",
        "frame_eval": ROOT / "outputs" / "b1_rule_based_frames" / "evaluation.json",
        "slot_eval": ROOT / "outputs" / "b1_rule_based_slots" / "evaluation.json",
        "req_eval": ROOT / "outputs" / "b1_rule_based_requirements" / "evaluation.json",
    },
    "b1_noisy": {
        "label": "B1",
        "condition": "noisy",
        "frame_eval": ROOT / "outputs" / "b1_rule_based_frames_noisy" / "evaluation.json",
        "slot_eval": ROOT / "outputs" / "b1_rule_based_slots_noisy" / "evaluation.json",
        "req_eval": ROOT / "outputs" / "b1_rule_based_requirements_noisy" / "evaluation.json",
    },
    "b2_clean": {
        "label": "B2",
        "condition": "clean",
        "frame_eval": ROOT / "outputs" / "b2_keyword_normalized_frames" / "evaluation.json",
        "slot_eval": ROOT / "outputs" / "b2_keyword_normalized_slots" / "evaluation.json",
        "req_eval": ROOT / "outputs" / "b2_keyword_normalized_requirements" / "evaluation.json",
    },
    "b2_noisy": {
        "label": "B2",
        "condition": "noisy",
        "frame_eval": ROOT / "outputs" / "b2_keyword_normalized_frames_noisy" / "evaluation.json",
        "slot_eval": ROOT / "outputs" / "b2_keyword_normalized_slots_noisy" / "evaluation.json",
        "req_eval": ROOT / "outputs" / "b2_keyword_normalized_requirements_noisy" / "evaluation.json",
    },
}


def maybe_load_latest_g1() -> dict | None:
    latest_path = ROOT / "outputs" / "g1_gemini_latest_run.json"
    if not latest_path.exists():
        return None
    latest = load_json(latest_path)
    run_dir = Path(latest["run_dir"])
    if not run_dir.exists():
        return None
    return {
        "run_id": latest["run_id"],
        "run_dir": run_dir,
        "clean_frame_eval": load_json(run_dir / "clean" / "metrics" / "frame_evaluation.json"),
        "clean_slot_eval": load_json(run_dir / "clean" / "metrics" / "slot_evaluation.json"),
        "clean_req_eval": load_json(run_dir / "clean" / "metrics" / "requirement_evaluation.json"),
        "noisy_frame_eval": load_json(run_dir / "noisy" / "metrics" / "frame_evaluation.json"),
        "noisy_slot_eval": load_json(run_dir / "noisy" / "metrics" / "slot_evaluation.json"),
        "noisy_req_eval": load_json(run_dir / "noisy" / "metrics" / "requirement_evaluation.json"),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_sample_id(sample_id: str) -> str:
    return sample_id[:-6] if sample_id.endswith("_noisy") else sample_id


def format_float(value: float) -> str:
    return f"{value:.4f}"


def bullet_list(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def summarize_gold_slots(sample: dict) -> list[str]:
    slots = sample["slots"]
    auth = slots["authentication"]
    lines = [
        f"System type: `{slots['system_type']['value']}`",
        "User roles: " + ", ".join(role["value"] for role in slots["user_roles"]),
        f"Authentication required: `{str(auth['required']).lower()}`",
        "Authentication methods: " + ", ".join(auth["methods"]),
        f"Functional capabilities ({len(slots['functional_capabilities'])}):",
    ]
    for capability in slots["functional_capabilities"]:
        lines.append(f"{capability['actor']} -> {capability['action']}")
    lines.append(
        f"Performance constraints ({len(slots['performance_constraints'])}): "
        + "; ".join(item["text"] for item in slots["performance_constraints"])
    )
    lines.append(
        f"Security constraints ({len(slots['security_constraints'])}): "
        + "; ".join(item["text"] for item in slots["security_constraints"])
    )
    return lines


def summarize_predicted_slots(prediction: dict) -> list[str]:
    slots = prediction["slots"]
    auth = slots["authentication"]
    lines = [
        f"System type: `{slots['system_type']['value']}`",
        "User roles: " + ", ".join(role["value"] for role in slots["user_roles"]) if slots["user_roles"] else "User roles: none",
        f"Authentication required: `{str(auth['required']).lower()}`",
        "Authentication methods: " + ", ".join(auth["methods"]) if auth["methods"] else "Authentication methods: none",
        f"Functional capabilities ({len(slots['functional_capabilities'])}):",
    ]
    if slots["functional_capabilities"]:
        for capability in slots["functional_capabilities"]:
            lines.append(f"{capability['actor']} -> {capability['action']}")
    else:
        lines.append("none")
    lines.append(
        f"Performance constraints ({len(slots['performance_constraints'])}): "
        + ("; ".join(item["text"] for item in slots["performance_constraints"]) if slots["performance_constraints"] else "none")
    )
    lines.append(
        f"Security constraints ({len(slots['security_constraints'])}): "
        + ("; ".join(item["text"] for item in slots["security_constraints"]) if slots["security_constraints"] else "none")
    )
    return lines


def summarize_requirements(prediction: dict) -> list[str]:
    requirements = prediction["requirements"]
    functional = requirements["functional"]
    non_functional = requirements["non_functional"]
    lines = [
        f"Functional requirements ({len(functional)}):",
    ]
    if functional:
        lines.extend(item["text"] for item in functional)
    else:
        lines.append("none")
    lines.append(f"Non-functional requirements ({len(non_functional)}):")
    if non_functional:
        lines.extend(f"{item['category']}: {item['text']}" for item in non_functional)
    else:
        lines.append("none")
    return lines


def get_per_sample(eval_payload: dict, sample_id: str) -> dict:
    for item in eval_payload["per_sample"]:
        if item["sample_id"] == sample_id:
            return item
    raise KeyError(f"Sample {sample_id} not found")


def collect_summary_metrics(slot_eval: dict, req_eval: dict, frame_eval: dict | None = None) -> dict:
    return {
        "frame_f1": frame_eval["aggregate"]["overall_f1_macro"] if frame_eval else None,
        "slot_f1": slot_eval["aggregate"]["overall_f1_macro"],
        "capability_f1": slot_eval["aggregate"]["per_slot"]["functional_capabilities"]["f1_macro"],
        "security_f1": slot_eval["aggregate"]["per_slot"]["security_constraints"]["f1_macro"],
        "requirement_f1": req_eval["aggregate"]["overall_f1_macro"],
        "coverage": req_eval["aggregate"]["coverage_macro"],
        "hallucination_rate": req_eval["aggregate"]["hallucination_rate_macro"],
    }


def main() -> int:
    baseline_metrics: dict[str, dict] = {}
    for key, config in BASELINES.items():
        frame_eval = load_json(config["frame_eval"])
        slot_eval = load_json(config["slot_eval"])
        req_eval = load_json(config["req_eval"])
        baseline_metrics[key] = collect_summary_metrics(slot_eval, req_eval, frame_eval)

    g1_latest = maybe_load_latest_g1()
    if g1_latest is not None:
        baseline_metrics["g1_clean"] = collect_summary_metrics(
            g1_latest["clean_slot_eval"],
            g1_latest["clean_req_eval"],
            g1_latest["clean_frame_eval"],
        )
        baseline_metrics["g1_noisy"] = collect_summary_metrics(
            g1_latest["noisy_slot_eval"],
            g1_latest["noisy_req_eval"],
            g1_latest["noisy_frame_eval"],
        )

    noisy_sample_id = "manual_001_amazing_lunch_indicator_noisy"
    gold_sample = load_json(ROOT / "synthetic" / "pilot_noisy" / f"{noisy_sample_id}.json")
    b1_slots = load_json(ROOT / "outputs" / "b1_rule_based_slots_noisy" / f"{noisy_sample_id}.json")
    b1_requirements = load_json(ROOT / "outputs" / "b1_rule_based_requirements_noisy" / f"{noisy_sample_id}.json")
    b2_slots = load_json(ROOT / "outputs" / "b2_keyword_normalized_slots_noisy" / f"{noisy_sample_id}.json")
    b2_requirements = load_json(ROOT / "outputs" / "b2_keyword_normalized_requirements_noisy" / f"{noisy_sample_id}.json")

    b1_slot_eval = load_json(ROOT / "outputs" / "b1_rule_based_slots_noisy" / "evaluation.json")
    b1_req_eval = load_json(ROOT / "outputs" / "b1_rule_based_requirements_noisy" / "evaluation.json")
    b2_slot_eval = load_json(ROOT / "outputs" / "b2_keyword_normalized_slots_noisy" / "evaluation.json")
    b2_req_eval = load_json(ROOT / "outputs" / "b2_keyword_normalized_requirements_noisy" / "evaluation.json")
    g1_combined = None
    g1_sample_frame = None
    g1_sample_slot = None
    g1_sample_req = None
    if g1_latest is not None:
        g1_combined = load_json(
            g1_latest["run_dir"] / "noisy" / "combined" / f"{noisy_sample_id}.json"
        )
        g1_sample_frame = get_per_sample(g1_latest["noisy_frame_eval"], noisy_sample_id)
        g1_sample_slot = get_per_sample(g1_latest["noisy_slot_eval"], noisy_sample_id)
        g1_sample_req = get_per_sample(g1_latest["noisy_req_eval"], noisy_sample_id)

    b1_sample_slot = get_per_sample(b1_slot_eval, noisy_sample_id)
    b1_sample_req = get_per_sample(b1_req_eval, noisy_sample_id)
    b2_sample_slot = get_per_sample(b2_slot_eval, noisy_sample_id)
    b2_sample_req = get_per_sample(b2_req_eval, noisy_sample_id)

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_size": 3,
        "best_baseline": {
            "name": "B2",
            "condition": "noisy",
            **baseline_metrics["b2_noisy"],
        },
        "best_llm": (
            {
                "name": "G1",
                "condition": "noisy",
                **baseline_metrics["g1_noisy"],
            }
            if "g1_noisy" in baseline_metrics
            else None
        ),
        "baseline_metrics": baseline_metrics,
        "walkthrough_sample": {
            "sample_id": noisy_sample_id,
            "source": gold_sample["source"],
            "dialogue_turns": gold_sample["dialogue"],
            "gold_slot_summary": summarize_gold_slots(gold_sample),
            "b1": {
                "slot_metrics": b1_sample_slot["overall"],
                "requirement_metrics": b1_sample_req["overall"],
                "predicted_slot_summary": summarize_predicted_slots(b1_slots),
                "generated_requirement_summary": summarize_requirements(b1_requirements),
            },
            "b2": {
                "slot_metrics": b2_sample_slot["overall"],
                "requirement_metrics": b2_sample_req["overall"],
                "predicted_slot_summary": summarize_predicted_slots(b2_slots),
                "generated_requirement_summary": summarize_requirements(b2_requirements),
            },
            "g1": (
                {
                    "frame_metrics": g1_sample_frame["overall"],
                    "slot_metrics": g1_sample_slot["overall"],
                    "requirement_metrics": g1_sample_req["overall"],
                    "predicted_slot_summary": summarize_predicted_slots({"slots": g1_combined["slots"]}),
                    "generated_requirement_summary": summarize_requirements({"requirements": g1_combined["requirements"]}),
                }
                if g1_combined is not None
                else None
            ),
        },
        "warnings": [
            "B2 is the current best pilot system, but it is still a hand-built normalized rule baseline.",
            "The pilot set is only 3 source-grounded samples, so these scores are useful for proving the pipeline works, not for claiming broad generalization.",
            "The next stronger result is to scale the manual gold set and then keep the same evaluation scripts unchanged.",
        ],
    }

    lines: list[str] = []
    lines.append("# Pilot Flow Report")
    lines.append("")
    lines.append("## Read This First")
    lines.append("")
    lines.append("This is the clearest current path through the project:")
    lines.extend(
        bullet_list(
            [
                "Use the public source corpora in `raw_sources/public/` as the trusted origin.",
                "Use the manual gold samples in `raw_sources/manual_gold/` and the noisy variants in `synthetic/pilot_noisy/` as the current pilot benchmark.",
                "Run the full benchmark with `python3 scripts/run_pilot_pipeline.py`.",
                "Run the Gemini LLM benchmark with `python3 scripts/run_g1_gemini_pipeline.py` after setting the Gemini env vars.",
                "Read the current best summary in `outputs/baseline_comparison_summary.md`.",
            ]
        )
    )
    lines.append("")
    lines.append("## Current Best Result")
    lines.append("")
    lines.extend(
        bullet_list(
            [
                f"Best pilot baseline: `B2` on the noisy condition.",
                f"Slot F1: `{format_float(report_payload['best_baseline']['slot_f1'])}`",
                f"Requirement F1: `{format_float(report_payload['best_baseline']['requirement_f1'])}`",
                f"Coverage: `{format_float(report_payload['best_baseline']['coverage'])}`",
                f"Hallucination rate: `{format_float(report_payload['best_baseline']['hallucination_rate'])}`",
            ]
        )
    )
    if report_payload["best_llm"] is not None:
        lines.append("")
        lines.extend(
            bullet_list(
                [
                    f"Best pilot LLM result: `G1` on the noisy condition.",
                    f"Frame F1: `{format_float(report_payload['best_llm']['frame_f1'])}`",
                    f"Slot F1: `{format_float(report_payload['best_llm']['slot_f1'])}`",
                    f"Requirement F1: `{format_float(report_payload['best_llm']['requirement_f1'])}`",
                    f"Coverage: `{format_float(report_payload['best_llm']['coverage'])}`",
                    f"Hallucination rate: `{format_float(report_payload['best_llm']['hallucination_rate'])}`",
                ]
            )
        )
    lines.append("")
    lines.append("## Full Flow")
    lines.append("")
    lines.append("1. Trusted public requirements documents are downloaded into `raw_sources/public/`.")
    lines.append("2. Selected documents are converted manually into source-grounded gold samples in `raw_sources/manual_gold/`.")
    lines.append("3. The same gold samples are paraphrased into noisy dialogue variants in `synthetic/pilot_noisy/`.")
    lines.append("4. The main LLM path uses Gemini to extract requirement frames from the dialogue.")
    lines.append("5. Frames are projected deterministically into slots and then into FRs and NFRs.")
    lines.append("6. Frame, slot, and requirement outputs are scored with the evaluation scripts in `scripts/`.")
    lines.append("")
    lines.append("## Baseline Results")
    lines.append("")
    lines.append("| Baseline | Condition | Frame F1 | Slot F1 | Capability F1 | Security F1 | Requirement F1 | Coverage | Hallucination |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    table_keys = ["b1_clean", "b1_noisy", "b2_clean", "b2_noisy"]
    if "g1_clean" in baseline_metrics:
        table_keys.extend(["g1_clean", "g1_noisy"])
    for key in table_keys:
        metrics = baseline_metrics[key]
        if key.startswith("g1_"):
            label = "G1"
            condition = "clean" if key.endswith("clean") else "noisy"
        else:
            config = BASELINES[key]
            label = config["label"]
            condition = config["condition"]
        lines.append(
            f"| {label} | {condition} | "
            f"{format_float(metrics['frame_f1']) if metrics['frame_f1'] is not None else 'n/a'} | "
            f"{format_float(metrics['slot_f1'])} | "
            f"{format_float(metrics['capability_f1'])} | "
            f"{format_float(metrics['security_f1'])} | "
            f"{format_float(metrics['requirement_f1'])} | "
            f"{format_float(metrics['coverage'])} | "
            f"{format_float(metrics['hallucination_rate'])} |"
        )
    lines.append("")
    lines.append("## Why B1 Fails and B2 Works On The Pilot")
    lines.append("")
    lines.extend(
        bullet_list(
            [
                "B1 mostly copies literal phrases from the dialogue. When the user paraphrases the same intent, it misses roles, capabilities, and constraints.",
                "B2 adds normalized phrase mapping for the fixed six-question chatbot flow. That makes it robust to the current noisy paraphrases.",
                "G1 is the real LLM system. It is measurably weaker than B2 on this pilot, which is expected because B2 is a hand-built benchmark tied to the pilot schema.",
                "This is a good pilot result, but it is still a controlled benchmark. The next validation step is scaling the gold set, not changing the metrics.",
            ]
        )
    )
    lines.append("")
    lines.append(f"## Walkthrough Sample: `{noisy_sample_id}`")
    lines.append("")
    source = gold_sample["source"]
    lines.append(
        f"Source anchor: `{source['dataset']}` document `{source['document_id']}` with source sentence ids "
        + ", ".join(source["source_sentence_ids"])
    )
    lines.append("")
    lines.append("### Dialogue")
    lines.append("")
    for turn in gold_sample["dialogue"]:
        lines.append(f"- `{turn['turn_id']}` `{turn['role']}`: {turn['text']}")
    lines.append("")
    lines.append("### Gold Slots")
    lines.append("")
    lines.extend(bullet_list(report_payload["walkthrough_sample"]["gold_slot_summary"]))
    lines.append("")
    lines.append("### B1 Output On This Same Noisy Dialogue")
    lines.append("")
    lines.extend(
        bullet_list(
            [
                f"Slot F1: `{format_float(b1_sample_slot['overall']['f1'])}`",
                f"Requirement F1: `{format_float(b1_sample_req['overall']['f1'])}`",
                f"Coverage: `{format_float(b1_sample_req['overall']['coverage'])}`",
                f"Hallucination rate: `{format_float(b1_sample_req['overall']['hallucination_rate'])}`",
            ]
        )
    )
    lines.append("")
    lines.append("Predicted slots:")
    lines.extend(bullet_list(report_payload["walkthrough_sample"]["b1"]["predicted_slot_summary"]))
    lines.append("")
    lines.append("Generated requirements:")
    lines.extend(bullet_list(report_payload["walkthrough_sample"]["b1"]["generated_requirement_summary"]))
    lines.append("")
    lines.append("### B2 Output On This Same Noisy Dialogue")
    lines.append("")
    lines.extend(
        bullet_list(
            [
                f"Slot F1: `{format_float(b2_sample_slot['overall']['f1'])}`",
                f"Requirement F1: `{format_float(b2_sample_req['overall']['f1'])}`",
                f"Coverage: `{format_float(b2_sample_req['overall']['coverage'])}`",
                f"Hallucination rate: `{format_float(b2_sample_req['overall']['hallucination_rate'])}`",
            ]
        )
    )
    lines.append("")
    lines.append("Predicted slots:")
    lines.extend(bullet_list(report_payload["walkthrough_sample"]["b2"]["predicted_slot_summary"]))
    lines.append("")
    lines.append("Generated requirements:")
    lines.extend(bullet_list(report_payload["walkthrough_sample"]["b2"]["generated_requirement_summary"]))
    lines.append("")
    if report_payload["walkthrough_sample"]["g1"] is not None:
        lines.append("### G1 Output On This Same Noisy Dialogue")
        lines.append("")
        lines.extend(
            bullet_list(
                [
                    f"Frame F1: `{format_float(report_payload['walkthrough_sample']['g1']['frame_metrics']['f1'])}`",
                    f"Slot F1: `{format_float(report_payload['walkthrough_sample']['g1']['slot_metrics']['f1'])}`",
                    f"Requirement F1: `{format_float(report_payload['walkthrough_sample']['g1']['requirement_metrics']['f1'])}`",
                    f"Coverage: `{format_float(report_payload['walkthrough_sample']['g1']['requirement_metrics']['coverage'])}`",
                    f"Hallucination rate: `{format_float(report_payload['walkthrough_sample']['g1']['requirement_metrics']['hallucination_rate'])}`",
                ]
            )
        )
        lines.append("")
        lines.append("Predicted slots:")
        lines.extend(bullet_list(report_payload["walkthrough_sample"]["g1"]["predicted_slot_summary"]))
        lines.append("")
        lines.append("Generated requirements:")
        lines.extend(bullet_list(report_payload["walkthrough_sample"]["g1"]["generated_requirement_summary"]))
        lines.append("")
    lines.append("## What To Use Now")
    lines.append("")
    lines.extend(
        bullet_list(
            [
                "Use `B2` as the current best pilot baseline.",
                "Use `B1` as the weak comparison baseline in the paper.",
                "Use `G1` as the current native Gemini LLM system and report it separately from the hand-built B2 benchmark.",
                "Use the existing evaluation scripts unchanged when you scale from 3 pilot samples to a larger gold set.",
            ]
        )
    )
    lines.append("")
    lines.append("## Important Limits")
    lines.append("")
    lines.extend(bullet_list(report_payload["warnings"]))
    lines.append("")

    OUTPUT_JSON.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD.relative_to(ROOT)} and {OUTPUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
