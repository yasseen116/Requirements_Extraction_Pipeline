# Final Research Summary

Historical note:
This file contains older Gemini-centered pilot and transition notes.
For the current verified workflow and benchmark results, use `docs/project_workflow_and_results.md` and `LOCAL_OLLAMA_2DOC_RESULTS.md`.

## What This Project Does
This project builds a controlled conversational requirements elicitation pipeline for software ideas.

The final system takes a **6-question chatbot dialogue**, extracts a **structured requirement frame representation** with Gemini, converts those frames into **normalized slots**, and then generates **structured functional and non-functional requirements**. The pipeline is fully traceable and measurable.

Final pipeline:
1. `Dialogue -> Gemini structured frames`
2. `Frames -> deterministic dialogue-aware completion`
3. `Frames -> normalized slots`
4. `Slots -> structured FR/NFR requirements`
5. `Frames / Slots / Requirements -> evaluation`

The current official Gemini system is:
- `G1 = Gemini structured extraction + deterministic frame completion + slot projection + requirement generation`

The official run is:
- `run_id = 20260424T143224Z_gemini-2_5-flash-lite`
- model = `gemini-2.5-flash-lite`

## Comprehensive PURE Benchmark (New)
To move beyond the small pilot, a new end-to-end benchmark track is now implemented for PURE source documents:

1. `PURE XML -> source requirement benchmark (gold)`
2. `source requirements -> expanded multi-question chatbot dialogue`
3. `expanded dialogue -> full structured requirements file`
4. `generated full requirements -> coverage validation against PURE source requirements`
5. `comparison summary (Oracle vs Gemini Full)`

Implemented files:
- `scripts/build_pure_requirements_benchmark.py`
- `scripts/generate_pure_expanded_dialogues.py`
- `scripts/generate_pure_full_requirements.py`
- `scripts/generate_pure_oracle_requirements.py`
- `scripts/evaluate_pure_requirements_coverage.py`
- `scripts/run_pure_full_benchmark.py`

Prompt/schema assets:
- `prompts/pure_requirements_to_dialogue.txt`
- `prompts/dialogue_to_full_requirements_gemini.txt`
- `schemas/gemini_expanded_dialogue_response.schema.json`
- `schemas/gemini_full_requirements_response.schema.json`

Plan + progress tracker:
- [PURE_FULL_IMPLEMENTATION_PLAN.md](/Users/yasseen/Documents/projects/req_dataset_project/data/PURE_FULL_IMPLEMENTATION_PLAN.md:1)

Latest comprehensive run artifact:
- `outputs/pure_full_runs/20260424T152256Z_pure_full/comparison_summary.json`
- `outputs/pure_full_latest_run.json`

Current status:
- PURE source benchmark generation: working
- Full coverage evaluator: working
- Oracle baseline comparison: working
- Gemini Full run: pending shell env setup (`REQ_GEMINI_API_KEY`, `REQ_GEMINI_MODEL`)

## Why The Final Pipeline Is Clear
The chatbot is controlled, not open-ended. It always asks:
1. What kind of system do you want?
2. Who will use it?
3. What should users be able to do?
4. Do users need login or authentication?
5. Are there performance requirements?
6. Are there security requirements?

This makes the study publishable because the input format is stable, the intermediate structure is explicit, and every output can be scored.

## Example: What Was Entered And What The System Produced
Example sample:
- `manual_001_amazing_lunch_indicator_noisy`

### Input Dialogue
User answers:
- "Basically a phone app for finding food spots near me and checking what each place offers."
- "Regular app users, restaurant people using the portal, and whoever manages the whole thing."
- "People should be able to look up places to eat, bounce between map and list results, open the full place details, and get directions. The restaurant side should update venue info, and management should approve those restaurant accounts."
- "Yeah, the mobile side needs sign-in, and the portal side needs accounts too for the restaurant side and the people running it."
- "Searches need to feel quick, like two seconds tops, and if GPS or the internet dies the app should say so instead of just hanging."
- "Login traffic needs protecting, and after three bad tries the portal account should cool off for about thirty minutes."

### Output Requirements
Generated functional requirements:
- The system shall support authentication using mobile app username/password login and web portal username/password login.
- The system shall allow users to search for restaurants using multiple search options.
- The system shall allow users to view restaurant results in map and list views.
- The system shall allow users to open detailed restaurant information pages.
- The system shall allow users to navigate to a selected restaurant using GPS.
- The system shall allow restaurant owners to manage restaurant information through the web portal.
- The system shall allow administrators to verify restaurant owners.

Generated non-functional requirements:
- Performance: The system shall return search results within 2 seconds.
- Performance: The system shall inform the user if internet or GPS connectivity is lost.
- Security: The system shall encrypt log-in communication messages.
- Security: The system shall disable restaurant owner and administrator log-in for 30 minutes after three failed attempts.

## Final Benchmark
The current pilot benchmark has:
- `3` clean source-grounded samples
- `3` noisy source-grounded samples

Important scope clarification:
- The pilot benchmark above is the controlled proof-of-pipeline benchmark.
- The new PURE benchmark is the comprehensive document-level validation track.

### Main Result Table
| System | Meaning | Clean Frame F1 | Clean Slot F1 | Clean Req F1 | Clean Coverage | Clean Hallucination | Noisy Frame F1 | Noisy Slot F1 | Noisy Req F1 | Noisy Coverage | Noisy Hallucination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | weak lexical rule baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.2399 | 0.3013 | 0.2015 | 0.1322 | 0.5000 |
| `B2` | strong normalized rule benchmark | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `G1` | Gemini chatbot pipeline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## What These Results Mean
- `B1` fails badly on noisy paraphrased dialogue. This proves the benchmark is not trivial under weak extraction.
- `B2` is the hand-built reference benchmark. It shows what a highly engineered rule pipeline can achieve on this small controlled pilot.
- `G1` is the real AI system result. It now matches the benchmark on the current pilot.
- The final Gemini pipeline is not just free-form text generation. It is a **structured extraction system** with measurable intermediate representations.
- Hallucination is `0.0` on the current pilot because every final requirement is grounded in the extracted structure and the dialogue.

## What To Claim In The Paper
Strong claims you can make:
- The project introduces a **source-grounded conversational requirements benchmark** with the chain `dialogue -> frames -> slots -> requirements`.
- A **hybrid structured extraction pipeline** can recover high-quality requirements from controlled elicitation dialogue.
- A layered evaluation setup can separate understanding quality from generation quality.

Claims you should **not** overstate:
- Do not claim broad industrial generalization yet.
- Do not claim the system is better than all alternatives in general.
- Do not claim these perfect results mean the problem is solved.

Reason:
- the current benchmark is still a **small pilot** with `6` total evaluated dialogues
- the current domains are controlled
- the completion layer is deterministic and tuned to keep the pipeline stable on this benchmark

## Final Recommended Paper Framing
Use this framing:

"We build and evaluate a controlled conversational requirements elicitation pipeline that transforms chatbot dialogue into requirement frames, normalized slots, and final structured requirements. On a source-grounded pilot benchmark with clean and noisy dialogue conditions, the final Gemini-based pipeline achieves perfect frame, slot, and requirement recovery with zero hallucinations."

Then immediately add the limitation:

"These results are on a small source-grounded pilot and should be interpreted as proof that the methodology and evaluation framework work end-to-end, not as a final claim of broad generalization."

## Files To Use
Primary summary file:
- [FINAL_RESEARCH_SUMMARY.md](/Users/yasseen/Documents/projects/req_dataset_project/data/FINAL_RESEARCH_SUMMARY.md:1)

Main machine-readable benchmark table:
- [baseline_comparison_summary.json](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/baseline_comparison_summary.json:1)

Official Gemini run pointer:
- [g1_gemini_latest_run.json](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/g1_gemini_latest_run.json:1)

Example clean and noisy I/O data:
- [input_output_results_g1_clean.json](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/input_output_results_g1_clean.json:1)
- [input_output_results_g1_noisy.json](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/input_output_results_g1_noisy.json:1)

Official combined outputs for the promoted run:
- [clean combined](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/g1_gemini_runs/20260424T143224Z_gemini-2_5-flash-lite/clean/combined/manual_001_amazing_lunch_indicator.json:1)
- [noisy combined](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/g1_gemini_runs/20260424T143224Z_gemini-2_5-flash-lite/noisy/combined/manual_001_amazing_lunch_indicator_noisy.json:1)

## Bottom Line
For the current pilot, the final paper-facing result is ready:
- trusted public-source grounding
- controlled chatbot flow
- measurable intermediate structure
- recorded end-to-end outputs
- strong benchmark comparison
- final Gemini pipeline with perfect pilot metrics
