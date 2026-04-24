# PURE Full Benchmark Plan And Progress

Historical note:
This file documents an older Gemini-oriented implementation plan.
For the current verified workflow and local benchmark results, use `docs/project_workflow_and_results.md` and `LOCAL_OLLAMA_2DOC_RESULTS.md`.

## Objective
Build a comprehensive, measurable benchmark that generates a full requirements file from rich chatbot dialogue and validates coverage against trusted PURE source requirements.

Target comparison:
- `Oracle` source projection baseline (upper bound check)
- `Gemini Full` expanded-dialogue-to-full-requirements system

Primary metric focus:
- source coverage recall
- generated precision
- F1
- hallucination rate
- unmatched source/generation analysis

## Phases
| Phase | Goal | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | Create trusted PURE requirement ground truth | document-level benchmark JSON from PURE XML | Completed |
| 2 | Expand from 6 questions to comprehensive elicitation | multi-question dialogue generator prompt/schema | Completed |
| 3 | Generate full requirements file (not only FR/NFR core) | full structured requirements generator prompt/schema | Completed |
| 4 | Validate against PURE source requirements | coverage evaluator with per-document error analysis | Completed |
| 5 | Run end-to-end benchmark and compare systems | one runner with comparison summary output | Completed |
| 6 | Run Gemini live and analyze final model metrics | Gemini benchmark outputs + comparison vs oracle | Pending API env |

## Implemented Components
- PURE benchmark builder: `scripts/build_pure_requirements_benchmark.py`
- Expanded dialogue generation: `scripts/generate_pure_expanded_dialogues.py`
- Full requirements generation: `scripts/generate_pure_full_requirements.py`
- Oracle baseline generator: `scripts/generate_pure_oracle_requirements.py`
- Coverage evaluator: `scripts/evaluate_pure_requirements_coverage.py`
- End-to-end runner: `scripts/run_pure_full_benchmark.py`

New prompts and schemas:
- `prompts/pure_requirements_to_dialogue.txt`
- `prompts/dialogue_to_full_requirements_gemini.txt`
- `schemas/gemini_expanded_dialogue_response.schema.json`
- `schemas/gemini_full_requirements_response.schema.json`

## Current Run Artifact
Most recent comprehensive run:
- `outputs/pure_full_runs/20260424T152256Z_pure_full/comparison_summary.json`

Current state:
- source benchmark generation works
- oracle baseline works
- coverage evaluator works
- Gemini stage was skipped because `REQ_GEMINI_API_KEY` and `REQ_GEMINI_MODEL` were not set in shell env

## How To Run Full Comparison
1. Set Gemini env vars:
```bash
export REQ_GEMINI_API_KEY="YOUR_KEY"
export REQ_GEMINI_MODEL="gemini-2.5-flash-lite"
```
2. Run the benchmark:
```bash
python3 scripts/run_pure_full_benchmark.py --max-samples 10 --min-requirements 10 --match-threshold 0.55
```
3. Inspect outputs:
- latest pointer: `outputs/pure_full_latest_run.json`
- summary: `outputs/pure_full_runs/<run_id>/comparison_summary.json`
- source benchmark: `outputs/pure_full_runs/<run_id>/source_requirements`
- generated full requirements: `outputs/pure_full_runs/<run_id>/generated_requirements`
- coverage evaluation: `outputs/pure_full_runs/<run_id>/metrics/gemini_coverage.json`

## Method Note
The oracle baseline is expected to score near-perfect because it is a source projection sanity-check. The real research metric of interest is the `Gemini Full` run measured against the same PURE source benchmark.
