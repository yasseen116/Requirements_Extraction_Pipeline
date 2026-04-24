# Project Structure, Context, And Current Benchmarks

Date: `2026-04-25`

This file is the single current-state reference for the project.

It summarizes:

- what the project is trying to prove
- which parts of the repo are current vs historical
- the current directory structure
- the trusted data assets
- the active pipeline scripts and prompts
- the best and latest benchmark results
- the earlier pilot baseline results

## 1. Project Context

This project studies a controlled `dialogue -> requirements` pipeline for software requirements engineering.

The main research question is:

Can a controlled conversational elicitation pipeline recover trusted source requirements from dialogue, and can we measure coverage, omissions, and hallucinations in a rigorous way?

The current project focus is:

- source-grounded requirement recovery
- measurable end-to-end benchmarks
- local and auditable LLM-based extraction
- explicit comparison between direct generation and dialogue-based generation

The project is not currently framed as:

- an autonomous free-form chatbot
- a code generation system
- a production SRS tool
- a final industrial generalization claim

## 2. What Is Current vs Historical

Current active track:

- local Ollama-based PURE benchmark
- best current model: `qwen2.5:7b-instruct`
- benchmark entry point: `scripts/run_pure_full_benchmark.py`
- current paper-facing workflow summary: `docs/project_workflow_and_results.md`

Still useful, but historical / earlier-stage:

- `FINAL_RESEARCH_SUMMARY.md`
- `PURE_FULL_IMPLEMENTATION_PLAN.md`
- Gemini pilot run files under `outputs/g1_gemini_runs/`
- pilot baseline summaries under `outputs/baseline_comparison_summary.json`

Important distinction:

- `outputs/pure_full_latest_run.json` points to the most recent experimental PURE run
- `outputs/pure_full_best_run.json` points to the best current paper-facing PURE run

## 3. Current Top-Level Structure

Main directories:

```text
docs/          Current explanations, workflow notes, and quickstarts
experiments/   Experiment configs or scratch experiment assets
final/         Finalized or paper-facing artifacts
outputs/       All generated benchmark runs, reports, and summaries
prompts/       LLM prompt templates
qc/            Quality-control artifacts
raw/           Earlier raw or intermediate assets
raw_sources/   Trusted public datasets and manually grounded source samples
schemas/       JSON schemas for sample and model outputs
scripts/       Benchmark, generation, evaluation, and reporting scripts
seed/          Earlier exploratory seed examples
splits/        Split manifests
synthetic/     Generated synthetic dialogues or pilot variants
```

Top-level files:

```text
CURRENT_FLOW_OF_WORK.txt
FINAL_RESEARCH_SUMMARY.md
LOCAL_OLLAMA_2DOC_RESULTS.md
PROJECT_STRUCTURE_CONTEXT_AND_BENCHMARKS.md
PURE_FULL_IMPLEMENTATION_PLAN.md
README.md
```

## 4. Important Current Subdirectories

### `raw_sources/`

Purpose:

- stores trusted source datasets and manually grounded gold samples

Important subdirectories:

- `raw_sources/public/`
- `raw_sources/manual_gold/`
- `raw_sources/pure_benchmark/`
- `raw_sources/public_manifests/`

### `outputs/`

Purpose:

- stores all generated runs, metrics, and benchmark summaries

Important subdirectories:

- `outputs/pure_full_runs/`
- `outputs/reuse_source_2docs/`
- `outputs/reuse_source_6docs/`
- `outputs/g1_gemini_runs/`
- `outputs/b1_rule_based_*`
- `outputs/b2_keyword_normalized_*`

Important generated report location:

- `outputs/pure_full_runs/<run_id>/reports/`

Important top-level output files:

- `outputs/pure_full_best_run.json`
- `outputs/pure_full_latest_run.json`
- `outputs/baseline_comparison_summary.json`
- `outputs/pilot_results_summary.json`
- `outputs/source_grounded_validation_summary.json`
- `outputs/public_source_inventory.json`

### `scripts/`

Purpose:

- all runnable benchmark logic

Current key scripts:

```text
scripts/run_pure_full_benchmark.py
scripts/build_run_report.py
scripts/build_pure_requirements_benchmark.py
scripts/generate_pure_controlled_dialogues.py
scripts/generate_pure_direct_requirements.py
scripts/generate_pure_full_requirements.py
scripts/generate_pure_oracle_requirements.py
scripts/evaluate_pure_dialogue_coverage.py
scripts/evaluate_pure_requirements_coverage.py
scripts/analyze_pure_errors.py
scripts/postprocess_pure_generated_requirements.py
scripts/llm_router.py
scripts/local_ollama_client.py
```

Other important script groups:

- pilot pipeline: `run_pilot_pipeline.py`, `run_g1_gemini_pipeline.py`
- rule baselines: `generate_b0_template_requirements.py`, `generate_b1_rule_based_slots.py`, `generate_b2_keyword_normalized_slots.py`
- reporting: `summarize_baseline_comparison.py`, `build_pilot_flow_report.py`, `build_input_output_results_report.py`
- data inventory / validation: `download_public_sources.py`, `inventory_public_sources.py`, `validate_source_grounded_samples.py`

### `prompts/`

Current prompt files:

```text
prompts/dialogue_to_frames_gemini.txt
prompts/dialogue_to_full_requirements_gemini.txt
prompts/dialogue_to_slots_openai_compatible.txt
prompts/pilot_dialogue_to_requirements_gemini.txt
prompts/pure_dialogue_turn_rewrite.txt
prompts/pure_requirement_group_to_answer.txt
prompts/pure_requirements_to_dialogue.txt
prompts/pure_source_to_full_requirements_gemini.txt
```

### `schemas/`

Current schema files:

```text
schemas/conversational_requirements_sample.schema.json
schemas/gemini_expanded_dialogue_response.schema.json
schemas/gemini_full_requirements_response.schema.json
schemas/gemini_requirement_frames_response.schema.json
```

## 5. Trusted Data Assets

Public source inventory currently tracked in `outputs/public_source_inventory.json`.

Summary of currently inventoried public datasets:

| Dataset | File Count | Notes |
| :--- | ---: | :--- |
| `pure` | 119 | Main trusted benchmark source for current end-to-end evaluation |
| `software_requirements_dataset` | 153 | Used for source-grounded manual samples and broader requirement assets |
| `promise_plus` | 2 | Additional requirements-related public data |
| `nice` | 1 | Additional public dataset |

Current source-grounded manual gold summary from `outputs/source_grounded_validation_summary.json`:

| Split | Count |
| :--- | ---: |
| `train` | 1 |
| `dev` | 1 |
| `test` | 1 |

Current manually grounded sample domains:

- `restaurant_discovery`
- `restaurant_ordering`
- `password_management`

## 6. Active PURE Benchmark Workflow

Current benchmark chain:

```text
trusted PURE source requirements
-> controlled dialogue generation
-> dialogue-to-full-requirements extraction
-> deterministic normalization and global deduplication
-> coverage / precision / F1 / hallucination evaluation
```

Current benchmark stages:

1. `scripts/build_pure_requirements_benchmark.py`
   - builds trusted gold source requirements

2. `scripts/generate_pure_controlled_dialogues.py`
   - converts source requirements into controlled elicitation dialogue
   - supports optional clarification rounds
   - current default: `--clarification-rounds 0`

3. `scripts/generate_pure_direct_requirements.py`
   - direct baseline from source requirements to generated requirements

4. `scripts/generate_pure_full_requirements.py`
   - main dialogue-to-full-requirements extractor
   - supports chunk-aware local generation

5. `scripts/postprocess_pure_generated_requirements.py`
   - deterministic cleanup and cross-category requirement deduplication

6. `scripts/evaluate_pure_dialogue_coverage.py`
   - dialogue-only coverage diagnostic

7. `scripts/evaluate_pure_requirements_coverage.py`
   - end-to-end precision / recall / F1 / hallucination scoring

8. `scripts/analyze_pure_errors.py`
   - omission and hallucination error analysis

## 7. Current Runtime / Model Context

Current preferred local setup:

- provider: `ollama`
- model: `qwen2.5:7b-instruct`
- benchmark source mode: cached trusted source JSON
- current fallback-safe path avoids raw XML rebuild dependence

Reason for cached-source mode:

- the Homebrew Python 3.14 environment currently has an `expat` linkage problem
- cached trusted source JSON avoids blocking on XML rebuild

Recommended environment:

```bash
export REQ_LLM_PROVIDER=ollama
export REQ_OLLAMA_MODEL=qwen2.5:7b-instruct
export REQ_OLLAMA_TIMEOUT_SECONDS=900
export REQ_OLLAMA_NUM_CTX=8192
export REQ_OLLAMA_NUM_PREDICT=4096
export REQ_OLLAMA_SEED=42
```

## 8. Current Best PURE Benchmark

Best current paper-facing PURE run pointer:

- `outputs/pure_full_best_run.json`

Best current run:

- run id: `20260424T205611Z_pure_full`
- run dir: `outputs/pure_full_runs/20260424T205611Z_pure_full/`
- metric file: `outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json`

This is the best current end-to-end full-requirements recovery result on the verified 2-document local PURE slice.

### Best Current 2-Document PURE Results

| Track | Micro Precision | Micro Recall | Micro F1 | Hallucination | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Dialogue-only lower bound | N/A | 0.7089 | N/A | N/A | Requirement content explicitly present in user turns |
| Direct baseline | 0.8852 | 0.6835 | 0.7714 | 0.0988 | Source requirements -> local LLM -> requirements |
| Conversational pipeline | **0.9559** | **0.8228** | **0.8844** | **0.0467** | Best current end-to-end result |
| Oracle | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Sanity-check upper bound |

Aggregate counts for the best conversational pipeline:

- sample count: `2`
- total source requirements: `79`
- total generated requirements: `68`
- total matched requirements: `65`

### Improvement Over Direct Baseline

| Metric | Direct | Best Conversational | Gain |
| :--- | ---: | ---: | ---: |
| Precision | 0.8852 | 0.9559 | +0.0706 |
| Recall | 0.6835 | 0.8228 | +0.1392 |
| F1 | 0.7714 | 0.8844 | +0.1129 |

## 9. Latest PURE Experimental Run

Latest run pointer:

- `outputs/pure_full_latest_run.json`

Latest experimental PURE run:

- run id: `20260424T211035Z_pure_full`
- run dir: `outputs/pure_full_runs/20260424T211035Z_pure_full/`

This was the clarification experiment, not the best current paper-facing run.

### Clarification Experiment

What changed:

- one additional clarification round was enabled in dialogue generation
- goal: increase dialogue coverage before final extraction

Clarification result:

| Setting | Dialogue Recall | Req Precision | Req Recall | Req F1 | Hallucination |
| :--- | ---: | ---: | ---: | ---: | ---: |
| No clarification | 0.7089 | **0.9559** | 0.8228 | **0.8844** | **0.0467** |
| One clarification round | **0.9367** | 0.8800 | **0.8354** | 0.8571 | 0.1350 |

Interpretation:

- clarification strongly improves dialogue coverage
- clarification slightly improves end-to-end recall
- but the current final extractor over-generates more when fed richer clarified dialogue
- so clarification is currently useful as an ablation and coverage diagnostic, not the best default end-to-end setting

## 10. Current Pilot Baseline Benchmarks

These are the earlier pilot/source-grounded baseline summaries from `outputs/baseline_comparison_summary.json`.

Important caution:

- these are not the same experiment as the current local PURE benchmark
- they are earlier small pilot baselines and should be reported separately

### Pilot Baseline Summary

| System | Condition | Frame F1 | Slot F1 | Requirement F1 | Coverage | Hallucination |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `B1` weak lexical rule baseline | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `B1` weak lexical rule baseline | noisy | 0.2399 | 0.3013 | 0.2015 | 0.1322 | 0.5000 |
| `B2` normalized rule benchmark | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `B2` normalized rule benchmark | noisy | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `G1` Gemini structured pipeline | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `G1` Gemini structured pipeline | noisy | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

Pilot noisy-case summary from `outputs/pilot_results_summary.json`:

- frame overall F1: `0.2399`
- slot overall F1: `0.3013`
- requirement overall F1: `0.2015`
- requirement coverage: `0.1322`
- requirement hallucination rate: `0.5000`

This noisy pilot summary corresponds to the weaker baseline behavior, not the current best PURE benchmark.

## 11. What Is Currently Working

Working well now:

- trusted public-source inventory and download tracking
- source-grounded manual sample validation
- cached-source PURE benchmarking
- local Ollama execution through `llm_router.py`
- chunk-aware dialogue-to-requirements generation
- deterministic global requirement deduplication
- direct baseline vs conversational pipeline comparison
- clarification-round ablation support

## 12. Main Current Limitations

Current limitations:

- best verified local PURE benchmark still covers `2` documents, not the full cached `6`
- latest run is not always the best run, so pointers must be interpreted carefully
- clarification improves dialogue coverage more than it improves final extracted requirement quality
- some historical summary files still exist and can confuse the current state if read without context
- raw XML rebuild is still affected by the local Python `expat` issue

Current modeling failure modes:

- omission of harder mixed-format requirements
- paraphrastic restatement that increases generated count without increasing matches enough
- occasional subject/name drift on complex requirements

## 13. Best Files To Read Right Now

Current main reference docs:

- `docs/project_workflow_and_results.md`
- `PROJECT_STRUCTURE_CONTEXT_AND_BENCHMARKS.md`
- `LOCAL_OLLAMA_2DOC_RESULTS.md`
- `CURRENT_FLOW_OF_WORK.txt`

Current best benchmark files:

- `outputs/pure_full_best_run.json`
- `outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/direct_coverage.json`
- `outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/dialogue_coverage_user_only.json`
- `outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json`

Clarification experiment files:

- `outputs/pure_full_latest_run.json`
- `outputs/pure_full_runs/20260424T211035Z_pure_full/metrics/dialogue_coverage_user_only.json`
- `outputs/pure_full_runs/20260424T211035Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json`

Pilot historical benchmark files:

- `outputs/baseline_comparison_summary.json`
- `outputs/pilot_results_summary.json`
- `outputs/g1_gemini_latest_run.json`

Trusted data inventory files:

- `outputs/public_source_inventory.json`
- `outputs/source_grounded_validation_summary.json`

## 14. Practical Bottom Line

If you need the current project state in one sentence:

The project currently has a working, measurable, source-grounded local PURE benchmark in which the best verified conversational pipeline beats the direct baseline on end-to-end full-requirements recovery, while a separate clarification ablation shows that dialogue coverage can be improved even further, though the final extractor still needs tightening to fully exploit that extra dialogue evidence.
