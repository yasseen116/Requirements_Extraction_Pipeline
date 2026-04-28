# Requirements Dataset Project: Current Workflow And Verified Results

This document is the current paper-facing overview of the project. It replaces older notes that focused on earlier Gemini-only runs and now also records the methodology fixes implemented on April 27-28, 2026.

## 1. Project Goal

The project studies a controlled `dialogue -> requirements` pipeline for software requirements engineering.

The research question is not "can an LLM write nice requirements text?"

The real question is:

Can a controlled conversational pipeline recover trusted source requirements from a dialogue transcript, and how well can we measure coverage, omissions, and hallucinations?

## 2. Current Benchmark Workflow

The benchmark is source-grounded and reproducible.

### Stage 1: Trusted Source Requirements

We start from public PURE requirement documents and convert them into structured gold requirement files.

Output:

- `source_requirements/*.json`

These are the trusted reference requirements used for evaluation.

### Stage 2: Controlled Dialogue Generation

We convert the source requirements into a controlled elicitation dialogue.

The dialogue is not a free-form chatbot session. It uses a fixed interviewer flow and groups requirements into question-answer chunks so the pipeline stays measurable.

The question-generation algorithm is part of the research method and is unified across providers.

Current documented algorithm:

1. Assign each gold requirement a primary semantic theme, and a secondary theme when the top-2 theme scores are very close.
2. After each user answer, recompute semantic coverage of every gold requirement against sentence/clause-level user evidence.
3. Rank the remaining uncovered themes by:
   - fixed theme priority
   - uncovered requirement count
   - average semantic gap
   - a cap on low-yield repeated questioning for the same theme
4. Select the top uncovered requirements for the chosen theme.
5. Ask the active LLM to generate exactly one natural follow-up question using:
   - the target theme
   - the most important uncovered requirement snippets
   - the recent dialogue history
   - an optional clarification focus hint for known hard clusters
6. Use fixed fallback question templates only if JSON question generation fails.

This algorithm is the same for Ollama and Gemini runs. Provider choice may affect model quality or latency, but not the question-selection logic or prompt construction.

Output:

- `expanded_dialogues/*.json`

### Stage 3: LLM Requirement Extraction

We feed the dialogue into the main extraction model and ask it to produce a structured requirements file.

The codebase now supports:

- hosted Gemini runs
- local Ollama runs

For the current verified local benchmark, the main model is:

- `qwen2.5:7b-instruct` via Ollama

The current extraction pipeline is proposition-first rather than direct final-requirement generation.

Current implemented extraction flow:

1. Build an evidence bank from sentence/clause-level user support units.
2. Group evidence units into theme-aware batches.
3. Give the model a scoped dialogue excerpt plus only the relevant evidence units for that batch.
4. Extract atomic propositions first.
5. Merge proposition duplicates conservatively.
6. Rewrite generic propositions using their supporting evidence.
7. Run one optional gap pass on semantically uncovered evidence.
8. Convert propositions back into the unchanged final requirements schema.

Important implementation details added on April 27-28, 2026:

- the local extractor no longer repeats the full dialogue transcript in every batch
- `goal_scope` is no longer processed as a standalone extraction batch when real requirement themes exist
- local Ollama calls now use bounded per-batch output caps
- local Ollama calls now use per-batch timeouts so pathological structured generations fall back instead of hanging indefinitely
- validation searches claimed `evidence_turns` first, then falls back once to the full dialogue

Output:

- `generated_requirements/*.json`
- `_artifacts/evidence_bank/*.json`
- `_artifacts/propositions/*.json`

### Stage 4: Evaluation

Generated requirements are compared against the trusted source requirements using the evaluation scripts in `scripts/`.

Main metrics:

- precision
- coverage recall
- F1
- hallucination rate

We also compute a dialogue-only diagnostic score:

- how much gold requirement content is explicitly recoverable from user turns alone

Important evaluation update:

- `sentence-transformers` is now treated as core benchmark infrastructure, not just an auxiliary metric
- the same semantic sentence/clause-level scorer is now used consistently for dialogue coverage, pipeline coverage, validation, and reporting
- the old dialogue metric bug that compared short gold requirements to whole user turns is fixed

This change materially affected the measured recall bound on the hard PURE document.

Before the sentence-level semantic scorer was standardized, the older hard-document evaluation path reported only `0.25` dialogue coverage on `pure_0000_cctns` in `20260426T224847Z_pure_full`.

After shifting to sentence-level support units with the same `0.55` semantic threshold, the comparable hard-document dialogue bound rose to `0.61` in `20260427T003223Z_pure_full`.

So the sentence-transformer scorer should be described in the paper as a methodological correction that changed the apparent coverage ceiling in a major way, not as a cosmetic metric swap.

This dialogue-only score should still be read as a diagnostic bound, not as a strict theoretical ceiling on final recall.

## 3. Current Best Paper-Facing Benchmark

The strongest current paper-facing result is still the verified 2-document local benchmark:

- provider: `ollama`
- model: `qwen2.5:7b-instruct`
- documents: `pure_0000_gamma_j`, `pure_1999_dii`
- source mode: cached trusted PURE source JSON
- extraction mode: chunk-aware dialogue-to-requirements generation
- deterministic cleanup: global cross-category requirement deduplication

Main best run path:

- `outputs/pure_full_runs/20260424T205611Z_pure_full/`

### Aggregate Results: Best Current 2-Document Recovery

| Track | What it measures | Micro Precision | Micro Recall | Micro F1 | Hallucination |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Dialogue-only lower bound | Requirement content explicitly present in user turns | N/A | 0.7089 | N/A | N/A |
| Direct local baseline | Source requirements -> local LLM -> requirements | 0.8852 | 0.6835 | 0.7714 | 0.0988 |
| Conversational local pipeline | Source -> dialogue -> local LLM -> requirements | **0.9559** | **0.8228** | **0.8844** | **0.0467** |
| Oracle | Perfect source reconstruction sanity check | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Clarification A/B Result On The 2-Document Slice

| Pipeline Setting | Dialogue Recall | Req Precision | Req Recall | Req F1 | Hallucination |
| :--- | ---: | ---: | ---: | ---: | ---: |
| No clarification round | 0.7089 | **0.9559** | 0.8228 | **0.8844** | **0.0467** |
| One clarification round | **0.9367** | 0.8800 | **0.8354** | 0.8571 | 0.1350 |

Interpretation:

- the clarification round strongly improves dialogue coverage
- but the current final extractor turns some of that richer dialogue into extra paraphrased outputs
- so the best current end-to-end full-requirements setting is still the no-clarification 2-document pipeline

## 4. Latest Hard-Slice Rerun After Method Fixes

On April 27-28, 2026, I reran the hardest single PURE document after fixing the local extractor and benchmark infrastructure.

Hard-slice rerun:

- run id: `20260427T213602Z_pure_full`
- provider: `ollama`
- model: `qwen2.5:7b-instruct`
- document: `pure_0000_cctns`
- dialogue question algorithm: `semantic_gap_llm_v1`
- direct baseline self-consistency: `3`
- conversational pipeline self-consistency: `1`

This rerun is important because it proves the updated local evidence-bank pipeline now completes on the hard document instead of hanging.

### Aggregate Results: Hard Single Document

| Track | What it measures | Micro Precision | Micro Recall | Micro F1 | Hallucination |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Dialogue-only lower bound | Requirement content explicitly present in user turns | N/A | 0.8500 | N/A | N/A |
| Direct local baseline | Source requirements -> local LLM -> requirements | 0.9417 | 0.9700 | 0.9557 | 0.0583 |
| Conversational local pipeline | Source -> dialogue -> local LLM -> requirements | 0.7857 | 0.5500 | 0.6471 | 0.2143 |
| Oracle | Perfect source reconstruction sanity check | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Conversational Diagnostics: Hard Single Document

| Diagnostic | Value |
| :--- | ---: |
| Dialogue recall | 0.8500 |
| Evidence units | 107 |
| Propositions | 70 |
| Final grounded requirements | 70 |
| Validation hallucinations | 0 |
| Gap-pass additions | 0 |
| Rewrite candidates | 13 |

### Theme-Level Coverage Detail: Hard Single Document

The hard rerun is not uniformly difficult across themes. The dialogue controller covered some requirement families very well and left others materially under-covered.

| Theme | Total Reqs | Covered | Recall |
| :--- | ---: | ---: | ---: |
| `usability_help_accessibility` | 51 | 50 | 0.9804 |
| `security_audit` | 11 | 11 | 1.0000 |
| `reporting_documentation` | 8 | 8 | 1.0000 |
| `user_roles_permissions` | 9 | 3 | 0.3333 |
| `availability_reliability` | 6 | 1 | 0.1667 |
| `deployment_environment_constraints` | 5 | 3 | 0.6000 |
| `maintainability_portability_testability` | 1 | 0 | 0.0000 |

This matters for the paper because the hard rerun was not a generic "long context failure". It was a selective coverage failure concentrated in a few semantic clusters, especially user roles and availability.

### Interpretation Of The Hard-Slice Rerun

- the infrastructure bug is fixed: the local evidence-bank proposition pipeline now completes on `pure_0000_cctns`
- the dialogue stage is no longer the primary bottleneck on this slice because user-turn coverage reached `0.85`
- the extractor is still the quality bottleneck because source recall stops at `0.55`
- all `70` extracted requirements were grounded in the dialogue, so the remaining failure mode is mostly omission or source drift, not unsupported extraction from nowhere
- the dialogue still contains off-domain and generic phrasing in places, which means a requirement can be grounded in the dialogue but still be a poor match to the original source requirement

An important nuance is that validation and source recall are now intentionally separated:

- validation grounded `70/70` conversational outputs in the dialogue
- source matching only accepted `55/70` of those outputs against the gold requirements

So `15` extracted items were supported by the dialogue but still failed the source-grounded benchmark because they were too generic, too paraphrastic, or semantically drifted away from the original requirement wording.

This means the current hard-slice problem is not "the run crashes" anymore.

The current problem is:

The local pipeline now runs end to end, but the dialogue-to-proposition-to-requirements mapping still loses too much source fidelity on the hardest document.

## 5. What The April 27-28, 2026 Improvements Changed

The latest implementation work introduced several concrete improvements that matter for the paper draft:

1. The semantic scorer is now shared across dialogue coverage, requirement coverage, validation, and reporting.
2. Dialogue coverage is now computed against sentence/clause support units rather than full user turns, which removed the previous under-reporting bug.
3. The question-generation algorithm is now explicitly unified across providers and recorded in run metadata.
4. The evidence-bank extractor now uses scoped dialogue excerpts per batch instead of repeating the full transcript every time.
5. The benchmark runner now preserves the intended Python environment instead of resolving the venv symlink away.
6. Error analysis is now compatible with the shared list-based semantic scorer.
7. Local Ollama proposition extraction now uses bounded per-batch output caps and timeouts, so pathological structured JSON calls do not stall the entire run indefinitely.
8. The hard-slice rerun now records per-theme dialogue coverage, evidence-bank counts, proposition counts, rewrite counts, and gap-pass counts, which makes failure analysis much more specific than the earlier run logs.

These changes improved methodological rigor even when they did not immediately improve the final hard-slice recall number.

## 6. What These Results Mean

There are now two different paper-relevant messages:

### Message A: Best Current Headline Result

On the verified 2-document slice, the controlled conversational pipeline still clearly outperforms the direct local baseline.

That remains the strongest current paper-facing result.

### Message B: Hard-Document Stress Test

On the hard single-document slice, the updated local memory-augmented proposition pipeline now completes reliably enough to evaluate.

However, it currently reaches:

- dialogue recall: `0.85`
- pipeline recall: `0.55`

That is below the target `0.60` hard-slice recall.

So the latest rerun should be interpreted as a successful infrastructure and methodology fix, but not yet as a new headline-quality extraction result.

## 7. Important Limitations And Paper Caution

Current limitations:

- the best current paper-facing benchmark still covers 2 PURE documents, not all 6
- the latest hard-slice rerun covers 1 difficult document and should be treated as a stress test, not a replacement for the stronger 2-document headline result
- the exact hard-slice rerun was completed after patching and resuming the pipeline stages in place
- the hard-slice conversational pipeline used `self-consistency=1` for the local extractor so the run would finish under bounded Ollama generation
- grounded-to-dialogue does not automatically imply grounded-to-source when the dialogue itself drifts semantically

Methodological caution:

- do not claim broad industrial generalization from the current local runs
- do not mix earlier Gemini experiments, the verified 2-document local slice, and the latest hard-slice rerun as if they were the same condition
- do not treat dialogue-only coverage as a strict ceiling on final source-grounded recall

## 8. Current Best Interpretation For The Paper

The strongest current paper-safe interpretation is:

"A controlled conversational pipeline using a local 7B instruct model can outperform a direct local baseline on a verified PURE benchmark slice, and the updated local evidence-bank extractor now completes end to end on a previously problematic hard document. However, on that hard document the main remaining bottleneck is extraction fidelity rather than dialogue coverage or benchmark infrastructure."

That statement is more accurate than claiming that the latest hard-slice rerun already solved recall.

## 9. Recommended Next Fixes

The next practical improvements are:

1. Reduce off-domain drift in the generated dialogue so grounded extraction is more source-faithful.
2. Tighten proposition synthesis so one dialogue sentence does not produce multiple loosely paraphrased requirements.
3. Improve user-role and access-control extraction, which is a major miss cluster on `pure_0000_cctns` with only `0.3333` dialogue recall in the latest hard rerun.
4. Improve availability and reliability elicitation, which is the weakest major theme in the hard rerun at `0.1667` dialogue recall.
5. Add stronger constraints on proposition wording so exact source terminology is preserved more often.
6. Make the gap pass useful on local runs; in the current hard rerun it added `0` new grounded requirements.
7. Rerun the fixed `paper_regression_3doc` slice after these quality fixes.
8. Only after that, rerun the full 6-document local benchmark.

## 10. Files To Use

Best current paper-facing 2-document result:

- [best pipeline run](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full:1)
- [best pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json:1)
- [best baseline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/direct_coverage.json:1)

Latest hard-slice rerun after method fixes:

- [rerun summary](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/comparison_summary.json:1)
- [rerun pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/metrics/pipeline_coverage.json:1)
- [rerun pipeline validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/metrics/pipeline_validation_report.json:1)
- [rerun dialogue coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/metrics/dialogue_coverage_user_only.json:1)
- [rerun summary PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/reports/summary.pdf:1)
- [rerun full-detail PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/reports/full_detail.pdf:1)

## 11. Reproduce The Local Runs

Recommended environment for current local Ollama runs:

```bash
export REQ_LLM_PROVIDER=ollama
export REQ_OLLAMA_MODEL=qwen2.5:7b-instruct
export REQ_OLLAMA_TIMEOUT_SECONDS=900
export REQ_OLLAMA_NUM_CTX=8192
export REQ_OLLAMA_NUM_PREDICT=2048
export REQ_OLLAMA_SEED=42
```

Example hard-slice benchmark launch:

```bash
python3 scripts/run_pure_full_benchmark.py \
  --python-bin data/venv/bin/python3 \
  --benchmark-slice hard_single_doc \
  --pipeline-preset local_recall \
  --dialogue-variant controlled \
  --llm-provider ollama \
  --clarification-rounds 2 \
  --theme-max-exchanges 3 \
  --target-dialogue-recall 0.82 \
  --max-turns 28
```

For the exact April 27-28, 2026 hard-slice rerun, the direct baseline remained at `self-consistency=3`, while the conversational local extractor was completed with `self-consistency=1` after the local proposition batching fixes were validated.
