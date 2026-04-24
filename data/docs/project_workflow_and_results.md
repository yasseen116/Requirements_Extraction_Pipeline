# Requirements Dataset Project: Current Workflow And Verified Results

This document is the current paper-facing overview of the project. It replaces older notes that focused on an earlier Gemini-only setup and older benchmark claims.

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

Output:

- `expanded_dialogues/*.json`

### Stage 3: LLM Requirement Extraction

We feed the dialogue into the main extraction model and ask it to produce a structured requirements file.

The codebase now supports:

- hosted Gemini runs
- local Ollama runs

For the current verified local benchmark, the main model is:

- `qwen2.5:7b-instruct` via Ollama

Important implementation detail:

- large PURE documents are processed with chunk-aware generation so the local 7B model is not unfairly penalized by oversized single-shot prompts

Output:

- `generated_requirements/*.json`

### Stage 4: Evaluation

Generated requirements are compared against the trusted source requirements using the evaluation scripts in `scripts/`.

Main metrics:

- precision
- coverage recall
- F1
- hallucination rate

We also compute a dialogue-only diagnostic score:

- how much gold requirement content is explicitly recoverable from user turns alone

This dialogue-only score is a lower-bound diagnostic, not a strict upper bound on final requirement recall.

## 3. Current Verified Local Benchmark

The current best verified 2-document local benchmark uses:

- provider: `ollama`
- model: `qwen2.5:7b-instruct`
- documents: `pure_0000_gamma_j`, `pure_1999_dii`
- source mode: cached trusted PURE source JSON
- extraction mode: chunk-aware dialogue-to-requirements generation
- deterministic cleanup: global cross-category requirement deduplication

Main best run path:

- `outputs/pure_full_runs/20260424T205611Z_pure_full/`

This setup does not depend on the currently broken Homebrew Python XML path because it reuses cached trusted source files.

## 4. Main Results

### Aggregate Results: Current Best Full-Requirement Recovery

| Track | What it measures | Micro Precision | Micro Recall | Micro F1 | Hallucination |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Dialogue-only lower bound | Requirement content explicitly present in user turns | N/A | 0.7089 | N/A | N/A |
| Direct local baseline | Source requirements -> local LLM -> requirements | 0.8852 | 0.6835 | 0.7714 | 0.0988 |
| Conversational local pipeline | Source -> dialogue -> local LLM -> requirements | **0.9559** | **0.8228** | **0.8844** | **0.0467** |
| Oracle | Perfect source reconstruction sanity check | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Improvement Over Direct Baseline

| Metric | Direct Baseline | Conversational Pipeline | Absolute Gain |
| :--- | ---: | ---: | ---: |
| Micro Precision | 0.8852 | 0.9559 | +0.0706 |
| Micro Recall | 0.6835 | 0.8228 | +0.1392 |
| Micro F1 | 0.7714 | 0.8844 | +0.1129 |

### Clarification A/B Result

I also tested whether adding one explicit clarification round improves full recovery.

| Pipeline Setting | Dialogue Recall | Req Precision | Req Recall | Req F1 | Hallucination |
| :--- | ---: | ---: | ---: | ---: | ---: |
| No clarification round | 0.7089 | **0.9559** | 0.8228 | **0.8844** | **0.0467** |
| One clarification round | **0.9367** | 0.8800 | **0.8354** | 0.8571 | 0.1350 |

Interpretation:

- the clarification round strongly improves dialogue coverage
- but the current final extractor turns some of that richer dialogue into extra paraphrased outputs
- so the best current end-to-end full-requirements setting is still the no-clarification pipeline
- the clarification mechanism is still useful as a research ablation because it proves the missing-coverage bottleneck can be moved upstream

## 5. What These Results Mean

- The local 7B model is usable for the project.
- The conversational pipeline outperformed the direct baseline on this verified 2-document benchmark.
- The current best conversational pipeline is stronger after deterministic global deduplication across output sections.
- A clarification pass can substantially increase dialogue coverage, but the current final extractor does not preserve that gain cleanly enough yet.
- The main remaining error modes are:
  - omission of harder mixed-format requirements
  - paraphrastic restatement of already-covered requirements
  - occasional cross-document or cross-subject naming drift on harder cases

## 6. Important Limitations

These results are promising, but they are not yet the final paper claim.

Current limitations:

- the verified local benchmark currently covers 2 PURE documents, not all 6
- some older project files still refer to earlier Gemini experiments and should be treated as historical, not final
- the Homebrew Python 3.14 environment currently has an `expat` linkage problem, so raw XML rebuilding is not reliable in that interpreter
- the current best metrics include deterministic postprocessing, so the paper should state that normalization and deduplication are part of the pipeline, not a hidden afterthought

Methodological caution:

- do not claim broad industrial generalization from the current 2-document local run
- do not mix older Gemini runs and current local Ollama runs as if they were the same experiment
- do not treat the dialogue-only coverage score as a strict ceiling on final recall

## 7. Current Best Interpretation For The Paper

The strongest current claim is:

"A controlled conversational pipeline using a local 7B instruct model can recover source-grounded software requirements with strong precision and recall on a verified PURE benchmark slice, and it clearly outperforms a direct generation baseline under the same local-model setting."

Best current paper-facing numbers on the verified 2-document slice:

- direct baseline micro F1: `0.7714`
- conversational pipeline micro F1: `0.8844`
- conversational pipeline micro recall: `0.8228`
- conversational pipeline hallucination rate: `0.0467`

## 8. Recommended Next Fixes

The next practical improvements are:

1. Run the same local benchmark on all 6 cached PURE documents.
2. Improve the final extractor so it can benefit from clarification-heavy dialogues without over-paraphrasing.
3. Add a second reported condition in the paper:
   - best end-to-end pipeline
   - clarification-enhanced high-coverage dialogue pipeline
4. Replace stale Gemini-centric documentation with provider-neutral wording across the repo.
5. Keep one primary summary file and mark older experimental summaries as historical.

## 9. Files To Use

Primary local benchmark summary:

- [LOCAL_OLLAMA_2DOC_RESULTS.md](/Users/yasseen/Documents/projects/req_dataset_project/data/LOCAL_OLLAMA_2DOC_RESULTS.md:1)

Best current machine-readable metrics:

- [best run pointer](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_best_run.json:1)
- [best pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json:1)
- [best baseline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/direct_coverage.json:1)
- [clarification pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T211035Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json:1)
- [clarification dialogue coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T211035Z_pure_full/metrics/dialogue_coverage_user_only.json:1)

Main metric files:

- [no-clarification dialogue coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/dialogue_coverage_user_only.json:1)
- [no-clarification pipeline coverage raw](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage.json:1)
- [no-clarification pipeline coverage postprocessed](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json:1)
- [clarification pipeline coverage raw](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T211035Z_pure_full/metrics/pipeline_coverage.json:1)
- [clarification pipeline coverage postprocessed](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T211035Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json:1)

Note:

- `gemini_coverage.json` is still present for backward compatibility
- for newer runs, the provider-neutral file name is `pipeline_coverage.json`

## 10. Reproduce The Local Run

Recommended environment:

```bash
export REQ_LLM_PROVIDER=ollama
export REQ_OLLAMA_MODEL=qwen2.5:7b-instruct
export REQ_OLLAMA_TIMEOUT_SECONDS=900
export REQ_OLLAMA_NUM_CTX=8192
export REQ_OLLAMA_NUM_PREDICT=4096
export REQ_OLLAMA_SEED=42
```

Example cached-source benchmark run:

```bash
python3 scripts/run_pure_full_benchmark.py \
  --reuse-source-dir outputs/reuse_source_2docs \
  --max-samples 2 \
  --dialogue-variant controlled
```

Example clarification benchmark run:

```bash
python3 scripts/run_pure_full_benchmark.py \
  --reuse-source-dir outputs/reuse_source_2docs \
  --max-samples 2 \
  --dialogue-variant controlled \
  --clarification-rounds 1
```
