# Requirements Dataset Project: Current Workflow And Verified Results

This document is the current paper-facing overview of the project. It replaces older notes that focused on earlier Gemini-only runs and now also records the methodology and recall-improvement fixes implemented on April 27-28, 2026.

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
4. If the global dialogue target is met but critical themes remain below their floor, force clarification rounds on those weak themes instead of stopping early.
5. Select the top uncovered requirements for the chosen theme.
6. Ask the active LLM to generate exactly one natural follow-up question using:
   - the target theme
   - the most important uncovered requirement snippets
   - the recent dialogue history
   - an optional clarification focus hint for known hard clusters
7. Ask the active LLM to generate one consolidated stakeholder answer that may preserve short exact phrases for critical technical details instead of paraphrasing them away.
8. Use fixed fallback question templates only if JSON question generation fails.

This algorithm is the same for Ollama and Gemini runs. Provider choice may affect model quality or latency, but not the question-selection logic or prompt construction.

The current controller version is:

- `semantic_gap_llm_v2`

Algorithm 1: Controlled dialogue generation controller

1. Initialize all gold requirements as uncovered and assign each one to one or two semantic themes.
2. Select the next theme using semantic uncoveredness, fixed theme priority, uncovered count, average gap score, and a low-yield repetition cap.
3. If the global recall target is reached but any critical theme remains below its configured recall floor, override the normal stop condition and force a clarification round on those weak themes.
4. Build one interviewer prompt from the top uncovered requirement snippets for that theme plus the recent dialogue history.
5. Ask the active LLM for exactly one follow-up question in JSON form.
6. Ask the active LLM for one consolidated user answer that compresses several target requirements into natural dialogue under the configured `max_reqs_per_answer` and `max_chars_per_answer` limits, while preserving short exact phrases for critical roles, interfaces, storage concepts, and negative constraints.
7. Split the new user answer into sentence/clause support units and recompute semantic coverage against every gold requirement with the shared sentence-transformer scorer.
8. Stop only when the target dialogue recall is reached and all critical theme floors are satisfied, or when the turn budget and clarification budget are both exhausted.

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
6. Detect grounded-but-generic propositions that drop important anchors such as numbers, negation, exclusivity, role names, interface phrases, storage phrases, and configuration terms.
7. Rewrite those generic propositions using only their supporting evidence while preserving exact anchors when possible.
8. Run one optional gap pass on semantically uncovered evidence.
9. Convert propositions back into the unchanged final requirements schema.

Important implementation details added on April 27-28, 2026:

- the local extractor no longer repeats the full dialogue transcript in every batch
- `goal_scope` is no longer processed as a standalone extraction batch when real requirement themes exist
- local Ollama calls now use bounded per-batch output caps
- local Ollama calls now use per-batch timeouts so pathological structured generations fall back instead of hanging indefinitely
- validation searches claimed `evidence_turns` first, then falls back once to the full dialogue
- the stakeholder-answer and extraction prompts now preserve short exact phrases such as `user profile`, `browser interface`, `specified users or user groups`, and `super-user` more deliberately
- the rewrite trigger is now more aggressive on missing numbers, negation, exclusivity, role labels, interface/storage phrases, and configuration anchors

Algorithm 2: Dialogue-to-requirements extractor

1. Build an evidence bank from user sentence/clause support units, preserving turn ids, local context, and semantic theme labels.
2. Create either one full-context batch or several theme-aware evidence-bank batches, depending on the active preset.
3. For each batch, prompt the model with the scoped dialogue context, the selected evidence units, and the running memory of previously extracted propositions.
4. Extract atomic propositions in JSON form and merge duplicate propositions at a conservative semantic threshold.
5. Update the running memory so later batches can reuse already recovered global facts without re-reading the whole transcript.
6. Flag grounded-but-generic propositions when they drop critical anchors such as numbers, negation, exclusivity, role names, interface/storage phrases, or configuration terms that were present in the evidence.
7. Run one constrained rewrite pass using only the supporting evidence for those flagged propositions, preferring narrower source-faithful wording over broad paraphrase.
8. Run one optional gap pass over the highest-novelty uncovered evidence units.
9. Convert the final proposition set into the unchanged structured requirements schema and validate every output against the dialogue before scoring it against the source requirements.

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

There are now two evaluation layers:

1. A deterministic semantic layer using the shared sentence-transformer scorer and one-to-one greedy matching at threshold `0.55`.
2. A standardized Gemini verification layer that re-judges shortlisted source/generated requirement pairs with a checklist rubric.

We also compute a dialogue-only diagnostic score:

- how much gold requirement content is explicitly recoverable from user turns alone

Important evaluation update:

- `sentence-transformers` is now treated as core benchmark infrastructure, not just an auxiliary metric
- the same semantic sentence/clause-level scorer is now used consistently for dialogue coverage, pipeline coverage, validation, and reporting
- the old dialogue metric bug that compared short gold requirements to whole user turns is fixed
- the reports can now include a second-layer Gemini judge using a checklist-style validator rather than a free-form scalar score

The current standardized LLM validator is:

- provider: `gemini`
- model: `gemini-2.5-flash`
- role: second-layer requirement matching and verification across all runs, including runs generated by Ollama or other providers
- default shortlist: top-`2` semantic candidates plus top-`1` lexical candidate per requirement direction before Gemini adjudication

The LLM validator should be described carefully in the paper:

- it is methodologically useful as a second verification layer
- it should not replace the deterministic semantic scorer as the sole benchmark metric
- it is most defensible when it uses a fixed rubric, structured outputs, and explicit partial-credit rules
- it should be treated as scalable adjudication support, not as a substitute for final human adjudication in high-stakes claims

This change materially affected the measured recall bound on the hard PURE document.

Before the sentence-level semantic scorer was standardized, the older hard-document evaluation path reported only `0.25` dialogue coverage on `pure_0000_cctns` in `20260426T224847Z_pure_full`.

After shifting to sentence-level support units with the same `0.55` semantic threshold, the comparable hard-document dialogue bound rose to `0.61` in `20260427T003223Z_pure_full`.

So the sentence-transformer scorer should be described in the paper as a methodological correction that changed the apparent coverage ceiling in a major way, not as a cosmetic metric swap.

This dialogue-only score should still be read as a diagnostic bound, not as a strict theoretical ceiling on final recall.

The new Gemini judge layer should also be interpreted carefully:

- strict LLM metrics count only `full` requirement recoveries
- weighted LLM metrics count `full = 1.0`, `partial = 0.5`, and `none = 0.0`
- the semantic layer remains the reproducible backbone of the benchmark
- the Gemini layer is used to surface near-miss recoveries and paraphrastic partial matches that the semantic threshold may miss
- because LLM-as-a-judge methods can be prompt-sensitive and attack-sensitive, this layer is explicitly reported as secondary rather than replacing the main scorer

Algorithm 3: End-to-end benchmark procedure

1. Build or reuse trusted source-grounded requirement JSON for each PURE document.
2. Generate a controlled elicitation dialogue with the unified semantic controller.
3. Score dialogue-only recoverability with the shared semantic scorer.
4. Run the direct baseline from source requirements to final requirements.
5. Run the conversational pipeline from dialogue to evidence bank to propositions to final requirements.
6. Validate conversational outputs against dialogue evidence.
7. Score direct and conversational outputs against the trusted source requirements.
8. Re-score shortlisted requirement matches with the standardized Gemini checklist judge.
9. Run error analysis, compare variants, and publish the selected run summary and report artifacts.

Algorithm 4: Dual-layer requirement matching and adjudication

1. Embed all source and generated requirements with the shared sentence-transformer scorer and build semantic candidate shortlists.
2. Run deterministic greedy one-to-one semantic matching at threshold `0.55` to compute the primary precision, recall, F1, and hallucination metrics.
3. For the second layer, build a small shortlist per requirement direction using top semantic candidates plus one lexical-backup candidate.
4. Ask the standardized Gemini validator to classify each shortlisted pair as `full`, `partial`, or `none` under a fixed checklist rubric focused on actors, conditions, constraints, and scope preservation.
5. Convert judge outputs into strict metrics with `full = 1.0`, `partial = 0.0`, `none = 0.0`.
6. Convert the same judge outputs into weighted metrics with `full = 1.0`, `partial = 0.5`, `none = 0.0`.
7. Report both layers together: the semantic layer as the main continuity metric, and the Gemini layer as a source-faithfulness diagnostic that distinguishes exact recovery from partial recovery.

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

### Frontier Gemini Comparison Run (April 28, 2026)

I then ran the same hard-slice benchmark with a frontier Gemini model while keeping the dialogue controller and extraction algorithm unchanged.

Gemini hard-slice run:

- run id: `20260428T015358Z_pure_full`
- provider: `gemini`
- model: `gemini-3.1-pro-preview`
- document: `pure_0000_cctns`
- dialogue question algorithm: `semantic_gap_llm_v1`
- Gemini preset selection: `gemini_evidence_bank`
- full-context conversational recall: `0.50`
- evidence-bank conversational recall: `0.55`

### Aggregate Results: Frontier Gemini Hard Single Document

| Track | What it measures | Micro Precision | Micro Recall | Micro F1 | Hallucination |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Dialogue-only lower bound | Requirement content explicitly present in user turns | N/A | 0.8400 | N/A | N/A |
| Direct Gemini baseline | Source requirements -> Gemini -> requirements | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Conversational Gemini pipeline | Source -> dialogue -> Gemini -> requirements | 0.6548 | 0.5500 | 0.5978 | 0.3452 |
| Oracle | Perfect source reconstruction sanity check | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Second-Layer Gemini Judge On The Same Gemini Run

The same `20260428T015358Z_pure_full` run now also has a standardized Gemini validation layer using `gemini-2.5-flash` as a fixed evaluator after the semantic scorer.

| Layer | Strict Precision | Strict Recall | Strict F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Conversational pipeline | 0.3690 | 0.3100 | 0.3370 | 0.5655 | 0.4750 | 0.5163 |

Interpretation:

- the semantic benchmark still reports conversational recall at `0.55`
- the Gemini judge only counts `31` conversational outputs as full source-faithful recoveries
- another `33` conversational outputs are judged as partial matches rather than full recoveries
- so the second layer strengthens the conclusion that the frontier conversational run is dominated by partial paraphrase recovery rather than exact source reconstruction

### Direct Comparison: Latest Ollama vs Frontier Gemini On The Same Hard Slice

| Metric | Ollama `20260427T213602Z` | Gemini `20260428T015358Z` |
| :--- | ---: | ---: |
| Dialogue recall | 0.8500 | 0.8400 |
| Conversational precision | 0.7857 | 0.6548 |
| Conversational recall | 0.5500 | 0.5500 |
| Conversational F1 | 0.6471 | 0.5978 |
| Generated conversational requirements | 70 | 84 |
| Grounded conversational requirements | 70 | 84 |
| Unmatched generated outputs | 15 | 29 |
| Near-match paraphrases | 9 | 27 |
| Generic-domain outputs | 5 | 1 |

This comparison matters because it shows that the frontier model did not materially improve the conversational benchmark on this document. The recall stayed exactly the same as the patched Ollama run, while precision became worse.

### Why The Frontier Model Did Not Materially Beat Ollama

The main explanation is that the bottleneck on this hard slice is no longer raw model capacity.

It is the information transformation performed by the pipeline itself.

More specifically:

1. The direct baseline proves the frontier model is strong enough.
   Gemini reached `1.00` precision and `1.00` recall when it worked directly from the trusted source requirements, so the model itself is not the limiting factor on this document.
2. The dialogue stage still drops the same hard requirement families.
   Gemini dialogue recall was `0.84`, which is effectively the same as the patched Ollama run at `0.85`, and the weakest themes stayed the same: `user_roles_permissions` at `0.3333` and `availability_reliability` at `0.1667`.
3. The controller stopped before clarification rounds were used.
   In both the patched Ollama run and the Gemini run, `clarification_rounds_requested=2` but `clarification_rounds_used=0`, so the controller reached the global recall target without ever forcing deeper probing of the weak clusters.
4. The dialogue is semantically rich but source-lossy.
   Gemini produced more support units than the Ollama run (`123` vs `107`), but many of those turns restated requirements in polished natural language rather than preserving the exact normative constraints from the source.
5. The extractor optimizes groundedness better than source fidelity.
   Gemini grounded `84/84` conversational outputs in the dialogue, but only `55/84` matched the source requirements. This means the frontier model generated more dialogue-supported requirements, but many of them were too generic or too paraphrastic to count as source recovery.
6. The error profile shifted toward paraphrase drift, not true hallucination.
   The unmatched Gemini outputs were dominated by `27` near-match paraphrases, compared with `9` in the patched Ollama run. That is a strong sign that the frontier model is better at generating plausible, fluent restatements than at preserving the exact requirement boundary needed by the benchmark.
7. Full context alone did not solve the problem.
   Gemini full-context extraction reached only `0.50` recall, while the evidence-bank path reached `0.55`. That means the long context window helped less than the retrieval-based structure, but even the better variant still did not cross the `0.60` target.
8. The genericity detector under-triggered on Gemini outputs.
   The Gemini evidence-bank run reported only `4` rewrite candidates, versus `13` in the patched Ollama run, despite generating more paraphrastic outputs. This suggests the current rewrite trigger catches obviously weak local outputs, but misses fluent frontier-model paraphrases that remain semantically too broad.
9. The second-layer Gemini judge confirms that many apparent semantic matches are only partial recoveries.
   On the same run, the fixed Gemini validator scored only `31` conversational outputs as full requirement recoveries and `33` as partial matches. This means the semantic metric is useful for benchmark continuity, but the validator layer exposes how much of the remaining output is still paraphrastic or detail-losing rather than fully source-faithful.

The paper implication is important:

The frontier Gemini model substantially improves the direct source-to-requirements task, but it does not materially improve the dialogue-mediated source-recovery task under the current pipeline. That means the dominant bottleneck is algorithmic and representational, not just model strength.

### Second Gemini Hard-Slice Rerun After Recall-Focused Fixes (April 28, 2026)

I then reran the same hard-slice benchmark after implementing the recall-focused controller and rewrite fixes.

Second Gemini hard-slice run:

- run id: `20260428T032711Z_pure_full`
- provider: `gemini`
- model: `gemini-3.1-pro-preview`
- validator model: `gemini-2.5-flash`
- document: `pure_0000_cctns`
- dialogue question algorithm: `semantic_gap_llm_v2`
- selected conversational variant by benchmark rule: `gemini_evidence_bank`
- best raw conversational recall variant: `gemini_full_context`

### Aggregate Results: Second Gemini Hard Single Document

| Track | What it measures | Micro Precision | Micro Recall | Micro F1 | Hallucination |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Dialogue-only lower bound | Requirement content explicitly present in user turns | N/A | 0.9400 | N/A | N/A |
| Direct Gemini baseline | Source requirements -> Gemini -> requirements | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Conversational Gemini pipeline | Source -> dialogue -> Gemini -> requirements | 0.8272 | 0.6700 | 0.7403 | 0.1728 |
| Oracle | Perfect source reconstruction sanity check | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Second-Layer Gemini Judge On The Second Gemini Run

The same `20260428T032711Z_pure_full` run also has the standardized Gemini validation layer using `gemini-2.5-flash` as a fixed evaluator after the semantic scorer.

| Layer | Strict Precision | Strict Recall | Strict F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Conversational pipeline | 0.4691 | 0.3800 | 0.4199 | 0.6420 | 0.5200 | 0.5746 |

### Conversational Diagnostics: Second Gemini Hard Single Document

| Diagnostic | Value |
| :--- | ---: |
| Dialogue recall | 0.9400 |
| Evidence units | 108 |
| Propositions | 81 |
| Final grounded requirements | 81 |
| Validation hallucinations | 0 |
| Gap-pass additions | 0 |
| Rewrite candidates | 42 |

### Direct Comparison: First Gemini Run vs Second Gemini Run

| Metric | First Gemini `20260428T015358Z` | Second Gemini `20260428T032711Z` |
| :--- | ---: | ---: |
| Dialogue recall | 0.8400 | 0.9400 |
| Conversational semantic precision | 0.6548 | 0.8272 |
| Conversational semantic recall | 0.5500 | 0.6700 |
| Conversational semantic F1 | 0.5978 | 0.7403 |
| Gemini strict recall | 0.3100 | 0.3800 |
| Gemini weighted recall | 0.4750 | 0.5200 |
| Full conversational recoveries | 31 | 38 |
| Partial conversational recoveries | 33 | 28 |

### Why The Second Gemini Run Improved

The second Gemini run is important because it shows that the earlier `0.55` conversational ceiling was not a hard model limit.

The main reasons for the improvement are:

1. The controller changed from `semantic_gap_llm_v1` to `semantic_gap_llm_v2`.
   The new controller does not stop just because global recall is good enough. It now forces clarification rounds when critical themes remain under-covered.
2. The weak dialogue themes were materially repaired.
   The latest run pushed `user_roles_permissions` from the earlier `0.3333` failure state to `1.0000`, and `availability_reliability` from `0.1667` to `0.8333`.
3. The stakeholder-answer prompt became less paraphrase-happy for critical details.
   Preserving short exact phrases made it easier for the downstream extractor to recover narrower source-faithful propositions.
4. The rewrite trigger became more sensitive to dropped anchors.
   It now catches missing roles, negative constraints, interface/storage phrases, exclusivity markers, and configuration terms that were being smoothed away in the first Gemini run.
5. The extractor gained recall without losing control of groundedness.
   All `81` conversational outputs remained grounded in the dialogue, while semantic source recall rose from `0.55` to `0.67`.

### What The Second Gemini Run Still Does Not Solve

The hard-slice semantic target is now met, but the run is still not equivalent to perfect source reconstruction.

- the standardized semantic benchmark now clears the `0.60` target on the selected Gemini variant
- the stricter Gemini judge still reports only `0.38` strict recall and `0.52` weighted recall
- this means many outputs are now close enough to count semantically, but are still partial rather than fully source-faithful under a checklist reading
- the current benchmark selector still chose `gemini_evidence_bank`, even though `gemini_full_context` had slightly higher raw recall at `0.68`, because the selection rule requires at least a `0.02` recall advantage to switch variants

## 5. What The April 27-28, 2026 Improvements And Recall Fixes Changed

The latest implementation work introduced several concrete improvements that matter for the paper draft:

1. The semantic scorer is now shared across dialogue coverage, requirement coverage, validation, and reporting.
2. Dialogue coverage is now computed against sentence/clause support units rather than full user turns, which removed the previous under-reporting bug.
3. The question-generation algorithm is now explicitly unified across providers and recorded in run metadata.
4. The dialogue controller is now `semantic_gap_llm_v2`, which forces clarification rounds when critical themes remain below their recall floors.
5. The stakeholder-answer prompt now preserves short exact source phrases for critical roles, interfaces, storage concepts, and negative constraints instead of paraphrasing them away by default.
6. The evidence-bank extractor now uses scoped dialogue excerpts per batch instead of repeating the full transcript every time.
7. The benchmark runner now preserves the intended Python environment instead of resolving the venv symlink away.
8. Error analysis is now compatible with the shared list-based semantic scorer.
9. Local Ollama proposition extraction now uses bounded per-batch output caps and timeouts, so pathological structured JSON calls do not stall the entire run indefinitely.
10. The rewrite trigger is now more aggressive on missing anchors such as numbers, negation, exclusivity, role labels, interface/storage phrases, and configuration wording.
11. The hard-slice runs now record per-theme dialogue coverage, evidence-bank counts, proposition counts, rewrite counts, and gap-pass counts, which makes failure analysis much more specific than the earlier run logs.
12. A standardized Gemini second-layer validator now re-judges shortlisted requirement pairs across all runs, including Ollama-generated runs.
13. Reports are now generated automatically after completed runs and include both the deterministic semantic layer and the fixed Gemini validation layer.
14. The second Gemini hard-slice rerun demonstrates that the current algorithm can now exceed the `0.60` semantic recall target on the hard document.

These changes improved both methodological rigor and actual hard-slice recovery.

## 6. What These Results Mean

There are now four different paper-relevant messages:

### Message A: Best Current Headline Result

On the verified 2-document slice, the controlled conversational pipeline still clearly outperforms the direct local baseline.

That remains the strongest current paper-facing result for a compact verified slice.

### Message B: Local Hard-Document Stress Test

On the hard single-document slice, the updated local memory-augmented proposition pipeline now completes reliably enough to evaluate.

However, the local Ollama path still reaches only:

- dialogue recall: `0.85`
- pipeline recall: `0.55`

So the local hard-slice result is best interpreted as a successful infrastructure and methodology fix, but not yet as a headline-quality hard-slice recovery result.

### Message C: First Frontier Comparison

The first frontier Gemini run changed the direct baseline dramatically, but not the conversational benchmark.

That initial comparison showed that the hard-slice gap should not be framed as "local models are too weak". It isolated the bottleneck as the pipeline itself.

### Message D: Recall-Fixed Frontier Rerun

After the recall-focused controller and rewrite fixes, the second Gemini rerun materially improved the same hard-slice conversational benchmark:

- dialogue recall: `0.94`
- selected conversational semantic recall: `0.67`
- selected conversational semantic precision: `0.8272`
- selected conversational semantic F1: `0.7403`

So the hard-slice semantic target is now met with the updated Gemini pipeline.

However, the stricter second-layer judge still reports:

- strict recall: `0.38`
- weighted recall: `0.52`

So the main remaining bottleneck is no longer broad omission alone. It is exact source-faithfulness and requirement-boundary preservation.

## 7. Important Limitations And Paper Caution

Current limitations:

- the best current paper-facing benchmark still covers 2 PURE documents, not all 6
- the latest frontier hard-slice rerun still covers only 1 difficult document and should be treated as a stress test, not a replacement for the stronger 2-document headline result
- the exact hard-slice local rerun was completed after patching and resuming the pipeline stages in place
- the hard-slice conversational local pipeline used `self-consistency=1` for the local extractor so the run would finish under bounded Ollama generation
- the semantic hard-slice target is now exceeded on the second Gemini rerun, but the stricter validator layer still shows that many apparent recoveries remain partial
- grounded-to-dialogue does not automatically imply grounded-to-source when the dialogue itself drifts semantically
- the latest Gemini hard-slice rerun selected `gemini_evidence_bank` by methodology rule even though `gemini_full_context` had slightly higher raw recall, so the reported selected variant and the best raw variant should not be conflated

Methodological caution:

- do not claim broad industrial generalization from the current local and frontier runs
- do not mix earlier Gemini experiments, the verified 2-document local slice, the local hard-slice rerun, and the second Gemini rerun as if they were the same condition
- do not treat dialogue-only coverage as a strict ceiling on final source-grounded recall
- do not collapse the semantic metric and the Gemini judge into one number; they measure related but different notions of recovery
- do not interpret the direct Gemini `1.00 / 1.00` result, the conversational semantic `0.67` result, and the conversational strict `0.38` result as contradictory; they isolate different stages and different definitions of success
- do not report the selected Gemini variant and the best raw Gemini variant interchangeably without stating the selection rule

## 8. Current Best Interpretation For The Paper

The strongest current paper-safe interpretation is:

"A controlled conversational pipeline can outperform a direct local baseline on a verified PURE benchmark slice. On a previously problematic hard PURE document, the updated frontier Gemini pipeline now exceeds the `0.60` semantic recall target, reaching `0.67` under the selected evidence-bank variant and `0.68` in the best raw full-context variant. However, a fixed second-layer Gemini judge still scores the selected conversational run at only `0.38` strict recall and `0.52` weighted recall, which indicates that the remaining bottleneck is exact source-faithfulness and requirement-boundary preservation rather than gross coverage failure or benchmark infrastructure."

That statement is more accurate than claiming either that the hard-slice problem is fully solved or that the frontier model made no practical progress.

## 9. Recommended Next Fixes

The next practical improvements are:

1. Reduce off-domain drift in the generated dialogue so grounded extraction is more source-faithful.
2. Tighten proposition synthesis so one dialogue sentence does not produce multiple loosely paraphrased requirements.
3. Improve the remaining weaker dialogue themes in the latest Gemini rerun, especially `reporting_documentation` at `0.7500`, `other_constraints` at `0.7778`, `deployment_environment_constraints` at `0.8000`, and the remaining unrecovered availability item.
4. Improve exact-faithfulness for audit detail, complaint workflow, alerting/reporting detail, and user-profile-storage requirements that still degrade from `full` to `partial` under the Gemini judge.
5. Add even stronger constraints on proposition wording so exact source terminology is preserved more often.
6. Revisit the Gemini variant-selection rule if the research goal for a given run is pure best recall rather than preset-selection consistency.
7. Improve the genericity detector further so fluent frontier-model paraphrases are still flagged for rewrite when they drift away from the benchmark requirement boundary.
8. Make the gap pass useful on both local and Gemini runs; in the current hard-slice runs it still added `0` new grounded requirements.
9. Rerun the fixed `paper_regression_3doc` slice with `semantic_gap_llm_v2` and the standardized Gemini second-layer validator.
10. Only after that, rerun the full 6-document benchmark.

## 10. Files To Use

Best current paper-facing 2-document result:

- [best pipeline run](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full:1)
- [best pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/pipeline_coverage_postprocessed_v3.json:1)
- [best baseline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260424T205611Z_pure_full/metrics/direct_coverage.json:1)

Local hard-slice rerun after infrastructure fixes:

- [local hard-slice summary](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/comparison_summary.json:1)
- [local hard-slice pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/metrics/pipeline_coverage.json:1)
- [local hard-slice pipeline validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/metrics/pipeline_validation_report.json:1)
- [local hard-slice dialogue coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/metrics/dialogue_coverage_user_only.json:1)
- [local hard-slice summary PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/reports/summary.pdf:1)
- [local hard-slice full-detail PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260427T213602Z_pure_full/reports/full_detail.pdf:1)

First frontier Gemini hard-slice comparison run:

- [first Gemini summary](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/comparison_summary.json:1)
- [first Gemini pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/metrics/pipeline_coverage.json:1)
- [first Gemini direct LLM validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/metrics/direct_coverage_llm.json:1)
- [first Gemini pipeline LLM validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/metrics/pipeline_coverage_llm.json:1)
- [first Gemini pipeline validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/metrics/pipeline_validation_report.json:1)
- [first Gemini error analysis](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/metrics/pipeline_error_analysis.json:1)
- [first Gemini dialogue coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/metrics/dialogue_coverage_user_only.json:1)
- [first Gemini summary PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/reports/summary.pdf:1)
- [first Gemini full-detail PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T015358Z_pure_full/reports/full_detail.pdf:1)

Second Gemini hard-slice rerun after recall fixes:

- [second Gemini summary](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/comparison_summary.json:1)
- [second Gemini pipeline coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/metrics/pipeline_coverage.json:1)
- [second Gemini direct LLM validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/metrics/direct_coverage_llm.json:1)
- [second Gemini pipeline LLM validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/metrics/pipeline_coverage_llm.json:1)
- [second Gemini pipeline validation](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/metrics/pipeline_validation_report.json:1)
- [second Gemini error analysis](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/metrics/pipeline_error_analysis.json:1)
- [second Gemini dialogue coverage](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/metrics/dialogue_coverage_user_only.json:1)
- [second Gemini summary PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/reports/summary.pdf:1)
- [second Gemini full-detail PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/reports/full_detail.pdf:1)
- [second Gemini changes log](</Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260428T032711Z_pure_full/Changes and Improvements.md:1>)

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

For the exact April 27-28, 2026 hard-slice local rerun, the direct baseline remained at `self-consistency=3`, while the conversational local extractor was completed with `self-consistency=1` after the local proposition batching fixes were validated.

Recommended environment for the frontier Gemini comparison and rerun:

```bash
export REQ_LLM_PROVIDER=gemini
export REQ_GEMINI_MODEL=gemini-3.1-pro-preview
export REQ_GEMINI_TIMEOUT_SECONDS=180
export REQ_GEMINI_MAX_RETRIES=8
export REQ_GEMINI_RETRY_BACKOFF_SECONDS=10
export REQ_GEMINI_CACHE_TTL_SECONDS=3600
export REQ_VALIDATOR_GEMINI_MODEL=gemini-2.5-flash
```

Example hard-slice Gemini launch:

```bash
python3 scripts/run_pure_full_benchmark.py \
  --python-bin data/venv/bin/python3 \
  --benchmark-slice hard_single_doc \
  --pipeline-preset auto \
  --dialogue-variant controlled \
  --llm-provider gemini \
  --self-consistency 1 \
  --clarification-rounds 2 \
  --theme-max-exchanges 3 \
  --target-dialogue-recall 0.82 \
  --max-turns 28
```

In the exact April 28, 2026 second Gemini rerun, the benchmark runner evaluated both `gemini_full_context` and `gemini_evidence_bank`, selected `gemini_evidence_bank` under the current benchmark rule because the full-context recall advantage was only `0.01`, ran `gemini-2.5-flash` as the fixed second-layer validator, and generated the final reports automatically at run completion.

## 12. Four-Document Cross-Model Paper Slice (April 29, 2026)

I then built a paper-facing 4-document comparison slice to compare the current Gemini and Ollama pipelines on the same source set.

Included documents:

- `pure_0000_cctns`
- `pure_0000_gamma_j`
- `pure_1999_dii`
- `pure_2005_microcare`

Compared runs:

- Gemini run: `20260429T001722Z_pure_full`
- Ollama run: `20260429T002305Z_pure_full`
- fixed validator model for both runs: `gemini-2.5-flash`

Important comparability note:

- for this 4-document comparison, the final paper report uses the completed `full_context` conversational artifacts for both models
- the long evidence-bank workers did not complete reliably on this slice during the same overnight run window
- so this 4-document comparison should be described as a matched full-context cross-model comparison, not as an evidence-bank-vs-evidence-bank comparison

### Aggregate Results: 4-Document Paper Slice

The main paper-safe headline table should emphasize semantic metrics plus weighted validator metrics.

| Model | Dialogue Recall | Direct F1 | Conversational Precision | Conversational Recall | Conversational F1 | Weighted Validator Recall | Weighted Validator F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini `gemini-2.5-pro` | **0.9118** | 0.7784 | **0.7157** | **0.6912** | **0.7032** | **0.5294** | **0.5387** |
| Ollama `qwen2.5:7b-instruct` | 0.8824 | **0.9203** | 0.4967 | 0.3725 | 0.4258 | 0.3211 | 0.3669 |

### Main Interpretation: 4-Document Paper Slice

- Gemini is clearly stronger on the dialogue-mediated end-to-end recovery condition on this 4-document slice.
- Ollama is clearly stronger on the direct source-to-requirements condition on the same slice.
- Gemini also achieved slightly better dialogue coverage before extraction, which means part of the gain is upstream elicitation quality rather than only downstream extraction quality.
- The weighted Gemini validator still favors Gemini on the conversational condition, which means the semantic advantage is not just a threshold artifact.

This produces a cleaner paper message than the earlier single-document comparisons alone:

The local model is highly competitive when it works directly from source requirements, but the frontier Gemini model is materially better once the task becomes dialogue-mediated source recovery.

### Per-Document Conversational Result Pattern

Gemini beat Ollama on conversational semantic recall on all 4 documents:

| Document | Gemini Conversational Recall | Ollama Conversational Recall |
| :--- | ---: | ---: |
| `pure_0000_cctns` | **0.7200** | 0.4100 |
| `pure_0000_gamma_j` | **0.6140** | 0.3684 |
| `pure_1999_dii` | **0.5909** | 0.0909 |
| `pure_2005_microcare` | **0.8400** | 0.4800 |

That matters because it shows the Gemini advantage is not coming from only one outlier document.

### Dialogue-Grounding Result

Both 4-document conversational runs remained fully grounded after dialogue validation:

- Gemini grounded `197/197` conversational outputs in the dialogue
- Ollama grounded `153/153` conversational outputs in the dialogue
- both runs therefore had `0` dialogue-grounding hallucinations after validation

So the cross-model gap on this slice is not "Gemini hallucinates less from nowhere."

It is:

- Gemini preserves more source-relevant requirement content through the dialogue-to-requirements transformation
- Ollama loses much more source fidelity after the dialogue stage even when the dialogue itself still covers a large fraction of the source

## 13. Paper-Safe Reporting Choice

For the main paper-facing comparison tables, the reporting policy should now be:

1. Keep the deterministic semantic metrics as the primary headline benchmark.
2. Keep the weighted Gemini validator metrics as the secondary headline diagnostic.
3. Do not foreground strict Gemini `full-only` recall in the main comparison tables.

The reason is methodological, not cosmetic:

- the strict judge is useful as a harsher appendix diagnostic
- but it is too brittle and too low-level to serve as the main comparative headline
- the weighted judge preserves the useful signal about partial vs full recovery without collapsing near-miss recoveries into a single harsh failure bucket

So the paper-facing reporting stack should be described as:

- primary layer: semantic source-grounded recovery
- secondary layer: weighted Gemini adjudication
- appendix-only layer: strict Gemini adjudication

The raw strict metrics are still preserved in the underlying JSON artifacts if needed for appendix discussion or reviewer questions.

## 14. Main Paper Artifacts

Current main paper comparison artifacts:

- [cross-model comparison PDF](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/paper_reports/20260429_4doc_gemini_vs_ollama/cross_model_comparison_report.pdf:1)
- [cross-model comparison markdown](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/paper_reports/20260429_4doc_gemini_vs_ollama/cross_model_comparison_report.md:1)
- [cross-model comparison summary JSON](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/paper_reports/20260429_4doc_gemini_vs_ollama/cross_model_comparison_summary.json:1)
- [Gemini 4-document run](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260429T001722Z_pure_full:1)
- [Ollama 4-document run](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/pure_full_runs/20260429T002305Z_pure_full:1)

I also created a cleaner Obsidian-ready manuscript draft intended to become the actual submit-ready paper:

- [paper_manuscript_obsidian.md](/Users/yasseen/Documents/projects/req_dataset_project/data/docs/paper_manuscript_obsidian.md:1)
