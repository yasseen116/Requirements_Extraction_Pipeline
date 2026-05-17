# Publishable Paper Upgrade Plan

This document is the revised full plan for turning the current benchmark work into a professional, submission-ready research paper while making explicit use of:

1. the strongest existing runs
2. the current methodology draft
3. the existing test suite
4. the current report-generation pipeline

It is intentionally grounded in the repository as it exists now, not in a hypothetical clean-slate paper.

## 1. Objective

The paper should make one clear scientific contribution:

> A controlled, source-grounded dialogue-to-requirements pipeline can be benchmarked rigorously, and model rankings differ substantially between direct source-to-requirements generation and dialogue-mediated source recovery.

To support that claim professionally, the paper must do three things well:

1. present the methodology clearly and conservatively
2. use the strongest existing runs without overclaiming
3. demonstrate reproducibility through artifacts, scripts, and tests

## 2. External Basis for the Plan

This plan follows four classes of external guidance and applies them directly to this project.

### 2.1 Empirical Software Engineering Reporting

Kitchenham et al. argue that reporting guidance should specify what information belongs in which section and avoid duplication or ambiguous structure.  
Source: [Evaluating Guidelines for Reporting Empirical Software Engineering Studies](https://link.springer.com/article/10.1007/s10664-007-9053-5)

Applied here:
- separate research questions, methodology, results, and threats to validity
- move implementation detail out of headline results unless it changes interpretation
- use a fixed paper structure rather than project-log narration

### 2.2 Observed Reporting Failures in SE Experiments

Revoredo et al. show that many software engineering experiment papers omit required information, lack a standard reporting sequence, and often fail to anchor claims in clear experimental structure.  
Source: [A Study into the Practice of Reporting Software Engineering Experiments](https://link.springer.com/article/10.1007/s10664-021-10007-3)

Applied here:
- use explicit slice descriptions
- state what each run is for
- separate compact verified results from stress tests and broader comparative slices
- add a reproducibility and artifact section as part of the paper, not only as repo documentation

### 2.3 Open Science and Artifact Expectations

ICSE 2025 states that artifact sharing should be the default and that papers should describe how artifacts are accessed.  
Source: [ICSE 2025 Research Track Open Science Policy](https://conf.researchr.org/track/icse-2025/icse-2025-research-track?track=ICSE+Research+Track+ICSE+SE+In+Practice)

Applied here:
- build a paper artifact package, not just code
- provide run IDs, configs, scripts, and outputs in a navigable structure
- make the paper’s reported numbers traceable to saved run artifacts

### 2.4 NLP Reproducibility and Benchmark Robustness

Magnusson et al. show that stronger reporting of metrics, infrastructure, and artifacts improves reproducibility.  
Source: [Reproducibility in NLP: What Have We Learned from the Checklist?](https://aclanthology.org/2023.findings-acl.809.pdf)

Ailem et al. show that benchmark-level model comparisons can be sensitive to benchmark composition and aggregation assumptions.  
Source: [Examining the Robustness of LLM Evaluation to the Distributional Assumptions of Benchmarks](https://aclanthology.org/2024.acl-long.560.pdf)

Applied here:
- report both aggregate and per-document results
- state clearly what slice each result comes from
- avoid relying on one metric or one document
- keep deterministic semantic scoring as the backbone metric

## 3. Assets We Already Have

The paper should be built from existing assets instead of re-explaining the entire project from scratch.

### 3.1 Core Methodology Draft

Primary source:
- [project_workflow_and_results.md](/Users/yasseen/Documents/projects/req_dataset_project/data/docs/project_workflow_and_results.md:1)

This file already contains:
- dialogue controller algorithm
- proposition extraction algorithm
- dual-layer evaluation design
- hard-slice interpretation
- 4-document cross-model interpretation
- key methodological cautions
- source references that can be reused in the paper

### 3.2 Existing Paper-Facing Manuscript

Current manuscript source:
- [paper_manuscript_obsidian.md](/Users/yasseen/Documents/projects/req_dataset_project/data/docs/paper_manuscript_obsidian.md:1)

This file is now the preferred narrative source, but it still needs more quantitative hardening and artifact integration.

### 3.3 Existing Report Artifacts

Primary cross-model report:
- [cross_model_comparison_report.pdf](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/paper_reports/20260429_4doc_gemini_vs_ollama/cross_model_comparison_report.pdf:1)
- [cross_model_comparison_report.md](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/paper_reports/20260429_4doc_gemini_vs_ollama/cross_model_comparison_report.md:1)
- [cross_model_comparison_summary.json](/Users/yasseen/Documents/projects/req_dataset_project/data/outputs/paper_reports/20260429_4doc_gemini_vs_ollama/cross_model_comparison_summary.json:1)

These should feed tables and appendix material rather than be treated as the paper itself.

### 3.4 Existing Reported Runs

The strongest currently useful runs are:

1. `20260424T205611Z_pure_full`
   - verified 2-document local benchmark
   - best compact local headline result

2. `20260427T213602Z_pure_full`
   - local hard-slice rerun
   - infrastructure-fix proof point

3. `20260428T015358Z_pure_full`
   - first frontier Gemini hard-slice comparison
   - useful for showing that model strength alone did not initially solve the pipeline

4. `20260428T032711Z_pure_full`
   - improved hard-slice Gemini rerun
   - strongest single-document frontier recovery result

5. `20260429T001722Z_pure_full`
   - Gemini 4-document paper slice

6. `20260429T002305Z_pure_full`
   - Ollama 4-document paper slice

### 3.5 Existing Test and Validation Assets

Primary test file:
- [tests/test_recall_pipeline.py](/Users/yasseen/Documents/projects/req_dataset_project/tests/test_recall_pipeline.py:1)

This suite already covers paper-relevant logic:

1. dialogue controller metadata exposure
2. support-unit splitting
3. contextual support-unit construction
4. semantic theme annotation
5. low-yield theme cap behavior
6. forced clarification logic
7. dialogue context scoping for evidence-bank extraction
8. batch construction rules
9. proposition deduplication
10. generic proposition detection
11. validation preference for claimed evidence turns
12. novelty gap selection
13. LLM candidate-pair generation
14. partial-credit matching logic
15. report model-metadata loading

Additional stability support already exists in:
- [run_pure_full_multi_seed.py](/Users/yasseen/Documents/projects/req_dataset_project/data/scripts/run_pure_full_multi_seed.py:1)

That script is especially important because it gives us a direct path to paper-quality variance reporting rather than single-run-only claims.

## 4. Paper Claim Architecture

The paper will be stronger if each claim is assigned to the run evidence best suited to support it.

### 4.1 Headline Claim

Recommended headline:

> A controlled conversational pipeline can outperform a direct local baseline on a verified compact PURE slice, while a broader 4-document comparison shows that dialogue-mediated recovery and direct structured generation produce different cross-model rankings.

Why this is strongest:
- it uses the clean verified 2-document result for the strongest positive claim
- it uses the 4-document slice for broader comparison
- it avoids claiming that the hard single-document frontier result is the sole headline

### 4.2 Supporting Claims

Claim A:
- controlled dialogue can preserve enough information for strong source recovery
- supported by `20260424T205611Z_pure_full`

Claim B:
- the hardest remaining problem is source-faithfulness, not unsupported hallucination
- supported by `20260428T032711Z_pure_full` and dialogue-grounding diagnostics

Claim C:
- direct-source generation and dialogue-mediated recovery should be treated as different benchmark conditions
- supported by the 4-document Gemini/Ollama comparison:
  - `20260429T001722Z_pure_full`
  - `20260429T002305Z_pure_full`

### 4.3 Claims to Avoid

Do not claim:
- full industrial generalization
- that weighted LLM adjudication is equivalent to human validation
- that the 4-document full-context comparison proves evidence-bank superiority or inferiority
- that the hard-slice problem is fully solved

## 5. Full Use of Existing Runs

The current paper should explicitly reuse the run history as a structured narrative, not as scattered historical notes.

### 5.1 Run Roles in the Paper

Use each run for one job only:

| Run | Role in paper |
| --- | --- |
| `20260424T205611Z` | compact verified positive headline |
| `20260427T213602Z` | local hard-slice stress baseline |
| `20260428T015358Z` | first frontier-model stress comparison |
| `20260428T032711Z` | improved hard-slice frontier rerun |
| `20260429T001722Z` | Gemini 4-document comparison arm |
| `20260429T002305Z` | Ollama 4-document comparison arm |

This prevents the manuscript from reading like an accumulated lab notebook.

### 5.2 Recommended Paper Result Layout

Main body:

1. Table: compact verified 2-document local result
2. Table: hard-slice single-document comparison
3. Table: 4-document aggregate cross-model result
4. Table: 4-document per-document recall breakdown

Appendix:

1. per-theme hard-slice coverage
2. weighted-vs-semantic disagreement audit
3. run configuration matrix
4. extended per-document diagnostics

## 6. Full Use of Existing Tests

The tests should not just remain engineering checks. They should support the paper’s reproducibility story.

### 6.1 What the Current Tests Already Prove

The current suite supports the following paper-facing claims:

| Test area | Paper relevance |
| --- | --- |
| dialogue controller metadata | the reported controller version is explicit and stable |
| forced clarification logic | the recall-improvement controller behavior is testable |
| support-unit segmentation | the unit of analysis for dialogue coverage is reproducible |
| evidence-bank context scoping | extraction context is not silently full-transcript by accident |
| proposition deduplication | duplicate handling is not ad hoc |
| generic proposition detection | rewrite triggering has a deterministic basis |
| validator candidate-pair generation | the secondary LLM judge uses a controlled shortlist |
| partial-credit matching | weighted adjudication logic is explicit, not hand-waved |
| report metadata loading | reported model/validator metadata is programmatically traceable |

### 6.2 How Tests Should Be Used in the Paper

The paper should not dump unit-test details into the main text. Instead:

1. mention in the artifact section that core benchmark logic is covered by regression tests
2. include the test categories in the artifact README
3. optionally add one appendix table mapping paper components to test coverage

### 6.3 Additional Tests Worth Adding

Before submission, add four more paper-facing tests:

1. **Cross-model summary consistency test**
   - verifies that aggregate report numbers equal the underlying per-document summaries

2. **Artifact-index completeness test**
   - verifies that all paper-reported run IDs have their required summary files

3. **Manuscript number consistency test**
   - checks that key values in the manuscript match the stored JSON summaries

4. **Report-generation smoke test**
   - ensures the report builders run successfully on the paper-report slice

These would materially strengthen the reproducibility story.

## 7. Professionalization Work Packages

### Work Package A. Freeze the Scientific Narrative

Deliverables:
- fixed abstract claim
- fixed introduction claim
- fixed conclusion claim

Action:
- choose one compact headline result and one broader comparison result
- remove any narrative drift between manuscript, draft, and report

### Work Package B. Harden the Quantitative Section

Deliverables:
- macro metrics where available
- one stability subsection
- one compact ablation subsection

Recommended implementation:
- use [run_pure_full_multi_seed.py](/Users/yasseen/Documents/projects/req_dataset_project/data/scripts/run_pure_full_multi_seed.py:1) on a small representative slice
- report mean and standard deviation for recall and F1
- add one table for:
  - no clarification vs clarification
  - full-context vs evidence-bank where comparable
  - semantic vs weighted interpretation

### Work Package C. Build the Artifact Package

Deliverables:
- `ARTIFACT_README.md`
- `artifact_index.json`
- paper run manifest
- claim-to-run mapping
- claim-to-test mapping

Recommended contents:

1. all run IDs cited in the manuscript
2. all report files cited in the manuscript
3. exact scripts used to produce them
4. environment assumptions
5. how to rerun or partially reproduce key results

### Work Package D. Build the Evidence Appendix

Deliverables:
- sample requirement-pair appendix
- semantic vs weighted disagreement appendix
- hard-slice theme appendix

Recommended scope:
- 20-30 examples
- stratified by full, partial, disagreement, and miss

### Work Package E. Final Venue Conversion

Deliverables:
- BibTeX
- static figures
- final PDF draft
- anonymization / disclosure pass if needed

## 8. Concrete Next Implementation Steps

The next high-value implementation sequence should be:

1. **Create an artifact index**
   - machine-readable index of all runs cited in the manuscript

2. **Add paper-facing reproducibility tests**
   - cross-model summary consistency
   - manuscript number consistency
   - report smoke test

3. **Add a Stability and Ablation subsection**
   - use existing multi-seed support where possible

4. **Add a claim-to-run and claim-to-test appendix**
   - this will make the paper feel much more disciplined

5. **Only after that, do wording polish**
   - wording polish is much less valuable before the quantitative and artifact structure is locked

## 9. Recommended Immediate Deliverables

If we continue from here, the strongest next concrete outputs are:

1. `data/docs/ARTIFACT_INDEX_PLAN.md` or a direct `artifact_index.json`
2. a manuscript section `6.5 Stability and Ablation`
3. new tests for report/manuscript consistency
4. an appendix-ready audit table for semantic vs weighted disagreements

Those four changes would make the paper substantially more professional than another round of prose-only editing.
