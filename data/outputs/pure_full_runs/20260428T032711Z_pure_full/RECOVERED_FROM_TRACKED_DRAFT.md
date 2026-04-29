# Recovered Run Snapshot

This folder is a tracked reconstruction of the deleted run `20260428T032711Z_pure_full`.

Important:

- The original generated artifacts were deleted by `git clean -fd` while `data/outputs/` was ignored.
- Git cannot restore those deleted untracked files.
- This reconstruction is based on the tracked paper draft in [data/docs/project_workflow_and_results.md](/Users/yasseen/Documents/projects/req_dataset_project/data/docs/project_workflow_and_results.md:417).
- It preserves the key metrics, models, and interpretation of the `.67` recall run so you still have a stable reference in the repository.

Known original run facts from the tracked draft:

- run id: `20260428T032711Z_pure_full`
- provider: `gemini`
- generation model: `gemini-3.1-pro-preview`
- validator model: `gemini-2.5-flash`
- document: `pure_0000_cctns`
- dialogue question algorithm: `semantic_gap_llm_v2`
- selected conversational variant: `gemini_evidence_bank`
- best raw conversational variant: `gemini_full_context`

Recovered headline metrics:

- dialogue recall: `0.94`
- direct baseline precision / recall / F1: `1.00 / 1.00 / 1.00`
- conversational semantic precision / recall / F1: `0.8272 / 0.67 / 0.7403`
- conversational hallucination rate: `0.1728`
- Gemini strict precision / recall / F1: `0.4691 / 0.38 / 0.4199`
- Gemini weighted precision / recall / F1: `0.6420 / 0.52 / 0.5746`

Recovered diagnostics:

- evidence units: `108`
- propositions: `81`
- final grounded requirements: `81`
- validation hallucinations: `0`
- gap-pass additions: `0`
- rewrite candidates: `42`

Theme-level dialogue coverage recovered from the tracked draft:

- `user_roles_permissions`: `1.0000`
- `functional_capabilities`: `1.0000`
- `workflows_business_rules`: `0.4000`
- `data_validation`: `1.0000`
- `interfaces_integrations`: `1.0000`
- `performance_capacity`: `0.2500`
- `availability_reliability`: `0.8333`
- `security_audit`: `1.0000`
- `usability_help_accessibility`: `1.0000`
- `deployment_environment_constraints`: `0.8000`
- `maintainability_portability_testability`: `0.0000`
- `reporting_documentation`: `0.7500`
- `other_constraints`: `0.6667`

What cannot be recovered from Git alone:

- the original PDFs
- the original full metric JSON bundle
- the original generated requirements files
- the original raw responses

If you need the full real artifact set again, the only faithful path is to rerun the benchmark from commit `5e55113`.

