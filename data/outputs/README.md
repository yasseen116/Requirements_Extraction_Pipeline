## Output Preservation Policy

This directory used to be fully ignored by Git. That made `git clean -fd` delete benchmark runs, reports, and summaries because they were untracked generated artifacts.

Current policy:

- `data/outputs/pure_full_runs/` is now intended to be trackable.
- Commit benchmark runs you need to preserve, especially:
  - `comparison_summary.json`
  - `run_config.json`
  - `metrics/*.json`
  - `reports/*`
  - any human-written `Changes and Improvements` notes
- raw provider dumps such as `*.raw_response.json` are still ignored globally and should stay that way unless you explicitly need them.

Practical rule:

- If a run matters for the paper, compare, or reproducibility, commit its folder under `data/outputs/pure_full_runs/` before cleanup.

