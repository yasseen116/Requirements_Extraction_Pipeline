# Gemini Primary Quickstart

## What This Adds

The project now supports a native Gemini main system:

- `G1`: `dialogue -> frames -> slots -> requirements`

This is different from the older generic API path because it is:

- native Gemini
- frame-first
- recorded with per-run manifests and condition reports

## Required Environment Variables

```bash
export REQ_GEMINI_API_KEY="your-key"
export REQ_GEMINI_MODEL="gemini-2.5-flash"
```

Optional:

```bash
export REQ_GEMINI_TEMPERATURE="0.0"
export REQ_GEMINI_DEBUG_RAW="false"
```

## Dry Run

This prints the exact prompt and JSON schema for one sample without making an API call:

```bash
python3 scripts/generate_g1_gemini_frames.py --dry-run --max-samples 1
```

## Run The Full Gemini Pipeline

```bash
python3 scripts/run_g1_gemini_pipeline.py
```

This creates:

- `outputs/g1_gemini_runs/<run_id>/manifest.json`
- `clean/frames`, `clean/combined`, `clean/slots`, `clean/requirements`
- `noisy/frames`, `noisy/combined`, `noisy/slots`, `noisy/requirements`
- readable reports in each condition directory
- an updated top-level `outputs/baseline_comparison_summary.md`

## What To Read First

1. `outputs/g1_gemini_latest_run.json`
2. `<run_dir>/clean/reports/summary_clean.md`
3. `<run_dir>/noisy/reports/summary_noisy.md`
4. `<run_dir>/noisy/reports/input_output_noisy.md`

## Compact Trace Default

By default the pipeline saves:

- normalized frame outputs
- derived slots
- generated requirements
- per-sample and aggregate metrics
- run metadata

It does not save raw Gemini response bodies unless:

```bash
export REQ_GEMINI_DEBUG_RAW="true"
```
