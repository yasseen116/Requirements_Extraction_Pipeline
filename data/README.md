# Requirements Dataset Project

## Recommended Local LLM Setup

For this machine, the strongest practical local setup is:

- provider: `ollama`
- model: `qwen2.5:7b-instruct`

This keeps the project fully local while still giving a capable structured extraction model.

## Start Ollama

Make sure Ollama is installed and running:

```bash
ollama --version
ollama serve
```

In another terminal, pull the model:

```bash
ollama pull qwen2.5:7b-instruct
```

## Quality-Focused Environment

Use these settings for the research pipeline:

```bash
export REQ_LLM_PROVIDER=ollama
export REQ_OLLAMA_MODEL=qwen2.5:7b-instruct
export REQ_OLLAMA_TIMEOUT_SECONDS=900
export REQ_OLLAMA_NUM_CTX=8192
export REQ_OLLAMA_NUM_PREDICT=4096
export REQ_OLLAMA_SEED=42
```

Notes:

- `REQ_OLLAMA_NUM_CTX=8192` is the safe default for a 16 GB machine.
- If the machine stays stable and you want more context, try `12288` or `16384`.
- Keep the seed fixed for reproducible research runs.

## Run The Pilot

```bash
python3 scripts/run_pilot_pipeline.py
```

## Run The PURE Full Benchmark

Recommended cached-source run:

```bash
python3 scripts/run_pure_full_benchmark.py \
  --reuse-source-dir outputs/reuse_source_6docs \
  --max-samples 6 \
  --dialogue-variant controlled
```

If you need to override the interpreter used by child scripts:

```bash
export REQ_PYTHON_BIN=/usr/bin/python3
```

This is useful on this machine because the current Homebrew Python 3.14 build has an `expat` linkage issue for XML parsing.

## What The Ollama Path Now Does

When `REQ_LLM_PROVIDER=ollama` is active, the benchmark now uses:

- deterministic generation by default (`temperature=0.0`)
- longer local timeouts for large documents
- chunked source-to-requirements generation for large PURE documents
- chunked dialogue-to-requirements generation for long elicitation transcripts

This is important because a 7B local model should not be evaluated with oversized single-shot prompts that make it fail for avoidable reasons.

## Main Outputs

After a run, inspect:

- `outputs/pure_full_latest_run.json`
- `outputs/pure_full_runs/<run_id>/metrics/dialogue_coverage_user_only.json`
- `outputs/pure_full_runs/<run_id>/metrics/direct_coverage.json`
- `outputs/pure_full_runs/<run_id>/metrics/pipeline_coverage.json`
- `outputs/pure_full_runs/<run_id>/metrics/gemini_coverage.json`

`pipeline_coverage.json` is the provider-neutral file name.

`gemini_coverage.json` is still written for backward compatibility, even when the provider is Ollama.
