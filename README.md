## Requirements Extraction Research Project

This repository contains a research pipeline for extracting **structured software requirements** from elicitation dialogues, and evaluating them against gold references.

### Repository structure (high level)
- `data/raw_sources/`: upstream corpora and benchmark gold exports
- `data/synthetic/`: generated dialogue variants (pilot + benchmark)
- `data/scripts/`: main pipeline + evaluation scripts
- `prompts/`: prompt templates used during dialogue generation / extraction
- `paper_artifacts/`: small, stable bundle of paper-cited outputs (tracked)

### Quickstart
- **Pilot pipeline (B0/B1/B2 baselines + reports)**:

```bash
python3 data/scripts/run_pilot_pipeline.py
```

- **PURE full benchmark (writes timestamped run under `data/outputs/pure_full_runs/`)**:

```bash
python3 data/scripts/run_pure_full_benchmark.py --max-samples 6
```

### Gemini configuration
The PURE runner conditionally executes Gemini-based stages when the env vars are present:
- `REQ_GEMINI_API_KEY`
- `REQ_GEMINI_MODEL`

### Local (offline) model configuration (Ollama)
You can run the pipeline without any cloud API by using Ollama.

1. Install and start Ollama (macOS):

```bash
brew install ollama
ollama serve
```

2. Pull a model (recommended for 16GB RAM):

```bash
ollama pull qwen2.5:7b-instruct
```

3. Run pipelines using Ollama:

```bash
export REQ_LLM_PROVIDER=ollama
export REQ_OLLAMA_MODEL=qwen2.5:7b-instruct

python3 data/scripts/run_pilot_pipeline.py
python3 data/scripts/run_pure_full_benchmark.py --seed 1 --dialogue-variant transcript_paraphrase
```

### Notes for research reporting
- The benchmark includes a **dialogue coverage upper bound** (how much of the gold is explicitly present in user turns). Always report this alongside end-to-end extraction scores.
- Generated run outputs under `data/outputs/` are treated as local artifacts and are ignored by default; paper-facing outputs belong in `paper_artifacts/`.

