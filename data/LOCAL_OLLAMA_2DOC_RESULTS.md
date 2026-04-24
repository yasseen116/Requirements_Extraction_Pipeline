# Ollama 2-Document PURE Benchmark

Run id: `20260424T191659Z_pure_full`
Model: `qwen2.5:7b-instruct`
Provider: `ollama`
Documents: `pure_0000_gamma_j`, `pure_1999_dii`

## What Was Measured
- Dialogue-only lower bound: how much source requirement content made it into the generated user dialogue.
- Direct baseline: source requirements -> local LLM -> generated requirements.
- Conversational pipeline: source requirements -> controlled dialogue -> local LLM -> generated requirements.

## Main Results
| Track | Precision | Recall | F1 | Hallucination |
| --- | ---: | ---: | ---: | ---: |
| Dialogue-only lower bound | n/a | 0.7089 | n/a | n/a |
| Direct baseline | 0.8852 | 0.6835 | 0.7714 | 0.0988 |
| Conversational pipeline | 0.9028 | 0.8228 | 0.8609 | 0.0829 |

## Interpretation
- The local conversational pipeline beat the local direct baseline on this 2-document run.
- Direct baseline micro F1: `0.7714`
- Conversational pipeline micro F1: `0.8609`
- Absolute gain: `+0.0895`
- Dialogue-only lower bound was `0.7089`, and the final pipeline recall was `0.8228`.
- In this project, the dialogue coverage score is a diagnostic lower bound based on user-turn overlap, not a strict mathematical ceiling.
- The conversational pipeline can exceed that dialogue-only recall because the final requirement generator consolidates and paraphrases dialogue evidence better than the simple overlap check captures.
- Hallucination stayed relatively low at `0.0829`.

## What This Means For The Paper
- The local 7B model is usable for this project.
- The benchmark is now measurable without paid API dependency.
- The main remaining weakness is omission, not massive hallucination.
- The largest next-step opportunity is improving dialogue coverage on harder documents.

## Files
- `outputs/pure_full_runs/20260424T191659Z_pure_full/comparison_summary.json`
- `outputs/pure_full_runs/20260424T191659Z_pure_full/metrics/dialogue_coverage_user_only.json`
- `outputs/pure_full_runs/20260424T191659Z_pure_full/metrics/direct_coverage.json`
- `outputs/pure_full_runs/20260424T191659Z_pure_full/metrics/pipeline_coverage.json`
