# PURE 4-Document Cross-Model Paper Report

Generated: `2026-04-28T23:21:45.522256+00:00`

## Included Documents

| Sample ID | Document ID | Source Requirements |
| --- | --- | --- |
| pure_0000_cctns | 0000_cctns | 100 |

## Aggregate Comparison

| Model | Generation Model | Dialogue Recall | Direct P | Direct R | Direct F1 | Pipeline P | Pipeline R | Pipeline F1 | LLM Strict R | LLM Weighted R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-smoke | gemini-3.1-pro-preview | 0.9000 | 1.0000 | 1.0000 | 1.0000 | 0.9403 | 0.6300 | 0.7545 | 0.3500 | 0.4650 |
| ollama-smoke | qwen2.5:7b-instruct | 0.9000 | 0.9600 | 0.9600 | 0.9600 | 0.6515 | 0.4300 | 0.5181 | N/A | N/A |

## Per-Document Dialogue Coverage

| Sample ID | Reqs | gemini-smoke Recall | gemini-smoke Units | ollama-smoke Recall | ollama-smoke Units |
| --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 100 | 0.9000 | 103 | 0.9000 | 128 |

## Per-Document Direct Semantic Metrics

| Sample ID | gemini-smoke P | gemini-smoke R | gemini-smoke F1 | ollama-smoke P | ollama-smoke R | ollama-smoke F1 |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 1.0000 | 1.0000 | 1.0000 | 0.9600 | 0.9600 | 0.9600 |

## Per-Document Conversational Semantic Metrics

| Sample ID | gemini-smoke P | gemini-smoke R | gemini-smoke F1 | ollama-smoke P | ollama-smoke R | ollama-smoke F1 |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 0.9403 | 0.6300 | 0.7545 | 0.6515 | 0.4300 | 0.5181 |

## Per-Document Conversational Validator Metrics

| Sample ID | gemini-smoke Strict R | gemini-smoke Weighted R | gemini-smoke Weighted F1 | ollama-smoke Strict R | ollama-smoke Weighted R | ollama-smoke Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 0.3500 | 0.4650 | 0.5569 | N/A | N/A | N/A |

## Per-Document Conversational Diagnostics

| Sample ID | gemini-smoke Evidence | gemini-smoke Props | gemini-smoke Grounded | ollama-smoke Evidence | ollama-smoke Props | ollama-smoke Grounded |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 103 | 67 | 67 | 128 | 66 | 66 |

## Run Paths

- `gemini-smoke`: `outputs/pure_full_runs/20260428T201035Z_pure_full`
- `ollama-smoke`: `outputs/pure_full_runs/20260428T205422Z_pure_full`
