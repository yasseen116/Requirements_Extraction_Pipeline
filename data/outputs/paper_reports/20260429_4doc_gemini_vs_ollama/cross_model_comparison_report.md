# PURE 4-Document Cross-Model Paper Report

Generated: `2026-04-29T03:29:51.415807+00:00`

## Headline Findings

- Higher dialogue recall: Gemini-2.5-Pro-FullContext **0.9118** vs Ollama-Qwen2.5-7B-FullContext 0.8824
- Stronger direct baseline F1: Gemini-2.5-Pro-FullContext 0.7784 vs Ollama-Qwen2.5-7B-FullContext **0.9203**
- Stronger conversational semantic recall: Gemini-2.5-Pro-FullContext **0.6912** vs Ollama-Qwen2.5-7B-FullContext 0.3725
- Stronger conversational semantic F1: Gemini-2.5-Pro-FullContext **0.7032** vs Ollama-Qwen2.5-7B-FullContext 0.4258
- Stronger weighted conversational validator recall: Gemini-2.5-Pro-FullContext **0.5294** vs Ollama-Qwen2.5-7B-FullContext 0.3211

## Included Documents

| Sample ID | Document ID | Source Requirements |
| --- | --- | --- |
| pure_0000_cctns | 0000_cctns | 100 |
| pure_0000_gamma_j | 0000_gamma_j | 57 |
| pure_1999_dii | 1999_dii | 22 |
| pure_2005_microcare | 2005_microcare | 25 |

## Aggregate Headline Metrics

| Model | Generation Model | **Dialogue Recall** | Direct P | **Direct R** | Direct F1 | Pipeline P | **Pipeline R** | Pipeline F1 | **LLM Weighted R** | LLM Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini-2.5-Pro-FullContext | gemini-2.5-pro | 0.9118 | 0.9257 | 0.6716 | 0.7784 | 0.7157 | 0.6912 | 0.7032 | 0.5294 | 0.5387 |
| Ollama-Qwen2.5-7B-FullContext | qwen2.5:7b-instruct | 0.8824 | 0.9676 | 0.8775 | 0.9203 | 0.4967 | 0.3725 | 0.4258 | 0.3211 | 0.3669 |

## Per-Document Dialogue Coverage

| Sample ID | Reqs | **Gemini-2.5-Pro-FullContext Recall** | Gemini-2.5-Pro-FullContext Units | **Ollama-Qwen2.5-7B-FullContext Recall** | Ollama-Qwen2.5-7B-FullContext Units |
| --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 100 | **0.9400** | 105 | 0.9000 | 128 |
| pure_0000_gamma_j | 57 | **0.8596** | 60 | 0.8246 | 88 |
| pure_1999_dii | 22 | **0.8636** | 32 | **0.8636** | 64 |
| pure_2005_microcare | 25 | **0.9600** | 38 | **0.9600** | 47 |

## Per-Document Direct Semantic Metrics

| Sample ID | Gemini-2.5-Pro-FullContext P | **Gemini-2.5-Pro-FullContext R** | Gemini-2.5-Pro-FullContext F1 | Ollama-Qwen2.5-7B-FullContext P | **Ollama-Qwen2.5-7B-FullContext R** | Ollama-Qwen2.5-7B-FullContext F1 |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 0.9592 | 0.4700 | 0.6309 | 0.9600 | **0.9600** | 0.9600 |
| pure_0000_gamma_j | 0.9630 | **0.9123** | 0.9369 | 1.0000 | 0.7895 | 0.8824 |
| pure_1999_dii | 0.8947 | **0.7727** | 0.8293 | 1.0000 | **0.7727** | 0.8718 |
| pure_2005_microcare | 0.8077 | **0.8400** | 0.8235 | 0.9130 | **0.8400** | 0.8750 |

## Per-Document Conversational Semantic Metrics

| Sample ID | Gemini-2.5-Pro-FullContext P | **Gemini-2.5-Pro-FullContext R** | Gemini-2.5-Pro-FullContext F1 | Ollama-Qwen2.5-7B-FullContext P | **Ollama-Qwen2.5-7B-FullContext R** | Ollama-Qwen2.5-7B-FullContext F1 |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 0.7912 | **0.7200** | 0.7539 | 0.5775 | 0.4100 | 0.4795 |
| pure_0000_gamma_j | 0.7292 | **0.6140** | 0.6667 | 0.4286 | 0.3684 | 0.3962 |
| pure_1999_dii | 0.5909 | **0.5909** | 0.5909 | 0.1429 | 0.0909 | 0.1111 |
| pure_2005_microcare | 0.5833 | **0.8400** | 0.6885 | 0.6316 | 0.4800 | 0.5455 |

## Per-Document Conversational Weighted Validator Metrics

| Sample ID | **Gemini-2.5-Pro-FullContext Weighted R** | Gemini-2.5-Pro-FullContext Weighted F1 | **Ollama-Qwen2.5-7B-FullContext Weighted R** | Ollama-Qwen2.5-7B-FullContext Weighted F1 |
| --- | --- | --- | --- | --- |
| pure_0000_cctns | **0.5700** | 0.5969 | 0.3500 | 0.4094 |
| pure_0000_gamma_j | **0.5351** | 0.5810 | 0.3333 | 0.3585 |
| pure_1999_dii | **0.3409** | 0.3409 | 0.2045 | 0.2500 |
| pure_2005_microcare | **0.5200** | 0.4262 | 0.2800 | 0.3182 |

## Per-Document Conversational Diagnostics

| Sample ID | Gemini-2.5-Pro-FullContext Evidence | Gemini-2.5-Pro-FullContext Props | Gemini-2.5-Pro-FullContext Grounded | Ollama-Qwen2.5-7B-FullContext Evidence | Ollama-Qwen2.5-7B-FullContext Props | Ollama-Qwen2.5-7B-FullContext Grounded |
| --- | --- | --- | --- | --- | --- | --- |
| pure_0000_cctns | 105 | 91 | 91 | 128 | 71 | 71 |
| pure_0000_gamma_j | 60 | 48 | 48 | 88 | 49 | 49 |
| pure_1999_dii | 32 | 22 | 22 | 64 | 14 | 14 |
| pure_2005_microcare | 38 | 36 | 36 | 47 | 19 | 19 |

## Run Paths

- `Gemini-2.5-Pro-FullContext`: `outputs/pure_full_runs/20260429T001722Z_pure_full`
- `Ollama-Qwen2.5-7B-FullContext`: `outputs/pure_full_runs/20260429T002305Z_pure_full`
