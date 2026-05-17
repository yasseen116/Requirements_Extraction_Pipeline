# Qwen 4-Document Stability and Anchor Ablation

Generated at UTC: `2026-05-16T23:59:16.382159+00:00`

## Stability Summary

| Condition | Docs | Runs | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen full pipeline | 4 | 3 | 0.8824 +/- 0.0000 | 0.4857 +/- 0.0000 | 0.4167 +/- 0.0000 | 0.4485 +/- 0.0000 | N/A - Gemini validator disabled today |

## Per-Run Stability

| Run | Seed | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260516T223352Z_pure_full` | 1 | 0.8824 | 0.4857 | 0.4167 | 0.4485 |
| `20260516T224146Z_pure_full` | 2 | 0.8824 | 0.4857 | 0.4167 | 0.4485 |
| `20260516T224947Z_pure_full` | 3 | 0.8824 | 0.4857 | 0.4167 | 0.4485 |

## Anchor Preservation Ablation

| Variant | Docs | Runs | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 | Main interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full pipeline | 4 | 3 | 0.8824 +/- 0.0000 | 0.4857 +/- 0.0000 | 0.4167 +/- 0.0000 | 0.4485 +/- 0.0000 | Baseline repeated Qwen pipeline |
| No anchor preservation | 4 | 1 | 0.8824 | 0.4857 | 0.4167 | 0.4485 | Tests loss of requirement-specific anchors |

## Ablation Delta

- Baseline comparison run: `20260516T223352Z_pure_full`
- Ablation run: `20260516T225754Z_pure_full`
- Semantic F1 delta: 0.0000
- Semantic recall delta: 0.0000
- Semantic precision delta: 0.0000

## Existing Qwen Reference Run

| Run | Dialogue Recall | Semantic Precision | Semantic Recall | Semantic F1 |
| --- | ---: | ---: | ---: | ---: |
| `20260429T002305Z_pure_full` | 0.8824 | 0.4967 | 0.3725 | 0.4258 |

## Notes

- New runs intentionally disabled the Gemini validator.
- Weighted F1 is therefore not computed for stability or ablation rows.
- Existing reference weighted scores remain available in the original 4-document comparison artifacts.
