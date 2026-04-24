# Easy Pipeline Flow

## What the project does
The pipeline transforms controlled chatbot dialogue into structured requirements through four stages:

1. **Public source document**
2. **Manual gold sample**
3. **Predicted slots from dialogue**
4. **Generated requirements from slots**

The current pilot is intentionally small so that each stage can be inspected directly.

## Where each stage lives
- Public source corpora: `raw_sources/public/`
- Manual gold conversational samples: `raw_sources/manual_gold/`
- Noisy pilot variants: `synthetic/pilot_noisy/`
- Slot predictions: `outputs/*slots*/`
- Requirement predictions: `outputs/*requirements*/`
- Final summaries: `outputs/pilot_results_summary.md` and `outputs/baseline_comparison_summary.md`

## Baselines
- **B0**: gold slots -> template requirements
  - Purpose: upper bound for stage 2 only.
- **B1**: dialogue -> slots with simple lexical rules
  - Purpose: weak baseline that should fail under paraphrase.
- **B2**: dialogue -> slots with normalized keyword/paraphrase rules
  - Purpose: stronger interpretable baseline for the fixed-flow chatbot setting.
- **G1**: dialogue -> Gemini frames -> derived slots -> deterministic requirements
  - Purpose: main LLM system with a measurable intermediate representation.

## One-command run
Run:

```bash
python3 scripts/run_pilot_pipeline.py
```

This executes:
- validation of manual gold samples
- B0 generation and evaluation
- noisy pilot generation
- B1 clean/noisy slot extraction and requirement evaluation
- B2 clean/noisy slot extraction and requirement evaluation
- summary report generation
- a readable end-to-end report in `outputs/pilot_flow_report.md`

Optional API-based baseline:

```bash
python3 scripts/run_g1_gemini_pipeline.py
```

Setup details are in `docs/gemini_primary_quickstart.md`.

## What results to read first
1. `outputs/pilot_flow_report.md`
2. `outputs/baseline_comparison_summary.md`
3. `outputs/pilot_results_summary.md`
4. `outputs/b2_keyword_normalized_slots_noisy/evaluation.json`
5. `outputs/b2_keyword_normalized_requirements_noisy/evaluation.json`

## How to interpret the outputs
- If clean scores are perfect, the pilot is still too easy.
- The noisy condition is the real stress test.
- Slot failures matter most because requirement generation depends on them.
- High hallucination with low coverage means the system is both missing gold content and inventing unsupported requirements.

## Recommended current story
- `B1` proves that literal rule matching is too weak for messy dialogue.
- `B2` is the current best pilot system for the fixed six-question flow.
- `G1` is the main LLM path and should be compared against `B1` and `B2` with the same metrics.
- The next research step after the pilot is to scale the gold set while keeping the same evaluation pipeline and reporting format.
