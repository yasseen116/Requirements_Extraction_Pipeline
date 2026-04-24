# Source-Grounded Dataset Strategy

## Why this layer exists
The project needs a trusted origin for requirement content so the evaluation does not rest on synthetic text alone. Public, DOI-backed datasets provide that origin layer. The conversational dataset used by the project should therefore be **derived from public requirement sources**, not invented from scratch.

## Dataset roles
| Dataset | Project role | Why it matters |
| --- | --- | --- |
| `PURE` | Primary source corpus | Public requirements documents suitable for building source-grounded dialogue/slot/requirement samples. |
| `Software Requirements Data Set` | Supporting source | Preserves text conversions and multiple source formats that help ingestion and traceability. |
| `Promise+` | External evaluation | Useful for FR/NFR validation and classification-oriented sanity checks. |
| `NICE` | External evaluation | Useful for NFR subtype checks, especially performance and security reporting. |

## Intended data flow
1. Start from a source document and a source sentence span in a trusted public dataset.
2. Convert that source material into a normalized scenario card.
3. Author a controlled dialogue that expresses the same content through the fixed chatbot flow.
4. Annotate gold slots with evidence turn ids.
5. Generate gold functional and non-functional requirements with `source_slots` traceability.
6. Use synthetic augmentation only on train/dev descendants of the manual gold set.

## Required provenance fields
Every manually curated sample should include:
- `source.dataset`
- `source.dataset_doi`
- `source.record_url`
- `source.document_id`
- `source.source_sentence_ids`
- `metadata.source_type`
- `metadata.parent_id`

## Split policy
- Split by **source document family**, not by individual dialogue.
- Keep all synthetic descendants in the same split as their parent manual sample.
- Never reuse held-out test documents as prompt exemplars.

## Validation policy
The public source is the **trust anchor**, but the project still needs a manually built gold layer. Validation should compare:
- dialogue -> predicted slots vs gold slots
- predicted requirements vs gold requirements
- predicted requirements vs public-source evidence

This separation lets the project measure coverage and hallucination without pretending the public corpora already contain dialogues.

## Caution
PURE is useful, but its record explicitly notes uncertainty about original document rights. Use citation, provenance, and restrained redistribution practices.
