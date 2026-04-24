# Controlled Conversational Requirements Elicitation v1

## Problem definition
This project studies whether a controlled AI chatbot can transform messy user dialogue into a traceable set of software requirements through a staged pipeline: `dialogue -> normalized slots -> structured FR/NFR outputs`.

## What v1 does
- Uses a fixed six-question chatbot flow with limited clarifiers.
- Focuses on English, multi-domain software ideas.
- Produces functional requirements plus two non-functional categories: performance and security.
- Evaluates slot extraction, requirement generation, traceability, coverage, consistency, and hallucination.

## What v1 does not claim
- It is not a free-form autonomous elicitation agent.
- It does not generate a full industrial SRS from one short conversation.
- It does not replace human analysts or domain experts.
- It does not use synthetic data as the only source of truth.

## Main methodological decision
The benchmark must be anchored in trusted public requirements datasets. Public-source documents provide authentic requirement content, while manual annotations create the dialogue, slot, and requirement layers required for evaluation.

## Research questions
1. How accurately can the system recover normalized slots from messy dialogue?
2. Does an explicit slot stage improve requirement generation quality over direct dialogue-to-requirements generation?
3. How robust is the system to noisy or incomplete user answers?
4. Which failures dominate: missing evidence, poor normalization, actor-action errors, or unsupported generated requirements?

## Threats to validity to control early
- train/test leakage through shared source-document families
- synthetic test sets that inflate performance
- weak metrics that reward paraphrase while missing unsupported content
- overclaiming completeness from underspecified dialogue
