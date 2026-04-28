from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "scripts"))

from coverage_scorer import CoverageScorer, split_support_text
from generate_pure_controlled_dialogues import annotate_requirements, choose_target_theme
from generate_pure_full_requirements import (
    build_theme_batches,
    merge_proposition_entries,
    novelty_gap_units,
    render_dialogue_context,
)
import validate_pure_extracted_requirements as validation_module


class FakeScorer:
    similarity_method = "fake"

    def __init__(self, matrix: list[list[float]] | None = None) -> None:
        self.matrix = matrix or []

    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> list[list[float]]:
        return self.matrix

    def similarity_row(self, query: str, candidates: list[str]) -> list[float]:
        row = []
        for candidate in candidates:
            row.append(1.0 if query == candidate else 0.0)
        return row


class ValidationFakeScorer:
    similarity_method = "fake"

    def build_dialogue_support_units(self, payload: dict, *, user_only: bool = True) -> list[dict]:
        return CoverageScorer(context_window=1).build_dialogue_support_units(payload, user_only=user_only)

    def best_unit_for_query(self, query: str, units: list[dict], *, candidate_turn_ids=None, contextualized: bool = True):
        filtered = units if not candidate_turn_ids else [unit for unit in units if unit.get("turn_id") in candidate_turn_ids]
        filtered = filtered or units
        for unit in filtered:
            if query == unit.get("text"):
                return 1.0, unit
        return 0.0, filtered[0] if filtered else None


class RecallPipelineTests(unittest.TestCase):
    def test_dialogue_dry_run_exposes_unified_question_algorithm_metadata(self) -> None:
        script_path = ROOT / "data" / "scripts" / "generate_pure_controlled_dialogues.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "sample.json"
            sample_path.write_text(
                json.dumps(
                    {
                        "sample_id": "pure_test_sample",
                        "source": {"title": "Test System", "document_id": "DOC-TEST"},
                        "ground_truth_requirements": [
                            {"req_id": "REQ-001", "text": "Users can save defaults.", "category": "functional"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(script_path), "--input-dir", tmp_dir, "--max-samples", "1", "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
            )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["question_algorithm_version"], "semantic_gap_llm_v1")
        self.assertNotIn("question_generation", payload)

    def test_split_support_text_uses_sentences_and_semicolons(self) -> None:
        parts = split_support_text("Save defaults in the user profile; avoid horizontal scrolling. Support browser access too.")
        self.assertEqual(
            parts,
            [
                "Save defaults in the user profile",
                "avoid horizontal scrolling.",
                "Support browser access too.",
            ],
        )

    def test_contextual_support_units_include_neighboring_clauses(self) -> None:
        scorer = CoverageScorer(context_window=1)
        payload = {
            "dialogue": [
                {
                    "turn_id": 2,
                    "role": "user",
                    "text": "Save defaults in the user profile; avoid horizontal scrolling. Support browser access too.",
                }
            ],
            "dialogue_generation": {"trace": [{"user_turn_id": 2, "theme": "usability_help_accessibility"}]},
        }
        units = scorer.build_dialogue_support_units(payload, user_only=True)
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0]["trace_theme"], "usability_help_accessibility")
        self.assertIn("avoid horizontal scrolling.", units[0]["context_text"])
        self.assertIn("Save defaults in the user profile", units[1]["context_text"])

    def test_annotate_requirements_supports_secondary_theme_on_small_margin(self) -> None:
        requirements = [{"req_id": "AUTO-0001", "text": "User settings must be saved in the user profile."}]
        scorer = FakeScorer(matrix=[[0.82, 0.80] + [0.1] * 11])
        annotated = annotate_requirements(requirements, scorer)
        self.assertEqual(annotated[0]["primary_theme"], "user_roles_permissions")
        self.assertEqual(annotated[0]["secondary_theme"], "functional_capabilities")

    def test_choose_target_theme_respects_low_yield_cap(self) -> None:
        uncovered = [
            {"req_id": "1", "primary_theme": "deployment_environment_constraints", "secondary_theme": None, "best_score": 0.2},
            {"req_id": "2", "primary_theme": "functional_capabilities", "secondary_theme": None, "best_score": 0.3},
            {"req_id": "3", "primary_theme": "functional_capabilities", "secondary_theme": None, "best_score": 0.25},
        ]
        theme = choose_target_theme(uncovered, theme_exchange_counts={"deployment_environment_constraints": 3}, theme_max_exchanges=3)
        self.assertEqual(theme, "functional_capabilities")

    def test_render_dialogue_context_scopes_to_relevant_exchange_for_evidence_bank(self) -> None:
        sample = {
            "dialogue": [
                {"turn_id": 1, "role": "bot", "text": "What is the scope?"},
                {"turn_id": 2, "role": "user", "text": "This is the scope."},
                {"turn_id": 3, "role": "bot", "text": "How should defaults work?"},
                {"turn_id": 4, "role": "user", "text": "Save defaults in the user profile."},
                {"turn_id": 5, "role": "bot", "text": "What about browser support?"},
                {"turn_id": 6, "role": "user", "text": "Support a browser interface too."},
            ],
            "dialogue_generation": {
                "trace": [
                    {"bot_turn_id": 1, "user_turn_id": 2, "theme": "goal_scope"},
                    {"bot_turn_id": 3, "user_turn_id": 4, "theme": "data_validation"},
                    {"bot_turn_id": 5, "user_turn_id": 6, "theme": "interfaces_integrations"},
                ]
            },
        }
        context = render_dialogue_context(
            sample,
            [{"unit_id": "4:1", "turn_id": 4, "text": "Save defaults in the user profile.", "context_text": "Save defaults in the user profile."}],
            extraction_mode="evidence_bank",
        )
        self.assertIn("3. bot: How should defaults work?", context)
        self.assertIn("4. user: Save defaults in the user profile.", context)
        self.assertNotIn("6. user: Support a browser interface too.", context)

    def test_build_theme_batches_skips_goal_scope_when_other_themes_exist(self) -> None:
        batches = build_theme_batches(
            [
                {"unit_id": "2:1", "turn_id": 2, "sentence_index": 0, "trace_theme": "goal_scope"},
                {"unit_id": "4:1", "turn_id": 4, "sentence_index": 0, "trace_theme": "functional_capabilities"},
            ],
            top_k=14,
            overlap=2,
            chunked=True,
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["theme"], "functional_capabilities")

    def test_merge_proposition_entries_uses_high_threshold_dedup(self) -> None:
        scorer = FakeScorer()
        propositions = [
            {"id": "P-001", "category": "functional", "text": "Users can configure saved defaults.", "priority": "medium", "evidence_turns": [2], "source_unit_ids": ["2:1"]},
            {"id": "P-002", "category": "functional", "text": "Users can configure saved defaults.", "priority": "medium", "evidence_turns": [4], "source_unit_ids": ["4:1"]},
        ]
        merged, removed = merge_proposition_entries(propositions, scorer, threshold=0.90)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["evidence_turns"], [2, 4])
        self.assertEqual(len(removed), 1)

    def test_validate_items_prefers_claimed_evidence_turns(self) -> None:
        dialogue_payload = {
            "dialogue": [
                {"turn_id": 2, "role": "user", "text": "Users need their defaults saved in the profile."},
                {"turn_id": 4, "role": "user", "text": "The support tools should also work through a browser interface."},
            ]
        }
        items = [
            {
                "id": "DATA-001",
                "category": "data",
                "text": "Users need their defaults saved in the profile.",
                "priority": "medium",
                "evidence_turns": [2],
                "nfr_category": None,
            }
        ]
        original_scorer = validation_module._SCORER
        validation_module._SCORER = ValidationFakeScorer()
        try:
            result = validation_module.validate_items(dialogue_payload, items, threshold=0.25)
        finally:
            validation_module._SCORER = original_scorer
        grounded = result["grounded_items"][0]
        self.assertEqual(grounded["best_supporting_turn_id"], 2)
        self.assertTrue(grounded["matched_claimed_turns"])

    def test_novelty_gap_units_selects_uncovered_evidence(self) -> None:
        scorer = FakeScorer()
        units = [
            {"unit_id": "2:1", "context_text": "Save defaults in the user profile.", "text": "Save defaults in the user profile."},
            {"unit_id": "4:1", "context_text": "Avoid horizontal scrolling in the interface.", "text": "Avoid horizontal scrolling in the interface."},
        ]
        propositions = [{"text": "Save defaults in the user profile."}]
        gap_units = novelty_gap_units(units, propositions, scorer, top_k=1)
        self.assertEqual(gap_units[0]["unit_id"], "4:1")


if __name__ == "__main__":
    unittest.main()
