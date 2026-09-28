import unittest
from pathlib import Path

import numpy as np

from reason_prompt import reasonseg_best_flow as flow
from reason_prompt import reasonseg_best_flow_reference as core


ROOT = Path(__file__).resolve().parents[1]


class ReasonCandidateBoundaryTests(unittest.TestCase):
    def test_candidate_masks_info_preserves_empty_masks_and_excludes_bbox_evidence(self):
        nonempty = np.array([[True, False], [False, False]])
        empty = np.zeros((2, 2), dtype=bool)
        full_des = np.array([[False, True], [True, False]])
        text_top3 = [
            {
                "mask": nonempty,
                "confidence": 0.8,
                "pipeline": "identity",
                "prompt": {"role": "canonical_name", "text": "red cup"},
            },
            {
                "mask": empty,
                "confidence": 0.2,
                "pipeline": "relation",
                "prompt": {"role": "contrast_view", "text": "cup on table"},
            },
        ]
        full_result = {
            "full_des": {
                "text": "the red cup",
                "scope": "whole_object",
                "target_entity": "cup",
            },
            "bbox_candidates": [{"id": "must-not-leak"}],
        }

        records = core._candidate_masks_info(text_top3, full_des, full_result)

        self.assertEqual([item["id"] for item in records], ["Text Top1", "Text Top2", "FullDes"])
        self.assertEqual([item["source"] for item in records], ["text", "text", "full_des"])
        self.assertEqual(records[0]["pipeline"], "identity")
        self.assertEqual(records[0]["role"], "canonical_name")
        self.assertEqual(records[0]["text"], "red cup")
        self.assertTrue(all(item["mask"].dtype == np.bool_ for item in records))
        self.assertTrue(np.array_equal(records[0]["mask"], nonempty))
        self.assertTrue(np.array_equal(records[1]["mask"], empty))
        self.assertTrue(np.array_equal(records[2]["mask"], full_des))
        self.assertNotIn("must-not-leak", repr(records))

    def test_reason_source_contains_no_final_vote_implementation(self):
        self.assertFalse(
            (ROOT / "reason_prompt/reasonseg_shortlong_bbox_experiment.py").exists()
        )
        core_source = (ROOT / "reason_prompt/reasonseg_best_flow_reference.py").read_text()
        for forbidden in (
            "def unique_winner(",
            "def stable_vote(",
            "def build_vote_feature(",
            "def refine_vote(",
            "def training_free_vote(",
            "def select_best_mask(",
        ):
            self.assertNotIn(forbidden, core_source)
        flow_source = (ROOT / "reason_prompt/reasonseg_best_flow.py").read_text()
        self.assertNotIn("reasonseg_shortlong_bbox_experiment", flow_source)
        self.assertTrue(hasattr(flow, "run_reasonseg_candidate_flow"))
        self.assertFalse(hasattr(flow, "run_reasonseg_best_flow"))


if __name__ == "__main__":
    unittest.main()
