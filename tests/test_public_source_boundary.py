import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "reasonseg_shortlong_bbox_experiment",
    "def unique_winner(",
    "def stable_vote(",
    "def build_vote_feature(",
    "def select_best_mask(",
    "def training_free_vote(",
    "def _consensus_winner(",
    "def ERI_point_extractor(",
    "def mask_self_refine(",
    "def predict_point(",
    "def extract_best_mask(",
    "def save_mask_comparison_board(",
    "def save_semantic_mask_views(",
    "def save_absolute_mask_views(",
    "class DINOv3Engine",
    "final_output_mask.png",
)
REMOVED_ERI_KEYS = {
    "reason_sample_point_num",
    "reason_sample_point_strategy",
    "reason_spatial_iou",
    "reason_bbox_rank_weights",
    "reason_structural_certificate_bbox_cap",
    "reason_a34_near_duplicate_iou",
    "refer_sample_point_num",
    "refer_sample_point_strategy",
}


class PublicSourceBoundaryTests(unittest.TestCase):
    def test_removed_vote_and_msr_symbols_are_absent(self):
        production_files = [
            path
            for path in ROOT.rglob("*.py")
            if "tests" not in path.parts
            and "docs" not in path.parts
            and ".git" not in path.parts
        ]
        production_files.extend((ROOT / "configs").glob("*.yaml"))

        violations = []
        for path in production_files:
            text = path.read_text(encoding="utf-8")
            for symbol in FORBIDDEN:
                if symbol in text:
                    violations.append(f"{path.relative_to(ROOT)}: {symbol}")

        self.assertEqual(violations, [])

    def test_config_contains_no_deleted_pipeline_sections(self):
        config = yaml.safe_load((ROOT / "configs" / "7B.yaml").read_text())

        self.assertNotIn("dinov3", config)
        self.assertNotIn("MSR", config["tarot_sam3"])
        eri = config["tarot_sam3"]["ERI"]
        self.assertTrue(REMOVED_ERI_KEYS.isdisjoint(eri))
        self.assertEqual(eri["refer_reference_exact_copy_block_iou"], 0.9)


if __name__ == "__main__":
    unittest.main()
