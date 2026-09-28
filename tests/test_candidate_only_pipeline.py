import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


def _load_tarot_module():
    sam3 = types.ModuleType("sam3")
    sam3.__path__ = []
    sam3_model = types.ModuleType("sam3.model")
    sam3_model.__path__ = []
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = lambda *args, **kwargs: None
    processor = types.ModuleType("sam3.model.sam3_image_processor")
    processor.Sam3Processor = object
    sys.modules.update({
        "sam3": sam3,
        "sam3.model": sam3_model,
        "sam3.model_builder": builder,
        "sam3.model.sam3_image_processor": processor,
    })
    import tarot_sam3

    return tarot_sam3


tarot = _load_tarot_module()


class FakeLogger:
    def __init__(self):
        self.entries = []

    def log(self, phase, message):
        self.entries.append((phase, message))


def _image(path: Path):
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(path)


def _model_config():
    return SimpleNamespace(
        tarot_sam3=SimpleNamespace(
            Reason=SimpleNamespace(
                reason_sam_confidence=0.6,
                refer_sam_confidence=0.6,
            ),
            ERI=SimpleNamespace(refer_filter_bbox_iou_thresh=0.3),
            MSR=SimpleNamespace(
                reason_dilate_kernel_size=3,
                reason_dilate_iterations=1,
                reason_refine_iter_num=1,
                refer_dilate_kernel_size=3,
                refer_dilate_iterations=1,
                refer_refine_iter_num=1,
            ),
        )
    )


def _bare_model():
    model = tarot.TarotSAM3.__new__(tarot.TarotSAM3)
    model.cfg = _model_config()
    model.qwen = SimpleNamespace(
        temperature=None,
        max_tokens=None,
        load_image=lambda path: None,
    )
    model.text_qwen = object()
    model.sam3 = SimpleNamespace(conf_thresh=None, load_image=lambda path: None)
    model.dino = SimpleNamespace(load_image=lambda path: None)
    return model


class CandidateOnlyPipelineTests(unittest.TestCase):
    def test_reason_process_returns_runner_candidates_without_selecting(self):
        masks = [
            {"id": "Text Top1", "source": "text", "mask": np.ones((3, 4), dtype=bool)},
            {"id": "FullDes", "source": "full_des", "mask": np.zeros((3, 4), dtype=bool)},
        ]
        model = _bare_model()
        model.reasonseg_candidate_runner = lambda **kwargs: {
            "candidate_masks_info": masks,
            "full_des": {"text": "red cup", "scope": "whole_object"},
        }
        model.reasonseg_runner = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy best-mask runner called")
        )
        model._consensus_winner = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("final vote called")
        )
        model.mask_self_refine = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("MSR called")
        )
        logger = FakeLogger()

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            _image(image_path)
            result = model.process_image(
                str(image_path), "the red cup", True, logger, directory, False
            )

        self.assertIs(result, masks)
        self.assertEqual(model.full_description, "red cup")
        self.assertEqual(model.target_scope, "whole_object")
        self.assertTrue(any("Text Top1" in message for _, message in logger.entries))

    def test_refer_interpreter_returns_prevote_masks_info(self):
        text_mask = np.array([[True, False], [False, False]])
        raw_mask = np.array([[False, True], [False, False]])
        bbox_mask = np.array([[True, True], [False, False]])
        text_records = [{
            "id": "full_text_raw_0",
            "source": "text",
            "mask": text_mask,
            "raw_masks_info": [{"mask": raw_mask, "conf": 0.5}],
        }]
        bbox_candidate = {
            "method": "full",
            "source": "bbox",
            "mask": bbox_mask,
            "conf": 0.9,
        }
        model = _bare_model()
        model.reason_seg = False
        model.visualize = False
        model.image = np.zeros((2, 2, 3), dtype=np.uint8)
        model.logger = FakeLogger()
        model._run_full_sl_v2_live = lambda: {
            "full": {"bbox": [0, 0, 2, 1]},
            "short": {"description": "cup", "bbox": [0, 0, 2, 1]},
            "long": {"description": "red cup", "bbox": [0, 0, 2, 1]},
        }
        model._generate_bbox_candidates = lambda bboxes: [bbox_candidate]
        model._initial_text_candidates = lambda bboxes, archive_context=None: list(text_records)
        model._select_bbox_candidate = lambda candidates, masks: (bbox_mask, "full")
        model._consensus_winner = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("final vote called")
        )
        model.target_scope = "whole_object"
        model.full_des_bbox = None

        with patch.object(tarot, "filter_masks_by_bboxes", side_effect=lambda masks, *a, **k: masks):
            records = model.expression_reasoning_interpreter()

        self.assertEqual([item["id"] for item in records], ["full_text_raw_0", "Best BBox"])
        self.assertIs(records[0]["raw_masks_info"][0]["mask"], raw_mask)
        self.assertTrue(np.array_equal(records[1]["mask"], bbox_mask))

    def test_refer_text_candidates_keep_parent_metadata_and_empty_mask(self):
        empty = np.zeros((2, 2), dtype=bool)
        nonempty = np.array([[True, False], [False, False]])
        raw = {"mask": nonempty, "conf": 0.7}
        mask_records = iter([
            {"id": "full_text", "source": "text", "mask": empty,
             "raw_masks_info": [], "conf": 0.0},
            {"id": "more_text_0", "source": "text", "mask": empty,
             "raw_masks_info": [], "conf": 0.0},
            {"id": "more_text_0", "source": "text", "mask": nonempty,
             "raw_masks_info": [raw], "conf": 0.7},
        ])
        generated_prompts = []
        generated_text = iter(["", "the crimson cup"])
        model = _bare_model()
        model.reason_seg = False
        model.visualize = False
        model.image = np.zeros((2, 2, 3), dtype=np.uint8)
        model.logger = FakeLogger()
        model.query = "the red cup"
        model.full_description = "red cup"
        model.target_entity = "cup"
        model.target_scope = "whole_object"
        model.description_mask_extractor = (
            lambda description, candidate_id, return_info=False: next(mask_records)
        )

        def generate(prompt):
            generated_prompts.append(prompt)
            return next(generated_text)

        model.qwen.generate = generate
        with (
            patch.object(tarot, "generate_target_name_candidates", return_value=["cup"]),
            patch.object(tarot, "split_phases", return_value=[]),
            patch.object(
                tarot,
                "filter_masks_by_bboxes",
                side_effect=lambda records, *args, **kwargs: [
                    record for record in records if np.any(record["mask"])
                ],
            ),
        ):
            records = model._initial_text_candidates([[0, 0, 2, 2]])

        self.assertEqual([record["id"] for record in records], ["full_text", "more_text_0"])
        self.assertFalse(np.any(records[0]["mask"]))
        self.assertIs(records[1]["raw_masks_info"][0], raw)
        self.assertIn("{'red cup'}", generated_prompts[-1])

    def test_refer_generation_errors_propagate(self):
        model = _bare_model()
        model.logger = FakeLogger()
        model.query = "the cup"
        model._run_full_v4_live = lambda: (_ for _ in ()).throw(
            RuntimeError("full-v4 failed")
        )
        with self.assertRaisesRegex(RuntimeError, "full-v4 failed"):
            model.ref_reasoning_prompt()

        model._full_v4_row = {
            "descriptions": {"v4_full": "red cup"},
            "bbox": {"v4_full": [0, 0, 2, 2]},
        }
        model.image = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch.object(tarot, "evaluate", side_effect=RuntimeError("full-sl failed")):
            with self.assertRaisesRegex(RuntimeError, "full-sl failed"):
                model._run_full_sl_v2_live()

    def test_refer_process_returns_empty_candidate_list(self):
        model = _bare_model()
        model.ref_reasoning_prompt = lambda: None
        model.expression_reasoning_interpreter = lambda: []
        logger = FakeLogger()

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            _image(image_path)
            result = model.process_image(
                str(image_path), "the cup", False, logger, directory, False
            )

        self.assertEqual(result, [])
        self.assertTrue(any("candidate" in message.lower() for _, message in logger.entries))

    def test_candidate_generation_error_propagates(self):
        model = _bare_model()

        def fail(**kwargs):
            raise RuntimeError("candidate generation failed")

        model.reasonseg_candidate_runner = fail
        model.reasonseg_runner = fail
        logger = FakeLogger()

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            _image(image_path)
            with self.assertRaisesRegex(RuntimeError, "candidate generation failed"):
                model.process_image(
                    str(image_path), "the cup", True, logger, directory, False
                )


if __name__ == "__main__":
    unittest.main()
