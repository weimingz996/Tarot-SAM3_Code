import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils import save_candidate_masks


class FakeLogger:
    def __init__(self):
        self.entries = []

    def log(self, phase, message):
        self.entries.append((phase, message))


class CandidateCliOutputTests(unittest.TestCase):
    def test_save_candidate_masks_uses_stable_unique_names(self):
        candidates = [
            {
                "id": "Text / Candidate",
                "source": "text",
                "mask": np.array([[True, False], [False, False]]),
            },
            {
                "id": "Text / Candidate",
                "source": "full_des",
                "mask": np.zeros((2, 2), dtype=bool),
            },
        ]
        logger = FakeLogger()

        with tempfile.TemporaryDirectory() as directory:
            paths = save_candidate_masks(candidates, directory, logger)

            self.assertEqual(
                paths,
                [
                    "candidate_masks/01_text_candidate.png",
                    "candidate_masks/02_text_candidate.png",
                ],
            )
            self.assertTrue(all((Path(directory) / path).is_file() for path in paths))

        messages = [message for _, message in logger.entries]
        self.assertTrue(any("source=text" in message and "pixels=1" in message for message in messages))
        self.assertTrue(any("source=full_des" in message and "pixels=0" in message for message in messages))

    def test_save_candidate_masks_accepts_empty_list(self):
        logger = FakeLogger()

        with tempfile.TemporaryDirectory() as directory:
            paths = save_candidate_masks([], directory, logger)

            self.assertEqual(paths, [])
            self.assertTrue((Path(directory) / "candidate_masks").is_dir())


if __name__ == "__main__":
    unittest.main()
