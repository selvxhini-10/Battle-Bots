from pathlib import Path
import json
import unittest

import numpy as np

from solemates.matching import find_pairs
from solemates.perception import segment_socks
from solemates.smart import _select_sam_masks
from solemates.synthetic import make_scene


class SmartFallbackTest(unittest.TestCase):
    def test_sam_postprocessing_filters_background_and_duplicates(self):
        shape = (100, 120)
        sock = np.zeros(shape, dtype=bool)
        sock[20:60, 25:65] = True
        duplicate = np.zeros(shape, dtype=bool)
        duplicate[21:61, 25:65] = True
        background = np.ones(shape, dtype=bool)
        proposals = [
            {"segmentation": sock, "area": int(sock.sum()), "predicted_iou": 0.95, "stability_score": 0.98},
            {"segmentation": duplicate, "area": int(duplicate.sum()), "predicted_iou": 0.90, "stability_score": 0.94},
            {"segmentation": background, "area": int(background.sum()), "predicted_iou": 0.99, "stability_score": 0.99},
        ]
        selected = _select_sam_masks(
            proposals,
            shape,
            {"minimum_area_px": 100, "maximum_area_fraction": 0.5, "mask_iou_threshold": 0.7},
        )
        self.assertEqual(len(selected), 1)

    def test_clip_embedding_contributes_to_pair_score(self):
        root = Path(__file__).parents[1]
        cfg = json.loads((root / "config.json").read_text())
        image, _ = make_scene(*cfg["workspace"]["image_size"])
        socks = segment_socks(image, cfg["perception"])
        for sock in socks:
            sock.embedding = np.array([1.0, 0.0]) if sock.id in (1, 3) else np.array([0.0, 1.0])
        pairs, _, scores = find_pairs(socks, cfg["matching"]["clip_weights"], 0.0)
        self.assertIn("clip", scores[0].components)
        self.assertEqual(len(pairs), 2)


if __name__ == "__main__":
    unittest.main()
