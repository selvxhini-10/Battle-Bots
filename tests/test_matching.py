from pathlib import Path
import json
import unittest

from solemates.matching import find_pairs
from solemates.perception import segment_socks
from solemates.synthetic import make_scene


def config():
    return json.loads((Path(__file__).parents[1] / "config.json").read_text())


class MatchingTest(unittest.TestCase):
    def test_synthetic_scene_finds_expected_pairs_and_single(self):
        cfg = config()
        image, labels = make_scene(*cfg["workspace"]["image_size"])
        socks = segment_socks(image, cfg["perception"])
        pairs, singles, _ = find_pairs(
            socks, cfg["matching"]["weights"], cfg["matching"]["match_threshold"]
        )
        paired_labels = {frozenset((labels[p.first_id], labels[p.second_id])) for p in pairs}
        self.assertEqual(len(socks), 5)
        self.assertEqual(
            paired_labels,
            {frozenset(("coral-stripes",)), frozenset(("blue-dots",))},
        )
        self.assertEqual([labels[sock_id] for sock_id in singles], ["green-single"])


if __name__ == "__main__":
    unittest.main()
