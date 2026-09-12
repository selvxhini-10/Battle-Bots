from pathlib import Path
import json
import unittest

from solemates.app import SolematesApp
from solemates.models import RunState
from solemates.synthetic import make_scene


class PipelineTest(unittest.TestCase):
    def test_pipeline_reaches_done_without_holding_socks(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "config.json").read_text())
        image, _ = make_scene(*config["workspace"]["image_size"])
        app = SolematesApp(config)
        socks, pairs, singles, _ = app.analyze(image)
        app.execute(socks, pairs, singles)
        self.assertEqual(app.events[-1].state, RunState.DONE)
        self.assertEqual(app.robot.held, {"left": None, "right": None})
        self.assertEqual(len(app.robot.joints["left"]), 7)
        self.assertEqual(len(app.robot.joints["right"]), 7)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(singles), 1)


if __name__ == "__main__":
    unittest.main()
