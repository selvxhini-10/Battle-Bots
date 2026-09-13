import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from solemates.app import main
from calibrate import compute_homography, save_to_config
from solemates.synthetic import make_scene

ROOT = Path(__file__).parents[1]


class CalibratedSimulationTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "config.json").read_text())

    def test_perspective_image_calibration_to_simulation(self):
        """Fit and save calibration, then exercise the image CLI with real simulated IK."""
        cfg = self.config()
        original, _ = make_scene()
        src = np.float32([[0, 0], [960, 0], [960, 640], [0, 640]])
        dst = np.float32([[80, 70], [1020, 20], [1060, 700], [30, 730]])
        warp = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(original, warp, (1100, 760), borderValue=(242, 242, 242))
        expected = np.array([[.7/960, 0, -.35], [0, -.6/640, .3], [0, 0, 1]]) @ np.linalg.inv(warp)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(cfg))
            matrix, _ = compute_homography(
                dst, [[-.35, .3], [.35, .3], [.35, -.3], [-.35, -.3]])
            with contextlib.redirect_stdout(io.StringIO()):
                save_to_config(matrix, config_path, (1100, 760))
            path = folder / "table.png"
            cv2.imwrite(str(path), image)
            output = folder / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--image", str(path), "--config", str(config_path), "--output", str(output)])
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["sock_count"], 5)
            self.assertEqual(len(report["pairs"]), 2)
            observations = next(e["data"]["socks"] for e in report["events"] if "socks" in e["data"])
            by_id = {s["id"]: s for s in observations}
            plans = [e["data"]["plan"] for e in report["events"] if "plan" in e["data"]]
            self.assertEqual(len(plans), 5)  # Both paired socks and the single.
            self.assertEqual({p["sock_id"] for p in plans}, set(by_id))
            for event_index, event in enumerate(report["events"]):
                if "plan" not in event["data"]:
                    continue
                plan = event["data"]["plan"]
                sock = by_id[plan["sock_id"]]
                target = expected @ [*sock["grasp_px"], 1]
                xy = target[:2] / target[2]
                linear = [-.35 + sock["grasp_px"][0] * .7 / 1100,
                          .3 - sock["grasp_px"][1] * .6 / 760]
                self.assertGreater(np.linalg.norm(xy - linear), .001)
                for phase in ("pregrasp", "grasp", "lift"):
                    pose = plan[phase]
                    np.testing.assert_allclose([pose["x"], pose["y"]], xy, atol=1e-7)
                # These are the actual simulated moves, after the real IK solver.
                for offset, phase in ((1, "pregrasp"), (2, "grasp"), (4, "lift")):
                    move = report["events"][event_index + offset]["data"]
                    self.assertEqual(move["pose"], plan[phase])
                    self.assertIn("joints", move)
                    self.assertLessEqual(move["ik_error_m"], .015)
            moves = [e["data"] for e in report["events"] if "ik_error_m" in e["data"]]
            self.assertEqual(len(moves), 33)
            self.assertTrue(all(m["ik_error_m"] <= .015 for m in moves))


if __name__ == "__main__":
    unittest.main()
