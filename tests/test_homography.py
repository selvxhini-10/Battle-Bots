import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from calibrate import compute_homography, save_to_config
from solemates.app import SolematesApp, main
from solemates.planning import PixelTableTransform
from solemates.synthetic import make_scene


ROOT = Path(__file__).parents[1]
BOUNDS = (-0.35, 0.35, -0.30, 0.30)


class HomographyTest(unittest.TestCase):
    def test_perspective_fit_maps_independent_point(self):
        expected = np.array([[0.001, 0.0002, -0.3], [0.0001, -0.001, 0.2],
                             [0.0004, 0.0002, 1.]])
        pixels = np.array([[100, 100], [800, 80], [850, 550], [70, 500]])
        projected = np.column_stack([pixels, np.ones(4)]) @ expected.T
        table = projected[:, :2] / projected[:, 2:]
        matrix, _ = compute_homography(pixels, table)
        transform = PixelTableTransform((960, 640), BOUNDS, matrix)
        target = expected @ [430, 310, 1]
        np.testing.assert_allclose(transform.point((430, 310)), target[:2] / target[2], atol=1e-7)

    def test_direction_matches_projected_line(self):
        matrix = [[0.001, 0.0002, -0.3], [0.0001, -0.001, 0.2], [0.0004, 0.0002, 1.]]
        transform = PixelTableTransform((960, 640), BOUNDS, matrix)
        pixel = np.array([430., 310.])
        angle = 0.7
        delta = np.array(transform.point(pixel + [math.cos(angle), math.sin(angle)])) - transform.point(pixel)
        self.assertAlmostEqual(transform.angle(pixel, angle), math.atan2(delta[1], delta[0]))

    def test_linear_fallback(self):
        transform = PixelTableTransform((960, 640), BOUNDS)
        np.testing.assert_allclose(transform.point((480, 320)), [0, 0])
        self.assertAlmostEqual(transform.angle((480, 320), math.pi / 2), -math.pi / 2)

    def test_invalid_calibrations(self):
        for matrix in (np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                PixelTableTransform((960, 640), BOUNDS, matrix)
        with self.assertRaisesRegex(ValueError, "Image size"):
            PixelTableTransform((960, 640), BOUNDS, np.eye(3), (640, 480))
        with self.assertRaises(ValueError):
            compute_homography([[0, 0], [1, 0], [2, 0], [0, 1]], [[0, 0], [1, 0], [1, 1], [0, 1]])
        transform = PixelTableTransform((960, 640), BOUNDS, [[1, 0, 0], [0, 1, 0], [1, 0, -100]])
        with self.assertRaises(ValueError):
            transform.point((100, 50))

    def test_saved_calibration_drives_execution(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "config.json"
            path.write_text((ROOT / "config.json").read_text())
            matrix = np.array([[0.0005, 0, -0.25], [0, -0.0005, 0.2], [0, 0, 1.]])
            save_to_config(matrix, path, (960, 640))
            config = json.loads(path.read_text())
            self.assertIn("matching", config)
            app = SolematesApp(config)
            socks, pairs, singles, _ = app.analyze(make_scene()[0])
            app.robot = Mock()
            app.execute(socks, pairs, singles)
            first_sock = next(sock for sock in socks if sock.id == pairs[0].first_id)
            expected = matrix @ [*first_sock.grasp_px, 1]
            pose = app.robot.move_end_effector.call_args_list[0].args[1]
            np.testing.assert_allclose([pose.x, pose.y], expected[:2])

    def test_synthetic_cli_ignores_real_camera_calibration(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "config.json"
            config = json.loads((ROOT / "config.json").read_text())
            config["camera"] = {"homography": np.eye(3).tolist(), "image_size": [12, 12]}
            path.write_text(json.dumps(config))
            output = Path(directory) / "output"
            main(["--synthetic", "--config", str(path), "--output", str(output)])
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["events"][-1]["state"], "done")
            self.assertEqual(report["sock_count"], 5)
