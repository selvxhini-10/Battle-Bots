import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2

from solemates.app import main
from solemates.kinematics import KinematicModel


ROOT = Path(__file__).parents[1]


class AppArtifactsTest(unittest.TestCase):
    def read_artifacts(self, output):
        for name in ("scene.png", "matches.png"):
            image = cv2.imread(str(output / name))
            self.assertIsNotNone(image, name)
            self.assertEqual(image.shape, (640, 960, 3))
        report = json.loads((output / "run.json").read_text())
        self.assertEqual(report["sock_count"], 5)
        self.assertEqual(len(report["pairs"]), 2)
        self.assertEqual(len(report["singles"]), 1)
        self.assertEqual(len(report["all_scores"]), 10)
        return report

    def run_cli(self, output, *args):
        return subprocess.run(
            [sys.executable, "-m", "solemates", "--synthetic",
             "--detector", "threshold", "--matcher", "classical",
             "--output", str(output), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )

    def test_successful_cli_saves_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            result = self.run_cli(output)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = self.read_artifacts(output)
            self.assertEqual(report["status"], "completed")
            self.assertIsNone(report["error"])
            self.assertEqual(report["events"][-1]["state"], "done")
            moves = [event for event in report["events"] if "ik_error_m" in event["data"]]
            self.assertEqual(len(moves), 33)
            self.assertTrue(all(event["data"]["ik_error_m"] <= 0.015 for event in moves))

    def test_unreachable_cli_saves_artifacts_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            config = json.loads((ROOT / "config.json").read_text())
            config["workspace"]["safe_height_m"] = 10.0
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            result = self.run_cli(output, "--config", str(config_path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("IK could not reach", result.stderr)
            self.assertIn(str(output), result.stderr)
            report = self.read_artifacts(output)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["type"], "RuntimeError")
            self.assertIn("IK could not reach", report["error"]["message"])
            self.assertEqual(report["events"][-1]["message"], "Execution failed")
            self.assertEqual(report["events"][-1]["data"]["error"], report["error"])
            self.assertNotIn("done", [event["state"] for event in report["events"]])

    def test_failure_preserves_prior_moves_and_skips_optional_export(self):
        original = KinematicModel.solve_position
        calls = 0

        def fail_second_move(model, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Injected IK failure")
            return original(model, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(KinematicModel, "solve_position", fail_second_move), \
                    patch("solemates.app.save_rerun") as export:
                with self.assertRaisesRegex(SystemExit, "Injected IK failure"):
                    main(["--synthetic", "--output", str(output), "--rrd"])
                export.assert_not_called()
            report = self.read_artifacts(output)
            self.assertEqual(calls, 2)
            self.assertEqual(sum("joints" in event["data"] for event in report["events"]), 1)
            steps = [event["step"] for event in report["events"]]
            self.assertEqual(steps, list(range(1, len(steps) + 1)))

    def test_analyze_only_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("solemates.app.SolematesApp.execute") as execute, \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["--synthetic", "--analyze-only", "--output", str(output)])
                execute.assert_not_called()
            report = self.read_artifacts(output)
            self.assertEqual(report["status"], "analyzed")
            self.assertIsNone(report["error"])


if __name__ == "__main__":
    unittest.main()
