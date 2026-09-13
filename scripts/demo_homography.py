#!/usr/bin/env python3
"""Create a known perspective camera view and run it through calibrated simulation.

The perspective quadrilateral is a demonstration fixture, not real-camera calibration.
The source config and its camera calibration are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibrate import compute_homography, save_to_config
from solemates.app import main as run
from solemates.synthetic import make_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output" / "homography")
    parser.add_argument("--rrd", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path == args.config.resolve():
        parser.error("Output config must differ from the source config")
    config = json.loads(args.config.read_text())
    config["execution"]["backend"] = "simulated"
    checkpoint = Path(config["perception"]["sam"]["checkpoint"])
    if not checkpoint.is_absolute():
        config["perception"]["sam"]["checkpoint"] = str(args.config.resolve().parent / checkpoint)
    width, height = config["workspace"]["image_size"]
    image, _ = make_scene(width, height)
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # Known projective camera fixture. Coordinates scale with the configured image.
    pixels = np.float32([[.07, .09], [.93, .03], [.96, .92], [.03, .96]]) * [width, height]
    warp = cv2.getPerspectiveTransform(source, pixels.astype(np.float32))
    image = cv2.warpPerspective(image, warp, (width, height), borderValue=(242, 242, 242))
    xmin, xmax, ymin, ymax = config["workspace"]["table_bounds_m"]
    table = [[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
    matrix, _ = compute_homography(pixels, table)
    config_path.write_text(json.dumps(config, indent=2))
    save_to_config(matrix, config_path, (width, height))
    (output / "correspondences.json").write_text(json.dumps({
        "pixel_points": pixels.tolist(), "table_points_m": table, "image_size": [width, height],
    }, indent=2))
    camera_path = output / "camera.png"
    cv2.imwrite(str(camera_path), image)
    argv = ["--image", str(camera_path), "--config", str(config_path),
            "--output", str(output)]
    if args.rrd:
        argv.append("--rrd")
    run(argv)


if __name__ == "__main__":
    main()
