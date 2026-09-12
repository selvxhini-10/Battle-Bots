"""
calibrate.py — Step 1: Camera calibration for Solemates

Compute a pixel→table homography by clicking 4 known points.
Supports:
  - Laptop/desktop webcam:  python calibrate.py --camera 0
  - Phone as IP camera:     python calibrate.py --url http://192.168.x.x:8080/video
  - Static photo:           python calibrate.py --image path/to/photo.jpg

Phone IP camera setup (free apps):
  iOS:     "DroidCam" or "Camo" or "EpocCam"
  Android: "DroidCam" or "IP Webcam"
  Open the app, note the URL shown (e.g. http://192.168.1.5:8080/video)
  Make sure your phone and laptop are on the same WiFi network.

Usage:
  1. Print or tape 4 markers on your table at known positions (measure in meters
     from robot base, e.g. corners of a 30x40cm rectangle).
  2. Run the script and click the 4 markers IN THE SAME ORDER as REAL_POINTS.
  3. The homography is saved to config.json automatically.

EDIT THIS before running:
"""

# ── Configure these 4 real-world table positions (meters from robot base) ──────
# Order: top-left, top-right, bottom-right, bottom-left  (or any consistent order)
# Measure from the robot's arm_base frame origin on the table surface.
REAL_POINTS = [
    [-0.20,  0.20],   # marker 1: far-left
    [ 0.20,  0.20],   # marker 2: far-right
    [ 0.20, -0.20],   # marker 3: near-right
    [-0.20, -0.20],   # marker 4: near-left
]
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

clicked_points = []
display_image = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_image
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        idx = len(clicked_points)
        print(f"  Clicked point {idx}: pixel ({x}, {y})  →  table {REAL_POINTS[idx-1]} m")
        # Draw dot and label
        cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(display_image, str(idx), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Solemates Calibration", display_image)


def grab_frame(args) -> np.ndarray:
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            sys.exit(f"Could not read image: {args.image}")
        return frame
    url = args.url or args.camera
    cap = cv2.VideoCapture(url if args.url else int(url))
    if not cap.isOpened():
        sys.exit(f"Could not open camera: {url}")
    # Drain buffer so we get a fresh frame
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Failed to capture frame")
    return frame


def compute_homography(pixel_pts, real_pts):
    src = np.float32(pixel_pts)
    dst = np.float32(real_pts)
    H, mask = cv2.findHomography(src, dst)
    inliers = int(mask.sum()) if mask is not None else 4
    return H, inliers


def validate_homography(H, pixel_pts, real_pts):
    """Reproject and report error in mm."""
    src = np.float32(pixel_pts).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - np.float32(real_pts), axis=1) * 1000  # mm
    for i, (err, rp) in enumerate(zip(errors, real_pts)):
        print(f"  Marker {i+1}: target {rp} m  →  reprojected error {err:.1f} mm")
    return float(errors.mean())


def save_to_config(H, config_path: Path):
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    config.setdefault("camera", {})["homography"] = H.tolist()
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\nHomography saved to {config_path}")


def main():
    global display_image, clicked_points

    parser = argparse.ArgumentParser(description="Solemates camera calibration")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", default="0", help="webcam index (default: 0)")
    src.add_argument("--url", help="IP camera stream URL, e.g. http://192.168.1.5:8080/video")
    src.add_argument("--image", type=Path, help="static image file")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-image", type=Path, default=Path("demo_output/calibration.png"))
    args = parser.parse_args()

    print("Grabbing frame...")
    frame = grab_frame(args)
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    display_image = frame.copy()
    cv2.namedWindow("Solemates Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Solemates Calibration", min(w, 1200), min(h, 800))
    cv2.setMouseCallback("Solemates Calibration", mouse_callback)

    print("\n── Solemates Camera Calibration ──")
    print("Click these 4 table markers IN ORDER:")
    for i, pt in enumerate(REAL_POINTS, 1):
        print(f"  {i}. table position {pt} m")
    print("\nPress Q to quit without saving.")

    # Draw instruction overlay
    overlay = display_image.copy()
    cv2.putText(overlay, "Click 4 markers in order (see terminal)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    display_image = overlay

    while True:
        cv2.imshow("Solemates Calibration", display_image)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            print("Aborted.")
            break
        if len(clicked_points) == 4:
            print("\nAll 4 points collected. Computing homography...")
            H, inliers = compute_homography(clicked_points, REAL_POINTS)
            print(f"Homography computed ({inliers}/4 inliers):\n{H}\n")
            mean_err = validate_homography(H, clicked_points, REAL_POINTS)
            print(f"\nMean reprojection error: {mean_err:.1f} mm", end="  ")
            if mean_err < 5:
                print("✓ excellent")
            elif mean_err < 15:
                print("✓ acceptable")
            else:
                print("⚠ high — check marker positions or re-click")

            save_to_config(H, args.config)

            # Save annotated image
            args.output_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output_image), display_image)
            print(f"Annotated frame saved to {args.output_image}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()