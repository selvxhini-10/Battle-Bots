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
  2. Run the script and click the 4 markers in the same order as camera.table_points_m in your config.
  3. The homography is saved to config.json automatically.

"""

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
        print(f"  Clicked point {idx}: pixel ({x}, {y})  →  table {param[idx-1]} m")
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
    for points in (src, dst):
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Calibration requires four finite pixel and table point pairs")
        for omitted in range(4):
            triangle = np.delete(points, omitted, axis=0)
            if np.linalg.matrix_rank(triangle[1:] - triangle[0]) < 2:
                raise ValueError("Invalid calibration: markers must be distinct with no three collinear")
    H, mask = cv2.findHomography(src, dst)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
        raise ValueError("Invalid calibration: click four distinct, non-collinear markers")
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


def save_to_config(H, config_path: Path, image_size=None):
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    config.setdefault("camera", {})["homography"] = H.tolist()
    if image_size is not None:
        config["camera"]["image_size"] = list(image_size)
    else:
        config["camera"].pop("image_size", None)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\nHomography saved to {config_path}")


def main():
    global display_image, clicked_points

    parser = argparse.ArgumentParser(description="Solemates camera calibration")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", default="0", help="webcam index (default: 0)")
    src.add_argument("--url", help="IP camera stream URL, e.g. http://192.168.1.5:8080/video")
    src.add_argument("--image", type=Path, help="static image file")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--output-image", type=Path, default=Path("demo_output/calibration.png"))
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text())
        real_points = config["camera"]["table_points_m"]
        # Reuse the calibration geometry checks before opening a camera or GUI.
        compute_homography([[0, 0], [1, 0], [1, 1], [0, 1]], real_points)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"Set camera.table_points_m to four finite, distinct marker XY pairs "
                     f"in meters with no three collinear in {args.config}: {exc}")

    clicked_points = []
    print("Grabbing frame...")
    frame = grab_frame(args)
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    display_image = frame.copy()
    cv2.namedWindow("Solemates Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Solemates Calibration", min(w, 1200), min(h, 800))
    cv2.setMouseCallback("Solemates Calibration", mouse_callback, real_points)

    print("\n── Solemates Camera Calibration ──")
    print("Click these 4 table markers IN ORDER:")
    for i, pt in enumerate(real_points, 1):
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
            try:
                H, inliers = compute_homography(clicked_points, real_points)
            except ValueError as exc:
                print(f"{exc}. Please click the four markers again.")
                clicked_points = []
                display_image = overlay.copy()
                continue
            print(f"Homography computed ({inliers}/4 inliers):\n{H}\n")
            mean_err = validate_homography(H, clicked_points, real_points)
            print(f"\nMean fitting error: {mean_err:.1f} mm")
            print("These four points were used to fit the matrix; this is not an independent accuracy check.")
            print("Check additional measured table points before using the calibration for robot motion.")

            save_to_config(H, args.config, (w, h))

            # Save annotated image
            args.output_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output_image), display_image)
            print(f"Annotated frame saved to {args.output_image}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()