"""
phone_camera.py — Use DroidCam (iOS/Android) as an overhead camera for Solemates.

Your DroidCam details:
  WiFi IP:   100.66.208.88
  Port:      4747
  Video URL: http://100.66.208.88:4747/video

Quick test (run this file directly):
  python phone_camera.py

Then in any Solemates command, use --url instead of --camera:
  python calibrate.py --url http://100.66.208.88:4747/video
  python -m solemates --url http://100.66.208.88:4747/video --analyze-only

Tips:
  - DroidCam must be open and "Start" pressed on the phone
  - Phone and laptop must be on the same WiFi network
  - For overhead use: prop phone above the table (tape to a box, lamp, etc.)
  - Landscape orientation gives a wider field of view
  - In DroidCam app: Settings → Video Quality → 720p or 1080p for better detection
"""

import cv2
import sys

DROIDCAM_URL = "http://100.66.208.88:4747/mjpegfeed"


def open_droidcam(url: str = DROIDCAM_URL, retries: int = 3) -> cv2.VideoCapture:
    """Open DroidCam stream with retries."""
    for attempt in range(1, retries + 1):
        print(f"Connecting to DroidCam ({url}) attempt {attempt}/{retries}...")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            # Drain the buffer to get a fresh frame
            for _ in range(5):
                cap.read()
            print("Connected!")
            return cap
        cap.release()
    raise ConnectionError(
        f"Could not connect to DroidCam at {url}\n"
        "Check:\n"
        "  1. DroidCam app is open and 'Start' is pressed\n"
        "  2. Phone and laptop are on the same WiFi\n"
        f"  3. IP and port are correct ({url})"
    )


def capture_frame(url: str = DROIDCAM_URL) -> "np.ndarray":
    """Grab a single frame from DroidCam."""
    import numpy as np
    cap = open_droidcam(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame from DroidCam")
    return frame


def live_preview(url: str = DROIDCAM_URL):
    """Show a live preview window. Press Q to quit, S to save a frame."""
    cap = open_droidcam(url)
    print("Live preview open. Press Q to quit, S to save frame.")
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed — reconnecting...")
            cap.release()
            cap = open_droidcam(url)
            continue
        # Overlay connection info
        cv2.putText(frame, f"DroidCam: {url}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Q=quit  S=save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("DroidCam Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = f"droidcam_frame_{saved}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            saved += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--capture" in sys.argv:
        frame = capture_frame()
        cv2.imwrite("droidcam_test.png", frame)
        print(f"Saved droidcam_test.png ({frame.shape[1]}x{frame.shape[0]})")
    else:
        live_preview()