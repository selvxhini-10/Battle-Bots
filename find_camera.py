# find_camera.py - paste this into a new file in Battle-Bots root and run it
import cv2

print("Scanning for cameras...")
found = []
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            print(f"  Camera {i}: WORKS ({frame.shape[1]}x{frame.shape[0]})")
            found.append(i)
        else:
            print(f"  Camera {i}: opens but no frame")
        cap.release()
    else:
        print(f"  Camera {i}: not found")

if found:
    print(f"\nUse: python -m solemates --camera {found[-1]} --analyze-only")
else:
    print("\nNo cameras found.")