# preview.py - drop in Battle-Bots root
import cv2
cap = cv2.VideoCapture(0)
print("Camera open - press SPACE to capture and analyze, Q to quit")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Solemates Camera - press SPACE to capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        cv2.imwrite("captured.jpg", frame)
        print("Saved captured.jpg")
        break
cap.release()
cv2.destroyAllWindows()