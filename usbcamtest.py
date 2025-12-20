import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 backend to avoid GStreamer issues

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

ret, frame = cap.read()
if ret:
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()
else:
    print("Failed to read frame")

cap.release()


