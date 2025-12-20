import cv2
import time

def print_camera_fps(camera_index=0, width=640, height=480, fps=90):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows

    # Set camera properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -2)  # Set exposure to -5

    print(f"Requested: {width}x{height} @ {fps} FPS")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Show the camera stream
        cv2.imshow('Camera Stream', frame)

        # Print FPS every 60 frames
        if frame_count % 60 == 0:
            curr_time = time.time()
            elapsed = curr_time - prev_time
            actual_fps = frame_count / elapsed
            print(f"Actual FPS: {actual_fps:.2f}")
            prev_time = curr_time
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print_camera_fps()