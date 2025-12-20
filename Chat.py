import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

def adjust_brightness_contrast(image, brightness=40, contrast=100):
    beta = brightness
    alpha = 1 + (contrast / 100.0)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter * perimeter))

def shape_confidence(contour):
    circ = circularity(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contour)
    solidity = area / hull_area if hull_area != 0 else 0

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertex_count = len(approx)
    if vertex_count < 8:
        return 0

    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    circle_fill = area / circle_area if circle_area != 0 else 0

    score = (0.5 * circ) + (0.3 * solidity) + (0.2 * circle_fill)

    if circle_fill < 0.7:
        score *= circle_fill

    return score

def slot_confidence(contour):
    area = cv2.contourArea(contour)
    if area == 0:
        return 0
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    if width < 1 or height < 1:
        return 0
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio < 2 or aspect_ratio > 5:
        return 0
    rect_area = width * height
    solidity = area / rect_area if rect_area != 0 else 0
    if solidity < 0.7:
        return 0
    perimeter = cv2.arcLength(contour, True)
    rect_perimeter = 2 * (width + height)
    perimeter_ratio = perimeter / rect_perimeter
    if perimeter_ratio < 0.8 or perimeter_ratio > 1.3:
        return 0

    # Score combines solidity and closeness of aspect ratio to ~3.5 (typical slot)
    score = solidity * (1 - abs(aspect_ratio - 3.5) / 2.5)
    return score

lower_orange = np.array([6, 180, 180])
upper_orange = np.array([24, 255, 255])

MIN_CONTOUR_AREA = 150
MIN_RADIUS = 7
MAX_RADIUS = 200

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is None:
            if cv2.waitKey(1) == 27:
                break
            continue

        frame = in_rgb.getCvFrame()
        enhanced = adjust_brightness_contrast(frame, brightness=40, contrast=100)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = 0
        best_shape = None

        output = enhanced.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius < MIN_RADIUS or radius > MAX_RADIUS:
                continue

            circle_score = shape_confidence(cnt)
            slot_score = slot_confidence(cnt)

            # Draw all circle candidates in green
            if circle_score > 0:
                center = (int(x), int(y))
                cv2.circle(output, center, int(radius), (0, 255, 0), 2)

            # Draw all slot candidates in cyan
            if slot_score > 0:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(output, [box], 0, (255, 255, 0), 2)

            max_score = max(circle_score, slot_score)
            if max_score > best_score:
                best_score = max_score
                best_contour = cnt
                best_shape = 'Circle' if circle_score > slot_score else 'Slot'

        if best_contour is not None:
            if best_shape == 'Circle':
                ((x, y), radius) = cv2.minEnclosingCircle(best_contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output, center, radius, (0, 255, 255), 3)
                label_pos = (center[0] - 60, center[1] - radius - 10)
            else:
                rect = cv2.minAreaRect(best_contour)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(output, [box], 0, (0, 255, 255), 3)
                center = (int(rect[0][0]), int(rect[0][1]))
                label_pos = (center[0] - 60, center[1] - 20)

            cv2.putText(output, f"Pickleball {best_shape}: {best_score:.2f}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Circle + Slot Pickleball Detection", output)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
