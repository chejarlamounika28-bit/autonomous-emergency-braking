import cv2
import numpy as np

# ---------- Region of Interest ----------
def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# ---------- Hough Line Transform ----------
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    return lines

# ---------- Draw Lines ----------
def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    if lines is None:
        return

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# ---------- Camera ----------
cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny
    edges = cv2.Canny(blur, 50, 150)

    # 4. ROI Triangle
    h, w = frame.shape[:2]
    roi_vertices = np.array([[
        (0, h),
        (w//2, int(h*0.6)),
        (w, h)
    ]], dtype=np.int32)

    roi = roi_mask(edges, roi_vertices)

    # 5. Hough Lines
    lines = hough_lines(roi, 2, np.pi/180, 50, 100, 50)

    # 6. Draw
    line_img = np.zeros_like(frame)
    draw_lines(line_img, lines)

    # 7. Overlay
    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

    cv2.imshow("Lane Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # smaller delay for live camera
        break

cap.release()
cv2.destroyAllWindows()
