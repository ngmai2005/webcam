import cv2
import numpy as np

# ================= CONFIG =================
WIDTH, HEIGHT = 1280, 720
CAM_INDEX = 0
FLIP_MODE = 1

points = []

# ================= MOUSE CLICK =================
def click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Point {len(points)}: {x}, {y}")

# ================= CAMERA =================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("CALIBRATION")
cv2.setMouseCallback("CALIBRATION", click)

print("👉 Click 4 điểm theo thứ tự:")
print("TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔴 FLIP TRƯỚC KHI CLICK
    frame = cv2.flip(frame, FLIP_MODE)

    for p in points:
        cv2.circle(frame, tuple(p), 6, (0, 255, 0), -1)

    cv2.imshow("CALIBRATION", frame)

    if len(points) == 4:
        np.save("calibration_points.npy", np.array(points, dtype=np.float32))
        print("✅ Đã lưu calibration_points.npy")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
