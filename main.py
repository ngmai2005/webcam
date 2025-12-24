import cv2
import numpy as np
import time
import os
import math
import threading
import csv
from datetime import datetime
from playsound import playsound

# ================== CONFIG ==================
WIDTH, HEIGHT = 1280, 720
IMAGE_SIZE = 340          # phóng to bia
CAM_INDEX = 0             # cam rời thường là 1
SHOT_COOLDOWN = 0.12
TARGET_TIME = 7           # 7s mỗi bia
MAX_SHOTS = 16

FLIP_MODE = 1
TARGET_CENTER = (WIDTH // 2, HEIGHT // 2)

MODE = "THI"              # "THI" hoặc "HUAN_LUYEN"

# ================== LOAD TARGET IMAGES ==================
TARGET_IMAGES = []
for i in range(1, 5):
    img = cv2.imread(f"images/target{i}.png")
    if img is None:
        print(f"❌ Thiếu images/target{i}.png")
        exit()
    TARGET_IMAGES.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))

# ================== SOUND ==================
def play_hit_sound():
    threading.Thread(
        target=playsound,
        args=("sounds/hit.mp3",),
        daemon=True
    ).start()

# ================== CALIBRATION ==================
def load_matrix():
    if not os.path.exists("calibration_points.npy"):
        print("❌ Chưa calibrate")
        exit()

    src = np.load("calibration_points.npy", allow_pickle=True)
    src = np.array(src, dtype=np.float32)

    dst = np.float32([
        [0, 0],
        [WIDTH, 0],
        [WIDTH, HEIGHT],
        [0, HEIGHT]
    ])

    return cv2.getPerspectiveTransform(src, dst)

# ================== LASER DETECTION ==================
def detect_laser(frame, M):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 200])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 120, 200])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, 1)
    mask = cv2.dilate(mask, kernel, 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 8:
        return None

    m = cv2.moments(c)
    if m["m00"] == 0:
        return None

    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])

    pt = np.float32([[[x, y]]])
    mapped = cv2.perspectiveTransform(pt, M)

    return int(mapped[0][0][0]), int(mapped[0][0][1])

# ================== ZONE ==================
def get_zone(dist):
    if dist < 30:
        return "TAM"
    elif dist < 70:
        return "GIUA"
    return "NGOAI"

# ================== MAIN ==================
def main():
    M = load_matrix()
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    # ===== CSV =====
    csv_file = open("report.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["time","target","x","y","zone","score"])

    cv2.namedWindow("Laser Trainer", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Laser Trainer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_target = 0
    target_start = time.time()
    score = 0
    shots = 0
    last_shot_time = 0

    move_x = TARGET_CENTER[0] - IMAGE_SIZE//2
    direction = 1

    print("🎯 BAT DAU")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, FLIP_MODE)
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        now = time.time()
        laser = detect_laser(frame, M)

        # ===== TIME OUT =====
        if now - target_start > TARGET_TIME:
            current_target += 1
            target_start = now

        if current_target >= 4:
            break

        img = TARGET_IMAGES[current_target]

        # ===== POSITION =====
        if current_target == 3:
            move_x += direction * 5
            if move_x < 100 or move_x > WIDTH - IMAGE_SIZE - 100:
                direction *= -1
            tx = move_x
        else:
            tx = TARGET_CENTER[0] - IMAGE_SIZE//2

        ty = TARGET_CENTER[1] - IMAGE_SIZE//2
        canvas[ty:ty+IMAGE_SIZE, tx:tx+IMAGE_SIZE] = img

        # ===== LASER =====
        if laser and shots < MAX_SHOTS:
            lx, ly = laser
            cv2.circle(canvas, (lx, ly), 6, (0,0,255), -1)

            inside = tx < lx < tx+IMAGE_SIZE and ty < ly < ty+IMAGE_SIZE

            if inside and now - last_shot_time > SHOT_COOLDOWN:
                last_shot_time = now
                shots += 1
                play_hit_sound()

                cx, cy = tx+IMAGE_SIZE//2, ty+IMAGE_SIZE//2
                dist = math.hypot(lx-cx, ly-cy)

                if MODE == "THI":
                    shot_score = 1
                    score += 1
                else:
                    shot_score = 1
                    score += 1

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    current_target+1,
                    lx, ly,
                    get_zone(dist),
                    shot_score
                ])

                current_target += 1
                target_start = now

        # ===== UI =====
        cv2.putText(canvas, f"SCORE: {score}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(canvas, f"SHOT: {shots}/{MAX_SHOTS}", (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.imshow("Laser Trainer", canvas)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('r'):
            current_target = 0
            score = 0
            shots = 0
            target_start = time.time()

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

# ================== RUN ==================
if __name__ == "__main__":
    main()
