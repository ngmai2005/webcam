import cv2
import numpy as np
import time
import math
import csv
import os
import threading
from playsound import playsound

# ================= CONFIG =================
WIDTH, HEIGHT = 1280, 720
IMAGE_SIZE = 420
CAM_INDEX = 1
FLIP_MODE = 1
SHOT_COOLDOWN = 0.12
MAX_BULLETS = 16
TARGET_TIME = 7
REPORT_FILE = "report.csv"

# ================= SOUND =================
def play_hit_sound():
    threading.Thread(
        target=playsound,
        args=("sounds/hit.mp3",),
        daemon=True
    ).start()

# ================= LOAD TARGET IMAGES =================
TARGET_IMAGES = []
for i in range(1, 5):
    img = cv2.imread(f"images/target{i}.png")
    if img is None:
        print(f"❌ Thiếu images/target{i}.png")
        exit()
    TARGET_IMAGES.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))

# ================= CALIBRATION =================
def load_matrix():
    if not os.path.exists("calibration_points.npy"):
        return None
    try:
        src = np.load("calibration_points.npy").astype(np.float32)
        dst = np.float32([
            [0, 0],
            [WIDTH, 0],
            [WIDTH, HEIGHT],
            [0, HEIGHT]
        ])
        return cv2.getPerspectiveTransform(src, dst)
    except:
        return None

# ================= LASER DETECTION =================
def detect_laser(frame, M):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Bắt điểm sáng mạnh (laser đỏ / hồng)
    lower = np.array([0, 120, 230])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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

    if M is not None:
        pt = np.float32([[[x, y]]])
        mapped = cv2.perspectiveTransform(pt, M)
        return int(mapped[0][0][0]), int(mapped[0][0][1])

    return x, y

# ================= RESET =================
def reset_state():
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["Time", "X", "Y", "Target", "Hit", "Score"]
        )
    return 0, 0, 0, time.time(), False, []

# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    cv2.namedWindow("Laser Trainer", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Laser Trainer",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    M = load_matrix()

    current_target, score, bullets, target_start, laser_active, hit_points = reset_state()
    last_shot = 0
    move_x, direction = 100, 1

    print("🎯 BẮT ĐẦU MÔ PHỎNG")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame_raw = cv2.flip(frame_raw, FLIP_MODE)

        # detect laser trên frame gốc
        laser = detect_laser(frame_raw, M)

        frame = cv2.resize(frame_raw, (WIDTH, HEIGHT))
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        now = time.time()

        # Hết thời gian bia
        if now - target_start > TARGET_TIME:
            current_target += 1
            target_start = now
            hit_points.clear()

        if current_target < 4 and bullets < MAX_BULLETS:
            img = TARGET_IMAGES[current_target]

            if current_target == 3:  # bia di chuyển
                move_x += direction * 6
                if move_x < 50 or move_x > WIDTH - IMAGE_SIZE - 50:
                    direction *= -1
                tx = move_x
            else:
                tx = WIDTH // 2 - IMAGE_SIZE // 2

            ty = HEIGHT // 2 - IMAGE_SIZE // 2
            canvas[ty:ty + IMAGE_SIZE, tx:tx + IMAGE_SIZE] = img

            # Vẽ các điểm trúng đã lưu
            for px, py in hit_points:
                cv2.circle(canvas, (px, py), 10, (0, 0, 255), 2)
                cv2.circle(canvas, (px, py), 4, (0, 0, 0), -1)

            if laser:
                lx, ly = laser
                inside = tx < lx < tx + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE

                if not laser_active and now - last_shot > SHOT_COOLDOWN:
                    bullets += 1
                    last_shot = now
                    laser_active = True

                    hit = inside
                    if hit:
                        score += 1
                        hit_points.append((lx, ly))
                        current_target += 1
                        target_start = now
                        play_hit_sound()
                    else:
                        hit_points.append((lx, ly))

                    with open(REPORT_FILE, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [time.strftime("%H:%M:%S"), lx, ly,
                             current_target + 1, hit, score]
                        )

                cv2.circle(canvas, (lx, ly), 6, (0, 0, 255), -1)
            else:
                laser_active = False

        # ================= HUD =================
        cv2.putText(canvas, f"SCORE: {score}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(canvas, f"BULLETS: {bullets}/{MAX_BULLETS}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ================= KẾT THÚC =================
        if bullets >= MAX_BULLETS or current_target >= 4:
            if score >= 4:
                rank = "GIOI"
            elif score == 3:
                rank = "KHA"
            elif score == 2:
                rank = "TRUNG BINH"
            else:
                rank = "KEM"

            cv2.putText(canvas, f"KET QUA: {rank}",
                        (WIDTH // 2 - 250, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 255), 4)

            cv2.putText(canvas, "NHAN R DE RESET",
                        (WIDTH // 2 - 260, HEIGHT // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 3)

        cv2.imshow("Laser Trainer", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):
            current_target, score, bullets, target_start, laser_active, hit_points = reset_state()

    cap.release()
    cv2.destroyAllWindows()

# ================= RUN =================
if __name__ == "__main__":
    main()
