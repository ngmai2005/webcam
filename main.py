import cv2
import numpy as np
import time
import threading
import os
import pyttsx3
from playsound import playsound


# ================= CONFIG =================
WIDTH, HEIGHT = 1280, 720
IMAGE_SIZE = 420
CAM_INDEX = 0
FLIP_MODE = 1

SHOT_COOLDOWN = 0.12
MAX_BULLETS = 16
TARGET_TIME = 15
MODE_TEST = False  # False = chế độ thi, True = chế độ tập

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

laser_history = []
laser_armed = True

# ================= DETECT_LASER =================

def detect_laser(frame, M):
    global laser_history, laser_armed

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- MASK MÀU LASER ---
    mask_red = (
        cv2.inRange(hsv, (0, 120, 200), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 120, 200), (180, 255, 255))
    )
    mask_green = cv2.inRange(hsv, (35, 120, 200), (90, 255, 255))
    mask = mask_red | mask_green

    # --- LỌC NHIỄU ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        laser_history.clear()
        laser_armed = True
        return None

    # --- LẤY ĐỐM SÁNG NHẤT ---
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 2 or area > 80:
        return None

    m = cv2.moments(c)
    if m["m00"] == 0:
        return None

    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])

    if hsv[y, x][2] < 200:
        return None

    # chỉ cần thấy laser là bắn
    if not laser_armed:
        return None

    if not laser_armed:
        return None

    laser_armed = False
    laser_history.clear()

    if M is not None:
        pt = np.float32([[[x, y]]])
        mapped = cv2.perspectiveTransform(pt, M)
        return int(mapped[0][0][0]), int(mapped[0][0][1])

    laser_history.append((x, y))
    if len(laser_history) > 5:
        laser_history.pop(0)

    if not laser_armed:
        dx = max(p[0] for p in laser_history) - min(p[0] for p in laser_history)
        dy = max(p[1] for p in laser_history) - min(p[1] for p in laser_history)
        if dx < 3 and dy < 3:
            laser_armed = True

    return x, y

# ================= MAIN =================

def main():
    can_shoot = True
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Laser Trainer", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Laser Trainer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    M = load_matrix()

    score = 0
    bullets = 0
    hits = 0
    phase = 1
    round_id = 1
    MAX_ROUND = 2
    target_start = time.time()
    last_shot = 0
    hit_points = []

    move_x, direction = 100, 1
    show_2, show_3 = False, False
    key = -1

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame_raw, FLIP_MODE)
        H, W = frame.shape[:2]
        laser = detect_laser(frame, M)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        now = time.time()

        # ================= PHASE LOGIC =================
        # ---- PHASE 1: BIA 1 ----
        if phase == 1:
            tx = W // 2 - IMAGE_SIZE // 2
            ty = H // 2 - IMAGE_SIZE // 2
            canvas[ty:ty + IMAGE_SIZE, tx:tx + IMAGE_SIZE] = TARGET_IMAGES[0]

            if can_shoot and laser and now - last_shot > SHOT_COOLDOWN:
                lx, ly = laser
                bullets += 1
                last_shot = now
                if tx < lx < tx + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    laser_armed = True
                    phase = 2
                    target_start = now
                    show_2 = True
                    show_3 = False

            elif now - target_start >= TARGET_TIME:
                phase = 2
                target_start = now
                show_2 = True
                show_3 = False

        # ---- PHASE 2: BIA 2 & 3 ----
        elif phase == 2:
            ty = H // 2 - IMAGE_SIZE // 2
            tx2 = 150
            tx3 = W - IMAGE_SIZE - 150

            if now - target_start >= 5:
                show_3 = True

            if show_2:
                canvas[ty:ty + IMAGE_SIZE, tx2:tx2 + IMAGE_SIZE] = TARGET_IMAGES[1]
            if show_3:
                canvas[ty:ty + IMAGE_SIZE, tx3:tx3 + IMAGE_SIZE] = TARGET_IMAGES[2]

            if can_shoot and laser and now - last_shot > SHOT_COOLDOWN:
                lx, ly = laser
                bullets += 1
                last_shot = now

                if show_2 and tx2 < lx < tx2 + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    laser_armed = True
                    show_2 = False

                elif show_3 and tx3 < lx < tx3 + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    laser_armed = True
                    phase = 4
                    target_start = now
                    move_x = 100
                    direction = 1

            elif now - target_start >= TARGET_TIME:
                phase = 4
                target_start = now
                move_x = 100
                direction = 1

        # ---- PHASE 4: BIA 4 DI CHUYỂN ----
        elif phase == 4:
            move_x += direction * 6
            if move_x < 50 or move_x > W - IMAGE_SIZE - 50:
                direction *= -1
            ty = H // 2 - IMAGE_SIZE // 2
            canvas[ty:ty + IMAGE_SIZE, move_x:move_x + IMAGE_SIZE] = TARGET_IMAGES[3]

            if can_shoot and laser and now - last_shot > SHOT_COOLDOWN:
                lx, ly = laser
                bullets += 1
                last_shot = now

                if move_x < lx < move_x + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    laser_armed = True
                    if round_id < MAX_ROUND:
                        round_id += 1
                        phase = 1
                        target_start = now
                        show_2 = show_3 = False
                        move_x = 100
                        direction = 1
                    else:
                        phase = 99

            elif now - target_start >= TARGET_TIME:
                if round_id < MAX_ROUND:
                    round_id += 1
                    phase = 1
                    target_start = now
                    show_2 = show_3 = False
                    move_x = 100
                    direction = 1
                else:
                    phase = 99

        # ================= HIỆU ỨNG TRÚNG =================
        for px, py in hit_points:
            cv2.circle(canvas, (px, py), 10, (0,0,255), 2)
            cv2.circle(canvas, (px, py), 4, (0,0,0), -1)
        hit_points = [p for p in hit_points if now - target_start < 0.5]

        # ================= HUD =================
        cv2.putText(canvas, f"SCORE: {score}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(canvas, f"SHOT: {bullets}/{MAX_BULLETS}", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ================= INIT TTS =================
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)

        # ================= RESULT =================
        if phase == 99 or bullets >= MAX_BULLETS:
            canvas[:] = (0,0,0)
            if score >= 4:
                rank = "GIOI"
            elif score == 3:
                rank = "KHA"
            elif score == 2:
                rank = "DAT"
            else:
                rank = "KHONG DAT"

            cv2.putText(canvas, f"KET QUA: {rank}", (WIDTH // 2 - 250, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.putText(canvas, f"DIEM: {score}", (WIDTH // 2 - 250, HEIGHT // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(canvas, f"Trung: {hits}/{bullets}", (WIDTH // 2 - 250, HEIGHT // 2 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(canvas, "NHAN R DE RESET", (WIDTH // 2 - 260, HEIGHT // 2 + 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # ================= PHÁT TTS =================
            threading.Thread(target=lambda: engine.say(rank) or engine.runAndWait(), daemon=True).start()

        cv2.imshow("Laser Trainer", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):
            # reset toàn bộ
            score = 0
            bullets = 0
            hits = 0
            phase = 1
            round_id = 1
            target_start = time.time()
            show_2 = show_3 = False
            move_x = 100
            direction = 1
            laser_armed = True
            hit_points.clear()
            last_shot = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
