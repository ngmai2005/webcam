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

# ================= DETECT_LASER =================

def detect_laser(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # lọc điểm sáng mạnh
    bright = cv2.inRange(v, 240, 255)
    sat = cv2.inRange(s, 120, 255)
    base_mask = cv2.bitwise_and(bright, sat)

    # lọc màu laser
    red1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 120, 200), (180, 255, 255))
    green = cv2.inRange(hsv, (35, 120, 200), (90, 255, 255))
    pink = cv2.inRange(hsv, (140, 80, 200), (165, 255, 255))

    color_mask = red1 | red2 | green | pink

    laser_mask = cv2.bitwise_and(base_mask, color_mask)

    kernel = np.ones((3, 3), np.uint8)
    laser_mask = cv2.morphologyEx(laser_mask, cv2.MORPH_OPEN, kernel)
    laser_mask = cv2.dilate(laser_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        laser_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return False, None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 4 or area > 250:
        return False, None

    (x, y), r = cv2.minEnclosingCircle(c)

    return True, (int(x), int(y))

# ================= MAIN =================

def main():
    round_id = 1
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
    MAX_ROUND = 1
    target_start = time.time()
    last_shot = 0
    hit_points = []

    move_x, direction = 100, 1
    show_2, show_3 = False, False
    key = -1

    laser_prev = False

    # ================= INIT TTS =================

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    tts_spoken = False

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame_raw, FLIP_MODE)
        H, W = frame.shape[:2]
        laser_detected, laser_pos = detect_laser(frame)
        laser_now = laser_detected

        if laser_detected:
            lx, ly = laser_pos
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        now = time.time()

        # ================= PHASE LOGIC =================
        # ---- PHASE 1: BIA 1 ----

        if phase == 1:
            tx = W // 2 - IMAGE_SIZE // 2
            ty = H // 2 - IMAGE_SIZE // 2
            canvas[ty:ty + IMAGE_SIZE, tx:tx + IMAGE_SIZE] = TARGET_IMAGES[0]

            if can_shoot and bullets < MAX_BULLETS and laser_now and not laser_prev and now - last_shot > SHOT_COOLDOWN:
                bullets += 1
                last_shot = now

                if laser_detected:
                    lx, ly = laser_pos

                if tx < lx < tx + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
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

            if can_shoot and bullets < MAX_BULLETS and laser_now and not laser_prev and now - last_shot > SHOT_COOLDOWN:
                bullets += 1
                last_shot = now

                if laser_detected:
                    lx, ly = laser_pos

                if show_2 and tx2 < lx < tx2 + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    show_2 = False
                    show_3 = True

                elif show_3 and tx3 < lx < tx3 + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
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

            if can_shoot and bullets < MAX_BULLETS and laser_now and not laser_prev and now - last_shot > SHOT_COOLDOWN:
                bullets += 1
                last_shot = now

                if laser_detected:
                    lx, ly = laser_pos

                if move_x < lx < move_x + IMAGE_SIZE and ty < ly < ty + IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
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
                    if round_id < MAX_ROUND:
                        show_2 = show_3 = False
                    move_x = 100
                    direction = 1
                else:
                    phase = 99
        laser_prev = laser_now

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
            if not tts_spoken:
                threading.Thread(
                    target=lambda: (engine.say(rank), engine.runAndWait()),
                    daemon=True
                ).start()
                tts_spoken = True

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
            hit_points.clear()
            last_shot = 0
            tts_spoken = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()