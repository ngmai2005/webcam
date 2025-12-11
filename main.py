# Tên file: main.py (FINAL - Bia 3 Chuyển Động)
import cv2
import numpy as np
import os
import math
import time

# ====================================================================
# I. KHỐI CẤU HÌNH & TẢI ẢNH (CONFIG & GAME STATE)
# ====================================================================

# Kích thước khung hình Calibrate (1280x720)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Trạng thái Game
CURRENT_TARGET_INDEX = 0
MAX_TARGETS = 3
TOTAL_SCORE = 0
RANGE_FACTOR = 1.0

# Tọa độ gốc của các Bia
TARGET_POSITIONS_STATIC = {
    0: (640, 360),  # Bia 1: Chính giữa (Tĩnh)
    1: (640, 360),  # Bia 2: Chính giữa (Tĩnh)
    2: (640, 360)  # Bia 3: Vị trí gốc (Anchor/Tâm)
}

# Quản lý Bia di động (ĐÃ KHÔI PHỤC LOGIC CHUYỂN ĐỘNG)
TARGET3_X_OFFSET = 0
TARGET3_MOTION_SPEED = 3
TARGET3_DIRECTION = 1
MOTION_BOUNDARY_LIMIT = 500  # Giới hạn di chuyển (Rất rộng)

SHOW_CAMERA_WINDOW = False

# --- TẢI ẢNH BÊN NGOÀI VÒNG LẶP ---
TARGET_IMAGES = None
try:
    # KÍCH THƯỚC ẢNH ĐƯỢC TẢI (300 pixels)
    IMAGE_SIZE = 300
    img_files = ['images/target1.png', 'images/target2.png', 'images/target3.png']
    loaded_imgs = []

    for file in img_files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Không tìm thấy file: {file}")
        loaded_imgs.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))

    TARGET_IMAGES = loaded_imgs
    print("✅ Tải ảnh bia thành công.")

except Exception as e:
    print(f"❌ Lỗi tải ảnh (dùng fallback hình vuông): {e}")
    TARGET_IMAGES = None


# ====================================================================
# II. UTILITIES & THUẬT TOÁN HỖ TRỢ
# ====================================================================

def load_calibration_matrix():
    """Tải 4 điểm từ file .npy và tính Ma trận Perspective M"""
    if not os.path.exists('calibration_points.npy'):
        print("❌ Không tìm thấy calibration_points.npy. Hãy chạy calibration.py.")
        return None
    try:
        source_points = np.load('calibration_points.npy').astype(np.float32)
        DEST_POINTS = np.float32([
            [0, 0], [TARGET_WIDTH, 0],
            [TARGET_WIDTH, TARGET_HEIGHT], [0, TARGET_HEIGHT]
        ])
        M = cv2.getPerspectiveTransform(source_points, DEST_POINTS)
        return M
    except Exception as e:
        return None


def find_and_map_laser(frame, M):
    """Tìm Laser IR (chỉ dùng ngưỡng sáng) và ánh xạ sang tọa độ Bia chuẩn"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask_bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    mask_combined = cv2.GaussianBlur(mask_bright, (7, 7), 0)
    _, mask_combined = cv2.threshold(mask_combined, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 6: return None

    M_dot = cv2.moments(largest)
    if M_dot["m00"] == 0: return None

    cam_x = int(M_dot["m10"] / M_dot["m00"])
    cam_y = int(M_dot["m01"] / M_dot["m00"])

    try:
        pts = np.float32([[[cam_x, cam_y]]])
        transformed = cv2.perspectiveTransform(pts, M)
        final_x = int(transformed[0, 0, 0])
        final_y = int(transformed[0, 0, 1])
        final_x = max(0, min(TARGET_WIDTH - 1, final_x))
        final_y = max(0, min(TARGET_HEIGHT - 1, final_y))
        return (final_x, final_y, cam_x, cam_y)
    except Exception as e:
        return None


def update_target_3_position():
    """Cập nhật vị trí ngang của Bia số 3 (Di động)"""
    global TARGET3_X_OFFSET, TARGET3_DIRECTION
    TARGET3_X_OFFSET += TARGET3_MOTION_SPEED * TARGET3_DIRECTION
    if TARGET3_X_OFFSET > MOTION_BOUNDARY_LIMIT:
        TARGET3_DIRECTION = -1
    elif TARGET3_X_OFFSET < -MOTION_BOUNDARY_LIMIT:
        TARGET3_DIRECTION = 1
    ox, oy = TARGET_POSITIONS_STATIC[2]
    return (ox + TARGET3_X_OFFSET, oy)


def get_current_target_pos():
    """Trả về tọa độ (x, y) của bia đang hoạt động. Bia 3 có chuyển động."""
    if CURRENT_TARGET_INDEX == 2:
        # BIA 3: Dùng hàm cập nhật vị trí chuyển động
        return update_target_3_position()
    elif 0 <= CURRENT_TARGET_INDEX < 2:
        # BIA 1 & 2: Dùng vị trí tĩnh (trung tâm)
        return TARGET_POSITIONS_STATIC[CURRENT_TARGET_INDEX]
    return (0, 0)


def overlay_image_alpha(bg, fg, x, y):
    """Ghép fg (có alpha) lên bg tại (x,y)"""
    h, w = fg.shape[0], fg.shape[1]
    if x >= bg.shape[1] or y >= bg.shape[0]: return bg
    x1, x2 = max(x, 0), min(x + w, bg.shape[1])
    y1, y2 = max(y, 0), min(y + h, bg.shape[0])
    x1_fg, x2_fg = x1 - x, x2 - x
    y1_fg, y2_fg = y1 - y, y2 - y
    if x1 >= x2 or y1 >= y2: return bg

    if fg.shape[2] == 4:
        alpha = fg[y1_fg:y2_fg, x1_fg:x2_fg, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = (alpha * fg[y1_fg:y2_fg, x1_fg:x2_fg, c] + (1 - alpha) * bg[y1:y2, x1:x2, c])
    else:
        bg[y1:y2, x1:x2] = fg[y1_fg:y2_fg, x1_fg:x2_fg]
    return bg


def draw_target_graphics(canvas, target_index, target_x, target_y, is_active=False):
    """Vẽ hình ảnh Bia (Đã phóng to và bỏ vòng tròn)."""
    global TARGET_IMAGES

    # RADIUS fallback (150)
    RADIUS = 150

    # 1. VẼ HÌNH ẢNH (Ưu tiên)
    if TARGET_IMAGES is not None and target_index < len(TARGET_IMAGES):
        img = TARGET_IMAGES[target_index]
        h, w = img.shape[:2]
        xs = int(target_x - w // 2);
        ys = int(target_y - h // 2)
        overlay_image_alpha(canvas, img, xs, ys)
    else:
        # Fallback
        color = (255, 100, 100) if target_index == 0 else (100, 255, 100)
        cv2.circle(canvas, (int(target_x), int(target_y)), RADIUS, color, -1)

    # 2. Vẽ tâm bia (Đã tăng kích thước)
    cv2.circle(canvas, (int(target_x), int(target_y)), 12, (255, 255, 255), -1)


# ====================================================================
# IV. VÒNG LẶP CHÍNH (MAIN EXECUTION)
# ====================================================================

def main_loop():
    global CURRENT_TARGET_INDEX, TOTAL_SCORE

    M = load_calibration_matrix()
    if M is None: return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở camera (index 0). Thử thay 0 -> 1 hoặc 2.")
        return

    cv2.namedWindow('Laser Trainer UI', cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret: break

        result = find_and_map_laser(frame, M)
        score = 0

        # 1. Game over
        if CURRENT_TARGET_INDEX >= MAX_TARGETS:
            canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
            cv2.putText(canvas, f'GAME OVER - Total Score: {TOTAL_SCORE}', (TARGET_WIDTH // 4, TARGET_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('Laser Trainer UI', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 2. Xử lý Trúng đích
        if result:
            final_x, final_y, cam_x, cam_y = result
            # Lấy vị trí hiện tại (có thể chuyển động nếu là Bia 3)
            target_x, target_y = get_current_target_pos()
            distance = math.hypot(final_x - target_x, final_y - target_y)
            HIT_ZONE_RADIUS = 50 / RANGE_FACTOR

            if distance < 15:
                score = 10
            elif distance < HIT_ZONE_RADIUS:
                score = 5

            if score > 0:
                print(f"Bia {CURRENT_TARGET_INDEX + 1} HIT! Điểm {score}")
                TOTAL_SCORE += score
                CURRENT_TARGET_INDEX += 1  # CHỈ NHẢY KHI TRÚNG
                time.sleep(0.8)

        # 3. Draw UI canvas
        canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

        # Vẽ bia đang hoạt động
        if CURRENT_TARGET_INDEX < MAX_TARGETS:
            i = CURRENT_TARGET_INDEX
            # Cần gọi get_current_target_pos() để lấy vị trí chuyển động của Bia 3
            tx, ty = get_current_target_pos()
            draw_target_graphics(canvas, i, tx, ty, is_active=True)

        # Vẽ vết đạn (chỉ vẽ nếu bắn trúng)
        if result and score > 0:
            cv2.circle(canvas, (final_x, final_y), 8, (0, 0, 255), -1)

            # HUD
        cv2.putText(canvas, f'Total: {TOTAL_SCORE}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(canvas, f'Target: {CURRENT_TARGET_INDEX + 1}/{MAX_TARGETS}', (TARGET_WIDTH - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow('Laser Trainer UI', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_loop()