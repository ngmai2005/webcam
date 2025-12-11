import cv2
import numpy as np

# Danh sách lưu trữ 4 tọa độ góc bia (Nguồn)
source_points = []
CALIBRATION_POINTS_COUNT = 4


def mouse_callback(event, x, y, flags, param):
    """Xử lý sự kiện click chuột để lưu tọa độ"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(source_points) < CALIBRATION_POINTS_COUNT:
            source_points.append([x, y])
            print(f"Đã ghi nhận điểm {len(source_points)}: ({x}, {y})")


def run_calibration():
    """Chức năng chạy Calibration 4 điểm"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # FIX cho Windows tránh crash khi mở webcam

    if not cap.isOpened():
        print("❌ Lỗi: Không mở được Camera. Hãy thử dùng camera index khác (0 → 1 → 2...)")
        return

    cv2.namedWindow('Calibration Window')
    cv2.setMouseCallback('Calibration Window', mouse_callback)

    print("\n--- BẮT ĐẦU HIỆU CHỈNH ---")
    print("Click 4 góc màn hình theo thứ tự:")
    print("1️⃣ Trái-Trên")
    print("2️⃣ Phải-Trên")
    print("3️⃣ Phải-Dưới")
    print("4️⃣ Trái-Dưới")
    print("Nhấn Q để thoát.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được khung hình từ camera!")
            break

        # Vẽ các điểm đã chọn
        for point in source_points:
            cv2.circle(frame, tuple(point), 6, (0, 255, 0), -1)

        # Thêm số điểm đã chọn
        cv2.putText(frame,
                    f'Diem: {len(source_points)}/{CALIBRATION_POINTS_COUNT}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow('Calibration Window', frame)

        # Đủ 4 điểm thì break
        if len(source_points) == CALIBRATION_POINTS_COUNT:
            break

        # Nhấn Q để thoát sớm
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lưu file calibration
    if len(source_points) == CALIBRATION_POINTS_COUNT:
        np.save("calibration_points.npy",
                np.array(source_points, dtype=np.float32))
        print("\n✅ Đã lưu calibration_points.npy thành công!")
    else:
        print("\n❌ Chưa chọn đủ 4 điểm! Không lưu.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_calibration()
