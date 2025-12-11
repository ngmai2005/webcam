# Tên file: create_sample.py
import numpy as np
import os


def create_sample_calibration_file():
    """
    Tạo một file calibration_points.npy mẫu để kiểm tra main.py.

    LƯU Ý: Đây là dữ liệu giả định! Tọa độ thực tế sẽ phụ thuộc vào vị trí camera của bạn.
    """

    # Giả sử kích thước khung hình Camera là 640x480.
    # Các tọa độ (x, y) này được giả định hơi méo (skewed)

    # Cấu trúc: [Top-Left], [Top-Right], [Bottom-Right], [Bottom-Left]
    sample_points = np.array([
        [150.0, 100.0],  # Góc trên trái
        [550.0, 110.0],  # Góc trên phải (hơi cao hơn một chút)
        [540.0, 400.0],  # Góc dưới phải
        [160.0, 390.0]  # Góc dưới trái (hơi lệch vào trong)
    ], dtype=np.float32)

    filename = 'calibration_points.npy'

    # Lý do: np.save lưu mảng numpy vào file dưới định dạng .npy
    np.save(filename, sample_points)

    print("-------------------------------------------------------")
    print(f"✅ Đã tạo file mẫu: {filename}")
    print(f"   Vị trí: {os.path.abspath(filename)}")
    print("   Bây giờ bạn có thể chạy file main.py để kiểm tra!")
    print("-------------------------------------------------------")


if __name__ == '__main__':
    create_sample_calibration_file()