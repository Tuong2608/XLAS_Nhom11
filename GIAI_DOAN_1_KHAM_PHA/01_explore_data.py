"""
GIAI ĐOẠN 2 - Bước 1: Khám phá dữ liệu (EDA - Exploratory Data Analysis)
Đề tài: Phát hiện cháy rừng trên ảnh vệ tinh

Mục tiêu của file này:
- Kiểm tra cấu trúc thư mục dataset
- Đếm số ảnh mỗi lớp (wildfire / nowildfire) để xem dữ liệu có bị mất cân bằng không
- Hiển thị vài ảnh mẫu của mỗi lớp để hiểu dữ liệu trông như thế nào
- Kiểm tra kích thước ảnh

Cách chạy:  python 01_explore_data.py
"""

import os
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# CẤU HÌNH - sửa đường dẫn này cho khớp máy của bạn
# ============================================================
DATA_DIR = Path("./data")          # thư mục gốc chứa train/valid/test
SPLITS = ["train", "valid", "test"]
CLASSES = ["wildfire", "nowildfire"]


def kiem_tra_cau_truc():
    """Kiểm tra xem các thư mục cần thiết có tồn tại không."""
    print("=" * 60)
    print("KIỂM TRA CẤU TRÚC THƯ MỤC")
    print("=" * 60)
    if not DATA_DIR.exists():
        print(f"[LỖI] Không tìm thấy thư mục {DATA_DIR.resolve()}")
        print("      Hãy tải dataset và giải nén vào đúng vị trí.")
        return False

    ok = True
    for split in SPLITS:
        for cls in CLASSES:
            folder = DATA_DIR / split / cls
            status = "OK" if folder.exists() else "THIẾU"
            if not folder.exists():
                ok = False
            print(f"  [{status}] {folder}")
    return ok


def dem_anh():
    """Đếm số ảnh mỗi lớp trong mỗi split. Giúp phát hiện mất cân bằng dữ liệu."""
    print("\n" + "=" * 60)
    print("THỐNG KÊ SỐ LƯỢNG ẢNH")
    print("=" * 60)
    tong_ket = {}
    for split in SPLITS:
        print(f"\n--- {split.upper()} ---")
        for cls in CLASSES:
            folder = DATA_DIR / split / cls
            if not folder.exists():
                continue
            # đếm các file ảnh phổ biến
            so_anh = len([f for f in folder.iterdir()
                          if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            tong_ket[(split, cls)] = so_anh
            print(f"  {cls:12s}: {so_anh:6d}  ảnh")
    return tong_ket


def kiem_tra_kich_thuoc():
    """Lấy mẫu vài ảnh để xem kích thước và số kênh màu."""
    print("\n" + "=" * 60)
    print("KIỂM TRA KÍCH THƯỚC ẢNH (lấy mẫu)")
    print("=" * 60)
    kich_thuoc = Counter()
    folder = DATA_DIR / "train" / "wildfire"
    if not folder.exists():
        return
    files = [f for f in folder.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png")][:50]
    for f in files:
        with Image.open(f) as img:
            kich_thuoc[(img.size, img.mode)] += 1
    for (size, mode), count in kich_thuoc.most_common():
        print(f"  Kích thước {size}, chế độ màu {mode}: {count} ảnh")


def hien_thi_mau():
    """Hiển thị 4 ảnh wildfire và 4 ảnh nowildfire để quan sát trực quan."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row, cls in enumerate(CLASSES):
        folder = DATA_DIR / "train" / cls
        if not folder.exists():
            continue
        files = [f for f in folder.iterdir()
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")][:4]
        for col, f in enumerate(files):
            img = Image.open(f)
            axes[row, col].imshow(img)
            axes[row, col].set_title(cls, fontsize=11)
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig("mau_anh.png", dpi=100)
    print("\nĐã lưu ảnh mẫu vào file 'mau_anh.png'")
    plt.show()


if __name__ == "__main__":
    if kiem_tra_cau_truc():
        dem_anh()
        kiem_tra_kich_thuoc()
        hien_thi_mau()
        print("\nXong! Hãy mở file 'mau_anh.png' để xem ảnh mẫu.")
    else:
        print("\nVui lòng sửa đường dẫn DATA_DIR hoặc tải lại dataset.")
