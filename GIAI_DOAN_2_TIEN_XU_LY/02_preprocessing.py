"""
GIAI ĐOẠN 2 - Bước 2: Tiền xử lý ảnh (Image Preprocessing)
Đề tài: Phát hiện cháy rừng trên ảnh vệ tinh

Đây là phần TRỌNG TÂM của môn Xử lý ảnh số.
File này chứa các kỹ thuật tiền xử lý và tăng cường dữ liệu, đồng thời
tạo ra DataLoader để nạp dữ liệu vào model ở các giai đoạn sau.

GIẢI THÍCH CÁC KỸ THUẬT (dùng để viết vào báo cáo):

1. Resize - đưa mọi ảnh về cùng kích thước (224x224) vì mạng CNN
   yêu cầu input cố định. 224 là kích thước chuẩn của các model
   pretrained như ResNet, MobileNet.

2. Normalize (chuẩn hóa) - đưa giá trị pixel từ [0,255] về phân phối
   chuẩn theo mean/std của ImageNet. Giúp model hội tụ nhanh và ổn định hơn.

3. Data Augmentation (tăng cường dữ liệu) - tạo biến thể của ảnh bằng
   cách xoay, lật, đổi độ sáng. Mục đích: tăng tính đa dạng dữ liệu,
   chống overfitting. CHỈ áp dụng cho tập train, KHÔNG áp dụng cho
   valid/test (vì test phải phản ánh dữ liệu thật).

Cách chạy thử:  python 02_preprocessing.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# CẤU HÌNH
# ============================================================
DATA_DIR = Path("./data")
IMG_SIZE = 224           # kích thước ảnh đầu vào cho CNN
BATCH_SIZE = 32          # số ảnh xử lý cùng lúc; giảm xuống 16 nếu GPU thiếu VRAM

# Giá trị mean/std chuẩn của ImageNet - dùng khi transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def transform_train():
    """
    Pipeline tiền xử lý cho tập HUẤN LUYỆN.
    Có data augmentation để tăng đa dạng và chống overfitting.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),       # 1. resize về 224x224
        transforms.RandomHorizontalFlip(p=0.5),        # 2. lật ngang ngẫu nhiên
        transforms.RandomVerticalFlip(p=0.3),          #    lật dọc (ảnh vệ tinh nhìn từ trên nên hợp lý)
        transforms.RandomRotation(degrees=20),         # 3. xoay ngẫu nhiên +-20 độ
        transforms.ColorJitter(brightness=0.2,         # 4. thay đổi độ sáng/tương phản
                               contrast=0.2),
        transforms.ToTensor(),                         # 5. chuyển sang tensor [0,1]
        transforms.Normalize(IMAGENET_MEAN,            # 6. chuẩn hóa
                             IMAGENET_STD),
    ])


def transform_eval():
    """
    Pipeline cho tập VALID và TEST.
    KHÔNG augment - chỉ resize và chuẩn hóa, để đánh giá trung thực.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tao_dataloaders():
    """
    Tạo 3 DataLoader cho train/valid/test.
    ImageFolder tự động gán nhãn dựa trên tên thư mục con.
    Trả về dict các loader và danh sách tên lớp.
    """
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=transform_train())
    valid_ds = datasets.ImageFolder(DATA_DIR / "valid", transform=transform_eval())
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=transform_eval())

    # num_workers tăng tốc nạp dữ liệu; để 2 trên Colab, có thể tăng trên laptop
    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        "valid": DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }
    return loaders, train_ds.classes


if __name__ == "__main__":
    # Chạy thử để kiểm tra pipeline hoạt động
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị đang dùng: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loaders, classes = tao_dataloaders()
    print(f"\nCác lớp: {classes}")  # ví dụ ['nowildfire', 'wildfire']

    # Lấy 1 batch để kiểm tra
    images, labels = next(iter(loaders["train"]))
    print(f"Kích thước 1 batch ảnh: {images.shape}")   # [32, 3, 224, 224]
    print(f"Kích thước 1 batch nhãn: {labels.shape}")  # [32]
    print(f"Khoảng giá trị pixel sau chuẩn hóa: "
          f"[{images.min():.2f}, {images.max():.2f}]")
    print("\nPipeline tiền xử lý hoạt động tốt! Sẵn sàng cho giai đoạn 4.")
