"""
DATALOADER CHO DATASET WILDFIRES-CEMS
File này là nền tảng cho cả hai hướng:
- Người A (truyền thống): lặp qua dataset, lấy ảnh + mask để tính chỉ số NBR
- Người B (deep learning): feed batch vào U-Net để train

GIẢI THÍCH THIẾT KẾ (đưa vào báo cáo):

1. Mỗi sample = một thư mục, có S2L2A.tif (input 12 band) + DEL.tif (mask).
   Các file phụ khác (CM, GRA, LC...) bỏ qua khi train.

2. Vấn đề: ảnh gốc có kích thước khác nhau (mẫu vừa rồi 801x767, mẫu khác
   có thể 500x500 hay 1024x1024...). U-Net cần input cùng kích thước.
   Giải pháp:
   - TRAIN: random crop 256x256 từ ảnh gốc (kết hợp augmentation)
            -> mỗi epoch, mỗi ảnh cho ra patch khác nhau, tăng đa dạng
   - VAL/TEST: center crop hoặc resize về 256x256 (đánh giá nhất quán)

3. Augmentation chỉ cho train, gồm: lật ngang, lật dọc, xoay 90 độ.
   Ảnh vệ tinh nhìn từ trên xuống nên lật/xoay không phá vỡ tính tự nhiên.

4. Chuẩn hóa: giá trị raw đã là 0-2 (reflectance), clip về [0, 1.5] rồi
   chia 1.5 để đưa về [0, 1]. Không dùng ImageNet mean/std vì đây là ảnh
   đa phổ, không phải RGB tự nhiên.

CÁCH CHẠY THỬ:
  python 06_dataloader.py
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio


DATA_ROOT = Path("./wildfires-cems")
DEFAULT_PATCH_SIZE = 256


def tim_tat_ca_samples(split_dir: Path) -> List[Path]:
    """
    Quét thư mục split và trả về danh sách các thư mục sample có
    đủ cả S2L2A.tif và DEL.tif (bỏ qua sample thiếu).
    """
    samples = []
    if not split_dir.exists():
        return samples
    for emsr in split_dir.iterdir():
        if not emsr.is_dir():
            continue
        for aoi in emsr.iterdir():
            if not aoi.is_dir():
                continue
            for sample in aoi.iterdir():
                if not sample.is_dir():
                    continue
                s2 = list(sample.glob("*_S2L2A.tif"))
                de = list(sample.glob("*_DEL.tif"))
                if s2 and de:
                    samples.append(sample)
    return samples


def doc_sample(sample_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc một sample, trả về (image [12,H,W] float32, mask [H,W] uint8)."""
    s2_file = next(sample_dir.glob("*_S2L2A.tif"))
    del_file = next(sample_dir.glob("*_DEL.tif"))
    with rasterio.open(s2_file) as src:
        image = src.read().astype(np.float32)
    with rasterio.open(del_file) as src:
        mask = src.read(1).astype(np.uint8)
    return image, mask


def chuan_hoa(image: np.ndarray) -> np.ndarray:
    """Chuẩn hóa reflectance về [0, 1]. Clip ở 1.5 để loại outlier."""
    image = np.clip(image, 0.0, 1.5) / 1.5
    return image


def random_crop(image: np.ndarray, mask: np.ndarray,
                size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop ngẫu nhiên patch kích thước size x size từ image và mask cùng vị trí."""
    _, h, w = image.shape
    if h < size or w < size:
        # ảnh quá nhỏ, pad lên cho đủ
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
        _, h, w = image.shape
    top = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    image = image[:, top:top + size, left:left + size]
    mask = mask[top:top + size, left:left + size]
    return image, mask


def center_crop(image: np.ndarray, mask: np.ndarray,
                size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop chính giữa, dùng cho val/test để kết quả nhất quán."""
    _, h, w = image.shape
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
        _, h, w = image.shape
    top = (h - size) // 2
    left = (w - size) // 2
    image = image[:, top:top + size, left:left + size]
    mask = mask[top:top + size, left:left + size]
    return image, mask


def augment(image: np.ndarray, mask: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmentation đồng bộ cho image và mask:
    - Lật ngang với xác suất 0.5
    - Lật dọc với xác suất 0.5
    - Xoay 0/90/180/270 độ ngẫu nhiên (k * 90)
    """
    if np.random.rand() < 0.5:
        image = image[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()
    if np.random.rand() < 0.5:
        image = image[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()
    k = np.random.randint(0, 4)
    if k > 0:
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return image, mask


class CEMSDataset(Dataset):
    """
    Dataset PyTorch cho Wildfires-CEMS.

    Tham số:
      split: 'train' | 'val' | 'test'
      patch_size: kích thước patch sau crop (mặc định 256)
      augment_data: bật augmentation (chỉ nên bật cho train)
      crop_mode: 'random' (cho train) hoặc 'center' (cho val/test)
      full_image: nếu True thì trả nguyên ảnh không crop (cho người A
                  dùng baseline NBR, không cần kích thước cố định)
    """

    def __init__(self, split: str,
                 patch_size: int = DEFAULT_PATCH_SIZE,
                 augment_data: bool = False,
                 crop_mode: str = "random",
                 full_image: bool = False):
        self.split_dir = DATA_ROOT / split
        self.samples = tim_tat_ca_samples(self.split_dir)
        self.patch_size = patch_size
        self.augment_data = augment_data
        self.crop_mode = crop_mode
        self.full_image = full_image
        if not self.samples:
            raise RuntimeError(f"Không tìm thấy sample nào trong {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        image, mask = doc_sample(sample_dir)
        image = chuan_hoa(image)

        if self.full_image:
            # Trả nguyên ảnh, không crop, không augment.
            # Dùng cho baseline truyền thống (Người A).
            pass
        else:
            # Crop về kích thước cố định để feed vào U-Net (Người B).
            if self.crop_mode == "random":
                image, mask = random_crop(image, mask, self.patch_size)
            else:
                image, mask = center_crop(image, mask, self.patch_size)
            if self.augment_data:
                image, mask = augment(image, mask)

        # Convert sang torch tensor
        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t, str(sample_dir)


def tao_dataloaders(batch_size: int = 8, patch_size: int = DEFAULT_PATCH_SIZE,
                    num_workers: int = 0):
    """
    Factory tạo 3 DataLoader cho train/val/test (cho người B - U-Net).
    Người A có thể tạo dataset riêng với full_image=True (xem demo bên dưới).
    """
    train_ds = CEMSDataset("train", patch_size=patch_size,
                           augment_data=True, crop_mode="random")
    val_ds = CEMSDataset("val", patch_size=patch_size,
                         augment_data=False, crop_mode="center")
    test_ds = CEMSDataset("test", patch_size=patch_size,
                          augment_data=False, crop_mode="center")

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }
    return loaders


# ============================================================
# DEMO: chạy thử để xác nhận pipeline hoạt động
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM TRA DATALOADER CEMS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")

    # === Test 1: DataLoader cho U-Net (Người B) ===
    print("\n--- Pipeline cho U-Net (Người B) ---")
    loaders = tao_dataloaders(batch_size=4, patch_size=256, num_workers=0)
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(loaders[split].dataset)} sample, "
              f"{len(loaders[split])} batch")

    images, masks, paths = next(iter(loaders["train"]))
    print(f"\n  Batch train đầu tiên:")
    print(f"    images.shape: {images.shape}      (mong đợi [4, 12, 256, 256])")
    print(f"    masks.shape : {masks.shape}        (mong đợi [4, 256, 256])")
    print(f"    images range: [{images.min():.3f}, {images.max():.3f}]  "
          f"(mong đợi gần [0, 1])")
    print(f"    unique mask : {torch.unique(masks).tolist()}     "
          f"(mong đợi [0, 1])")
    print(f"    Tỉ lệ pixel cháy trong batch: "
          f"{(masks == 1).float().mean().item() * 100:.1f}%")

    # === Test 2: Dataset cho baseline truyền thống (Người A) ===
    print("\n--- Pipeline cho baseline NBR (Người A) ---")
    ds_full = CEMSDataset("test", full_image=True)
    img, msk, path = ds_full[0]
    print(f"  Lấy 1 sample full nguyên ảnh (cho người A):")
    print(f"    image.shape: {img.shape}  (12 band, kích thước gốc)")
    print(f"    mask.shape : {msk.shape}")
    print(f"    Đường dẫn  : {path}")

    print("\n" + "=" * 60)
    print("DATALOADER HOẠT ĐỘNG TỐT!")
    print("=" * 60)
    print("- Người A: dùng CEMSDataset(split='test', full_image=True)")
    print("           để chạy baseline NBR trên từng ảnh nguyên kích thước.")
    print("- Người B: dùng tao_dataloaders() để có batch đã augment, crop.")
