"""
GIAI ĐOẠN 3 - BASELINE V3: NBR + NDWI (LOẠI MẶT NƯỚC)

PHÁT HIỆN TỪ V2:
Nhìn ảnh kết quả V2, prediction gán cả sông/hồ là cháy do NBR không
ổn định trên mặt nước (cả NIR và SWIR2 đều rất thấp).

CẢI TIẾN V3:
Thêm chỉ số NDWI để phát hiện mặt nước, loại các pixel nước ra khỏi
mask cháy trước khi morphology. Đây là cách chuyên gia viễn thám
thường làm.

NDWI = (Green - NIR) / (Green + NIR)
- Mặt nước có NDWI > 0 (Green > NIR)
- Đất khô có NDWI < 0 (NIR > Green)

PIPELINE V3:
  Input ảnh đa phổ
     ↓
  [1] Lọc Gaussian
     ↓
  [2] Tính NBR và NDWI song song
     ↓
  [3] Mask_fire_raw = (NBR < threshold_NBR)
  [3'] Mask_water = (NDWI > threshold_NDWI)
     ↓
  [4] Mask_fire = Mask_fire_raw AND (NOT Mask_water)   <- loại nước
     ↓
  [5] Morphology opening + closing
     ↓
  Output: mask cháy không bao gồm mặt nước

CÁCH CHẠY:
  python 07c_baseline_v3.py
"""

from pathlib import Path
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_score,
                              recall_score, f1_score)

from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
dataloader = import_module("06_dataloader")
CEMSDataset = dataloader.CEMSDataset


# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_DIR = Path("./ket_qua_baseline_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

B3_GREEN_IDX = 2     # band B3
B8_NIR_IDX = 7       # band B8
B12_SWIR2_IDX = 11   # band B12

EPS = 1e-8

# Ngưỡng từ V2 đã tìm được
NBR_THRESHOLD = -0.10
# Ngưỡng NDWI cho mặt nước (chuẩn viễn thám)
NDWI_THRESHOLD = 0.0


# ============================================================
# PIPELINE V3
# ============================================================
def tinh_chi_so(image: np.ndarray):
    """Tính cả NBR và NDWI từ ảnh đa phổ."""
    green = cv2.GaussianBlur(image[B3_GREEN_IDX], (5, 5), 0)
    nir = cv2.GaussianBlur(image[B8_NIR_IDX], (5, 5), 0)
    swir2 = cv2.GaussianBlur(image[B12_SWIR2_IDX], (5, 5), 0)
    nbr = np.clip((nir - swir2) / (nir + swir2 + EPS), -1, 1)
    ndwi = np.clip((green - nir) / (green + nir + EPS), -1, 1)
    return nbr, ndwi


def du_doan_v3(image: np.ndarray,
               nbr_thr: float = NBR_THRESHOLD,
               ndwi_thr: float = NDWI_THRESHOLD):
    """Pipeline V3: NBR + loại mặt nước bằng NDWI."""
    nbr, ndwi = tinh_chi_so(image)

    # Mask ứng viên cháy theo NBR
    mask_fire_raw = (nbr < nbr_thr).astype(np.uint8)
    # Mask mặt nước theo NDWI
    mask_water = (ndwi > ndwi_thr).astype(np.uint8)
    # Loại mặt nước ra khỏi mask cháy
    mask_fire = mask_fire_raw * (1 - mask_water)

    # Morphology
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask, nbr, ndwi, mask_water


# ============================================================
# ĐÁNH GIÁ
# ============================================================
def danh_gia(test_ds, save_examples=5):
    print(f"Đánh giá V3 trên {len(test_ds)} mẫu test...")
    all_pred, all_gt = [], []
    ious, dices = [], []
    start = time.time()

    for idx in range(len(test_ds)):
        image, mask_gt, path = test_ds[idx]
        image = image.numpy() if hasattr(image, "numpy") else image
        mask_gt = mask_gt.numpy() if hasattr(mask_gt, "numpy") else mask_gt
        mask_gt = mask_gt.astype(np.uint8)

        mask_pred, nbr, ndwi, mask_water = du_doan_v3(image)

        inter = ((mask_pred == 1) & (mask_gt == 1)).sum()
        union = ((mask_pred == 1) | (mask_gt == 1)).sum()
        ious.append(inter / (union + EPS))
        dices.append(2 * inter / (mask_pred.sum() + mask_gt.sum() + EPS))

        all_pred.append(mask_pred[::4, ::4].flatten())
        all_gt.append(mask_gt[::4, ::4].flatten())

        if idx < save_examples:
            ve_minh_hoa_v3(image, mask_gt, mask_pred, nbr, ndwi, mask_water,
                            Path(path).name, ious[-1], dices[-1])

        if (idx + 1) % 20 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(test_ds)}...")

    elapsed = time.time() - start
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    precision = precision_score(all_gt, all_pred, zero_division=0)
    recall = recall_score(all_gt, all_pred, zero_division=0)
    f1 = f1_score(all_gt, all_pred, zero_division=0)
    cm = confusion_matrix(all_gt, all_pred)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ BASELINE V3 (NBR<{NBR_THRESHOLD} AND NDWI≤{NDWI_THRESHOLD})")
    print("=" * 60)
    print(f"Thời gian: {elapsed:.1f}s")
    print(f"IoU       : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Dice      : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"\nConfusion matrix:")
    print(f"                  Pred=KhongChay   Pred=Chay")
    print(f"  GT=KhongChay    {cm[0,0]:>12d}   {cm[0,1]:>9d}")
    print(f"  GT=Chay         {cm[1,0]:>12d}   {cm[1,1]:>9d}")

    ve_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix_v3.png")

    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write(f"BASELINE V3 - NBR<{NBR_THRESHOLD} AND loại mặt nước NDWI>{NDWI_THRESHOLD}\n")
        f.write("=" * 50 + "\n")
        f.write(f"IoU      : {np.mean(ious):.4f} ± {np.std(ious):.4f}\n")
        f.write(f"Dice     : {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")


def _stretch(arr):
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def ve_minh_hoa_v3(image, mask_gt, mask_pred, nbr, ndwi, mask_water,
                    ten, iou, dice):
    rgb = np.stack([_stretch(image[3]), _stretch(image[2]),
                    _stretch(image[1])], axis=-1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Baseline V3 · {ten} · IoU={iou:.3f} · Dice={dice:.3f}",
                 fontsize=13)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB"); axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(nbr, cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[0, 1].set_title("NBR"); axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(ndwi, cmap="Blues", vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title("NDWI (>0 = nước)"); axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(mask_water, cmap="Blues")
    axes[1, 0].set_title("Mask mặt nước (loại bỏ)"); axes[1, 0].axis("off")

    axes[1, 1].imshow(mask_pred, cmap="hot")
    axes[1, 1].set_title("Prediction V3"); axes[1, 1].axis("off")

    axes[1, 2].imshow(mask_gt, cmap="hot")
    axes[1, 2].set_title("Ground Truth"); axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v3_{ten}.png", dpi=80, bbox_inches="tight")
    plt.close()


def ve_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["KhongChay", "Chay"])
    ax.set_yticklabels(["KhongChay", "Chay"])
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix · Baseline V3")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color=color, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE V3: NBR + NDWI (LOẠI MẶT NƯỚC)")
    print("=" * 60)
    test_ds = CEMSDataset("test", full_image=True)
    danh_gia(test_ds)
    print(f"\nXong! Mở thư mục '{OUTPUT_DIR}' để xem.")
