"""
GIAI ĐOẠN 3 - BASELINE TRUYỀN THỐNG (Người A)
Đề tài: Phát hiện cháy rừng trên ảnh vệ tinh

Pipeline xử lý ảnh truyền thống KHÔNG dùng deep learning:
  Input ảnh đa phổ
     ↓
  [1] Tách kênh (extract bands B8 NIR + B12 SWIR2)
     ↓
  [2] Lọc Gaussian (giảm nhiễu pixel)
     ↓
  [3] Tính chỉ số NBR = (NIR - SWIR2) / (NIR + SWIR2)
     ↓
  [4] Histogram equalization (tăng tương phản giữa cháy/không cháy)
     ↓
  [5] Thresholding Otsu (tự động tìm ngưỡng tối ưu)
     ↓
  [6] Morphological opening + closing (loại nhiễu, lấp lỗ)
     ↓
  Output: mask cháy nhị phân

PSEUDO-CODE (dùng cho báo cáo):
  function PredictFireMask(image_12bands):
      nir   ← image_12bands[7]     // band B8
      swir2 ← image_12bands[11]    // band B12
      nir   ← GaussianBlur(nir, kernel=5)
      swir2 ← GaussianBlur(swir2, kernel=5)
      nbr   ← (nir - swir2) / (nir + swir2 + ε)
      nbr_u8 ← scale_to_uint8(nbr, [-1, 1] → [0, 255])
      nbr_eq ← EqualizeHist(nbr_u8)
      threshold ← OtsuThreshold(nbr_eq)
      mask_raw ← (nbr_eq < threshold)    // pixel tối hơn = đã cháy
      mask     ← MorphologyOpen(mask_raw, k=3)   // loại đốm nhỏ
      mask     ← MorphologyClose(mask, k=5)      // lấp lỗ trong vùng cháy
      return mask

CÁCH CHẠY:
  python 07_baseline_traditional.py
"""

from pathlib import Path
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_score,
                              recall_score, f1_score)

# Import từ file dataloader đã viết ở bước 06
from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
dataloader = import_module("06_dataloader")
CEMSDataset = dataloader.CEMSDataset


# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_DIR = Path("./ket_qua_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)

# Vị trí band trong mảng (index từ 0)
B8_NIR_IDX = 7
B12_SWIR2_IDX = 11

EPS = 1e-8


# ============================================================
# CÁC BƯỚC PIPELINE
# ============================================================
def buoc_1_tach_kenh(image: np.ndarray):
    """Tách lấy hai band quan trọng nhất cho phát hiện cháy."""
    nir = image[B8_NIR_IDX]
    swir2 = image[B12_SWIR2_IDX]
    return nir, swir2


def buoc_2_loc_gaussian(nir: np.ndarray, swir2: np.ndarray, kernel_size: int = 5):
    """Lọc Gaussian giảm nhiễu pixel. Kernel 5x5 là cân bằng tốt."""
    nir_smoothed = cv2.GaussianBlur(nir, (kernel_size, kernel_size), 0)
    swir2_smoothed = cv2.GaussianBlur(swir2, (kernel_size, kernel_size), 0)
    return nir_smoothed, swir2_smoothed


def buoc_3_tinh_nbr(nir: np.ndarray, swir2: np.ndarray):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Giá trị trong [-1, 1]."""
    nbr = (nir - swir2) / (nir + swir2 + EPS)
    return np.clip(nbr, -1, 1)


def buoc_4_equalize_histogram(nbr: np.ndarray):
    """
    Histogram equalization: tăng tương phản, làm vùng cháy/không cháy
    cách xa nhau hơn trên thang giá trị. CV2 cần uint8 nên scale trước.
    Trả về cả nbr_u8 ban đầu và nbr_eq sau equalize để minh họa.
    """
    nbr_u8 = ((nbr + 1) * 127.5).astype(np.uint8)  # [-1,1] -> [0,255]
    nbr_eq = cv2.equalizeHist(nbr_u8)
    return nbr_u8, nbr_eq


def buoc_5_threshold_otsu(nbr_eq: np.ndarray):
    """
    Otsu's thresholding: tự động tìm ngưỡng tối ưu chia ảnh thành 2 class.
    Cháy có NBR thấp -> sau scale là pixel TỐI hơn -> lấy pixel < threshold.
    """
    threshold, mask = cv2.threshold(nbr_eq, 0, 1,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return int(threshold), mask.astype(np.uint8)


def buoc_6_morphology(mask: np.ndarray, k_open: int = 3, k_close: int = 5):
    """
    Opening (erode + dilate): loại các đốm nhỏ giả (false positive nhiễu).
    Closing (dilate + erode): lấp các lỗ nhỏ trong vùng cháy lớn.
    """
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
    return mask_opened, mask_closed


def pipeline_du_doan(image: np.ndarray):
    """Chạy toàn bộ pipeline. Trả về mask cuối + các bước trung gian."""
    nir, swir2 = buoc_1_tach_kenh(image)
    nir_s, swir2_s = buoc_2_loc_gaussian(nir, swir2)
    nbr = buoc_3_tinh_nbr(nir_s, swir2_s)
    nbr_u8, nbr_eq = buoc_4_equalize_histogram(nbr)
    threshold, mask_thresh = buoc_5_threshold_otsu(nbr_eq)
    mask_opened, mask_final = buoc_6_morphology(mask_thresh)
    return {
        "nir": nir, "swir2": swir2,
        "nbr": nbr, "nbr_u8": nbr_u8, "nbr_eq": nbr_eq,
        "threshold": threshold,
        "mask_thresh": mask_thresh,
        "mask_opened": mask_opened,
        "mask_final": mask_final,
    }


# ============================================================
# ĐÁNH GIÁ
# ============================================================
def tinh_iou(pred: np.ndarray, gt: np.ndarray):
    """IoU = TP / (TP + FP + FN)"""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return intersection / (union + EPS)


def tinh_dice(pred: np.ndarray, gt: np.ndarray):
    """Dice = 2*TP / (2*TP + FP + FN)"""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    return 2 * intersection / (pred_b.sum() + gt_b.sum() + EPS)


def danh_gia_toan_bo(dataset, save_examples: int = 5):
    """
    Chạy pipeline trên toàn bộ test set, tính các chỉ số.
    Đồng thời lưu ra hình minh họa cho save_examples mẫu đầu tiên.
    """
    print(f"\nĐánh giá pipeline trên {len(dataset)} mẫu test...")

    ious, dices = [], []
    # Tích lũy pixel cho confusion matrix toàn cục
    all_pred = []
    all_gt = []

    start_time = time.time()

    for idx in range(len(dataset)):
        image, mask_gt, path = dataset[idx]
        image = image.numpy() if hasattr(image, "numpy") else image
        mask_gt = mask_gt.numpy() if hasattr(mask_gt, "numpy") else mask_gt
        mask_gt = mask_gt.astype(np.uint8)

        result = pipeline_du_doan(image)
        mask_pred = result["mask_final"]

        ious.append(tinh_iou(mask_pred, mask_gt))
        dices.append(tinh_dice(mask_pred, mask_gt))

        # Lưu pixel cho confusion matrix (subsample để tiết kiệm RAM)
        step = 4
        all_pred.append(mask_pred[::step, ::step].flatten())
        all_gt.append(mask_gt[::step, ::step].flatten())

        # Lưu hình minh họa cho vài mẫu đầu
        if idx < save_examples:
            ten_mau = Path(path).name
            ve_chi_tiet_mot_mau(image, mask_gt, result,
                                ten_mau, ious[-1], dices[-1])

        if (idx + 1) % 10 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(dataset)} mẫu...")

    elapsed = time.time() - start_time

    # Tính chỉ số tổng quát
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    precision = precision_score(all_gt, all_pred, zero_division=0)
    recall = recall_score(all_gt, all_pred, zero_division=0)
    f1 = f1_score(all_gt, all_pred, zero_division=0)
    cm = confusion_matrix(all_gt, all_pred)

    print("\n" + "=" * 60)
    print("KẾT QUẢ BASELINE TRUYỀN THỐNG")
    print("=" * 60)
    print(f"Thời gian xử lý: {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/mẫu)")
    print(f"IoU trung bình       : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Dice trung bình      : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Precision (pixel)    : {precision:.4f}")
    print(f"Recall    (pixel)    : {recall:.4f}")
    print(f"F1-score  (pixel)    : {f1:.4f}")
    print(f"\nConfusion matrix (pixel-level):")
    print(f"                  Pred=KhongChay   Pred=Chay")
    print(f"  GT=KhongChay    {cm[0, 0]:>12d}   {cm[0, 1]:>9d}")
    print(f"  GT=Chay         {cm[1, 0]:>12d}   {cm[1, 1]:>9d}")

    # Vẽ confusion matrix và lưu
    ve_confusion_matrix(cm, f"{OUTPUT_DIR}/confusion_matrix_baseline.png")

    # Lưu kết quả số ra file để báo cáo dùng
    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ BASELINE TRUYỀN THỐNG\n")
        f.write("=" * 40 + "\n")
        f.write(f"Số mẫu test: {len(dataset)}\n")
        f.write(f"IoU      : {np.mean(ious):.4f} ± {np.std(ious):.4f}\n")
        f.write(f"Dice     : {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")

    return {"iou": np.mean(ious), "dice": np.mean(dices),
            "precision": precision, "recall": recall, "f1": f1, "cm": cm}


# ============================================================
# TRỰC QUAN HÓA
# ============================================================
def _stretch(arr):
    """Co giãn về [0, 1] để hiển thị đẹp."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def ve_chi_tiet_mot_mau(image, mask_gt, result, ten_mau, iou, dice):
    """Vẽ 8 ô: input RGB, NIR, SWIR2, NBR, NBR sau equalize, threshold,
    sau morphology, ground truth."""
    rgb = np.stack([_stretch(image[3]), _stretch(image[2]),
                    _stretch(image[1])], axis=-1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Pipeline truyền thống · {ten_mau} · "
                 f"IoU={iou:.3f} · Dice={dice:.3f}",
                 fontsize=14)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("(1) Input RGB"); axes[0, 0].axis("off")

    axes[0, 1].imshow(_stretch(result["nir"]), cmap="gray")
    axes[0, 1].set_title("(2) Band NIR (B8)"); axes[0, 1].axis("off")

    axes[0, 2].imshow(_stretch(result["swir2"]), cmap="gray")
    axes[0, 2].set_title("(3) Band SWIR2 (B12)"); axes[0, 2].axis("off")

    im = axes[0, 3].imshow(result["nbr"], cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[0, 3].set_title("(4) NBR"); axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    axes[1, 0].imshow(result["nbr_eq"], cmap="gray")
    axes[1, 0].set_title("(5) NBR sau Histogram Equalize")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result["mask_thresh"], cmap="hot")
    axes[1, 1].set_title(f"(6) Sau Otsu (T={result['threshold']})")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(result["mask_final"], cmap="hot")
    axes[1, 2].set_title("(7) Sau Morphology")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(mask_gt, cmap="hot")
    axes[1, 3].set_title("(8) Ground Truth")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"chi_tiet_{ten_mau}.png", dpi=80,
                bbox_inches="tight")
    plt.close()


def ve_confusion_matrix(cm, out_path):
    """Vẽ confusion matrix dạng heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    classes = ["KhongChay", "Chay"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix · Baseline truyền thống")
    # Ghi số vào từng ô
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE TRUYỀN THỐNG: NBR + EQUALIZE + OTSU + MORPHOLOGY")
    print("=" * 60)

    # Dùng full_image để có ảnh nguyên kích thước
    test_ds = CEMSDataset("test", full_image=True)
    print(f"Số mẫu test: {len(test_ds)}")
    print(f"Thư mục lưu kết quả: {OUTPUT_DIR.resolve()}")

    ket_qua = danh_gia_toan_bo(test_ds, save_examples=5)

    print(f"\nHoàn tất! Mở thư mục '{OUTPUT_DIR}' để xem:")
    print("  - chi_tiet_*.png: minh họa pipeline cho 5 mẫu đầu")
    print("  - confusion_matrix_baseline.png")
    print("  - ket_qua_so.txt: số liệu để đưa vào báo cáo")
