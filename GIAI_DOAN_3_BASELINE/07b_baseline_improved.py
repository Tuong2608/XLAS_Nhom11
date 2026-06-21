"""
GIAI ĐOẠN 3 - BASELINE TRUYỀN THỐNG (PHIÊN BẢN V2 - CẢI TIẾN)

Phát hiện từ V1 (file 07_baseline_traditional.py):
- Otsu thresholding cho Precision quá thấp (0.15) vì tham lam gán nhãn cháy
- Lý do: Otsu tự động tìm ngưỡng tách 2 phần nhưng không biết bối cảnh
- Các pixel đô thị, nông nghiệp, đất trống có NBR thấp tương tự cháy

CẢI TIẾN V2:
- BỎ Otsu, dùng NGƯỠNG CỐ ĐỊNH dựa trên kiến thức viễn thám
- Trong tài liệu USGS, ngưỡng NBR < -0.1 đến -0.25 thường dùng cho cháy
- File này thử NHIỀU ngưỡng, chọn cái cho F1 cao nhất trên validation set,
  rồi báo cáo kết quả trên test set với ngưỡng đó (đúng quy trình ML)

PSEUDO-CODE V2:
  function FindBestThreshold(val_set, candidates):
      best_thr, best_f1 = None, 0
      for thr in candidates:
          f1 = EvaluateAll(val_set, threshold=thr)
          if f1 > best_f1: best_thr, best_f1 = thr, f1
      return best_thr

  function PredictWithFixedThreshold(image, threshold):
      nbr = ComputeNBR(image)  // [-1, 1]
      mask = (nbr < threshold)
      mask = MorphologyOpen(mask, k=3)
      mask = MorphologyClose(mask, k=5)
      return mask

CÁCH CHẠY:
  python 07b_baseline_improved.py
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
OUTPUT_DIR = Path("./ket_qua_baseline_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

B8_NIR_IDX = 7
B12_SWIR2_IDX = 11
EPS = 1e-8

# Các ngưỡng để thử trên validation set
NGUONG_THU = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40]


# ============================================================
# PIPELINE
# ============================================================
def tinh_nbr_co_loc(image: np.ndarray):
    """Tách kênh + lọc Gaussian + tính NBR. Trả về NBR [-1, 1]."""
    nir = image[B8_NIR_IDX]
    swir2 = image[B12_SWIR2_IDX]
    nir = cv2.GaussianBlur(nir, (5, 5), 0)
    swir2 = cv2.GaussianBlur(swir2, (5, 5), 0)
    nbr = (nir - swir2) / (nir + swir2 + EPS)
    return np.clip(nbr, -1, 1)


def du_doan_voi_nguong(image: np.ndarray, threshold: float):
    """Pipeline đầy đủ với ngưỡng cố định cho trước."""
    nbr = tinh_nbr_co_loc(image)
    mask_raw = (nbr < threshold).astype(np.uint8)
    # Morphology làm sạch
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask, nbr


# ============================================================
# ĐÁNH GIÁ
# ============================================================
def danh_gia_ngưỡng(dataset, threshold, subsample_step=4):
    """Chạy 1 ngưỡng trên cả dataset, trả về precision/recall/f1/iou."""
    all_pred, all_gt = [], []
    ious = []
    for idx in range(len(dataset)):
        image, mask_gt, _ = dataset[idx]
        image = image.numpy() if hasattr(image, "numpy") else image
        mask_gt = mask_gt.numpy() if hasattr(mask_gt, "numpy") else mask_gt
        mask_gt = mask_gt.astype(np.uint8)
        mask_pred, _ = du_doan_voi_nguong(image, threshold)

        # IoU per image
        inter = ((mask_pred == 1) & (mask_gt == 1)).sum()
        union = ((mask_pred == 1) | (mask_gt == 1)).sum()
        ious.append(inter / (union + EPS))

        # Pixel-level cho precision/recall/f1
        all_pred.append(mask_pred[::subsample_step, ::subsample_step].flatten())
        all_gt.append(mask_gt[::subsample_step, ::subsample_step].flatten())

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    return {
        "threshold": threshold,
        "iou": float(np.mean(ious)),
        "precision": float(precision_score(all_gt, all_pred, zero_division=0)),
        "recall": float(recall_score(all_gt, all_pred, zero_division=0)),
        "f1": float(f1_score(all_gt, all_pred, zero_division=0)),
    }


def tim_nguong_tot_nhat(val_ds):
    """Quét các ngưỡng trên val set, chọn ngưỡng F1 cao nhất."""
    print(f"\nQuét {len(NGUONG_THU)} ngưỡng trên tập VALIDATION...")
    print(f"{'Threshold':>10s} {'IoU':>8s} {'Precision':>10s} "
          f"{'Recall':>8s} {'F1':>8s}")
    print("-" * 50)
    ket_qua = []
    for thr in NGUONG_THU:
        r = danh_gia_ngưỡng(val_ds, thr)
        ket_qua.append(r)
        print(f"{thr:>10.2f} {r['iou']:>8.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f}")
    best = max(ket_qua, key=lambda r: r["f1"])
    print(f"\n=> Ngưỡng tốt nhất: {best['threshold']:.2f} (F1={best['f1']:.4f})")
    return best, ket_qua


def danh_gia_va_luu_minh_hoa(test_ds, best_threshold, save_examples=5):
    """Đánh giá cuối trên test set + lưu hình minh họa."""
    print(f"\nĐánh giá cuối trên TEST set với ngưỡng {best_threshold:.2f}...")
    all_pred, all_gt = [], []
    ious, dices = [], []
    start = time.time()

    for idx in range(len(test_ds)):
        image, mask_gt, path = test_ds[idx]
        image = image.numpy() if hasattr(image, "numpy") else image
        mask_gt = mask_gt.numpy() if hasattr(mask_gt, "numpy") else mask_gt
        mask_gt = mask_gt.astype(np.uint8)
        mask_pred, nbr = du_doan_voi_nguong(image, best_threshold)

        inter = ((mask_pred == 1) & (mask_gt == 1)).sum()
        union = ((mask_pred == 1) | (mask_gt == 1)).sum()
        ious.append(inter / (union + EPS))
        dices.append(2 * inter / (mask_pred.sum() + mask_gt.sum() + EPS))

        all_pred.append(mask_pred[::4, ::4].flatten())
        all_gt.append(mask_gt[::4, ::4].flatten())

        if idx < save_examples:
            ve_minh_hoa(image, mask_gt, mask_pred, nbr,
                        Path(path).name, ious[-1], dices[-1],
                        best_threshold)

    elapsed = time.time() - start
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    precision = precision_score(all_gt, all_pred, zero_division=0)
    recall = recall_score(all_gt, all_pred, zero_division=0)
    f1 = f1_score(all_gt, all_pred, zero_division=0)
    cm = confusion_matrix(all_gt, all_pred)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ BASELINE V2 (ngưỡng {best_threshold:.2f})")
    print("=" * 60)
    print(f"Thời gian: {elapsed:.1f}s ({elapsed/len(test_ds):.2f}s/mẫu)")
    print(f"IoU       : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Dice      : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"\nConfusion matrix:")
    print(f"                  Pred=KhongChay   Pred=Chay")
    print(f"  GT=KhongChay    {cm[0,0]:>12d}   {cm[0,1]:>9d}")
    print(f"  GT=Chay         {cm[1,0]:>12d}   {cm[1,1]:>9d}")

    ve_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix_v2.png")

    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write(f"BASELINE V2 - Ngưỡng tốt nhất: {best_threshold:.2f}\n")
        f.write("=" * 40 + "\n")
        f.write(f"IoU      : {np.mean(ious):.4f} ± {np.std(ious):.4f}\n")
        f.write(f"Dice     : {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")


def _stretch(arr):
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def ve_minh_hoa(image, mask_gt, mask_pred, nbr, ten, iou, dice, thr):
    rgb = np.stack([_stretch(image[3]), _stretch(image[2]),
                    _stretch(image[1])], axis=-1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Baseline V2 (thr={thr:.2f}) · {ten} · "
                 f"IoU={iou:.3f} · Dice={dice:.3f}", fontsize=13)
    axes[0].imshow(rgb); axes[0].set_title("Input RGB"); axes[0].axis("off")
    im = axes[1].imshow(nbr, cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[1].set_title("NBR"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(mask_pred, cmap="hot")
    axes[2].set_title("Prediction V2"); axes[2].axis("off")
    axes[3].imshow(mask_gt, cmap="hot")
    axes[3].set_title("Ground Truth"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"v2_{ten}.png", dpi=80, bbox_inches="tight")
    plt.close()


def ve_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    classes = ["KhongChay", "Chay"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix · Baseline V2")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color=color, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def ve_so_sanh_nguong(ket_qua_quet, out_path):
    """Vẽ biểu đồ Precision/Recall/F1 theo các ngưỡng - đẹp cho báo cáo."""
    thrs = [r["threshold"] for r in ket_qua_quet]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thrs, [r["precision"] for r in ket_qua_quet], "o-",
            label="Precision")
    ax.plot(thrs, [r["recall"] for r in ket_qua_quet], "s-", label="Recall")
    ax.plot(thrs, [r["f1"] for r in ket_qua_quet], "^-", label="F1")
    ax.plot(thrs, [r["iou"] for r in ket_qua_quet], "d-", label="IoU")
    ax.set_xlabel("Ngưỡng NBR")
    ax.set_ylabel("Metric value")
    ax.set_title("Ảnh hưởng của ngưỡng NBR đến các chỉ số (trên VAL set)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE V2: NGƯỠNG NBR CỐ ĐỊNH + MORPHOLOGY")
    print("=" * 60)

    val_ds = CEMSDataset("val", full_image=True)
    test_ds = CEMSDataset("test", full_image=True)
    print(f"Val: {len(val_ds)}, Test: {len(test_ds)} mẫu")

    # Bước 1: quét ngưỡng trên val
    best, ket_qua_quet = tim_nguong_tot_nhat(val_ds)
    ve_so_sanh_nguong(ket_qua_quet, OUTPUT_DIR / "anh_huong_nguong.png")

    # Bước 2: đánh giá cuối trên test với ngưỡng tốt nhất
    danh_gia_va_luu_minh_hoa(test_ds, best["threshold"])

    print(f"\nXong! Mở thư mục '{OUTPUT_DIR}' để xem kết quả.")
