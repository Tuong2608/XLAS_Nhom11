"""
GIAI ĐOẠN 5+ - KIỂM TRA PHÂN LOẠI CÓ/KHÔNG CHÁY (PHIÊN BẢN CÓ LƯU)

Khác phiên bản trước: lưu các patch test ra đĩa để
- Tái lập kết quả
- Làm bằng chứng cho thuyết trình
- Tái dùng cho web app demo sau

CẤU TRÚC LƯU:
  ket_qua_classification/
    patches/
      fire/         <- 65 patch CÓ cháy
        fire_001_EMSR207_AOI01_01.tif    (12 band gốc)
        fire_001_EMSR207_AOI01_01.png    (preview RGB)
        ...
      no_fire/      <- 91 patch KHÔNG cháy
        nofire_001_EMSR207_AOI01_01.tif
        nofire_001_EMSR207_AOI01_01.png
        ...
    danh_sach_test.csv       <- bảng metadata
    ket_qua_classification.txt
    cm_*.png

CÁCH CHẠY:
  python 10b_classification_save.py
"""

from pathlib import Path
import csv

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))

dataloader = import_module("06_dataloader")
baseline_v3 = import_module("07c_baseline_v3")
unet_module = import_module("08_train_unet")

CEMSDataset = dataloader.CEMSDataset
chuan_hoa = dataloader.chuan_hoa
du_doan_v3 = baseline_v3.du_doan_v3
UNet = unet_module.UNet


# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_DIR = Path("./ket_qua_classification")
PATCH_DIR = OUTPUT_DIR / "patches"
FIRE_DIR = PATCH_DIR / "fire"
NOFIRE_DIR = PATCH_DIR / "no_fire"
for d in [OUTPUT_DIR, PATCH_DIR, FIRE_DIR, NOFIRE_DIR]:
    d.mkdir(exist_ok=True, parents=True)

UNET_CHECKPOINT = Path("./ket_qua_unet/best_model.pt")
NGUONG_TI_LE_CHAY = 0.01
SO_PATCH_KHONG_CHAY_MOI_ANH = 1
PATCH_SIZE = 256
EPS = 1e-8


# ============================================================
# CROP PATCH (giống file 10 cũ nhưng trả thêm vị trí để lưu)
# ============================================================
def crop_patch_khong_chay(image, mask, patch_size=256, max_try=20):
    _, h, w = image.shape
    if h < patch_size or w < patch_size:
        return None, None, None
    for _ in range(max_try):
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        mask_patch = mask[top:top + patch_size, left:left + patch_size]
        if mask_patch.sum() == 0:
            image_patch = image[:, top:top + patch_size,
                                left:left + patch_size].copy()
            return image_patch, mask_patch.copy(), (top, left)
    return None, None, None


def crop_patch_co_chay(image, mask, patch_size=256, max_try=20):
    _, h, w = image.shape
    if h < patch_size or w < patch_size:
        return None, None, None
    nguong = patch_size * patch_size * 0.1
    for _ in range(max_try):
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        mask_patch = mask[top:top + patch_size, left:left + patch_size]
        if mask_patch.sum() > nguong:
            image_patch = image[:, top:top + patch_size,
                                left:left + patch_size].copy()
            return image_patch, mask_patch.copy(), (top, left)
    return None, None, None


# ============================================================
# LƯU PATCH RA ĐĨA
# ============================================================
def luu_patch(image_12band, mask, output_path_tif, output_path_png):
    """
    Lưu patch:
    - .tif: 12 band gốc để model load lại sau
    - .png: preview RGB để mở xem nhanh bằng mắt thường
    """
    # Lưu file TIF đa phổ
    with rasterio.open(
        output_path_tif, "w",
        driver="GTiff",
        height=image_12band.shape[1],
        width=image_12band.shape[2],
        count=image_12band.shape[0],
        dtype="float32",
    ) as dst:
        dst.write(image_12band)

    # Lưu file PNG preview (RGB từ band 4,3,2)
    def _stretch(arr):
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / (hi - lo + EPS), 0, 1)
    rgb = np.stack([_stretch(image_12band[3]),
                    _stretch(image_12band[2]),
                    _stretch(image_12band[1])], axis=-1)
    plt.imsave(output_path_png, rgb)


def predict_v3(image_raw_patch):
    mask, _, _, _ = du_doan_v3(image_raw_patch)
    return mask


def predict_unet(model, image_patch_normalized, device):
    model.eval()
    with torch.no_grad():
        patch_t = torch.from_numpy(image_patch_normalized).float()
        patch_t = patch_t.unsqueeze(0).to(device)
        logits = model(patch_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (probs > 0.5).astype(np.uint8)


def phan_loai_tu_mask(mask_pred, nguong=NGUONG_TI_LE_CHAY):
    ti_le = mask_pred.sum() / mask_pred.size
    return int(ti_le > nguong), float(ti_le)


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)

    print("=" * 60)
    print("CLASSIFICATION CẤP ẢNH (CÓ LƯU PATCH)")
    print("=" * 60)

    # Load U-Net
    checkpoint = torch.load(UNET_CHECKPOINT, weights_only=False,
                             map_location=device)
    model = UNet(in_channels=12, num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = CEMSDataset("test", full_image=True)
    print(f"Số ảnh test gốc: {len(test_ds)}")
    print(f"Thư mục lưu patch: {PATCH_DIR.resolve()}")

    # Tạo + lưu + chạy model song song
    print("\nĐang tạo test set, lưu và chạy model...")
    metadata = []   # list các dict để lưu CSV
    y_true, y_pred_v3, y_pred_unet = [], [], []
    fire_idx = 0
    nofire_idx = 0

    for sample_idx in range(len(test_ds)):
        _, _, path = test_ds[sample_idx]
        sample_dir = Path(path)
        s2_file = next(sample_dir.glob("*_S2L2A.tif"))
        del_file = next(sample_dir.glob("*_DEL.tif"))

        with rasterio.open(s2_file) as src:
            image_raw = src.read().astype(np.float32)
        with rasterio.open(del_file) as src:
            mask_gt = src.read(1).astype(np.uint8)

        ten_anh = sample_dir.name

        # ----- Patch CÓ cháy -----
        img_co, _, pos_co = crop_patch_co_chay(image_raw, mask_gt, PATCH_SIZE)
        if img_co is not None:
            fire_idx += 1
            ten_file = f"fire_{fire_idx:03d}_{ten_anh}"
            luu_patch(img_co, _,
                      FIRE_DIR / f"{ten_file}.tif",
                      FIRE_DIR / f"{ten_file}.png")

            mask_v3 = predict_v3(img_co)
            mask_unet = predict_unet(model, chuan_hoa(img_co), device)
            pred_v3, _ = phan_loai_tu_mask(mask_v3)
            pred_unet, _ = phan_loai_tu_mask(mask_unet)

            y_true.append(1)
            y_pred_v3.append(pred_v3)
            y_pred_unet.append(pred_unet)
            metadata.append({
                "filename": f"{ten_file}.tif",
                "type": "fire",
                "label": 1,
                "source": ten_anh,
                "position": f"{pos_co[0]},{pos_co[1]}",
                "pred_v3": pred_v3,
                "pred_unet": pred_unet,
            })

        # ----- Patch KHÔNG cháy -----
        for _ in range(SO_PATCH_KHONG_CHAY_MOI_ANH):
            img_khong, _, pos_khong = crop_patch_khong_chay(image_raw, mask_gt,
                                                             PATCH_SIZE)
            if img_khong is not None:
                nofire_idx += 1
                ten_file = f"nofire_{nofire_idx:03d}_{ten_anh}"
                luu_patch(img_khong, _,
                          NOFIRE_DIR / f"{ten_file}.tif",
                          NOFIRE_DIR / f"{ten_file}.png")

                mask_v3 = predict_v3(img_khong)
                mask_unet = predict_unet(model, chuan_hoa(img_khong),
                                          device)
                pred_v3, _ = phan_loai_tu_mask(mask_v3)
                pred_unet, _ = phan_loai_tu_mask(mask_unet)

                y_true.append(0)
                y_pred_v3.append(pred_v3)
                y_pred_unet.append(pred_unet)
                metadata.append({
                    "filename": f"{ten_file}.tif",
                    "type": "no_fire",
                    "label": 0,
                    "source": ten_anh,
                    "position": f"{pos_khong[0]},{pos_khong[1]}",
                    "pred_v3": pred_v3,
                    "pred_unet": pred_unet,
                })

        if (sample_idx + 1) % 20 == 0:
            print(f"  Đã xử lý {sample_idx + 1}/{len(test_ds)} ảnh gốc...")

    # Lưu CSV metadata
    with open(OUTPUT_DIR / "danh_sach_test.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    n_fire = sum(1 for m in metadata if m["label"] == 1)
    n_no_fire = sum(1 for m in metadata if m["label"] == 0)
    print(f"\n  Đã lưu {n_fire} patch CÓ cháy vào {FIRE_DIR}")
    print(f"  Đã lưu {n_no_fire} patch KHÔNG cháy vào {NOFIRE_DIR}")
    print(f"  Metadata: {OUTPUT_DIR}/danh_sach_test.csv")

    # In kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ CLASSIFICATION CẤP ẢNH")
    print("=" * 60)

    with open(OUTPUT_DIR / "ket_qua_classification.txt", "w",
              encoding="utf-8") as f:
        f.write("CLASSIFICATION CẤP ẢNH (CÓ/KHÔNG CHÁY)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tổng patch test : {len(metadata)}\n")
        f.write(f"  Có cháy   : {n_fire}\n")
        f.write(f"  Không cháy: {n_no_fire}\n\n")

        for ten, y_pred in [("Baseline V3", y_pred_v3),
                             ("U-Net", y_pred_unet)]:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            print(f"\n--- {ten} ---")
            print(f"  Accuracy : {acc*100:.2f}%")
            print(f"  Precision: {prec*100:.2f}%")
            print(f"  Recall   : {rec*100:.2f}%")
            print(f"  F1-score : {f1*100:.2f}%")
            print(f"  Confusion matrix:")
            print(f"    GT=KhongChay -> Pred KhongChay: {cm[0,0]}, "
                  f"Pred Chay: {cm[0,1]}")
            print(f"    GT=Chay      -> Pred KhongChay: {cm[1,0]}, "
                  f"Pred Chay: {cm[1,1]}")

            f.write(f"--- {ten} ---\n")
            f.write(f"  Accuracy : {acc*100:.2f}%\n")
            f.write(f"  Precision: {prec*100:.2f}%\n")
            f.write(f"  Recall   : {rec*100:.2f}%\n")
            f.write(f"  F1-score : {f1*100:.2f}%\n\n")

            ve_cm(cm, ten,
                  OUTPUT_DIR / f"cm_classification_{ten.replace(' ','_')}.png")

    print(f"\nXong! Kết quả ở '{OUTPUT_DIR}'.")
    print("Mở thư mục 'patches' để xem ảnh test (có cả .tif và .png).")


def ve_cm(cm, ten, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["KhongChay", "Chay"])
    ax.set_yticklabels(["KhongChay", "Chay"])
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix · {ten}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
