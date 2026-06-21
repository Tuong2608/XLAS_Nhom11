"""
GIAI ĐOẠN 5+ - KIỂM TRA PHÂN LOẠI CÓ/KHÔNG CHÁY CẤP ẢNH (3 MODEL)

Phiên bản cập nhật: so sánh CẢ BA phương pháp trên cùng một bộ patch test:
  - Baseline V3 (NBR + NDWI, rule-based)
  - U-Net (v2)
  - UPerNet (v2)

Vấn đề: dataset CEMS chỉ có ảnh CÓ cháy, không có ảnh KHÔNG cháy.
Để đánh giá "model có phân biệt cháy/không cháy không", ta cần cả hai loại.

Cách giải quyết:
- Patch CÓ cháy : crop vùng có >10% diện tích là cháy từ ảnh test.
- Patch KHÔNG cháy: crop vùng NGOÀI mask cháy (chuyên gia xác nhận không cháy)
  từ cùng các ảnh test.

Cách phân loại từ segmentation:
- Chạy model trên patch -> mask dự đoán.
- Nếu tỉ lệ pixel cháy dự đoán > NGUONG_TI_LE_CHAY thì kết luận "có cháy".
- So với nhãn thật để tính Accuracy / Precision / Recall / F1.

LƯU Ý: cả 3 model chạy trên CÙNG một bộ patch (sinh 1 lần, seed cố định),
nên so sánh là công bằng tuyệt đối.

CÁCH CHẠY:
  python 10_classification_test.py
"""

from pathlib import Path
import csv

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
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
OUTPUT_DIR.mkdir(exist_ok=True)

# Dùng checkpoint v2 cho cả hai model (đồng bộ với báo cáo)
UNET_CHECKPOINT = Path("./ket_qua_unet_v2/best_model.pt")
UPERNET_CHECKPOINT = Path("./ket_qua_upernet_v2/best_model.pt")

# Ngưỡng diện tích để kết luận "ảnh có cháy" (vùng cháy > 1% ảnh)
NGUONG_TI_LE_CHAY = 0.01
# Số patch "không cháy" tự sinh từ mỗi ảnh test
SO_PATCH_KHONG_CHAY_MOI_ANH = 1
PATCH_SIZE = 256
EPS = 1e-8

# Tên 3 model (giữ thứ tự xuyên suốt)
MODEL_NAMES = ["Baseline V3", "U-Net", "UPerNet"]


# ============================================================
# TỰ SINH PATCH
# ============================================================
def crop_patch_khong_chay(image, mask, patch_size=256, max_try=20):
    """Tìm patch KHÔNG chứa pixel cháy. Trả None nếu không tìm được."""
    _, h, w = image.shape
    if h < patch_size or w < patch_size:
        return None, None
    for _ in range(max_try):
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        mask_patch = mask[top:top + patch_size, left:left + patch_size]
        if mask_patch.sum() == 0:
            image_patch = image[:, top:top + patch_size, left:left + patch_size]
            return image_patch.copy(), mask_patch.copy()
    return None, None


def crop_patch_co_chay(image, mask, patch_size=256, max_try=20):
    """Tìm patch chứa cháy đáng kể (>10% diện tích)."""
    _, h, w = image.shape
    if h < patch_size or w < patch_size:
        return None, None
    nguong_min_chay = patch_size * patch_size * 0.1
    for _ in range(max_try):
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        mask_patch = mask[top:top + patch_size, left:left + patch_size]
        if mask_patch.sum() > nguong_min_chay:
            image_patch = image[:, top:top + patch_size, left:left + patch_size]
            return image_patch.copy(), mask_patch.copy()
    return None, None


# ============================================================
# DỰ ĐOÁN
# ============================================================
def predict_v3(image_raw_patch):
    mask_pred, _, _, _ = du_doan_v3(image_raw_patch)
    return mask_pred


def predict_dl(model, image_patch_normalized, device):
    """U-Net hoặc UPerNet trên patch 256x256 (đúng kích thước train)."""
    model.eval()
    with torch.no_grad():
        patch_t = torch.from_numpy(image_patch_normalized).float()
        patch_t = patch_t.unsqueeze(0).to(device)
        logits = model(patch_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (probs > 0.5).astype(np.uint8)


def phan_loai_tu_mask(mask_pred, nguong_ti_le=NGUONG_TI_LE_CHAY):
    """Quy đổi mask segmentation -> nhãn nhị phân cấp ảnh."""
    ti_le_chay = mask_pred.sum() / mask_pred.size
    return int(ti_le_chay > nguong_ti_le), float(ti_le_chay)


# ============================================================
# LOAD MODEL
# ============================================================
def load_models(device):
    for ck in [UNET_CHECKPOINT, UPERNET_CHECKPOINT]:
        if not ck.exists():
            print(f"[LỖI] Thiếu checkpoint: {ck}")
            print("      Hãy chạy 08_train_unet_v2.py và 12_train_upernet_v2.py trước.")
            sys.exit(1)

    ck_u = torch.load(UNET_CHECKPOINT, weights_only=False, map_location=device)
    m_unet = UNet(in_channels=12, num_classes=1).to(device)
    m_unet.load_state_dict(ck_u["model_state_dict"]); m_unet.eval()
    print(f"  U-Net  : best epoch {ck_u.get('epoch','?')}, "
          f"val Dice {ck_u.get('val_dice', float('nan')):.4f}")

    ck_p = torch.load(UPERNET_CHECKPOINT, weights_only=False, map_location=device)
    m_uper = smp.UPerNet(encoder_name="resnet50", encoder_weights=None,
                         in_channels=12, classes=1).to(device)
    m_uper.load_state_dict(ck_p["model_state_dict"]); m_uper.eval()
    print(f"  UPerNet: best epoch {ck_p.get('epoch','?')}, "
          f"val Dice {ck_p.get('val_dice', float('nan')):.4f}")
    return m_unet, m_uper


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)  # cố định để bộ patch tái lập và giống nhau cho 3 model

    print("=" * 60)
    print("KIỂM TRA PHÂN LOẠI CÓ/KHÔNG CHÁY CẤP ẢNH (3 MODEL)")
    print("=" * 60)
    print(f"Ngưỡng diện tích để coi là 'có cháy': {NGUONG_TI_LE_CHAY*100:.1f}%")
    print(f"Thiết bị: {device}")

    print("\nĐang load model...")
    model_unet, model_upernet = load_models(device)

    test_ds = CEMSDataset("test", full_image=True)
    print(f"\nTổng số ảnh test gốc: {len(test_ds)}")

    # ----- Sinh bộ patch (1 lần, dùng chung cho cả 3 model) -----
    samples_test = []
    print("Đang tạo test set có cháy + không cháy...")
    for idx in range(len(test_ds)):
        _, _, path = test_ds[idx]
        sample_dir = Path(path)
        s2_file = next(sample_dir.glob("*_S2L2A.tif"))
        del_file = next(sample_dir.glob("*_DEL.tif"))
        with rasterio.open(s2_file) as src:
            image_raw = src.read().astype(np.float32)
        with rasterio.open(del_file) as src:
            mask_gt = src.read(1).astype(np.uint8)

        img_co, _ = crop_patch_co_chay(image_raw, mask_gt, PATCH_SIZE)
        if img_co is not None:
            samples_test.append({"image_raw": img_co,
                                 "image_norm": chuan_hoa(img_co),
                                 "label": 1, "source": sample_dir.name})
        for _ in range(SO_PATCH_KHONG_CHAY_MOI_ANH):
            img_khong, _ = crop_patch_khong_chay(image_raw, mask_gt, PATCH_SIZE)
            if img_khong is not None:
                samples_test.append({"image_raw": img_khong,
                                     "image_norm": chuan_hoa(img_khong),
                                     "label": 0, "source": sample_dir.name})

    n_fire = sum(1 for s in samples_test if s["label"] == 1)
    n_no_fire = sum(1 for s in samples_test if s["label"] == 0)
    print(f"  Patch CÓ cháy   : {n_fire}")
    print(f"  Patch KHÔNG cháy: {n_no_fire}")
    print(f"  Tổng patch test : {len(samples_test)}")

    # ----- Chạy 3 model -----
    print("\nĐang chạy 3 model trên test set...")
    y_true = []
    y_pred = {m: [] for m in MODEL_NAMES}

    for s in samples_test:
        y_true.append(s["label"])

        mask_v3 = predict_v3(s["image_raw"])
        y_pred["Baseline V3"].append(phan_loai_tu_mask(mask_v3)[0])

        mask_unet = predict_dl(model_unet, s["image_norm"], device)
        y_pred["U-Net"].append(phan_loai_tu_mask(mask_unet)[0])

        mask_uper = predict_dl(model_upernet, s["image_norm"], device)
        y_pred["UPerNet"].append(phan_loai_tu_mask(mask_uper)[0])

    # ----- Metric -----
    print("\n" + "=" * 60)
    print("KẾT QUẢ CLASSIFICATION CẤP ẢNH (CÓ/KHÔNG CHÁY)")
    print("=" * 60)
    print(f"{'Model':<14}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    print("-" * 60)

    ket_qua = {}
    for m in MODEL_NAMES:
        acc = accuracy_score(y_true, y_pred[m])
        prec = precision_score(y_true, y_pred[m], zero_division=0)
        rec = recall_score(y_true, y_pred[m], zero_division=0)
        f1 = f1_score(y_true, y_pred[m], zero_division=0)
        cm = confusion_matrix(y_true, y_pred[m])
        ket_qua[m] = dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm)
        print(f"{m:<14}{acc*100:>9.2f}%{prec*100:>10.2f}%"
              f"{rec*100:>8.2f}%{f1*100:>8.2f}%")
        ve_cm(cm, m, OUTPUT_DIR / f"cm_classification_{m.replace(' ', '_')}.png")

    # ----- Lưu txt -----
    with open(OUTPUT_DIR / "ket_qua_classification.txt", "w",
              encoding="utf-8") as f:
        f.write("CLASSIFICATION CẤP ẢNH (CÓ/KHÔNG CHÁY) - 3 MODEL\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tổng patch test : {len(samples_test)}  "
                f"(có cháy {n_fire}, không cháy {n_no_fire})\n")
        f.write(f"Ngưỡng diện tích: {NGUONG_TI_LE_CHAY*100:.1f}%\n\n")
        f.write(f"{'Model':<14}{'Accuracy':>10}{'Precision':>11}"
                f"{'Recall':>9}{'F1':>9}\n")
        f.write("-" * 54 + "\n")
        for m in MODEL_NAMES:
            r = ket_qua[m]
            f.write(f"{m:<14}{r['acc']*100:>9.2f}%{r['prec']*100:>10.2f}%"
                    f"{r['rec']*100:>8.2f}%{r['f1']*100:>8.2f}%\n")
        f.write("\nConfusion matrix (GT hàng, Pred cột; 0=KhongChay, 1=Chay):\n")
        for m in MODEL_NAMES:
            cm = ket_qua[m]["cm"]
            f.write(f"\n{m}:\n")
            f.write(f"  GT=KhongChay : {cm[0,0]:>4d}  {cm[0,1]:>4d}\n")
            f.write(f"  GT=Chay      : {cm[1,0]:>4d}  {cm[1,1]:>4d}\n")

    # ----- Lưu CSV gọn cho báo cáo -----
    with open(OUTPUT_DIR / "bang_classification.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Accuracy(%)", "Precision(%)", "Recall(%)", "F1(%)"])
        for m in MODEL_NAMES:
            r = ket_qua[m]
            w.writerow([m, f"{r['acc']*100:.2f}", f"{r['prec']*100:.2f}",
                        f"{r['rec']*100:.2f}", f"{r['f1']*100:.2f}"])

    print(f"\nXong! Mở '{OUTPUT_DIR}' để xem:")
    print("  - ket_qua_classification.txt (bảng + confusion matrix)")
    print("  - bang_classification.csv    (bảng cho mục 5.4)")
    print("  - cm_classification_*.png    (3 confusion matrix)")


def ve_cm(cm, ten, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["KhongChay", "Chay"])
    ax.set_yticklabels(["KhongChay", "Chay"])
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix · {ten} (Classification)")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
