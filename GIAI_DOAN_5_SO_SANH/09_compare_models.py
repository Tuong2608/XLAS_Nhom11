"""
GIAI ĐOẠN 5 - SO SÁNH TRỰC QUAN 3 MODEL (V3 / U-Net v2 / UPerNet v2)
Đề tài: Phát hiện cháy rừng trên ảnh vệ tinh

File này tạo các hình so sánh chất lượng cao cho báo cáo:
- Mỗi mẫu: lưới 2x3 ô (gần vuông, không bị dài):
      Hàng trên : Input RGB | NBR | Ground Truth
      Hàng dưới : Baseline V3 | U-Net | UPerNet
- Tự động sinh bảng số liệu so sánh 3 model (IoU/Dice trung bình).
- Chọn các mẫu minh họa: cải thiện rõ nhất của học sâu so với baseline,
  kèm nhãn mức độ cháy để dễ chọn ví dụ cháy nhỏ / cháy lớn cho báo cáo.

CÁCH CHẠY:
  python 09_compare_models.py

OUTPUT:
  ket_qua_so_sanh/
    so_sanh_*.png      : hình so sánh 2x3 ô (cho báo cáo)
    bang_so_sanh.txt   : bảng số liệu
    bang_so_sanh.csv   : CSV (gồm cột muc_do để lọc cháy nhỏ/lớn)
"""

from pathlib import Path
import csv

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))

dataloader = import_module("06_dataloader")
baseline_v3 = import_module("07c_baseline_v3")
unet_module = import_module("08_train_unet")

CEMSDataset = dataloader.CEMSDataset
du_doan_v3 = baseline_v3.du_doan_v3
UNet = unet_module.UNet


# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_DIR = Path("./ket_qua_so_sanh")
OUTPUT_DIR.mkdir(exist_ok=True)

# Checkpoint v2 (đồng bộ với báo cáo)
UNET_CHECKPOINT = Path("./ket_qua_unet_v2/best_model.pt")
UPERNET_CHECKPOINT = Path("./ket_qua_upernet_v2/best_model.pt")

NUM_VISUALIZATIONS = 8    # số mẫu vẽ hình so sánh
PATCH_SIZE = 256
EPS = 1e-8

MODEL_NAMES = ["Baseline V3", "U-Net", "UPerNet"]


# ============================================================
# UTILITIES
# ============================================================
def tinh_iou_dice(pred, gt):
    pred_b = pred.astype(bool); gt_b = gt.astype(bool)
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    iou = inter / (union + EPS)
    dice = 2 * inter / (pred_b.sum() + gt_b.sum() + EPS)
    return iou, dice


def _stretch(arr):
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def muc_do_chay(ti_le):
    """Phân tầng mức độ cháy theo tỉ lệ pixel cháy của GT."""
    if ti_le <= 0:      return "Khong chay"
    if ti_le < 0.01:    return "Rat it (<1%)"
    if ti_le < 0.10:    return "It (1-10%)"
    if ti_le < 0.30:    return "Vua (10-30%)"
    if ti_le < 0.50:    return "Nhieu (30-50%)"
    return "Rat nhieu (>50%)"


# ============================================================
# DỰ ĐOÁN HỌC SÂU TRÊN ẢNH NGUYÊN KÍCH THƯỚC (sliding window)
# ============================================================
def du_doan_dl_full(model, image_normalized, device, patch_size=256):
    """U-Net / UPerNet: sliding window patch 256, chồng lấp 50%, ghép mask."""
    _, h, w = image_normalized.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_padded = np.pad(image_normalized, ((0, 0), (0, pad_h), (0, pad_w)),
                          mode="reflect")
    _, h_pad, w_pad = image_padded.shape
    full_mask = np.zeros((h_pad, w_pad), dtype=np.float32)
    count = np.zeros((h_pad, w_pad), dtype=np.float32)
    stride = patch_size // 2

    model.eval()
    with torch.no_grad():
        for top in range(0, h_pad - patch_size + 1, stride):
            for left in range(0, w_pad - patch_size + 1, stride):
                patch = image_padded[:, top:top + patch_size,
                                     left:left + patch_size]
                patch_t = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                probs = torch.sigmoid(model(patch_t)).squeeze().cpu().numpy()
                full_mask[top:top + patch_size, left:left + patch_size] += probs
                count[top:top + patch_size, left:left + patch_size] += 1

    full_mask = full_mask / np.maximum(count, 1)
    return (full_mask[:h, :w] > 0.5).astype(np.uint8)


# ============================================================
# VẼ HÌNH SO SÁNH (lưới 2x3, gần vuông)
# ============================================================
def ve_so_sanh(image_raw, mask_gt, masks, metrics, ten):
    """
    masks  : dict {ten_model: mask}
    metrics: dict {ten_model: (iou, dice)}
    Lưới 2x3: [RGB, NBR, GT] / [V3, U-Net, UPerNet]
    """
    rgb = np.stack([_stretch(image_raw[3]), _stretch(image_raw[2]),
                    _stretch(image_raw[1])], axis=-1)
    nir, swir2 = image_raw[7], image_raw[11]
    nbr = np.clip((nir - swir2) / (nir + swir2 + EPS), -1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    fig.suptitle(f"So sánh 3 phương pháp · {ten}", fontsize=13, y=0.99)

    axes[0, 0].imshow(rgb); axes[0, 0].set_title("Input RGB")
    im = axes[0, 1].imshow(nbr, cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[0, 1].set_title("NBR (đặc trưng đầu vào)")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    axes[0, 2].imshow(mask_gt, cmap="hot"); axes[0, 2].set_title("Ground Truth")

    for col, name in enumerate(MODEL_NAMES):
        iou, dice = metrics[name]
        axes[1, col].imshow(masks[name], cmap="hot")
        axes[1, col].set_title(f"{name}\nIoU={iou:.3f}  Dice={dice:.3f}")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"so_sanh_{ten}.png", dpi=110, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print("SO SÁNH 3 PHƯƠNG PHÁP (V3 / U-Net v2 / UPerNet v2) TRÊN TEST SET")
    print("=" * 64)
    print(f"Thiết bị: {device}")

    for ck in [UNET_CHECKPOINT, UPERNET_CHECKPOINT]:
        if not ck.exists():
            print(f"[LỖI] Thiếu checkpoint: {ck}")
            return

    print("\nĐang load model...")
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

    test_ds = CEMSDataset("test", full_image=True)
    print(f"\nSố mẫu test: {len(test_ds)}")

    results = []
    cache = {}   # idx -> (image_raw, mask_gt, masks)
    print("\nĐang chạy 3 model trên test set...")
    for idx in range(len(test_ds)):
        image_t, mask_gt_t, path = test_ds[idx]
        image = image_t.numpy()
        mask_gt = mask_gt_t.numpy().astype(np.uint8)
        sample_dir = Path(path)
        s2_file = next(sample_dir.glob("*_S2L2A.tif"))
        with rasterio.open(s2_file) as src:
            image_raw = src.read().astype(np.float32)

        mask_v3, _, _, _ = du_doan_v3(image_raw)
        mask_unet = du_doan_dl_full(m_unet, image, device, PATCH_SIZE)
        mask_uper = du_doan_dl_full(m_uper, image, device, PATCH_SIZE)
        masks = {"Baseline V3": mask_v3, "U-Net": mask_unet, "UPerNet": mask_uper}

        ti_le_gt = mask_gt.mean()
        iou_v3, dice_v3 = tinh_iou_dice(mask_v3, mask_gt)
        iou_u, dice_u = tinh_iou_dice(mask_unet, mask_gt)
        iou_p, dice_p = tinh_iou_dice(mask_uper, mask_gt)

        results.append({
            "idx": idx, "name": sample_dir.name,
            "muc_do": muc_do_chay(ti_le_gt), "gt_ratio": round(float(ti_le_gt), 4),
            "iou_v3": iou_v3, "dice_v3": dice_v3,
            "iou_unet": iou_u, "dice_unet": dice_u,
            "iou_upernet": iou_p, "dice_upernet": dice_p,
            "dl_improve": max(iou_u, iou_p) - iou_v3,
        })
        cache[idx] = (image_raw, mask_gt, masks,
                      {"Baseline V3": (iou_v3, dice_v3),
                       "U-Net": (iou_u, dice_u), "UPerNet": (iou_p, dice_p)})

        if idx < 3 or (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(test_ds)}] {sample_dir.name[:22]:22s} "
                  f"V3={iou_v3:.3f} U={iou_u:.3f} UPer={iou_p:.3f}")

    # ----- Tổng kết (chỉ tính trên ảnh có GT>0 để IoU có nghĩa) -----
    valid = [r for r in results if r["gt_ratio"] > 0]
    def mean(k, rs): return float(np.mean([r[k] for r in rs]))
    print("\n" + "=" * 64)
    print(f"TỔNG KẾT (IoU/Dice trung bình theo ảnh, {len(valid)} ảnh GT>0)")
    print("=" * 64)
    print(f"{'Model':>14}{'IoU':>10}{'Dice':>10}")
    print("-" * 34)
    for name, ki, kd in [("Baseline V3", "iou_v3", "dice_v3"),
                          ("U-Net", "iou_unet", "dice_unet"),
                          ("UPerNet", "iou_upernet", "dice_upernet")]:
        print(f"{name:>14}{mean(ki, valid):>10.4f}{mean(kd, valid):>10.4f}")

    with open(OUTPUT_DIR / "bang_so_sanh.txt", "w", encoding="utf-8") as f:
        f.write("SO SÁNH 3 PHƯƠNG PHÁP - IoU/Dice trung bình theo ảnh\n")
        f.write("=" * 50 + "\n")
        f.write(f"Số ảnh GT>0: {len(valid)}\n\n")
        f.write(f"{'Model':<14}{'IoU':<10}{'Dice':<10}\n")
        for name, ki, kd in [("Baseline V3", "iou_v3", "dice_v3"),
                              ("U-Net", "iou_unet", "dice_unet"),
                              ("UPerNet", "iou_upernet", "dice_upernet")]:
            f.write(f"{name:<14}{mean(ki, valid):<10.4f}{mean(kd, valid):<10.4f}\n")

    with open(OUTPUT_DIR / "bang_so_sanh.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # ----- Chọn mẫu minh họa: học sâu cải thiện rõ nhất so với V3 -----
    chosen = sorted(results, key=lambda r: -r["dl_improve"])[:NUM_VISUALIZATIONS]
    print(f"\nĐang vẽ {len(chosen)} hình so sánh (2x3) ...")
    print("  Gợi ý chọn cho báo cáo: 1 ảnh cháy nhỏ (U-Net thường thắng) + "
          "1 ảnh cháy lớn (UPerNet thường thắng).")
    for r in chosen:
        image_raw, mask_gt, masks, metrics = cache[r["idx"]]
        ve_so_sanh(image_raw, mask_gt, masks, metrics, r["name"])
        print(f"  - {r['name']:24s} [{r['muc_do']}]  "
              f"V3={r['iou_v3']:.3f} U={r['iou_unet']:.3f} UPer={r['iou_upernet']:.3f}")

    print(f"\nXong! Mở '{OUTPUT_DIR}':")
    print("  - so_sanh_*.png   (hình 2x3 cho báo cáo)")
    print("  - bang_so_sanh.csv (cột 'muc_do' giúp chọn ví dụ cháy nhỏ/lớn)")


if __name__ == "__main__":
    main()
