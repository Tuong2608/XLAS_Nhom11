"""
THỐNG KÊ DIỆN TÍCH CHÁY TRÊN TEST SET (3 MODEL)

So sánh khả năng ước tính diện tích cháy của 3 model:
- Baseline V3 (NBR + NDWI)
- U-Net (file 08)
- UPerNet (file 12)

QUY TRÌNH:
1. Với mỗi ảnh test (99 ảnh):
   - Đọc S2L2A và DEL (ground truth)
   - Tính pixel_area cho ảnh đó
   - Tính diện tích GT
   - Chạy V3 (rule-based, full image) -> tính diện tích
   - Chạy U-Net (sliding window) -> tính diện tích
   - Chạy UPerNet (sliding window) -> tính diện tích
2. Tính sai số: tuyệt đối (m²) và tương đối (%)
3. Lưu CSV bảng số liệu
4. Vẽ biểu đồ scatter: GT vs Predicted

CÁCH CHẠY:
  python 14_thong_ke_dien_tich_test.py
"""

from pathlib import Path
import csv
import time

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))

ts = import_module("13_tinh_dien_tich")
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
OUTPUT_DIR = Path("./ket_qua_dien_tich")
OUTPUT_DIR.mkdir(exist_ok=True)
UNET_CHECKPOINT    = Path("./ket_qua_unet_v2/best_model.pt")
UPERNET_CHECKPOINT = Path("./ket_qua_upernet_v2/best_model.pt")
PATCH_SIZE = 256
MIN_PIXELS = 100


# ============================================================
# SLIDING WINDOW CHO U-NET / UPERNET
# ============================================================
def predict_sliding(model, image_normalized, device, patch_size=256):
    """
    Chạy model trên ảnh nguyên kích thước bằng sliding window 50% overlap.
    image_normalized: shape (12, H, W), đã chuẩn hóa.
    """
    _, h, w = image_normalized.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_padded = np.pad(image_normalized,
                          ((0, 0), (0, pad_h), (0, pad_w)),
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
                logits = model(patch_t)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                full_mask[top:top + patch_size,
                          left:left + patch_size] += probs
                count[top:top + patch_size,
                      left:left + patch_size] += 1

    full_mask = full_mask / np.maximum(count, 1)
    return (full_mask[:h, :w] > 0.5).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("THỐNG KÊ DIỆN TÍCH CHÁY TRÊN TEST SET (3 MODEL)")
    print("=" * 60)
    print(f"Thiết bị: {device}")

    # ----- Load 2 model deep learning -----
    print("\nĐang load các model...")
    ck_unet = torch.load(UNET_CHECKPOINT, weights_only=False,
                          map_location=device)
    model_unet = UNet(in_channels=12, num_classes=1).to(device)
    model_unet.load_state_dict(ck_unet["model_state_dict"])
    model_unet.eval()
    print(f"  U-Net  : best epoch {ck_unet['epoch']}, "
          f"val Dice {ck_unet['val_dice']:.4f}")

    ck_uper = torch.load(UPERNET_CHECKPOINT, weights_only=False,
                          map_location=device)
    model_upernet = smp.UPerNet(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=12, classes=1,
    ).to(device)
    model_upernet.load_state_dict(ck_uper["model_state_dict"])
    model_upernet.eval()
    print(f"  UPerNet: best epoch {ck_uper['epoch']}, "
          f"val Dice {ck_uper['val_dice']:.4f}")

    # ----- Test dataset -----
    test_ds = CEMSDataset("test", full_image=True)
    print(f"\nSố ảnh test: {len(test_ds)}")

    # ----- Chạy 3 model trên test set -----
    print("\nĐang tính diện tích trên test set...")
    print(f"{'Idx':>4s} {'Tên':>22s} {'GT(ha)':>8s} "
          f"{'V3(ha)':>8s} {'U-Net(ha)':>10s} {'UPer(ha)':>9s}")
    print("-" * 75)

    results = []
    start = time.time()

    for idx in range(len(test_ds)):
        _, _, path = test_ds[idx]
        sample_dir = Path(path)
        s2_file = next(sample_dir.glob("*_S2L2A.tif"))
        del_file = next(sample_dir.glob("*_DEL.tif"))

        # Đọc ảnh và mask
        with rasterio.open(s2_file) as src:
            image_raw = src.read().astype(np.float32)
        with rasterio.open(del_file) as src:
            mask_gt = src.read(1).astype(np.uint8)

        # Tính pixel_area cho ảnh này
        pixel_area = ts.read_pixel_size_meters(s2_file)

        # Diện tích GT
        area_gt = ts.compute_total_area(mask_gt, pixel_area)

        # V3 prediction
        mask_v3, _, _, _ = du_doan_v3(image_raw)
        area_v3 = ts.compute_total_area(mask_v3, pixel_area)

        # U-Net và UPerNet cần ảnh chuẩn hóa
        image_norm = chuan_hoa(image_raw)
        mask_unet = predict_sliding(model_unet, image_norm, device,
                                     patch_size=PATCH_SIZE)
        area_unet = ts.compute_total_area(mask_unet, pixel_area)

        mask_upernet = predict_sliding(model_upernet, image_norm, device,
                                        patch_size=PATCH_SIZE)
        area_uper = ts.compute_total_area(mask_upernet, pixel_area)

        # Sai số tương đối (chỉ tính khi GT > 0)
        def rel_error(pred, gt):
            if gt == 0:
                return 0.0 if pred == 0 else float("inf")
            return abs(pred - gt) / gt * 100

        results.append({
            "idx": idx,
            "name": sample_dir.name,
            "pixel_area_m2": pixel_area,
            "gt_m2": area_gt["area_m2"],
            "gt_ha": area_gt["area_ha"],
            "v3_m2": area_v3["area_m2"],
            "v3_ha": area_v3["area_ha"],
            "v3_err_pct": rel_error(area_v3["area_m2"], area_gt["area_m2"]),
            "unet_m2": area_unet["area_m2"],
            "unet_ha": area_unet["area_ha"],
            "unet_err_pct": rel_error(area_unet["area_m2"], area_gt["area_m2"]),
            "upernet_m2": area_uper["area_m2"],
            "upernet_ha": area_uper["area_ha"],
            "upernet_err_pct": rel_error(area_uper["area_m2"], area_gt["area_m2"]),
        })

        if idx < 5 or (idx + 1) % 20 == 0:
            print(f"{idx:>4d} {sample_dir.name[:22]:>22s} "
                  f"{area_gt['area_ha']:>8.2f} "
                  f"{area_v3['area_ha']:>8.2f} "
                  f"{area_unet['area_ha']:>10.2f} "
                  f"{area_uper['area_ha']:>9.2f}")

    elapsed = time.time() - start
    print(f"\nHoàn thành {len(results)} ảnh trong {elapsed/60:.1f} phút.")

    # ----- Tổng kết -----
    valid = [r for r in results if r["gt_m2"] > 0 and
             not np.isinf(r["v3_err_pct"]) and
             not np.isinf(r["unet_err_pct"]) and
             not np.isinf(r["upernet_err_pct"])]

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT TRÊN {len(valid)} ẢNH CÓ GT > 0")
    print("=" * 60)
    print(f"{'Model':<15s} {'MAE (ha)':>12s} {'Sai số TB (%)':>15s} "
          f"{'Sai số TV (%)':>15s}")
    print("-" * 60)
    for key, name in [("v3", "Baseline V3"),
                       ("unet", "U-Net"),
                       ("upernet", "UPerNet")]:
        errors_m2 = [abs(r[f"{key}_m2"] - r["gt_m2"]) for r in valid]
        errors_pct = [r[f"{key}_err_pct"] for r in valid]
        mae_ha = np.mean(errors_m2) / 10_000
        mean_pct = np.mean(errors_pct)
        median_pct = np.median(errors_pct)
        print(f"{name:<15s} {mae_ha:>12.2f} {mean_pct:>15.1f} {median_pct:>15.1f}")

    # ----- Lưu CSV -----
    with open(OUTPUT_DIR / "thong_ke_dien_tich.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nĐã lưu CSV: {OUTPUT_DIR}/thong_ke_dien_tich.csv")

    # ----- Vẽ scatter plot: GT vs Predicted -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key, name, color in zip(
        axes,
        ["v3", "unet", "upernet"],
        ["Baseline V3", "U-Net", "UPerNet"],
        ["#E74C3C", "#3498DB", "#2ECC71"],
    ):
        gt_ha = [r["gt_ha"] for r in results]
        pred_ha = [r[f"{key}_ha"] for r in results]
        ax.scatter(gt_ha, pred_ha, alpha=0.6, c=color, s=40,
                   edgecolors="black", linewidth=0.5)

        # Đường y=x (dự đoán hoàn hảo)
        max_val = max(max(gt_ha), max(pred_ha)) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5,
                label="y = x (lý tưởng)")
        ax.set_xlabel("Diện tích GT (ha)")
        ax.set_ylabel("Diện tích dự đoán (ha)")
        ax.set_title(f"{name} · {len(results)} mẫu test")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_gt_vs_predicted.png",
                dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu biểu đồ: {OUTPUT_DIR}/scatter_gt_vs_predicted.png")

    # ----- Lưu kết quả tổng kết -----
    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write("THỐNG KÊ DIỆN TÍCH CHÁY TRÊN TEST SET\n")
        f.write("=" * 50 + "\n")
        f.write(f"Số ảnh test: {len(results)}\n")
        f.write(f"Số ảnh có GT > 0 (valid): {len(valid)}\n\n")
        f.write(f"{'Model':<15s} {'MAE (ha)':>12s} "
                f"{'Sai số TB (%)':>15s} {'Sai số TV (%)':>15s}\n")
        f.write("-" * 60 + "\n")
        for key, name in [("v3", "Baseline V3"),
                           ("unet", "U-Net"),
                           ("upernet", "UPerNet")]:
            errors_m2 = [abs(r[f"{key}_m2"] - r["gt_m2"]) for r in valid]
            errors_pct = [r[f"{key}_err_pct"] for r in valid]
            mae_ha = np.mean(errors_m2) / 10_000
            mean_pct = np.mean(errors_pct)
            median_pct = np.median(errors_pct)
            f.write(f"{name:<15s} {mae_ha:>12.2f} "
                    f"{mean_pct:>15.1f} {median_pct:>15.1f}\n")

    print(f"\nXong! Tất cả kết quả ở {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
