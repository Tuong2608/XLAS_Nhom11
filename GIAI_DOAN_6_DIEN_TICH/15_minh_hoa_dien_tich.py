"""
MINH HỌA DIỆN TÍCH CHÁY VỚI TỪNG CỤM RIÊNG BIỆT

Sinh các hình ảnh đẹp cho báo cáo và phần web app preview:
- Multi-panel: Input RGB | GT | V3 | U-Net | UPerNet
- Mỗi cụm cháy được tô MÀU RIÊNG (random)
- Hiển thị diện tích tổng và số cụm trên tiêu đề

Chọn 6 ảnh test có đa dạng diện tích (nhỏ, vừa, lớn).

CÁCH CHẠY:
  python 15_minh_hoa_dien_tich.py
"""

from pathlib import Path
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp

from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))

ts = import_module("13_tinh_dien_tich")
dataloader = import_module("06_dataloader")
baseline_v3 = import_module("07c_baseline_v3")
unet_module = import_module("08_train_unet")
thong_ke = import_module("14_thong_ke_dien_tich_test")

CEMSDataset = dataloader.CEMSDataset
chuan_hoa = dataloader.chuan_hoa
du_doan_v3 = baseline_v3.du_doan_v3
UNet = unet_module.UNet
predict_sliding = thong_ke.predict_sliding


OUTPUT_DIR = Path("./ket_qua_dien_tich/minh_hoa")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
UNET_CHECKPOINT = Path("./ket_qua_unet/best_model.pt")
UPERNET_CHECKPOINT = Path("./ket_qua_upernet/best_model.pt")

NUM_SAMPLES = 6
MIN_PIXELS = 100
EPS = 1e-8


# ============================================================
# UTILS
# ============================================================
def _stretch(arr):
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def tao_colormap_cum(n_clusters):
    """
    Tạo colormap để tô màu cho từng cụm cháy.
    Cụm 0 = nền (đen), cụm 1, 2, 3... có màu riêng.
    """
    # Lấy màu từ tab20 (palette 20 màu của matplotlib)
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    # Lặp lại nếu nhiều cụm hơn 20
    colors = [(0, 0, 0, 0)]  # nền trong suốt
    for i in range(n_clusters):
        colors.append(base_colors[i % 20])
    return ListedColormap(colors)


def ve_mask_voi_cum(ax, labels, components, title, pixel_area_m2):
    """
    Vẽ mask với mỗi cụm tô màu khác nhau, đánh số trên hình.
    """
    n_clusters = len(components)
    if n_clusters == 0:
        # Không có cụm nào (mask rỗng)
        ax.imshow(np.zeros_like(labels), cmap="gray")
        ax.set_title(f"{title}\nKhông phát hiện cháy")
        ax.axis("off")
        return

    cmap = tao_colormap_cum(n_clusters)
    # Mỗi cụm có id riêng; remap về 1, 2, 3... để dùng colormap
    remap = np.zeros_like(labels)
    for new_id, c in enumerate(components, start=1):
        remap[labels == c["id"]] = new_id

    ax.imshow(remap, cmap=cmap, vmin=0, vmax=n_clusters)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    # Đánh số từng cụm với diện tích
    for new_id, c in enumerate(components, start=1):
        cx, cy = c["centroid"]
        area_str = ts.format_area(c["area_m2"])
        # Vẽ box nhỏ tại centroid với label
        ax.annotate(
            f"#{new_id}\n{area_str}",
            xy=(cx, cy), xytext=(cx, cy),
            ha="center", va="center", fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="black", alpha=0.7,
                       edgecolor="white", linewidth=0.5),
        )


def ve_minh_hoa_anh(image_raw, mask_gt, mask_v3, mask_unet, mask_upernet,
                    pixel_area_m2, ten):
    """Vẽ 5 panel cho một ảnh."""
    # Tính cụm cho mỗi mask
    comp_gt, lbl_gt = ts.compute_components(mask_gt, pixel_area_m2, MIN_PIXELS)
    comp_v3, lbl_v3 = ts.compute_components(mask_v3, pixel_area_m2, MIN_PIXELS)
    comp_un, lbl_un = ts.compute_components(mask_unet, pixel_area_m2, MIN_PIXELS)
    comp_up, lbl_up = ts.compute_components(mask_upernet, pixel_area_m2, MIN_PIXELS)

    total_gt = ts.compute_total_area(mask_gt, pixel_area_m2)
    total_v3 = ts.compute_total_area(mask_v3, pixel_area_m2)
    total_un = ts.compute_total_area(mask_unet, pixel_area_m2)
    total_up = ts.compute_total_area(mask_upernet, pixel_area_m2)

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    fig.suptitle(
        f"Ước tính diện tích cháy · {ten} · "
        f"Pixel = {pixel_area_m2:.1f} m²",
        fontsize=13, y=1.02,
    )

    # Panel 1: Input RGB
    rgb = np.stack([_stretch(image_raw[3]),
                    _stretch(image_raw[2]),
                    _stretch(image_raw[1])], axis=-1)
    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB", fontsize=10)
    axes[0].axis("off")

    # Panel 2: GT
    ve_mask_voi_cum(
        axes[1], lbl_gt, comp_gt,
        f"GT · {len(comp_gt)} cụm · "
        f"{ts.format_area(total_gt['area_m2'])}",
        pixel_area_m2,
    )

    # Panel 3: V3
    err_v3 = abs(total_v3["area_m2"] - total_gt["area_m2"])
    err_v3_pct = err_v3 / (total_gt["area_m2"] + EPS) * 100
    ve_mask_voi_cum(
        axes[2], lbl_v3, comp_v3,
        f"V3 · {len(comp_v3)} cụm · "
        f"{ts.format_area(total_v3['area_m2'])}\n"
        f"Sai số: {err_v3_pct:.1f}%",
        pixel_area_m2,
    )

    # Panel 4: U-Net
    err_un = abs(total_un["area_m2"] - total_gt["area_m2"])
    err_un_pct = err_un / (total_gt["area_m2"] + EPS) * 100
    ve_mask_voi_cum(
        axes[3], lbl_un, comp_un,
        f"U-Net · {len(comp_un)} cụm · "
        f"{ts.format_area(total_un['area_m2'])}\n"
        f"Sai số: {err_un_pct:.1f}%",
        pixel_area_m2,
    )

    # Panel 5: UPerNet
    err_up = abs(total_up["area_m2"] - total_gt["area_m2"])
    err_up_pct = err_up / (total_gt["area_m2"] + EPS) * 100
    ve_mask_voi_cum(
        axes[4], lbl_up, comp_up,
        f"UPerNet · {len(comp_up)} cụm · "
        f"{ts.format_area(total_up['area_m2'])}\n"
        f"Sai số: {err_up_pct:.1f}%",
        pixel_area_m2,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"dien_tich_{ten}.png", dpi=80,
                bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("MINH HỌA DIỆN TÍCH CHÁY VỚI TỪNG CỤM")
    print("=" * 60)

    # Load 2 model
    print("\nĐang load model...")
    ck_unet = torch.load(UNET_CHECKPOINT, weights_only=False,
                          map_location=device)
    model_unet = UNet(in_channels=12, num_classes=1).to(device)
    model_unet.load_state_dict(ck_unet["model_state_dict"])
    model_unet.eval()

    ck_uper = torch.load(UPERNET_CHECKPOINT, weights_only=False,
                          map_location=device)
    model_upernet = smp.UPerNet(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=12, classes=1,
    ).to(device)
    model_upernet.load_state_dict(ck_uper["model_state_dict"])
    model_upernet.eval()

    # Test dataset
    test_ds = CEMSDataset("test", full_image=True)
    print(f"Số ảnh test: {len(test_ds)}")

    # Chọn 6 ảnh đa dạng diện tích: tính nhanh diện tích GT của tất cả,
    # rồi chọn các phân vị 10%, 30%, 50%, 70%, 90% + 1 ảnh ngẫu nhiên
    print("\nĐang chọn 6 ảnh đa dạng diện tích...")
    gt_areas = []
    for idx in range(len(test_ds)):
        _, _, path = test_ds[idx]
        del_file = next(Path(path).glob("*_DEL.tif"))
        with rasterio.open(del_file) as src:
            mask = src.read(1)
        gt_areas.append((idx, int((mask > 0).sum())))

    # Sắp xếp theo diện tích tăng dần
    gt_areas.sort(key=lambda x: x[1])
    # Chọn các phân vị
    n_total = len(gt_areas)
    indices_to_viz = [
        gt_areas[int(n_total * 0.10)][0],
        gt_areas[int(n_total * 0.30)][0],
        gt_areas[int(n_total * 0.50)][0],
        gt_areas[int(n_total * 0.70)][0],
        gt_areas[int(n_total * 0.90)][0],
        gt_areas[-1][0],   # ảnh có vùng cháy lớn nhất
    ]

    print(f"\nĐang vẽ {len(indices_to_viz)} hình minh họa...")
    for i, idx in enumerate(indices_to_viz):
        _, _, path = test_ds[idx]
        sample_dir = Path(path)
        s2_file = next(sample_dir.glob("*_S2L2A.tif"))
        del_file = next(sample_dir.glob("*_DEL.tif"))

        with rasterio.open(s2_file) as src:
            image_raw = src.read().astype(np.float32)
        with rasterio.open(del_file) as src:
            mask_gt = src.read(1).astype(np.uint8)

        pixel_area = ts.read_pixel_size_meters(s2_file)
        mask_v3, _, _, _ = du_doan_v3(image_raw)
        image_norm = chuan_hoa(image_raw)
        mask_unet = predict_sliding(model_unet, image_norm, device,
                                     patch_size=256)
        mask_upernet = predict_sliding(model_upernet, image_norm, device,
                                        patch_size=256)

        ve_minh_hoa_anh(image_raw, mask_gt, mask_v3, mask_unet,
                         mask_upernet, pixel_area, sample_dir.name)
        print(f"  [{i+1}/{len(indices_to_viz)}] Xong {sample_dir.name}")

    print(f"\nXong! Mở thư mục '{OUTPUT_DIR}' để xem.")


if __name__ == "__main__":
    main()
