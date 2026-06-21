"""
ĐỌC VÀ KHÁM PHÁ ẢNH SENTINEL-2 ĐA PHỔ
File này là khoảnh khắc lý thuyết biến thành thực hành.

NHIỆM VỤ:
1. Đọc file ảnh S2L2A.tif (12 band Sentinel-2)
2. Đọc mask DEL.tif (nhãn vùng cháy)
3. Hiển thị từng band riêng để hiểu trông thế nào
4. Tạo ảnh RGB thật (Red Green Blue)
5. Tạo ảnh "false color" dùng NIR-Red-Green để nổi rừng
6. Tính chỉ số NBR và NDVI, hiển thị
7. So sánh với mask ground-truth

KIẾN THỨC NỀN (đọc cùng code, để vào báo cáo):

Sentinel-2 Level 2A có 12 band trong file .tif này. Thứ tự band tùy
vào cách dataset lưu - hầu hết các dataset CEMS lưu theo thứ tự:
  Index 0  = B1   Coastal/Aerosol (60m)
  Index 1  = B2   Blue            (10m)
  Index 2  = B3   Green           (10m)
  Index 3  = B4   Red             (10m)   <- dùng cho NDVI và RGB
  Index 4  = B5   Red Edge 1      (20m)
  Index 5  = B6   Red Edge 2      (20m)
  Index 6  = B7   Red Edge 3      (20m)
  Index 7  = B8   NIR             (10m)   <- dùng cho NDVI, NBR, false color
  Index 8  = B8A  Narrow NIR      (20m)
  Index 9  = B9   Water Vapor     (60m)
  Index 10 = B11  SWIR 1          (20m)   <- dùng cho NBR
  Index 11 = B12  SWIR 2          (20m)   <- dùng cho NBR

Giá trị pixel là "reflectance" nhân 10000 (theo định dạng Sentinel-2).
Để hiển thị, ta chia cho 10000 rồi nhân lên cho rõ.

CÔNG THỨC CÁC CHỈ SỐ VIỄN THÁM:
  NDVI = (NIR - Red) / (NIR + Red)         -> đo lường thực vật
  NBR  = (NIR - SWIR2) / (NIR + SWIR2)     -> đo lường cháy
  
NBR cao (gần +1)  = rừng khỏe
NBR thấp (gần -1) = đất cháy / không có thực vật

CÁCH CHẠY:
  python 05_read_multispectral.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# CẤU HÌNH - đường dẫn sample mẫu (lấy từ file 04 ở bước trước)
# ============================================================
SAMPLE_DIR = Path("./wildfires-cems/train/EMSR218/AOI01/EMSR218_AOI01_01")
S2_FILE = SAMPLE_DIR / "EMSR218_AOI01_01_S2L2A.tif"
MASK_FILE = SAMPLE_DIR / "EMSR218_AOI01_01_DEL.tif"

# Vị trí các band trong file (theo index từ 1, vì rasterio đánh từ 1)
BAND_INDEX = {
    "B2_Blue":   2,
    "B3_Green":  3,
    "B4_Red":    4,
    "B8_NIR":    8,
    "B11_SWIR1": 11,
    "B12_SWIR2": 12,
}


def doc_anh_s2(path: Path):
    """Đọc toàn bộ ảnh Sentinel-2 12 band thành mảng numpy [12, H, W]."""
    with rasterio.open(path) as src:
        # rasterio đọc theo thứ tự (band, hàng, cột)
        # src.count là số band, src.height/width là kích thước
        print(f"  Số band: {src.count}")
        print(f"  Kích thước: {src.height} x {src.width}")
        print(f"  Kiểu dữ liệu: {src.dtypes[0]}")
        print(f"  CRS (hệ tọa độ địa lý): {src.crs}")
        anh = src.read()  # mảng (band, H, W)
    return anh


def doc_mask(path: Path):
    """Đọc mask nhị phân vùng cháy (1 = cháy, 0 = không cháy)."""
    with rasterio.open(path) as src:
        mask = src.read(1)  # đọc band đầu tiên
    return mask


def chuan_hoa_de_hien_thi(arr, percentile=2):
    """
    Chuẩn hóa mảng về [0, 1] để hiển thị bằng matplotlib.
    Dùng percentile clip để loại các giá trị cực đại làm méo hình.
    """
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, percentile)
    hi = np.percentile(arr, 100 - percentile)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    return arr


def tao_anh_rgb(anh_s2):
    """Tạo ảnh RGB thật bằng cách xếp B4-B3-B2."""
    red = anh_s2[BAND_INDEX["B4_Red"] - 1]
    green = anh_s2[BAND_INDEX["B3_Green"] - 1]
    blue = anh_s2[BAND_INDEX["B2_Blue"] - 1]
    rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)
    return chuan_hoa_de_hien_thi(rgb)


def tao_anh_false_color(anh_s2):
    """
    Ảnh false color NIR-Red-Green: rừng khỏe sẽ rất ĐỎ.
    Vùng cháy / đất trống sẽ nâu xám.
    Đây là cách viễn thám hay dùng để xem thực vật.
    """
    nir = anh_s2[BAND_INDEX["B8_NIR"] - 1]
    red = anh_s2[BAND_INDEX["B4_Red"] - 1]
    green = anh_s2[BAND_INDEX["B3_Green"] - 1]
    fcc = np.stack([nir, red, green], axis=-1)
    return chuan_hoa_de_hien_thi(fcc)


def tinh_ndvi(anh_s2):
    """NDVI = (NIR - Red) / (NIR + Red). Đo lượng thực vật."""
    nir = anh_s2[BAND_INDEX["B8_NIR"] - 1].astype(np.float32)
    red = anh_s2[BAND_INDEX["B4_Red"] - 1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi  # giá trị trong [-1, 1]


def tinh_nbr(anh_s2):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Đo lường cháy."""
    nir = anh_s2[BAND_INDEX["B8_NIR"] - 1].astype(np.float32)
    swir2 = anh_s2[BAND_INDEX["B12_SWIR2"] - 1].astype(np.float32)
    nbr = (nir - swir2) / (nir + swir2 + 1e-8)
    return nbr  # giá trị trong [-1, 1]


def ve_tat_ca(anh_s2, mask):
    """Vẽ tất cả các góc nhìn trong một figure để so sánh trực quan."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. RGB thật
    axes[0, 0].imshow(tao_anh_rgb(anh_s2))
    axes[0, 0].set_title("RGB thật (như mắt người nhìn)")
    axes[0, 0].axis("off")

    # 2. False color NIR-R-G
    axes[0, 1].imshow(tao_anh_false_color(anh_s2))
    axes[0, 1].set_title("False color (NIR-Red-Green)\nRừng = đỏ, cháy = nâu xám")
    axes[0, 1].axis("off")

    # 3. NDVI - thực vật
    ndvi = tinh_ndvi(anh_s2)
    im2 = axes[0, 2].imshow(ndvi, cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[0, 2].set_title("NDVI - chỉ số thực vật\nXanh = rừng, đỏ = đất trống")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # 4. NBR - chỉ số cháy
    nbr = tinh_nbr(anh_s2)
    im3 = axes[1, 0].imshow(nbr, cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[1, 0].set_title("NBR - chỉ số cháy\nĐỏ = cháy, xanh = rừng khỏe")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # 5. Mask ground-truth
    axes[1, 1].imshow(mask, cmap="hot")
    axes[1, 1].set_title("Mask ground-truth\n(Vàng = cháy, đen = không)")
    axes[1, 1].axis("off")

    # 6. Baseline đơn giản: ngưỡng NBR < -0.1
    baseline_pred = (nbr < -0.1).astype(np.uint8)
    axes[1, 2].imshow(baseline_pred, cmap="hot")
    axes[1, 2].set_title("Dự đoán BASELINE: NBR < -0.1\n(Đây là cách truyền thống làm)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("kham_pha_mau_dau_tien.png", dpi=100, bbox_inches="tight")
    print("\nĐã lưu hình ảnh ra 'kham_pha_mau_dau_tien.png'")
    plt.show()


def in_thong_ke(anh_s2, mask):
    """In vài thống kê cơ bản để hiểu dữ liệu."""
    print("\n" + "=" * 60)
    print("THỐNG KÊ CƠ BẢN")
    print("=" * 60)
    print(f"  Giá trị min/max của S2 raw: {anh_s2.min()} / {anh_s2.max()}")
    print(f"  Tỉ lệ pixel cháy trong mask: "
          f"{(mask > 0).sum() / mask.size * 100:.2f}%")
    nbr = tinh_nbr(anh_s2)
    print(f"  NBR trên vùng cháy (theo mask): "
          f"trung bình {nbr[mask > 0].mean():.3f}")
    print(f"  NBR trên vùng không cháy:      "
          f"trung bình {nbr[mask == 0].mean():.3f}")
    print("  -> Hai con số NBR này khác nhau rõ rệt là dấu hiệu tốt:")
    print("     band hồng ngoại thật sự phân biệt được cháy/không cháy.")


if __name__ == "__main__":
    print("=" * 60)
    print("ĐỌC THỬ MỘT MẪU ẢNH SENTINEL-2 ĐA PHỔ")
    print("=" * 60)
    print(f"File ảnh: {S2_FILE}")
    print(f"File mask: {MASK_FILE}\n")

    if not S2_FILE.exists() or not MASK_FILE.exists():
        print("[LỖI] Không tìm thấy file. Hãy kiểm tra đường dẫn SAMPLE_DIR.")
        exit(1)

    print("Đang đọc ảnh Sentinel-2...")
    anh_s2 = doc_anh_s2(S2_FILE)
    print(f"  Shape mảng đọc được: {anh_s2.shape}  "
          f"(nên là [12, H, W])")

    print("\nĐang đọc mask vùng cháy...")
    mask = doc_mask(MASK_FILE)
    print(f"  Shape mask: {mask.shape}")
    print(f"  Giá trị duy nhất trong mask: {np.unique(mask)}")

    in_thong_ke(anh_s2, mask)
    ve_tat_ca(anh_s2, mask)

    print("\n" + "=" * 60)
    print("HOÀN TẤT KHÁM PHÁ MẪU ĐẦU TIÊN!")
    print("=" * 60)
    print("Mở file 'kham_pha_mau_dau_tien.png' để xem 6 góc nhìn của 1 mẫu.")
    print("So sánh trực quan: ảnh RGB thường khó thấy cháy, NBR thì rõ ràng!")
