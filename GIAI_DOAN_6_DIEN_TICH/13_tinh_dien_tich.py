"""
MODULE TÍNH DIỆN TÍCH CHÁY TỪ MASK
File này là MODULE dùng chung cho file 14 (thống kê) và 15 (minh họa),
sau này dùng cả cho web app demo.

NGUYÊN LÝ TÍNH DIỆN TÍCH:
  Diện tích cháy = (Số pixel cháy) × (Diện tích thật mỗi pixel)

Diện tích pixel phụ thuộc CRS của ảnh:
- Nếu CRS là UTM/Mercator (đơn vị mét): đọc thẳng từ transform.
- Nếu CRS là WGS84 (đơn vị độ): chuyển đổi theo công thức xấp xỉ địa cầu,
  có hiệu chỉnh theo vĩ độ - vì 1 độ kinh tuyến co lại khi gần cực.

Công thức chuyển đổi (xấp xỉ địa cầu):
  1° kinh tuyến (longitude) ≈ 111,320 × cos(vĩ độ) mét
  1° vĩ tuyến  (latitude)   ≈ 110,540 mét (gần không đổi)

Sai số <1% trong phạm vi 1 ảnh — chấp nhận được cho bài toán.

CÁCH DÙNG:
  from importlib import import_module
  ts = import_module("13_tinh_dien_tich")
  
  pixel_area = ts.read_pixel_size_meters("anh.tif")
  total = ts.compute_total_area(mask, pixel_area)
  components, labels = ts.compute_components(mask, pixel_area, min_pixels=100)
"""

from pathlib import Path
import numpy as np
import cv2
import rasterio


# ============================================================
# HẰNG SỐ
# ============================================================
# Mặc định nếu không đọc được metadata (Sentinel-2 chuẩn 10m × 10m)
DEFAULT_PIXEL_AREA_M2 = 100.0

# Ngưỡng filter cụm nhỏ (~1 hecta nếu pixel ~100m²)
DEFAULT_MIN_PIXELS = 100

# Hệ số chuyển đổi địa cầu (mét/độ)
METERS_PER_DEGREE_LAT = 110_540
METERS_PER_DEGREE_LON_EQUATOR = 111_320


# ============================================================
# ĐỌC METADATA GEOTIFF -> DIỆN TÍCH PIXEL THẬT
# ============================================================
def read_pixel_size_meters(tif_path):
    """
    Đọc metadata GeoTIFF, trả về diện tích thật của 1 pixel (m²).
    Tự động xử lý CRS UTM (mét) và WGS84 (độ).
    """
    try:
        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

            # Kích thước pixel theo đơn vị CRS
            dx = abs(transform.a)  # ngang
            dy = abs(transform.e)  # dọc

            if crs is None:
                return DEFAULT_PIXEL_AREA_M2

            if crs.is_geographic:
                # WGS84 - đơn vị độ -> phải chuyển sang mét
                center_lat = (bounds.top + bounds.bottom) / 2
                lat_rad = np.radians(center_lat)
                dx_m = dx * METERS_PER_DEGREE_LON_EQUATOR * np.cos(lat_rad)
                dy_m = dy * METERS_PER_DEGREE_LAT
                return dx_m * dy_m
            else:
                # UTM/projected - đơn vị mét -> nhân thẳng
                return dx * dy
    except Exception as e:
        print(f"[Cảnh báo] Không đọc được metadata {tif_path}, "
              f"dùng mặc định 100 m²/pixel. Lỗi: {e}")
        return DEFAULT_PIXEL_AREA_M2


# ============================================================
# TÍNH TỔNG DIỆN TÍCH
# ============================================================
def compute_total_area(mask, pixel_area_m2):
    """
    Tính tổng diện tích cháy.
    Trả về dict với các đơn vị: m², hecta, km², số pixel.
    """
    n_pixels = int((mask > 0).sum())
    area_m2 = n_pixels * pixel_area_m2
    return {
        "n_pixels": n_pixels,
        "area_m2": area_m2,
        "area_ha": area_m2 / 10_000,
        "area_km2": area_m2 / 1_000_000,
    }


# ============================================================
# TÌM CÁC CỤM CHÁY RIÊNG BIỆT (CONNECTED COMPONENTS)
# ============================================================
def compute_components(mask, pixel_area_m2, min_pixels=DEFAULT_MIN_PIXELS):
    """
    Phân tách mask thành các cụm cháy riêng biệt (connected components),
    tính diện tích từng cụm, lọc các cụm quá nhỏ.

    Trả về:
      components: list các dict, mỗi cụm có id, n_pixels, area_m2,
                  area_ha, bbox, centroid
      labels: ảnh 2D gán nhãn từng cụm (0=nền, 1,2,3...=cụm)
              Dùng cho visualization tô màu từng cụm.
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    components = []
    # Bỏ qua label 0 (nền)
    for label_id in range(1, n_labels):
        n_pix = int(stats[label_id, cv2.CC_STAT_AREA])
        if n_pix < min_pixels:
            # Lọc cụm quá nhỏ - cũng xóa khỏi labels để không vẽ
            labels[labels == label_id] = 0
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        area_m2 = n_pix * pixel_area_m2

        components.append({
            "id": int(label_id),
            "n_pixels": n_pix,
            "area_m2": float(area_m2),
            "area_ha": float(area_m2 / 10_000),
            "bbox": (x, y, w, h),
            "centroid": (float(cx), float(cy)),
        })

    # Sắp xếp theo diện tích giảm dần (cụm lớn nhất lên đầu)
    components.sort(key=lambda c: -c["area_m2"])
    return components, labels


# ============================================================
# FORMAT ĐỊNH DẠNG HIỂN THỊ
# ============================================================
def format_area(area_m2):
    """Format diện tích đẹp tùy độ lớn (m² / ha / km²)."""
    if area_m2 >= 1_000_000:
        return f"{area_m2 / 1_000_000:.2f} km²"
    elif area_m2 >= 10_000:
        return f"{area_m2 / 10_000:.2f} ha"
    else:
        return f"{area_m2:.0f} m²"


# ============================================================
# DEMO / KIỂM TRA MODULE
# ============================================================
if __name__ == "__main__":
    test_path = Path("./wildfires-cems/train/EMSR218/AOI01/EMSR218_AOI01_01")
    if not test_path.exists():
        print(f"Không tìm thấy {test_path}")
        exit(1)

    s2_file = next(test_path.glob("*_S2L2A.tif"))
    del_file = next(test_path.glob("*_DEL.tif"))

    print("=" * 60)
    print("KIỂM TRA MODULE TÍNH DIỆN TÍCH")
    print("=" * 60)
    print(f"Ảnh thử: {test_path.name}")

    # Đọc pixel size
    pixel_area = read_pixel_size_meters(s2_file)
    print(f"\nDiện tích 1 pixel (m²): {pixel_area:.2f}")
    print(f"  -> kích thước pixel ≈ {pixel_area**0.5:.2f} m × {pixel_area**0.5:.2f} m")

    # Đọc mask
    with rasterio.open(del_file) as src:
        mask = src.read(1)
        print(f"\nMetadata ảnh:")
        print(f"  CRS         : {src.crs}")
        print(f"  Kích thước  : {src.width} × {src.height}")
        print(f"  Vùng địa lý : {src.bounds}")
        center_lat = (src.bounds.top + src.bounds.bottom) / 2
        print(f"  Vĩ độ trung tâm: {center_lat:.4f}°")

    # Tính tổng
    total = compute_total_area(mask, pixel_area)
    print(f"\nTỔNG DIỆN TÍCH CHÁY (GROUND TRUTH):")
    print(f"  Số pixel cháy : {total['n_pixels']:,}")
    print(f"  Diện tích m²  : {total['area_m2']:,.0f}")
    print(f"  Diện tích ha  : {total['area_ha']:.2f}")
    print(f"  Diện tích km² : {total['area_km2']:.4f}")
    print(f"  Format ngắn   : {format_area(total['area_m2'])}")

    # Tính từng cụm
    components, labels = compute_components(mask, pixel_area, min_pixels=100)
    print(f"\nCÁC CỤM CHÁY (>= 100 pixel = ~1 ha):")
    print(f"  Tổng số cụm: {len(components)}")
    for i, c in enumerate(components[:10]):
        print(f"  Cụm #{i+1} (id={c['id']}): "
              f"{format_area(c['area_m2'])}, "
              f"bbox=(x={c['bbox'][0]},y={c['bbox'][1]},"
              f"w={c['bbox'][2]},h={c['bbox'][3]})")
    if len(components) > 10:
        print(f"  ... và {len(components) - 10} cụm khác")

    print("\nMODULE HOẠT ĐỘNG TỐT!")
