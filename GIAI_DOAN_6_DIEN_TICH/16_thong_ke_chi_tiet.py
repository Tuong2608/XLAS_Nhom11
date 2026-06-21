"""
THỐNG KÊ ẢNH CÓ CHÁY / KHÔNG CHÁY TRONG TRAIN/VAL/TEST

Phân tích chi tiết:
- Tổng số ảnh mỗi tập
- Số ảnh CÓ cháy (GT có ít nhất 1 pixel cháy)
- Số ảnh KHÔNG cháy (GT toàn 0)
- Phân bố tỷ lệ pixel cháy theo từng ảnh (histogram)
- Phân loại ảnh theo mức độ cháy: <1%, 1-10%, 10-30%, 30-50%, >50%

Đây là số liệu cho báo cáo và để trả lời cô.

CÁCH CHẠY:
  python 16_thong_ke_chi_tiet.py
"""

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt


DATA_ROOT = Path("./wildfires-cems")
OUTPUT_DIR = Path("./ket_qua_thong_ke_chi_tiet")
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "val", "test"]


# ============================================================
# UTILS
# ============================================================
def tim_samples(split_dir):
    """Tìm tất cả file DEL.tif trong split."""
    samples = []
    if not split_dir.exists():
        return samples
    for emsr in split_dir.iterdir():
        if not emsr.is_dir(): continue
        for aoi in emsr.iterdir():
            if not aoi.is_dir(): continue
            for sample in aoi.iterdir():
                if not sample.is_dir(): continue
                del_files = list(sample.glob("*_DEL.tif"))
                if del_files:
                    samples.append({
                        "emsr": emsr.name,
                        "sample_name": sample.name,
                        "del_file": del_files[0],
                    })
    return samples


def phan_loai_muc_do_chay(ti_le_chay):
    """Phân loại ảnh theo mức độ cháy."""
    if ti_le_chay == 0:
        return "Không cháy"
    elif ti_le_chay < 0.01:
        return "Cháy rất ít (<1%)"
    elif ti_le_chay < 0.10:
        return "Cháy ít (1-10%)"
    elif ti_le_chay < 0.30:
        return "Cháy vừa (10-30%)"
    elif ti_le_chay < 0.50:
        return "Cháy nhiều (30-50%)"
    else:
        return "Cháy rất nhiều (>50%)"


CAC_MUC_DO = [
    "Không cháy",
    "Cháy rất ít (<1%)",
    "Cháy ít (1-10%)",
    "Cháy vừa (10-30%)",
    "Cháy nhiều (30-50%)",
    "Cháy rất nhiều (>50%)",
]


# ============================================================
# THỐNG KÊ MỘT SPLIT
# ============================================================
def thong_ke_split(split_name):
    """Trả về dict thống kê cho một split."""
    split_dir = DATA_ROOT / split_name
    samples = tim_samples(split_dir)

    n_total = len(samples)
    n_co_chay = 0
    n_khong_chay = 0
    tong_pixel = 0
    tong_pixel_chay = 0
    ti_le_chay_moi_anh = []
    phan_loai_count = {muc: 0 for muc in CAC_MUC_DO}
    danh_sach_khong_chay = []   # ghi nhận tên ảnh KHÔNG cháy
    set_emsr = set()             # các EMSR unique

    for s in samples:
        set_emsr.add(s["emsr"])
        with rasterio.open(s["del_file"]) as src:
            mask = src.read(1)
        n_pix = mask.size
        n_fire = int((mask > 0).sum())

        tong_pixel += n_pix
        tong_pixel_chay += n_fire
        ti_le = n_fire / n_pix
        ti_le_chay_moi_anh.append(ti_le)

        if n_fire > 0:
            n_co_chay += 1
        else:
            n_khong_chay += 1
            danh_sach_khong_chay.append(s["sample_name"])

        muc_do = phan_loai_muc_do_chay(ti_le)
        phan_loai_count[muc_do] += 1

    return {
        "split": split_name,
        "n_total": n_total,
        "n_emsr": len(set_emsr),
        "n_co_chay": n_co_chay,
        "n_khong_chay": n_khong_chay,
        "ti_le_co_chay": n_co_chay / n_total if n_total > 0 else 0,
        "tong_pixel": tong_pixel,
        "tong_pixel_chay": tong_pixel_chay,
        "ti_le_pixel_chay": tong_pixel_chay / tong_pixel if tong_pixel > 0 else 0,
        "ti_le_moi_anh": ti_le_chay_moi_anh,
        "phan_loai": phan_loai_count,
        "danh_sach_khong_chay": danh_sach_khong_chay,
    }


# ============================================================
# VẼ BIỂU ĐỒ
# ============================================================
def ve_histogram_phan_bo(thong_ke_all, out_path):
    """Vẽ histogram phân bố tỷ lệ pixel cháy cho 3 split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, tk, color in zip(axes, thong_ke_all,
                              ["#3498DB", "#F39C12", "#E74C3C"]):
        arr = np.array(tk["ti_le_moi_anh"]) * 100
        ax.hist(arr, bins=30, color=color, edgecolor="black", alpha=0.8)
        ax.set_xlabel("Tỷ lệ pixel cháy trong ảnh (%)")
        ax.set_ylabel("Số ảnh")
        ax.set_title(f"{tk['split'].upper()} · {tk['n_total']} ảnh\n"
                     f"({tk['n_co_chay']} có cháy, "
                     f"{tk['n_khong_chay']} không cháy)")
        if arr.size > 0:
            ax.axvline(arr.mean(), color="black", linestyle="--",
                       label=f"TB = {arr.mean():.1f}%")
            ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def ve_bar_phan_loai(thong_ke_all, out_path):
    """Vẽ bar chart phân loại ảnh theo mức độ cháy."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CAC_MUC_DO))
    width = 0.25
    colors = ["#3498DB", "#F39C12", "#E74C3C"]
    for i, (tk, color) in enumerate(zip(thong_ke_all, colors)):
        counts = [tk["phan_loai"][muc] for muc in CAC_MUC_DO]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, counts, width,
                       label=tk["split"].upper(), color=color)
        # Ghi số lên đỉnh cột
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         str(count), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(CAC_MUC_DO, rotation=15, ha="right")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân loại ảnh theo mức độ cháy (theo tỷ lệ pixel cháy)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("THỐNG KÊ ẢNH CÓ CHÁY / KHÔNG CHÁY TRONG WILDFIRES-CEMS")
    print("=" * 70)

    thong_ke_all = []
    for split in SPLITS:
        print(f"\nĐang thống kê {split.upper()}...")
        tk = thong_ke_split(split)
        thong_ke_all.append(tk)

    # ----- In bảng tổng quan -----
    print("\n" + "=" * 70)
    print("BẢNG TỔNG QUAN")
    print("=" * 70)
    print(f"{'Split':<8} {'Tổng':>6} {'EMSR':>6} {'Có cháy':>9} "
          f"{'Không cháy':>11} {'% có cháy':>10} {'% pixel cháy':>14}")
    print("-" * 70)
    for tk in thong_ke_all:
        print(f"{tk['split'].upper():<8} "
              f"{tk['n_total']:>6} "
              f"{tk['n_emsr']:>6} "
              f"{tk['n_co_chay']:>9} "
              f"{tk['n_khong_chay']:>11} "
              f"{tk['ti_le_co_chay']*100:>9.1f}% "
              f"{tk['ti_le_pixel_chay']*100:>13.2f}%")

    # Tổng cộng cả 3 tập
    total_n = sum(tk["n_total"] for tk in thong_ke_all)
    total_co_chay = sum(tk["n_co_chay"] for tk in thong_ke_all)
    total_khong_chay = sum(tk["n_khong_chay"] for tk in thong_ke_all)
    total_pixel = sum(tk["tong_pixel"] for tk in thong_ke_all)
    total_pixel_chay = sum(tk["tong_pixel_chay"] for tk in thong_ke_all)
    print("-" * 70)
    print(f"{'TỔNG':<8} "
          f"{total_n:>6} "
          f"{'-':>6} "
          f"{total_co_chay:>9} "
          f"{total_khong_chay:>11} "
          f"{total_co_chay/total_n*100:>9.1f}% "
          f"{total_pixel_chay/total_pixel*100:>13.2f}%")

    # ----- In phân loại chi tiết -----
    print("\n" + "=" * 70)
    print("PHÂN LOẠI ẢNH THEO MỨC ĐỘ CHÁY")
    print("=" * 70)
    print(f"{'Mức độ':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Tổng':>8}")
    print("-" * 70)
    for muc in CAC_MUC_DO:
        train_c = thong_ke_all[0]["phan_loai"][muc]
        val_c = thong_ke_all[1]["phan_loai"][muc]
        test_c = thong_ke_all[2]["phan_loai"][muc]
        total_c = train_c + val_c + test_c
        print(f"{muc:<25} {train_c:>8} {val_c:>8} {test_c:>8} {total_c:>8}")

    # ----- In danh sách ảnh không cháy -----
    print("\n" + "=" * 70)
    print("DANH SÁCH ẢNH KHÔNG CHÁY (GT = 0)")
    print("=" * 70)
    for tk in thong_ke_all:
        if tk["n_khong_chay"] > 0:
            print(f"\n{tk['split'].upper()} ({tk['n_khong_chay']} ảnh):")
            for name in tk["danh_sach_khong_chay"][:20]:
                print(f"  - {name}")
            if len(tk["danh_sach_khong_chay"]) > 20:
                print(f"  ... và {len(tk['danh_sach_khong_chay']) - 20} ảnh khác")

    # ----- Vẽ biểu đồ -----
    print("\n" + "=" * 70)
    print("Đang vẽ biểu đồ...")
    print("=" * 70)
    ve_histogram_phan_bo(thong_ke_all,
                          OUTPUT_DIR / "histogram_ti_le_chay.png")
    ve_bar_phan_loai(thong_ke_all,
                      OUTPUT_DIR / "bar_phan_loai_muc_do.png")
    print(f"  Histogram: {OUTPUT_DIR}/histogram_ti_le_chay.png")
    print(f"  Bar chart: {OUTPUT_DIR}/bar_phan_loai_muc_do.png")

    # ----- Lưu file text tổng kết -----
    with open(OUTPUT_DIR / "thong_ke_chi_tiet.txt", "w",
              encoding="utf-8") as f:
        f.write("THỐNG KÊ ẢNH CÓ CHÁY / KHÔNG CHÁY TRONG WILDFIRES-CEMS\n")
        f.write("=" * 70 + "\n\n")
        f.write("BẢNG TỔNG QUAN:\n")
        f.write(f"{'Split':<8} {'Tổng':>6} {'Có cháy':>9} "
                f"{'Không cháy':>11} {'%':>8}\n")
        f.write("-" * 50 + "\n")
        for tk in thong_ke_all:
            f.write(f"{tk['split'].upper():<8} {tk['n_total']:>6} "
                    f"{tk['n_co_chay']:>9} {tk['n_khong_chay']:>11} "
                    f"{tk['ti_le_co_chay']*100:>7.1f}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'TỔNG':<8} {total_n:>6} {total_co_chay:>9} "
                f"{total_khong_chay:>11} "
                f"{total_co_chay/total_n*100:>7.1f}%\n\n")

        f.write("\nPHÂN LOẠI THEO MỨC ĐỘ CHÁY:\n")
        f.write(f"{'Mức độ':<25} {'Train':>8} {'Val':>8} {'Test':>8} "
                f"{'Tổng':>8}\n")
        f.write("-" * 70 + "\n")
        for muc in CAC_MUC_DO:
            train_c = thong_ke_all[0]["phan_loai"][muc]
            val_c = thong_ke_all[1]["phan_loai"][muc]
            test_c = thong_ke_all[2]["phan_loai"][muc]
            total_c = train_c + val_c + test_c
            f.write(f"{muc:<25} {train_c:>8} {val_c:>8} "
                    f"{test_c:>8} {total_c:>8}\n")

    print(f"\nĐã lưu thống kê đầy đủ vào {OUTPUT_DIR}/thong_ke_chi_tiet.txt")
    print("\nXong!")


if __name__ == "__main__":
    main()
