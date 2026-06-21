"""
KIỂM TRA CẤU TRÚC DATASET WILDFIRES-CEMS SAU GIẢI NÉN

Chạy sau khi đã giải nén tarball bằng Git Bash.

NHIỆM VỤ:
- Đếm số sự kiện cháy (EMSR codes) trong mỗi tập train/val/test
- Đếm số sample (mỗi sample là một patch ảnh + nhãn)
- Liệt kê các file mỗi sample có
- Xác nhận cấu trúc đúng

CÁCH CHẠY:
  python 04_check_cems_structure.py
"""

from pathlib import Path
from collections import Counter

# Thư mục gốc của dataset sau giải nén
# Có thể là wildfires-cems/ hoặc wildfires-cems/data/ tùy giải nén
ROOT_CANDIDATES = [
    Path("./wildfires-cems"),
    Path("./wildfires-cems/data"),
]

SPLITS = ["train", "val", "test"]


def find_root():
    """Tìm thư mục gốc thật của dataset (chỗ chứa train/val/test)."""
    for cand in ROOT_CANDIDATES:
        if all((cand / s).exists() for s in SPLITS):
            return cand
    return None


def kiem_tra_split(root: Path, split: str):
    """Đếm số EMSR, AOI và sample của một split."""
    split_dir = root / split
    if not split_dir.exists():
        print(f"  [THIẾU] {split_dir}")
        return

    # Mỗi sub-folder cấp 1 là một sự kiện EMSR
    emsr_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    print(f"\n--- {split.upper()} ---")
    print(f"  Số sự kiện cháy (EMSR codes): {len(emsr_dirs)}")

    # Đếm tổng số sample (thư mục lá có file .tif)
    total_samples = 0
    file_types = Counter()
    for emsr in emsr_dirs:
        for aoi in emsr.iterdir():
            if not aoi.is_dir():
                continue
            for sample in aoi.iterdir():
                if not sample.is_dir():
                    continue
                tif_files = list(sample.glob("*.tif"))
                if tif_files:
                    total_samples += 1
                    for f in tif_files:
                        # Lấy hậu tố để biết loại file: S2L2A, DEL, GRA, ESA_LC, CM
                        suffix = f.stem.split("_")[-1]
                        file_types[suffix] += 1

    print(f"  Tổng số sample (patch): {total_samples}")
    print(f"  Các loại file .tif:")
    for ftype, cnt in file_types.most_common():
        print(f"    - {ftype:10s}: {cnt} file")


def lay_mau_sample(root: Path):
    """Lấy đường dẫn của 1 sample đầu tiên trong train để bước sau dùng thử."""
    train_dir = root / "train"
    for emsr in train_dir.iterdir():
        if not emsr.is_dir():
            continue
        for aoi in emsr.iterdir():
            if not aoi.is_dir():
                continue
            for sample in aoi.iterdir():
                if not sample.is_dir():
                    continue
                tifs = list(sample.glob("*.tif"))
                if tifs:
                    print("\n" + "=" * 60)
                    print("VÍ DỤ 1 SAMPLE (dùng để test đọc ảnh ở bước sau)")
                    print("=" * 60)
                    print(f"  Thư mục: {sample}")
                    for f in sorted(tifs):
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"    {f.name}  ({size_mb:.2f} MB)")
                    return
    print("\n[CẢNH BÁO] Không tìm thấy sample nào để minh họa.")


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM TRA CẤU TRÚC DATASET WILDFIRES-CEMS")
    print("=" * 60)

    root = find_root()
    if root is None:
        print("[LỖI] Không tìm thấy thư mục có train/val/test.")
        print("Hãy kiểm tra:")
        print("  1. Đã giải nén tarball bằng Git Bash chưa?")
        print("  2. Thư mục giải nén có ở đúng chỗ không?")
        print("Cấu trúc kỳ vọng:")
        print("  wildfires-cems/")
        print("    train/, val/, test/")
        exit(1)

    print(f"Thư mục gốc: {root.resolve()}\n")

    for split in SPLITS:
        kiem_tra_split(root, split)

    lay_mau_sample(root)
    print("\nHoàn tất kiểm tra cấu trúc.")
