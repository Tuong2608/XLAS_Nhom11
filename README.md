# Đồ Án Xử Lý Ảnh Số - Nhóm 11
## Phát hiện cháy rừng trên ảnh vệ tinh Sentinel-2

### Thông tin nhóm

| Thành viên | MSSV | Vai trò |
|------------|------|---------|
| Thành viên 1 | Trần Văn Tưởng | Nhóm trưởng |
| Thành viên 2 | Trần Quang Toản | Thành viên |

### Giảng viên hướng dẫn
**ThS. Võ Lê Phúc Hậu**

---

## Mục lục

1. [Giới thiệu đề tài](#giới-thiệu-đề-tài)
2. [Cài đặt](#cài-đặt)
3. [Cấu trúc project](#cấu-trúc-project)
4. [Dataset](#dataset)
5. [Các phương pháp](#các-phương-pháp)
6. [Kết quả](#kết-quả)
7. [Hướng dẫn chạy](#hướng-dẫn-chạy)

---

## Giới thiệu đề tài

### Bối cảnh

Cháy rừng là một trong những thảm họa thiên nhiên nghiêm trọng nhất, gây ra:
- Thiệt hại về kinh tế hàng tỷ đô la mỗi năm
- Mất môi trường sống của động thực vật
- Ô nhiễm không khí ảnh hưởng đến sức khỏe con người
- Phát thải khí nhà kính góp phần vào biến đổi khí hậu

### Mục tiêu

Xây dựng hệ thống **phát hiện và phân vùng cháy rừng** trên ảnh vệ tinh đa phổ Sentinel-2, với hai hướng tiếp cận:
1. **Hướng truyền thống**: Xử lý ảnh + chỉ số viễn thám (NBR, NDWI)
2. **Hướng học sâu**: U-Net và UPerNet cho semantic segmentation

### Dataset

Sử dụng dataset **Wildfires-CEMS** từ HuggingFace:
- **Nguồn**: [links-ads/wildfires-cems](https://huggingface.co/datasets/links-ads/wildfires-cems)
- **Loại ảnh**: Ảnh vệ tinh Sentinel-2 Level-2A (12 band đa phổ)
- **Độ phân giải**: 10m - 60m tùy band
- **Kích thước**: ~10m cho band RGB, NIR, SWIR

#### Các band của Sentinel-2

| Band | Tên | Độ phân giải |
|------|------|--------------|
| B1 | Coastal/Aerosol | 60m |
| B2 | Blue | 10m |
| B3 | Green | 10m |
| B4 | Red | 10m |
| B5-B7 | Red Edge | 20m |
| B8 | NIR | 10m |
| B8A | Narrow NIR | 20m |
| B9 | Water Vapor | 60m |
| B11 | SWIR-1 | 20m |
| B12 | SWIR-2 | 20m |

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- GPU với CUDA hỗ trợ (khuyến nghị RTX 3060 6GB hoặc cao hơn)
- 16GB RAM

### Các bước cài đặt

```bash
# 1. Clone repository
git clone https://github.com/Tuong2608/XLAS_Nhom11.git
cd XLAS_Nhom11

# 2. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Cài đặt PyTorch với CUDA (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Cài các thư viện khác
pip install -r requirements.txt

# 5. Tải dataset từ HuggingFace
python run.py
```

---

## Cấu trúc project

```
XLAS_Nhom11/
├── README.md                    # File này
├── requirements.txt              # Danh sách thư viện Python
├── run.py                       # Script tải dataset từ HuggingFace
├── checksems.py                 # Kiểm tra cấu trúc dataset
│
├── GIAI_DOAN_1_KHAM_PHA/       # Khám phá dữ liệu
│   └── 01_explore_data.py       # EDA - Phân tích dữ liệu ban đầu
│
├── GIAI_DOAN_2_TIEN_XU_LY/     # Tiền xử lý ảnh
│   ├── 02_preprocessing.py      # Resize, Normalize, Data Augmentation
│   ├── 05_read_multispectral.py # Đọc ảnh Sentinel-2 12 band
│   └── 06_dataloader.py         # DataLoader cho PyTorch
│
├── GIAI_DOAN_3_BASELINE/       # Phương pháp truyền thống
│   ├── 07_baseline_traditional.py   # V1: NBR + Gaussian + Otsu
│   ├── 07b_baseline_improved.py      # V2: Ngưỡng cố định + Validation
│   └── 07c_baseline_v3.py            # V3: NBR + NDWI (loại mặt nước)
│
├── GIAI_DOAN_4_HOC_SAU/        # Mô hình học sâu
│   ├── 08_train_unet.py         # U-Net v1 (baseline)
│   ├── 08_train_unet_v2.py      # U-Net v2 (cải tiến: warmup, cosine, early stop)
│   ├── 12_train_upernet_segformer.py # UPerNet & SegFormer (v1)
│   └── 12_train_upernet_v2.py   # UPerNet v2 (cải tiến)
│
├── GIAI_DOAN_5_SO_SANH/        # So sánh và đánh giá
│   ├── 09_compare_models.py      # So sánh trực quan 3 model
│   ├── 10_classification_test.py    # Classification có/không cháy
│   └── 10b_classification_save.py    # Lưu patch để demo
│
├── GIAI_DOAN_6_DIEN_TICH/      # Tính diện tích cháy
│   ├── 13_tinh_dien_tich.py     # Module tính diện tích
│   ├── 14_thong_ke_dien_tich_test.py # Thống kê trên test set
│   ├── 15_minh_hoa_dien_tich.py # Minh họa từng cụm cháy
│   └── 16_thong_ke_chi_tiet.py  # Thống kê chi tiết dataset
│
├── GIAI_DOAN_7_WEB_APP/        # Ứng dụng web demo
│   └── 17_web_app.py            # Streamlit web app
│
└── wildfires-cems/              # Dataset (sau khi tải về)
    ├── train/
    ├── val/
    └── test/
```

---

## Các phương pháp

### 1. Baseline Truyền thống (V3)

Pipeline hoàn chỉnh:

```
Input ảnh đa phổ (12 band)
    │
    ├── 1. Lọc Gaussian (kernel 5×5)
    │
    ├── 2. Tách kênh B8 (NIR) và B12 (SWIR-2)
    │
    ├── 3. Tính chỉ số NBR = (NIR - SWIR2) / (NIR + SWIR2)
    │
    ├── 4. Tính chỉ số NDWI = (Green - NIR) / (Green + NIR)
    │
    ├── 5. Mask_fire = (NBR < threshold) AND (NOT Mask_nước)
    │
    └── 6. Morphology Opening + Closing
            │
            └── Output: Mask cháy nhị phân
```

**Ngưỡng NBR**: -0.10 (tìm được qua validation)

**Chỉ số NDWI** giúp loại bỏ false positive từ mặt nước sông/hồ.

### 2. U-Net (Encoder-Decoder)

**Kiến trúc**:
- Encoder: 4 mức với DoubleConv (Conv-BN-ReLU-Conv-BN-ReLU)
- Decoder: 4 mức với ConvTranspose và Skip Connections
- Input: 12 channels × 256×256 pixels
- Output: 1 channel với Sigmoid

**Hyperparameters**:
| Tham số | Giá trị |
|---------|---------|
| Patch size | 256×256 |
| Batch size | 4 |
| Epochs | 80 |
| Learning rate | 5×10⁻⁴ |
| Optimizer | Adam |
| Scheduler | Warmup (5 epoch) + Cosine Annealing |
| Loss | BCE + Dice (0.5 each) |
| Early stopping | Patience 15 |

### 3. UPerNet (Pyramid Network)

**Kiến trúc**:
- Encoder: ResNet-50 (from scratch, không pretrained)
- Decoder: Feature Pyramid Network với Attention Mechanism
- Input: 12 channels × 256×256 pixels
- Output: 1 channel với Sigmoid

**Lý do chọn UPerNet**:
- Multi-scale feature fusion phù hợp với vùng cháy có kích thước đa dạng
- Attention mechanism giúp tập trung vào vùng cháy

---

## Kết quả

### Segmentation Performance

| Model | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| Baseline V3 | ~0.32 | ~0.48 | ~0.35 | ~0.75 |
| U-Net v2 | ~0.78 | ~0.87 | ~0.85 | ~0.90 |
| UPerNet v2 | ~0.79 | ~0.88 | ~0.87 | ~0.88 |

### Ước tính diện tích cháy

| Model | MAE (ha) | Sai số TB (%) |
|-------|----------|--------------|
| Baseline V3 | ~15.2 | ~45% |
| U-Net v2 | ~3.5 | ~12% |
| UPerNet v2 | ~3.2 | ~10% |

### Classification (Có/Không cháy)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Baseline V3 | ~72% | ~68% | ~85% | ~75% |
| U-Net | ~91% | ~89% | ~93% | ~91% |
| UPerNet | ~92% | ~91% | ~92% | ~91% |

---

## Hướng dẫn chạy

### 1. Tải Dataset

```bash
python run.py
```

### 2. Khám phá dữ liệu

```bash
python 01_explore_data.py
```

### 3. Chạy Baseline truyền thống

```bash
# V1: Otsu thresholding
python 07_baseline_traditional.py

# V2: Fixed threshold với validation
python 07b_baseline_improved.py

# V3: NBR + NDWI (khuyến nghị)
python 07c_baseline_v3.py
```

### 4. Train Deep Learning Models

```bash
# U-Net v2
python 08_train_unet_v2.py

# UPerNet v2
python 12_train_upernet_v2.py
```

### 5. So sánh các model

```bash
# So sánh trực quan
python 09_compare_models.py

# Classification test
python 10_classification_test.py
```

### 6. Thống kê diện tích

```bash
python 14_thong_ke_dien_tich_test.py
python 15_minh_hoa_dien_tich.py
```

### 7. Chạy Web App

```bash
streamlit run 17_web_app.py
```

Mở trình duyệt: http://localhost:8501

---

## Kiến thức nền tảng

### Chỉ số NBR (Normalized Burn Ratio)

$$NBR = \frac{NIR - SWIR2}{NIR + SWIR2}$$

| Giá trị NBR | Ý nghĩa |
|-------------|----------|
| Gần +1 | Rừng khỏe mạnh |
| Gần 0 | Đất trống, thực vật khô |
| Gần -1 | Vùng cháy hoặc không có thực vật |

### Chỉ số NDWI (Normalized Difference Water Index)

$$NDWI = \frac{Green - NIR}{Green + NIR}$$

| Giá trị NDWI | Ý nghĩa |
|-------------|----------|
| > 0 | Mặt nước |
| < 0 | Đất liền |

### Kiến trúc U-Net

```
Encoder (Down)
─────────────
Input [B, 12, 256, 256]
  ↓
Conv3×3 + BN + ReLU  →  [B, 64, 256, 256]
  ↓ MaxPool
Conv3×3 + BN + ReLU  →  [B, 128, 128, 128]
  ↓ MaxPool
Conv3×3 + BN + ReLU  →  [B, 256, 64, 64]
  ↓ MaxPool
Conv3×3 + BN + ReLU  →  [B, 512, 32, 32]
  ↓ MaxPool
Bottleneck           →  [B, 1024, 16, 16]

Decoder (Up)
─────────────
  ↑ ConvTranspose
Concat + Conv3×3     →  [B, 512, 32, 32]
  ↑ ConvTranspose
Concat + Conv3×3     →  [B, 256, 64, 64]
  ↑ ConvTranspose
Concat + Conv3×3     →  [B, 128, 128, 128]
  ↑ ConvTranspose
Concat + Conv3×3     →  [B, 64, 256, 256]
  ↓
Conv1×1              →  [B, 1, 256, 256]
```

---

## Các file kết quả

Sau khi chạy các script, kết quả sẽ được lưu trong các thư mục:

```
├── ket_qua_baseline/           # Kết quả baseline V1
├── ket_qua_baseline_v2/        # Kết quả baseline V2
├── ket_qua_baseline_v3/        # Kết quả baseline V3
├── ket_qua_unet/               # Kết quả U-Net v1
├── ket_qua_unet_v2/            # Kết quả U-Net v2
├── ket_qua_upernet/            # Kết quả UPerNet v1
├── ket_qua_upernet_v2/        # Kết quả UPerNet v2
├── ket_qua_so_sanh/            # Hình so sánh 3 model
├── ket_qua_classification/     # Kết quả phân loại
├── ket_qua_dien_tich/          # Thống kê diện tích
├── ket_qua_thong_ke_chi_tiet/  # Thống kê dataset
└── mau_anh.png                 # Ảnh mẫu từ EDA
```

---

## Đóng góp

Đồ án này được thực hiện bởi **Nhóm 11** - Lớp XLAS - Xử lý ảnh số.

### Giảng viên hướng dẫn
- **ThS. Võ Lê Phúc Hậu** - Khoa CNTT

---

## Liên hệ

Mọi thắc mắc về project, vui lòng liên hệ qua SĐT: 0948843367.

---

## License

MIT License
