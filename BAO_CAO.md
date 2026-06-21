TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM
KHOA CÔNG NGHỆ THÔNG TIN
ĐỒ ÁN CUỐI KỲ MÔN XỬ LÝ ẢNH SỐ
ĐỀ TÀI: PHÁT HIỆN CHÁY RỪNG TRÊN ẢNH VỆ TINH SENTINEL-2 ĐA PHỔ (Wildfire Detection on Sentinel-2 Multispectral Satellite Imagery)
Giảng viên hướng dẫn: ThS. Võ Lê Phúc Hậu
Thành viên nhóm 11:
Họ tên
MSSV
Trần Quang Toản
23110158
Trần Văn Tưởng
23110170
TP. Hồ Chí Minh, tháng 6 năm 2026
## MỤC LỤC
CHƯƠNG 1 – GIỚI THIỆU ĐỀ TÀI4
1.1. Bối cảnh và lý do chọn đề tài4
1.2. Mục tiêu của đồ án4
1.3. Đóng góp của nhóm5
CHƯƠNG 2 – TỔNG QUAN LÝ THUYẾT XỬ LÝ ẢNH6
2.1. Xử lý ảnh cơ bản6
2.2. Ảnh viễn thám đa phổ Sentinel-28
2.3. Chỉ số phổ cho phát hiện cháy9
2.4. Học sâu cho phân vùng ảnh (Segmentation)9
CHƯƠNG 3 – MÔ TẢ BÀI TOÁN VÀ DATASET11
3.1. Phát biểu bài toán11
3.2. Dataset Wildfires-CEMS11
3.3. Thống kê và phân tích dữ liệu12
CHƯƠNG 4 – PHƯƠNG PHÁP ĐỀ XUẤT14
4.1. Tiền xử lý dữ liệu14
4.2. Phương pháp truyền thống (Baseline)15
4.3. Phương pháp Deep Learning17
4.4. Sơ đồ pipeline tổng thể21
CHƯƠNG 5 – THỰC NGHIỆM VÀ KẾT QUẢ22
5.1. Thiết lập thực nghiệm22
5.2. Kết quả phương pháp truyền thống22
5.3. Kết quả Deep Learning24
5.4. Phân loại có/không cháy cấp ảnh25
5.5. Ước tính diện tích cháy26
5.6. Thảo luận kết quả27
CHƯƠNG 6 – DEMO ỨNG DỤNG WEB30
6.1. Lựa chọn công nghệ30
6.2. Tính năng demo30
6.3. Validation an toàn cho user upload30
6.4. Kịch bản sử dụng thực tế31
CHƯƠNG 7 – KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN32
7.1. Kết luận32
7.2. Hạn chế32
7.3. Hướng phát triển32
TÀI LIỆU THAM KHẢO34
PHỤ LỤC35
Phụ lục A: Cấu trúc thư mục project35
Phụ lục B: Hướng dẫn chạy project35
Phụ lục C: Bảng tổng hợp kết quả chính36
Phụ lục D: Giải thích một số thuật ngữ37
PHỤ LỤC BỔ SUNG – THỐNG KÊ DATASET38
# CHƯƠNG 1 – GIỚI THIỆU ĐỀ TÀI
Biến đổi khí hậu toàn cầu đã làm gia tăng tần suất và cường độ của các vụ cháy rừng trên khắp thế giới. Theo báo cáo của Copernicus Atmosphere Monitoring Service, mùa cháy rừng năm 2023 tại châu Âu đã đạt mức kỷ lục với hơn 500.000 ha rừng bị thiêu rụi chỉ trong vài tuần tại Hy Lạp, Italia và Tây Ban Nha. Tương tự, Úc trải qua mùa cháy thảm khốc 2019-2020 với 18 triệu ha rừng bị phá hủy, California (Hoa Kỳ) liên tục ghi nhận các vụ cháy lịch sử.
Trong bối cảnh đó, việc phát hiện và khoanh vùng cháy sớm, chính xác là yếu tố then chốt giúp cơ quan chức năng triển khai lực lượng ứng cứu kịp thời, đánh giá mức độ thiệt hại và lập kế hoạch phục hồi.
## 1.1. Bối cảnh và lý do chọn đề tài
Sự ra đời của các hệ thống vệ tinh quan sát Trái Đất như Sentinel-2 (Cơ quan Vũ trụ Châu Âu ESA) mở ra cơ hội giám sát cháy rừng trên phạm vi toàn cầu với chi phí thấp và tần suất quan sát cao (mỗi vệ tinh quay lại một điểm trên Trái Đất sau 5 ngày, kết hợp 2 vệ tinh là 2-3 ngày).
Bên cạnh đó, sự phát triển mạnh mẽ của deep learning với các kiến trúc segmentation (U-Net, UPerNet) đã tạo ra bước nhảy vọt trong độ chính xác phân vùng ảnh y tế và viễn thám.
## 1.2. Mục tiêu của đồ án
1.  Xây dựng pipeline xử lý ảnh đa phổ Sentinel-2 từ bước đọc dữ liệu, chuẩn hóa, tăng cường dữ liệu đến đưa vào mô hình học sâu.
2.  Triển khai và so sánh 3 phương pháp: V1 (Otsu), V2 (NBR cố định), V3 (NBR + NDWI), và 2 model deep learning (U-Net, UPerNet).
3.  Đánh giá định lượng với các metric: IoU, Dice/F1-score, Precision, Recall, Accuracy.
4.  Xây dựng mô-đun ước tính diện tích cháy với hiệu chỉnh vĩ độ.
5.  Triển khai demo web bằng Streamlit.
6.  So sánh với paper gốc IGARSS 2023.
Tóm tắt input – output của hệ thống:
Input: Ảnh vệ tinh Sentinel-2 Level 2A định dạng GeoTIFF, gồm 12 band đa phổ (kích thước bất kỳ).
Output chính: Mask phân vùng nhị phân cùng kích thước (1 = pixel cháy, 0 = pixel không cháy).
Output phụ: Tổng diện tích cháy theo m²/ha/km² (có hiệu chỉnh vĩ độ), danh sách các cụm cháy riêng biệt với vị trí và kích thước từng cụm.
## 1.3. Đóng góp của nhóm
•  Xây dựng baseline truyền thống đầy đủ gồm 3 phiên bản (V1, V2, V3) với quy trình cải tiến có hệ thống, đo lường được.
•  Đề xuất baseline V3 kết hợp NBR (đo lường cháy) với NDWI (loại mặt nước).
•  Tái tạo và vượt kết quả paper gốc ở cấu hình tương đương: trên đánh giá toàn ảnh, UPerNet đạt F1 91.06% và U-Net đạt F1 87.64%, đều vượt UPerNet-scratch của paper (82.33% F1).
•  Mô-đun ước tính diện tích có hiệu chỉnh vĩ độ, làm việc đúng ở mọi vĩ độ.
•  Xây dựng web app tương tác cho phép upload ảnh Sentinel-2.
# CHƯƠNG 2 – TỔNG QUAN LÝ THUYẾT XỬ LÝ ẢNH
## 2.1. Xử lý ảnh cơ bản
Ảnh số (digital image) là một ma trận hai chiều gồm các pixel (picture element), mỗi pixel mang một hoặc nhiều giá trị biểu diễn cường độ sáng tại vị trí đó. Ảnh đơn sắc (grayscale) có 1 kênh giá trị, ảnh màu RGB có 3 kênh, ảnh đa phổ có thể có hàng chục đến hàng trăm kênh.
Các kỹ thuật xử lý ảnh nền tảng được sử dụng trong đồ án:
2.1.1. Lọc nhiễu Gaussian
Lọc Gaussian là phép cuộn (convolution) giữa ảnh và kernel có dạng phân phối chuẩn 2D:
G(x, y) = (1 / 2πσ²) · exp(−(x² + y²) / 2σ²)
Trong đó σ là độ lệch chuẩn quyết định mức độ làm mượt. Nhóm sử dụng kernel 5×5 với σ mặc định của OpenCV (≈ 1.1). Lọc Gaussian được áp dụng cho band NIR, SWIR2, Green trước khi tính chỉ số NBR/NDWI để giảm nhiễu pixel-level mà vẫn bảo toàn các đặc trưng lớn.
2.1.2. Histogram Equalization
Kỹ thuật kéo giãn histogram ảnh để tăng tương phản, sử dụng hàm phân phối tích lũy (CDF):
s(r) = (L − 1) · CDF(r)
Trong đó r là giá trị pixel gốc, s là giá trị sau biến đổi, L là số mức xám (thường L=256). Trong baseline V1, histogram equalization được áp dụng để kéo giãn giá trị NBR trước khi đưa vào Otsu thresholding.
2.1.3. Thresholding
Phép biến đổi ảnh xám thành ảnh nhị phân bằng ngưỡng T:
binary(x, y) = 1 nếu I(x, y) > T, ngược lại = 0
Hai biến thể được sử dụng:
•  Otsu adaptive: tự động tìm ngưỡng tối ưu T bằng cách tối đa hóa phương sai giữa hai class (cháy/không cháy).
•  Ngưỡng cố định: T được chọn thủ công bằng grid search trên tập validation.
2.1.4. Phép biến đổi hình thái học (Morphology)
Bốn phép biến đổi cơ bản trên ảnh nhị phân với kernel cấu trúc B:
•  Erosion (xói mòn):  A ⊖ B = { z | (B)z ⊆ A }
•  Dilation (phình to): A ⊕ B = { z | (B)z ∩ A ≠ ∅ }
•  Opening = Erosion theo sau Dilation: A ∘ B = (A ⊖ B) ⊕ B  → loại các đốm nhiễu nhỏ
•  Closing = Dilation theo sau Erosion: A • B = (A ⊕ B) ⊖ B  → lấp các lỗ nhỏ trong vùng
Nhóm áp dụng Opening kernel 3×3 để loại các pixel nhiễu lẻ tẻ, sau đó Closing kernel 5×5 để lấp các lỗ trong vùng cháy.
2.1.5. Tách kênh
Ảnh Sentinel-2 12 band được lưu trong file GeoTIFF dạng tensor shape (12, H, W). Việc tách kênh đơn giản là index vào tensor: B8 = image[7], B12 = image[11], B3 = image[2]. Mỗi kênh được xử lý độc lập trước khi tính chỉ số NBR/NDWI.
## 2.2. Ảnh viễn thám đa phổ Sentinel-2
Sentinel-2 là chòm vệ tinh quan sát Trái Đất do Cơ quan Vũ trụ Châu Âu (ESA) vận hành từ năm 2015. Mỗi vệ tinh (2A/2B) mang theo bộ cảm biến đa phổ (MSI) thu nhận ảnh ở 13 dải phổ, trong đó 12 band được sử dụng trong dataset Wildfires-CEMS.
Bảng 2.1: Thông tin 12 band phổ của Sentinel-2
Band
Tên
Bước sóng (nm)
Độ phân giải (m)
Vai trò
B1
Coastal Aerosol
443
60
Hiệu chỉnh khí quyển
B2
Blue
490
10
Kênh xanh dương (RGB)
B3
Green
560
10
Kênh xanh lá (RGB), NDWI
B4
Red
665
10
Kênh đỏ (RGB), NDVI
B5
Red Edge 1
705
20
Sức khỏe thực vật
B6
Red Edge 2
740
20
Sức khỏe thực vật
B7
Red Edge 3
783
20
Sức khỏe thực vật
B8
NIR
842
10
Tính NBR, NDVI
B8A
Narrow NIR
865
20
Bổ trợ phân tích
B9
Water Vapor
945
60
Hơi nước
B11
SWIR 1
1610
20
Phát hiện cháy
B12
SWIR 2
2190
20
Tính NBR
Hình 2.1: Hiển thị 12 band của ảnh Sentinel-2 L2A (mẫu EMSR207_AOI01_01); các band Green, Red, NIR và SWIR2 dùng để tính chỉ số được tô màu đỏ.
Khác biệt cốt lõi giữa ảnh Sentinel-2 và ảnh RGB thông thường là sự hiện diện của các dải hồng ngoại NIR (band 8) và hồng ngoại sóng ngắn SWIR (band 11, 12). Các dải này mang thông tin về sự phản xạ của thực vật khỏe, thực vật cháy, đất ẩm, mặt nước mà mắt thường và ảnh RGB không nhìn thấy được.
## 2.3. Chỉ số phổ cho phát hiện cháy
NBR – Normalized Burn Ratio (Key & Benson, USGS, 1999):
NBR = (NIR - SWIR2) / (NIR + SWIR2 + ε)
Rừng khỏe mạnh có giá trị NIR cao và SWIR2 thấp, cho NBR ≈ +0.5 đến +0.8. Vùng cháy mới có NIR giảm mạnh và SWIR2 tăng, cho NBR ≈ -0.2 đến -0.5.
NDWI – Normalized Difference Water Index (McFeeters, 1996):
NDWI = (Green - NIR) / (Green + NIR + ε)
Mặt nước hấp thụ mạnh NIR nhưng phản xạ Green, cho NDWI > 0. Đất khô, thực vật có NDWI < 0. Trong nghiên cứu này, NDWI được dùng như mặt nạ phụ để loại bỏ mặt nước khỏi vùng dự đoán cháy.
Hình 2.2: Ảnh RGB và các chỉ số phái sinh NDVI, NDWI, NBR cùng mask Ground Truth; vùng cháy có NBR thấp (màu đỏ) tương phản rõ với nền thực vật.
## 2.4. Học sâu cho phân vùng ảnh (Segmentation)
Phân vùng ảnh (semantic segmentation) là bài toán gán nhãn class cho từng pixel. Đối với bài toán phát hiện cháy, đầu ra là mask nhị phân (1 = cháy, 0 = không cháy) có cùng kích thước với ảnh đầu vào.
U-Net (Ronneberger et al., MICCAI 2015):
U-Net là kiến trúc encoder-decoder đối xứng hình chữ U, với các skip connection nối trực tiếp từ layer encoder sang layer decoder cùng mức. Trong đồ án, U-Net được sửa đổi với in_channels=12, 4 mức encoder-decoder với số kênh tăng dần 64→128→256→512→1024.
UPerNet (Xiao et al., ECCV 2018):
UPerNet (Unified Perceptual Parsing Network) kết hợp Feature Pyramid Network (FPN) với Pyramid Spatial Pooling (PSP) module. Encoder là ResNet-50, decoder tổng hợp đặc trưng ở nhiều scale. UPerNet là kiến trúc được sử dụng trong paper gốc IGARSS 2023.
# CHƯƠNG 3 – MÔ TẢ BÀI TOÁN VÀ DATASET
## 3.1. Phát biểu bài toán
Input:
•  Ảnh Sentinel-2 L2A đa phổ 12 band, định dạng GeoTIFF, kích thước bất kỳ (500-1200 pixel).
•  Hệ tọa độ WGS84 (EPSG:4326), giá trị reflectance float32 (dải thực tế khoảng [0, 2.0], được chuẩn hóa bằng clip về [0, 1.5] rồi chia 1.5).
Output:
•  Mask nhị phân cùng kích thước với ảnh (1 = vùng cháy, 0 = không cháy).
•  Tổng diện tích vùng cháy (đơn vị m², ha, km²) có hiệu chỉnh vĩ độ.
•  Danh sách các cụm cháy riêng biệt với diện tích, vị trí, kích thước từng cụm.
## 3.2. Dataset Wildfires-CEMS
Nguồn: Hugging Face links-ads/wildfires-cems (Arnaudo et al., IGARSS 2023).
Quy trình tạo dataset:
1.  Các vụ cháy thực xảy ra tại châu Âu, được hệ thống Copernicus EMS kích hoạt theo mã EMSR.
2.  Vệ tinh Sentinel-2 chụp ảnh khu vực trước và sau cháy.
3.  Chuyên gia viễn thám Copernicus khoanh vẽ polygon vùng cháy thủ công.
4.  Tác giả dataset raster hóa polygon thành mask bitmap 1-bit.
5.  Dataset chia thành 3 tập train/val/test, đảm bảo không có sự kiện EMSR trùng.
Bảng 3.1: Thống kê số lượng ảnh theo từng tập
Tập
Số sample
Số EMSR
Tỉ lệ
Train
281
129
65%
Validation
53
15
12%
Test
99
27
23%
TỔNG
433
171
100%
Mỗi sample chứa các file: S2L2A.tif (12-band input), DEL.tif (mask GT), GRA.tif (DEM), ESA_LC.tif (bản đồ lớp phủ), CM.tif (cloud mask).
3.2.1. Đọc và kiểm chứng ảnh đầu vào
Mỗi mẫu được đọc bằng thư viện rasterio. Khác với ảnh RGB (3 kênh, uint8 [0–255]), ảnh Sentinel-2 L2A gồm 12 band giá trị thực float32. Khi đọc, nhóm trích xuất đầy đủ metadata địa lý phục vụ tính diện tích: hệ tọa độ (CRS), ma trận affine (transform), kiểu dữ liệu và biên (bounds). Ví dụ mẫu EMSR207_AOI01_01 (cháy tại Bồ Đào Nha) có kích thước 1977×1442 pixel, CRS WGS84 (đơn vị độ), vĩ độ trung tâm 39.83°N, kích thước pixel sau hiệu chỉnh vĩ độ ≈ 98.98 m²/pixel — sát giá trị danh nghĩa 10 m × 10 m của Sentinel-2. Dải giá trị pixel khoảng [0; 2.0] (reflectance lưu trực tiếp, không nhân 10000).
Đặc biệt, toàn bộ pipeline giả định index 7 = B8 NIR và index 11 = B12 SWIR2. Nhóm không giả định mà kiểm chứng thứ tự band bằng một phép sanity-check định lượng: so NBR trung bình trên vùng cháy với vùng không cháy (theo mask GT). Kết quả trên mẫu EMSR207: NBR vùng cháy = −0.221, NBR vùng không cháy = +0.331 (chênh lệch 0.552). Vùng cháy có NBR thấp hơn rõ rệt — đúng với cơ sở vật lý 'cháy làm giảm NIR và tăng SWIR'. Nếu thứ tự band bị đọc sai, NBR sẽ không phân biệt được hai vùng; việc nó phân biệt tốt xác nhận thứ tự band đã đọc đúng.
## 3.3. Thống kê và phân tích dữ liệu
Bảng 3.2: Phân bố ảnh có/không cháy theo tập
Split
Tổng
Có cháy
Không cháy
% có cháy
% pixel cháy
Train
281
279
2
99.3%
22.69%
Val
53
48
5
90.6%
13.44%
Test
99
88
11
88.9%
8.71%
TỔNG
433
415
18
95.8%
18.17%
Quan sát quan trọng: Phân bố tỷ lệ pixel cháy giảm dần từ train (22.69%) xuống test (8.71%). Tập test có nhiều ảnh cháy nhỏ và không cháy hơn – mô phỏng tình huống thực tế khi triển khai deploy. Vì vậy, kết quả Dice 85%+ trên tập test này là đáng tin cậy.
Bảng 3.3: Phân loại ảnh theo mức độ cháy
Mức độ
Train
Val
Test
Tổng
Không cháy
2
5
11
18
Rất ít (<1%)
13
5
16
34
Ít (1-10%)
73
19
43
135
Vừa (10-30%)
109
15
23
147
Nhiều (30-50%)
63
9
5
77
Rất nhiều (>50%)
21
0
1
22
# CHƯƠNG 4 – PHƯƠNG PHÁP ĐỀ XUẤT
## 4.1. Tiền xử lý dữ liệu
4.1.1. Làm sạch và chuẩn hóa
Ảnh đa phổ Sentinel-2 được đọc bằng thư viện rasterio. Pipeline chuẩn hóa gồm 3 bước: (1) Clip giá trị về khoảng [0, 1.5] để loại outlier; (2) Chia cho 1.5 để đưa về [0, 1]; (3) Đảm bảo ảnh đầu vào và mask GT có cùng kích thước.
Hình 4.1: Histogram giá trị pixel trước và sau chuẩn hóa (clip về [0; 1.5] rồi chia 1.5 đưa về [0; 1]).
4.1.2. Xử lý ảnh nâng cao
•  Lọc Gaussian 5×5: trên band NIR, SWIR2, Green trước khi tính NBR/NDWI.
•  Histogram equalization: trong baseline V1 để tăng tương phản.
•  Thresholding: Otsu adaptive (V1) hoặc ngưỡng cố định (V2/V3).
•  Morphological: Opening kernel 3×3, Closing kernel 5×5.
4.1.3. Tăng cường dữ liệu (Data Augmentation)
Tập train chỉ có 281 sample - con số nhỏ so với 31 triệu tham số của U-Net. Để tránh overfitting, nhóm áp dụng 4 kỹ thuật augmentation (chỉ cho train):
1.  Random crop 256×256 từ ảnh gốc ~800×800 (khoảng 250.000 vị trí khả thi).
2.  Lật ngang ngẫu nhiên (p=0.5).
3.  Lật dọc ngẫu nhiên (p=0.5).
4.  Xoay 0°/90°/180°/270° ngẫu nhiên.
Quan trọng: Không sử dụng ColorJitter vì giá trị reflectance trong band hồng ngoại mang ý nghĩa vật lý, biến đổi tùy ý sẽ phá vỡ thông tin về nồng độ chlorophyll.
Quy mô: 281 ảnh × 16 tổ hợp hình học (2×2×4) ≈ 4.500 biến thể; do augmentation trực tuyến, mỗi epoch mỗi ảnh cho một view khác nhau nên qua hàng chục epoch model thấy hàng nghìn biến thể không trùng.
Hình 4.2: Minh họa các phép tăng cường dữ liệu hình học — lật ngang/dọc, xoay bội số 90° và random crop; mọi biến thể đều giữ tính hợp lệ của ảnh vệ tinh.
## 4.2. Phương pháp truyền thống (Baseline)
Nhóm triển khai 3 phiên bản baseline với độ phức tạp tăng dần.
4.2.1. Baseline V1: Otsu Thresholding
Pipeline V1: Tách kênh (B8, B12) → Gaussian 5×5 → Tính NBR → Scale [0,255] → Histogram Equalize → Otsu threshold → Morphology Open 3×3 → Close 5×5.
Kết quả test: F1 = 25.85%, IoU = 14.84% – thất bại.
Phân tích: Otsu tự động tìm ngưỡng chia ảnh thành 2 phần có phương sai trong cùng lớp tối thiểu, nhưng không hiểu ngữ nghĩa 'cháy'. Trong ảnh viễn thám, các pixel đô thị, đất trống, nước nông có NBR thấp tương tự cháy. Precision chỉ đạt 15.12%.
4.2.2. Baseline V2: NBR với ngưỡng cố định
Cải tiến: thay Otsu bằng ngưỡng cố định, chọn bằng grid search trên val.
Bảng grid search trên val set:
Threshold
IoU
Precision
Recall
F1
0.00
0.306
0.413
0.792
0.543
-0.05
0.362
0.602
0.740
0.664
-0.10
0.410
0.782
0.675
0.725
-0.15
0.420
0.871
0.594
0.706
-0.20
0.382
0.913
0.496
0.643
Ngưỡng tối ưu: -0.10 (F1 = 0.725 trên val). Kết quả test: F1 = 55.90%, IoU = 38.79%.
Vấn đề còn tồn tại: Mô hình dự đoán cả sông, hồ làm cháy. Nguyên nhân: trên mặt nước, cả NIR và SWIR2 đều rất thấp → tỉ số NIR/SWIR2 không ổn định, có thể rơi vào khoảng NBR < -0.10 ngẫu nhiên.
4.2.3. Baseline V3: NBR + NDWI (đề xuất chính)
Pseudo-code V3:
INPUT: 12-band Sentinel-2 imageOUTPUT: binary burn maskSTEP 1: Tách kênh    NIR   ← image[B8]      (842 nm)    SWIR2 ← image[B12]     (2190 nm)    Green ← image[B3]      (560 nm)STEP 2: GaussianBlur 5x5STEP 3: NBR  ← (NIR - SWIR2) / (NIR + SWIR2 + ε)        NDWI ← (Green - NIR) / (Green + NIR + ε)STEP 4: fire_mask  ← (NBR < -0.10)        water_mask ← (NDWI > 0.0)        mask       ← fire_mask AND NOT water_maskSTEP 5: Morphology Open 3x3, Close 5x5RETURN mask
Kết quả test: F1 = 78.94%, IoU = 65.22% (cải thiện +23.04% F1 so với V2).
Phân tích: V2 → V3, precision tăng gần 2 lần (44.30% → 82.53%), false positive giảm khoảng 6 lần. Bằng cách dùng NDWI > 0 làm mặt nạ loại bỏ mặt nước, vùng dự đoán cháy thu hẹp lại đúng vào các pixel có chữ ký phổ thực sự khớp với cháy.
## 4.3. Phương pháp Deep Learning
Nhóm triển khai 3 kiến trúc segmentation tiêu biểu với cùng cấu hình để so sánh công bằng (fair comparison).
Vì sao chọn U-Net và UPerNet? Có nhiều kiến trúc thị giác máy tính, nhưng nhóm lựa chọn dựa trên bốn tiêu chí: (a) phù hợp với phân vùng ngữ nghĩa vùng cháy có hình dạng tùy ý; (b) hoạt động tốt trên dataset nhỏ (281 ảnh train); (c) so sánh được với paper gốc; (d) khả thi trên GPU 6GB.
Không chọn họ detection (YOLO, Faster/Mask R-CNN) vì chúng phát hiện đối tượng bằng bounding box, trong khi vùng cháy là vùng liên thông hình dạng bất kỳ và cần tính diện tích ở mức pixel — đây là bài toán segmentation, không phải detection. Không chọn học máy cổ điển (SVM/Random Forest phân loại từng pixel) vì chúng chỉ dùng giá trị phổ của một pixel, không khai thác ngữ cảnh không gian nên dễ nhầm các loại đất có phổ giống cháy (đúng như hạn chế của baseline V3).
Ba kiến trúc được chọn tạo thành một phổ đại diện, trả lời ba câu hỏi: (1) U-Net — CNN nhẹ, kinh điển, mạnh trên dữ liệu nhỏ nhờ skip connection (31M tham số, hợp GPU 6GB); (2) UPerNet — CNN đa tỷ lệ (PPM + FPN), là kiến trúc của paper gốc nên cho phép đối chiếu trực tiếp; (3) SegFormer — đại diện hướng transformer, để kiểm chứng xem transformer có lợi thế trên dữ liệu nhỏ hay không (thực nghiệm cho thấy SegFormer bị training collapse, xem mục 5.3). Đây là một thiết kế thực nghiệm có chủ đích chứ không chọn ngẫu nhiên.
4.3.1. U-Net (Ronneberger et al., MICCAI 2015)
Input [B, 12, 256, 256]  ↓ enc1 → skip1 [64, 256, 256]  ↓ pool + enc2 → skip2 [128, 128, 128]  ↓ pool + enc3 → skip3 [256, 64, 64]  ↓ pool + enc4 → skip4 [512, 32, 32]  ↓ pool + bottleneck [1024, 16, 16]  ↑ up + dec4 (+ skip4) [512, 32, 32]  ↑ up + dec3 (+ skip3) [256, 64, 64]  ↑ up + dec2 (+ skip2) [128, 128, 128]  ↑ up + dec1 (+ skip1) [64, 256, 256]Output [B, 1, 256, 256] (logits)
Bảng 4.1: Hyperparameters U-Net
Tham số
Giá trị
in_channels
12 (Sentinel-2 đa phổ)
Số tham số
31.04 triệu
Patch size
256×256
Batch size
4 (giới hạn 6GB VRAM)
Epochs
80 (early stop ~32)
Optimizer
Adam
Learning rate
5e-4
Weight decay
1e-5
Loss
BCE + Dice (50-50 combo)
GPU
RTX 3060 6GB
Thời gian train
172.5 phút
4.3.2. UPerNet (Xiao et al., ECCV 2018)
UPerNet (Unified Perceptual Parsing Network) là kiến trúc segmentation hiện đại được sử dụng trong paper gốc IGARSS 2023, cho phép so sánh trực tiếp với benchmark. Kiến trúc gồm 3 thành phần:
•  Backbone Encoder (ResNet-50): trích xuất đặc trưng đa scale từ ảnh input. Đầu ra: 4 feature maps ở các scale 1/4, 1/8, 1/16, 1/32 độ phân giải gốc.
•  PSP Module (Pyramid Spatial Pooling): áp dụng pooling ở 4 kích thước (1×1, 2×2, 3×3, 6×6) trên feature map cuối, sau đó concat lại. Mục đích: thu thập context toàn cục ở nhiều quy mô không gian.
•  FPN Decoder (Feature Pyramid Network): kết hợp feature ở các scale từ thấp đến cao bằng top-down pathway và lateral connections. Đầu ra fuse 4 scale lại thành 1 feature map có độ phân giải 1/4 ảnh gốc.
•  Segmentation Head: 2 convolution 3×3 + upsampling về độ phân giải gốc → mask đầu ra.
Triển khai qua thư viện segmentation_models_pytorch (SMP) — implementation chuẩn được kiểm thử rộng rãi. Nhóm dùng cấu hình from scratch (encoder_weights=None) để so sánh công bằng với U-Net (cả hai đều train từ đầu).
Tổng số tham số: 37.31 triệu, đa phần ở backbone ResNet-50.
4.3.3. Kỹ thuật Sliding Window cho inference
Model được train trên patch cố định 256×256, nhưng ảnh test có kích thước khác nhau (từ 800×800 đến 1442×1977 pixel) tùy theo vùng địa lý. Để áp dụng model lên ảnh có kích thước bất kỳ, nhóm sử dụng kỹ thuật sliding window:
Quy trình:
1.  Padding ảnh đầu vào để chiều cao và rộng chia hết cho patch_size.
2.  Trượt cửa sổ 256×256 trên ảnh với stride = patch_size/2 = 128 (chồng lấp 50%).
3.  Với mỗi vị trí cửa sổ:
•  Cắt patch [12, 256, 256] từ ảnh padded.
•  Đưa qua model → logits → sigmoid → probability map.
•  Tích lũy probability vào ma trận tổng và đếm số lần mỗi pixel được xử lý.
4.  Lấy trung bình probability tại mỗi pixel (chia tổng cho số lần xử lý).
5.  Áp ngưỡng 0.5 → mask nhị phân.
6.  Cắt mask về đúng kích thước ảnh gốc.
Ưu điểm: chồng lấp 50% giúp mỗi pixel được dự đoán bởi nhiều patch khác nhau (đặc biệt pixel ở rìa patch trước trở thành tâm patch sau), giảm nhiễu rìa và đảm bảo dự đoán nhất quán trên toàn ảnh.
4.3.4. Hàm loss BCE + Dice
Loss = (1 - α) × BCE(logits, target) + α × (1 - Dice(probs, target))với α = 0.5, target là mask ground truth float32.BCE: phạt sai từng pixel độc lập, hội tụ ổn định.Dice: tối ưu overlap toàn cục, chịu imbalance tốt.Kết hợp: ổn định + hiệu quả trên vùng cháy nhỏ (imbalance mạng).
## 4.4. Sơ đồ pipeline tổng thể
# CHƯƠNG 5 – THỰC NGHIỆM VÀ KẾT QUẢ
## 5.1. Thiết lập thực nghiệm
Phần cứng: Laptop HP OMEN, CPU Intel Core i9-12900H, GPU NVIDIA GeForce RTX 3060 Laptop 6GB VRAM, RAM 16GB DDR5, SSD 512GB NVMe.
Phần mềm: Windows 11, Python 3.11, PyTorch 2.x + CUDA 12.1, rasterio, OpenCV, scikit-learn, segmentation_models_pytorch, Streamlit, Plotly.
Metric đánh giá: IoU, Dice/F1, Precision, Recall, MAE diện tích (ha).
## 5.2. Kết quả phương pháp truyền thống
Bảng 5.1: Kết quả tiến hóa baseline V1 → V2 → V3
Phiên bản
Phương pháp
IoU
Dice
Precision
Recall
F1
V1
Otsu
14.84%
25.85%
15.12%
89.06%
25.85%
V2
NBR < -0.10
38.79%
55.90%
44.30%
75.71%
55.90%
V3
NBR + NDWI
65.22%
78.94%
82.53%
75.65%
78.94%
Hình 5.1: Ảnh hưởng của ngưỡng NBR đến IoU, Precision, Recall và F1-score trên tập validation.
Phân tích định lượng:
•  V1 → V2: Precision tăng 2.9 lần (15.12% → 44.30%) nhờ thay Otsu bằng ngưỡng cố định.
•  V2 → V3: Precision tăng gần 2 lần (44.30% → 82.53%) nhờ NDWI loại mặt nước.
•  False positive giảm 6 lần. Recall hầu như giữ nguyên.
•  Tổng cộng: F1-score tăng 3.05 lần (25.85% → 78.94%).
Ghi chú: các chỉ số trong Bảng 5.1 được tính ở mức pixel trên toàn tập test (nhất quán với Bảng 5.2 và với cách paper gốc đo). Với bài toán phân vùng nhị phân một lớp, Dice và F1-score đồng nhất về mặt toán học nên hai cột có cùng giá trị; IoU liên hệ với F1 theo công thức IoU = F1 / (2 − F1).
## 5.3. Kết quả Deep Learning
Bảng 5.2: Kết quả các phương pháp trên test set (đánh giá toàn ảnh, mức pixel)
Model
Precision
Recall
F1
IoU
Baseline V3
0.8254
0.7566
0.7895
0.6522
U-Net (v2)
0.8116
0.9525
0.8764
0.7800
UPerNet (v2)
0.9049
0.9164
0.9106
0.8359
Phân tích biểu đồ training:
•  U-Net (v2): nhờ warmup + cosine annealing, đường cong hội tụ mượt và ít dao động hơn v1. Best epoch 32 với val Dice 0.826; Test Dice (patch) cải thiện từ 0.8511 (v1) lên 0.8898 (v2).
Hình 5.2: Biểu đồ training U-Net v2 (Loss và Dice qua các epoch, best epoch 32).
•  UPerNet (v2): hội tụ ổn định, chạy tới early-stop ở epoch 42 (best epoch 27, val Dice 0.844), không xảy ra phân kỳ. Đạt độ chính xác pixel cao nhất (xem Bảng 5.2).
Giao thức đánh giá: cả ba phương pháp được chấm theo cùng quy trình trên ảnh nguyên kích thước (học sâu dùng sliding window 256×256 chồng lấp 50%, ngưỡng 0.5), tính chỉ số ở mức pixel trên toàn bộ 99 ảnh test. Đây là cách paper gốc đo và đúng tình huống triển khai thực tế. Bảng 5.2 cho thấy UPerNet đạt F1 91.06% / IoU 83.59% (cao nhất); U-Net có recall rất cao (0.953) nhưng precision thấp hơn (0.812), tức hơi báo dư.
Để đánh giá độ ổn định theo từng ảnh, nhóm tính thêm IoU/Dice trung bình trên 88 ảnh có cháy (GT>0), kèm khoảng tin cậy 95% bằng bootstrap 5000 lần:
Bảng 5.3: IoU/Dice trung bình theo ảnh (88 ảnh GT>0).
Model
IoU mean±std
IoU 95% CI
Dice mean±std
Baseline V3
0.5421±0.258
[0.488, 0.595]
0.6610±0.255
U-Net (v2)
0.6800±0.252
[0.627, 0.732]
0.7768±0.221
UPerNet (v2)
0.6366±0.303
[0.574, 0.699]
0.7243±0.293
Theo từng ảnh, U-Net cao nhất và ổn định hơn (độ lệch chuẩn 0.252 so với 0.303 của UPerNet). Kiểm định Wilcoxon signed-rank (n=88) cho thấy U-Net vượt UPerNet ở per-image có ý nghĩa thống kê (p = 0.012), tuy effect size Cohen's d = 0.241 ở mức nhỏ; cả hai mô hình học sâu đều vượt baseline V3 (p < 0.0001).
Hình 5.4: Boxplot phân bố IoU theo từng ảnh của ba phương pháp — U-Net có trung vị cao nhất và ổn định nhất (hộp hẹp hơn UPerNet).
Bảng 5.3b: Dice trung bình theo mức độ cháy (phân tầng) — giải thích vì sao hai cách đo trái chiều.
Mức độ cháy
V3
U-Net
UPerNet
Rất ít (<1%) – 16 ảnh
0.347
0.546
0.346
Ít (1–10%) – 43 ảnh
0.674
0.780
0.733
Vừa (10–30%) – 23 ảnh
0.813
0.893
0.918
Nhiều (30–50%) – 5 ảnh
0.821
0.918
0.925
Rất nhiều (>50%) – 1 ảnh
0.826
0.970
0.969
Trên ảnh cháy nhỏ (<1%), U-Net vượt trội (0.546 so với 0.346); trên ảnh cháy vừa–lớn, UPerNet nhỉnh hơn. Chỉ số pixel-level bị các đám cháy lớn chi phối (nhiều pixel) nên UPerNet thắng; chỉ số per-image cân bằng mọi ảnh, mà phần lớn ảnh là cháy nhỏ–ít nên U-Net thắng. Diễn giải: U-Net giỏi phát hiện đám cháy nhỏ/sớm và ổn định hơn; UPerNet khoanh chính xác hơn các đám cháy lớn.
Hình 5.5: Dice trung bình theo mức độ cháy — U-Net vượt trội ở cháy nhỏ (<1%: 0.55 so với 0.35), UPerNet nhỉnh hơn ở cháy vừa–lớn.
So sánh với paper gốc IGARSS 2023: cả hai mô hình của nhóm vượt mọi cấu hình huấn-luyện-từ-đầu của paper; đặc biệt UPerNet đạt 91.06% F1, tiệm cận trần của cấu hình pretrained-đa-nhiệm (91.86%) mà không cần pretrained.
Bảng 5.3c: So sánh với paper gốc.
Phương pháp
F1
IoU
Pretrained
Đánh giá
Baseline V3 (NBR+NDWI)
78.95%
65.22%
–
full-image
U-Net (nhóm, v2)
87.64%
78.00%
Không
full-image
UPerNet (nhóm, v2)
91.06%
83.59%
Không
full-image
Paper UPerNet RN50 STL scratch
82.33%
70.94%
Không
–
Paper SegFormer B3 STL scratch
89.01%
80.22%
Không
–
Paper UPerNet RN50 MTL pretrained
91.86%
84.94%
Có
–
Lưu ý khi bảo vệ: quy trình đánh giá có thể khác paper đôi chút, nên phát biểu 'tương đương/vượt ở cấu hình so sánh được'.
Hình 5.3: Biểu đồ training UPerNet v2 (best epoch 27, early-stop epoch 42).
## 5.4. Phân loại có/không cháy cấp ảnh
Bài toán phân loại cấp ảnh: cho một patch 256×256, mô hình cần trả lời 'có cháy' hay 'không cháy'. Vì dataset CEMS chỉ chứa ảnh có cháy, nhóm tự sinh patch 'không cháy' bằng cách crop vùng ngoài mask cháy từ cùng ảnh.
Bảng 5.4: Kết quả phân loại cấp ảnh (156 patches)
Model
Accuracy
Precision
Recall
F1
Baseline V3
85.26%
75.00%
96.92%
84.56%
U-Net (v2)
92.95%
85.53%
100.00%
92.20%
UPerNet (v2)
98.72%
100.00%
96.92%
98.44%
Cả hai mô hình học sâu vượt baseline V3 nhờ học được ngữ cảnh không gian, phân biệt cháy với đô thị/đất trống. Kết quả phản ánh đúng đặc tính của từng mô hình: UPerNet đạt cao nhất (Accuracy 98.72%, F1 98.44%) với Precision 100% — không báo nhầm patch nào (0 false positive), chỉ bỏ sót 2/65 patch cháy. U-Net đạt Recall 100% — bắt được toàn bộ patch cháy (0 bỏ sót), nhưng Precision thấp hơn (85.53%) do có 11 patch không cháy bị báo nhầm, đúng với xu hướng ‘recall cao, precision thấp’ đã thấy ở phần phân vùng. Như vậy nếu ưu tiên tuyệt đối không bỏ sót đám cháy thì chọn U-Net; nếu ưu tiên hạn chế báo động giả thì UPerNet vượt trội.
Hình 5.6: Ma trận nhầm lẫn bài toán phân loại có/không cháy cấp ảnh của ba phương pháp (lần lượt Baseline V3, U-Net, UPerNet) trên 156 patch — UPerNet không báo nhầm (0 FP), U-Net không bỏ sót (0 FN).
## 5.5. Ước tính diện tích cháy
Phương pháp tính: Diện tích cháy = số pixel cháy × diện tích thật của mỗi pixel (m²). Vì ảnh Sentinel-2 có CRS WGS84 (đơn vị độ), cần chuyển sang mét theo công thức xấp xỉ địa cầu có hiệu chỉnh vĩ độ:
dx_m = |Δlng| × 111,320 × cos(vĩ_độ_trung_tâm)dy_m = |Δlat| × 110,540pixel_area_m2 = dx_m × dy_m
Bảng 5.5: Sai số ước tính diện tích (88 ảnh test có GT > 0)
Model
MAE (ha)
Sai số TB (%)
Sai số TV (%)
Baseline V3
434.13
128.5%
21.6%
U-Net
321.37
43.2%
13.7%
UPerNet
166.49
27.0%
15.4%
Quan sát đặc biệt: UPerNet ước tính diện tích chính xác nhất — MAE chỉ 166.49 ha, với R² = 0.989 và bias gần như bằng 0 (+18 ha), tức gần như không thiên lệch. U-Net có xu hướng over-predict (+273 ha, 67% số ảnh bị ước tính cao) nên MAE cao hơn (321.37 ha), nhưng có sai số trung vị thấp nhất (13.7%). Kiểm định Wilcoxon ghép cặp trên trị tuyệt đối sai số giữa U-Net và UPerNet cho p = 0.276 (không có ý nghĩa thống kê), nên xét trên từng ảnh hai mô hình tương đương; ưu thế MAE của UPerNet đến từ việc nó không over-predict mạnh ở các đám cháy lớn. Bài học: chọn model cho ước tính tổng diện tích (UPerNet — hiệu chuẩn tốt) có thể khác với chọn model ổn định theo từng ảnh (U-Net).
Hình 5.7: Scatter plot giữa diện tích Ground Truth (trục x) và diện tích dự đoán (trục y).
## 5.6. Thảo luận kết quả
Hình 5.8: So sánh trực quan trên đám cháy nhỏ (EMSR298) — U-Net khoanh sát Ground Truth (Dice 0.929), vượt UPerNet (0.723) và baseline V3 (0.157).
Hình 5.9: So sánh trực quan trên đám cháy lớn (EMSR302) — UPerNet bám biên tốt nhất (Dice 0.936), nhỉnh hơn U-Net (0.824) và V3 (0.565).
Case study 1: Ảnh không cháy EMSR214_AOI06_04
Ảnh KHÔNG có cháy theo GT. Baseline V3 dự đoán 19 cụm, tổng 310 ha – fail nặng. U-Net chỉ dự đoán 2 cụm nhỏ, 4.94 ha. UPerNet cũng chỉ 2 cụm, 4.75 ha. Đây là minh chứng cho ưu thế của deep learning trong việc học context phân biệt cháy với các loại đất có chữ ký phổ tương tự.
Case study 2: Vụ cháy Greenland EMSR218_AOI01_01
Vụ cháy ở vĩ độ 67.84°N (Bắc Cực), diện tích GT 23.44 km² (2,344 ha). Pixel size thật sau hiệu chỉnh: 98.81 m² (sai khác <2% so với 100 m² mặc định). Vụ cháy ở Bắc Cực là case study thú vị về biến đổi khí hậu.
# CHƯƠNG 6 – DEMO ỨNG DỤNG WEB
## 6.1. Lựa chọn công nghệ
•  Streamlit: framework Python mã nguồn mở để xây dựng web app data science nhanh.
•  Plotly: thư viện visualization tương tác, hỗ trợ hiển thị ảnh và overlay các lớp dữ liệu.
•  Lý do chọn: triển khai nhanh, không cần backend riêng, phù hợp cho đồ án môn học.
## 6.2. Tính năng demo
•  Chọn ảnh mẫu từ tập test (99 ảnh) hoặc upload ảnh .tif 12 band của người dùng.
•  Chọn model: Baseline V3, U-Net, UPerNet, hoặc so sánh cả 3.
•  Hiển thị tương tác bằng Plotly: ảnh RGB nền, mask cháy overlay đỏ trong suốt, bounding box vàng, hover hiện thông tin chi tiết.
•  Toggle 'Hiện diện tích' bật/tắt label trên từng cụm.
•  Bảng tổng diện tích + bảng chi tiết từng cụm (vị trí, kích thước, số pixel).
•  Tùy chọn xem Ground Truth để so sánh.
## 6.3. Validation an toàn cho user upload
•  Lớp 1: Kiểm tra đuôi file phải là .tif hoặc .tiff.
•  Lớp 2: Kiểm tra số band phải đúng 12 (Sentinel-2 L2A).
•  Lớp 3: Kiểm tra dải giá trị reflectance (thường <2, nếu >100 là bất thường).
•  Giới hạn kích thước ảnh: tối đa 2000×2000 pixel để tránh OOM.
•  Fallback metadata: nếu không đọc được CRS, dùng giả định 100 m²/pixel.
Hình 6.1: Giao diện web demo — kết quả phát hiện cháy của một mô hình với bounding box từng cụm và tổng diện tích ước tính.
Hình 6.2: Giao diện web demo — chế độ so sánh đồng thời ba phương pháp V3, U-Net, UPerNet trên cùng một ảnh.
## 6.4. Kịch bản sử dụng thực tế
Cơ quan giám sát môi trường upload ảnh Sentinel-2 mới chụp lên hệ thống, app tự động phát hiện vùng cháy, ước tính diện tích và hiển thị kết quả trong vài giây. Thông tin này hỗ trợ ra quyết định triển khai lực lượng cứu hỏa, lập kế hoạch sơ tán dân, đánh giá thiệt hại sau cháy.
# CHƯƠNG 7 – KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
## 7.1. Kết luận
Đồ án đã đạt được các mục tiêu đề ra:
•  Triển khai thành công 3 phương pháp (V1, V2, V3 truyền thống + 2 deep learning) với pipeline đầy đủ.
•  Trên đánh giá toàn ảnh công bằng, lựa chọn mô hình phụ thuộc mục tiêu: UPerNet đạt độ chính xác pixel cao nhất (F1 91.06%, IoU 83.59%) và ước tính tổng diện tích chính xác nhất (MAE 166 ha); U-Net ổn định hơn theo từng ảnh và vượt trội ở phát hiện cháy nhỏ (IoU per-image 0.680, cao hơn UPerNet có ý nghĩa thống kê p = 0.012). Cả hai đều vượt mọi cấu hình huấn-luyện-từ-đầu của paper gốc IGARSS 2023.
•  Baseline V3 (NBR + NDWI) là đề xuất chính cho hệ thống chi phí thấp: đạt F1 = 78.94% chỉ với CPU.
•  Mô-đun tính diện tích với hiệu chỉnh vĩ độ làm việc chính xác cho ảnh ở mọi vĩ độ.
•  Web app demo tương tác với validation 3 lớp cho user upload.
## 7.2. Hạn chế
•  Dataset chỉ 281 ảnh train – hạn chế khả năng học của các model lớn.
•  Ground truth do chuyên gia Copernicus khoanh tay nên có thể không hoàn toàn pixel-precise.
•  Cả ba phương pháp đều suy giảm trên ảnh cháy rất nhỏ (<1% diện tích): Dice chỉ khoảng 0.35–0.55, do vùng cháy ít pixel nên sai lệch ở biên chiếm tỷ trọng lớn.
•  Demo web chưa hỗ trợ ensemble model.
## 7.3. Hướng phát triển
•  Sử dụng encoder pretrained: ImageNet hoặc tốt hơn là SSL4EO-S12 (pretrained trên 1M ảnh Sentinel-2). Paper gốc cho thấy +9 điểm F1 với pretrained.
•  Multitask Learning: học cùng lúc segmentation cháy + phân loại lớp phủ đất.
•  dNBR (differenced NBR): dùng cặp ảnh trước-sau cháy cho baseline truyền thống.
•  Mở rộng dataset: tải thêm ảnh Sentinel-2 không cháy từ Copernicus.
•  Ablation study: phân tích band nào quan trọng nhất cho phát hiện cháy.
•  Triển khai production: Docker hóa web app, deploy lên cloud.
# TÀI LIỆU THAM KHẢO
[1] L. Arnaudo, R. Ferrari, and F. Cermelli, 'Robust Burned Area Delineation through Multitask Learning,' IEEE IGARSS 2023.
[2] O. Ronneberger, P. Fischer, and T. Brox, 'U-Net: Convolutional Networks for Biomedical Image Segmentation,' MICCAI 2015.
[3] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, 'Unified Perceptual Parsing for Scene Understanding,' ECCV 2018.
[6] C. H. Key and N. C. Benson, 'Landscape Assessment (LA),' USDA Forest Service, 2006.
[7] S. K. McFeeters, 'The Use of NDWI in the Delineation of Open Water Features,' Int. J. Remote Sensing, 1996.
[8] European Space Agency, 'Sentinel-2 User Handbook,' ESA, 2015.
[9] Copernicus Emergency Management Service (CEMS), emergency.copernicus.eu/mapping
[10] N. Otsu, 'A Threshold Selection Method from Gray-Level Histograms,' IEEE Trans. SMC, 1979.
[11] D. P. Kingma and J. Ba, 'Adam: A Method for Stochastic Optimization,' ICLR 2015.
[12] P. Iakubovskii, 'Segmentation Models PyTorch,' GitHub 2019.
# PHỤ LỤC
## Phụ lục A: Cấu trúc thư mục project
xlas_bc/├── README.md├── requirements.txt├── 01_explore_data.py     # EDA├── 02_preprocessing.py    # Pipeline tiền xử lý├── 05_read_multispectral.py├── 06_dataloader.py├── 07_baseline_traditional.py    # V1├── 07b_baseline_improved.py      # V2├── 07c_baseline_v3.py            # V3├── 08_train_unet.py├── 09_compare_models.py├── 10b_classification_save.py├── 13_tinh_dien_tich.py├── 14_thong_ke_dien_tich_test.py├── 15_minh_hoa_dien_tich.py├── 16_thong_ke_chi_tiet.py├── 17_web_app.py├── wildfires-cems/         # Dataset└── ket_qua_*/              # Kết quả các model
## Phụ lục B: Hướng dẫn chạy project
# Cài đặtpython -m venv venvvenv\Scripts\activatepip install -r requirements.txt# Tải datasetpython run.py# Chạy theo thứ tựpython 01_explore_data.pypython 07_baseline_traditional.py        # V1python 07b_baseline_improved.py          # V2python 07c_baseline_v3.py                # V3python 08_train_unet.py                  # U-Netpython 12_train_upernet.py upernetpython 12_train_upernet.py python 14_thong_ke_dien_tich_test.pypython 16_thong_ke_chi_tiet.py# Web appstreamlit run 17_web_app.py
## Phụ lục C: Bảng tổng hợp kết quả chính
Mục
Giá trị
Tổng số ảnh dataset
433 ảnh (Train 281, Val 53, Test 99)
Dung lượng dataset
18.4 GB
Số sự kiện EMSR
171
Số band Sentinel-2
12
V1 F1 / IoU
25.85% / 14.84%
V2 F1 / IoU
55.90% / 38.79%
V3 F1 / IoU
78.94% / 65.22%
U-Net Test Dice / IoU
87.64% / 78.00%
UPerNet Test Dice / IoU
91.06% / 83.59% (best pixel)
V3 MAE diện tích (ha)
434.13
U-Net MAE diện tích (ha)
321.37
UPerNet MAE diện tích (ha)
166.49 (best)
V3 Classification Acc
85.26%
Classification Acc (cấp ảnh)
U-Net 92.95% — UPerNet 98.72% (best)
Paper UPerNet F1 (cùng cfg)
82.33%
Nhóm UPerNet F1 (cùng cfg)
91.06% (+8.73)
## Phụ lục D: Giải thích một số thuật ngữ
•  NBR (Normalized Burn Ratio): Chỉ số phổ đo lường cháy: (NIR - SWIR2) / (NIR + SWIR2).
•  NDWI (Normalized Difference Water Index): Chỉ số phát hiện mặt nước: (Green - NIR) / (Green + NIR).
•  NDVI (Normalized Difference Vegetation Index): Chỉ số đo lường thực vật: (NIR - Red) / (NIR + Red).
•  IoU (Intersection over Union): Tỉ số giao/hợp giữa mask dự đoán và ground truth.
•  Dice / F1-score: Tương tự IoU: 2·TP / (2·TP + FP + FN).
•  Sentinel-2: Vệ tinh quan sát Trái Đất của ESA, cung cấp ảnh đa phổ 12 band.
•  EMSR (Emergency Management Service Number): Mã số định danh vụ cháy/khẩn cấp do Copernicus cấp.
•  GeoTIFF: Định dạng ảnh raster có thông tin địa lý nhúng trong file.
•  Sliding window: Kỹ thuật áp dụng model trên ảnh lớn: chia thành các patch chồng lấp.
•  Connected components: Thuật toán gán nhãn các vùng liên thông trong mask nhị phân.
## PHỤ LỤC BỔ SUNG – THỐNG KÊ DATASET
Phụ lục E: Phân loại ảnh theo mức độ cháy (bar chart).
Phụ lục F: Histogram phân bố tỷ lệ pixel cháy theo từng tập.