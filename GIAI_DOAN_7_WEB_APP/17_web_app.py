"""
WEB APP DEMO - PHÁT HIỆN CHÁY RỪNG TRÊN ẢNH VỆ TINH

Tính năng:
- Chọn ảnh mẫu có sẵn HOẶC upload ảnh .tif Sentinel-2 12 band
- Chọn model (V3, U-Net, UPerNet, hoặc so sánh tất cả)
- Hiển thị tương tác bằng Plotly:
    * Mặc định: ảnh RGB + mask overlay sạch
    * Toggle "Hiện diện tích": label trên mỗi cụm
    * Hover lên cụm cháy: popup hiển thị chi tiết
- Bảng tổng diện tích + chi tiết từng cụm

CÁCH CHẠY:
  streamlit run 17_web_app.py

Mở browser: http://localhost:8501
"""

from pathlib import Path
import sys
import io

import numpy as np
import torch
import rasterio
import streamlit as st
import plotly.graph_objects as go
import segmentation_models_pytorch as smp

# Import modules của project
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

ts = import_module("13_tinh_dien_tich")
dataloader = import_module("06_dataloader")
baseline_v3 = import_module("07c_baseline_v3")
unet_module = import_module("08_train_unet")
thong_ke14 = import_module("14_thong_ke_dien_tich_test")

chuan_hoa = dataloader.chuan_hoa
du_doan_v3 = baseline_v3.du_doan_v3
UNet = unet_module.UNet
predict_sliding = thong_ke14.predict_sliding


# ============================================================
# CẤU HÌNH STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Phát hiện cháy rừng - Demo",
    page_icon="🔥",
    layout="wide",
)

UNET_CHECKPOINT = Path("./ket_qua_unet_v2/best_model.pt")
UPERNET_CHECKPOINT = Path("./ket_qua_upernet_v2/best_model.pt")

# Thư mục chứa ảnh mẫu (sẽ tìm từ test set)
DATA_ROOT = Path("./wildfires-cems")

# Giới hạn kích thước ảnh upload (để tránh OOM)
MAX_IMAGE_SIZE = 2000  # pixels mỗi chiều
MIN_PIXELS_CLUSTER = 100   # filter cụm <1 ha
EPS = 1e-8


# ============================================================
# LOAD MODEL (CACHE)
# ============================================================
@st.cache_resource
def load_models():
    """Load 2 model deep learning, cache để không load lại mỗi request."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = None
    if UNET_CHECKPOINT.exists():
        ck = torch.load(UNET_CHECKPOINT, weights_only=False,
                          map_location=device)
        unet = UNet(in_channels=12, num_classes=1).to(device)
        unet.load_state_dict(ck["model_state_dict"])
        unet.eval()

    upernet = None
    if UPERNET_CHECKPOINT.exists():
        ck = torch.load(UPERNET_CHECKPOINT, weights_only=False,
                          map_location=device)
        upernet = smp.UPerNet(
            encoder_name="resnet50", encoder_weights=None,
            in_channels=12, classes=1,
        ).to(device)
        upernet.load_state_dict(ck["model_state_dict"])
        upernet.eval()

    return unet, upernet, device


# ============================================================
# CHUẨN BỊ DANH SÁCH ẢNH MẪU
# ============================================================
@st.cache_data
def liet_ke_anh_mau():
    """Liệt kê các ảnh test có sẵn để chọn."""
    test_dir = DATA_ROOT / "test"
    if not test_dir.exists():
        return []
    samples = []
    for emsr in sorted(test_dir.iterdir()):
        if not emsr.is_dir(): continue
        for aoi in sorted(emsr.iterdir()):
            if not aoi.is_dir(): continue
            for sample in sorted(aoi.iterdir()):
                if not sample.is_dir(): continue
                s2 = list(sample.glob("*_S2L2A.tif"))
                de = list(sample.glob("*_DEL.tif"))
                if s2 and de:
                    samples.append({
                        "ten": sample.name,
                        "s2": s2[0],
                        "del": de[0],
                    })
    return samples


# ============================================================
# VALIDATE FILE UPLOAD
# ============================================================
def validate_upload(uploaded_file):
    """Trả về (image, mask_gt_or_none, pixel_area, lỗi_nếu_có)."""
    # Lớp 1: kiểm tra đuôi file
    name = uploaded_file.name.lower()
    if not (name.endswith(".tif") or name.endswith(".tiff")):
        return None, None, None, "File phải có đuôi .tif hoặc .tiff"

    try:
        # Đọc file bằng rasterio (qua memory)
        with rasterio.MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as src:
                image = src.read().astype(np.float32)
                # Lớp 2: kiểm tra số band
                if image.shape[0] != 12:
                    return None, None, None, (
                        f"Ảnh phải có 12 band (Sentinel-2 L2A). "
                        f"File này có {image.shape[0]} band."
                    )
                # Lớp 3: kiểm tra dải giá trị
                vmax = float(np.nanmax(image))
                if vmax > 100:
                    return None, None, None, (
                        f"Giá trị pixel bất thường (max={vmax:.0f}). "
                        f"Sentinel-2 L2A reflectance thường <2. "
                        f"File có thể không đúng định dạng."
                    )
                # Kiểm tra kích thước
                _, h, w = image.shape
                if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE:
                    return None, None, None, (
                        f"Ảnh quá lớn ({h}×{w}). "
                        f"Giới hạn: {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE}."
                    )
                # Lấy pixel area từ metadata (giả định 100 nếu không đọc được)
                try:
                    if src.crs and src.crs.is_geographic:
                        bounds = src.bounds
                        transform = src.transform
                        center_lat = (bounds.top + bounds.bottom) / 2
                        lat_rad = np.radians(center_lat)
                        dx_m = abs(transform.a) * 111320 * np.cos(lat_rad)
                        dy_m = abs(transform.e) * 110540
                        pixel_area = dx_m * dy_m
                    elif src.crs:
                        transform = src.transform
                        pixel_area = abs(transform.a) * abs(transform.e)
                    else:
                        pixel_area = ts.DEFAULT_PIXEL_AREA_M2
                except Exception:
                    pixel_area = ts.DEFAULT_PIXEL_AREA_M2

        return image, None, pixel_area, None
    except Exception as e:
        return None, None, None, f"Không đọc được file: {e}"


# ============================================================
# CHẠY MODEL
# ============================================================
def chay_model(model_name, image_raw, unet, upernet, device):
    """Chạy 1 model, trả về mask dự đoán."""
    if model_name == "V3":
        mask, _, _, _ = du_doan_v3(image_raw)
        return mask
    elif model_name == "U-Net":
        if unet is None:
            return None
        image_norm = chuan_hoa(image_raw)
        return predict_sliding(unet, image_norm, device, patch_size=256)
    elif model_name == "UPerNet":
        if upernet is None:
            return None
        image_norm = chuan_hoa(image_raw)
        return predict_sliding(upernet, image_norm, device, patch_size=256)


# ============================================================
# VẼ ẢNH TƯƠNG TÁC BẰNG PLOTLY
# ============================================================
def _stretch(arr):
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + EPS), 0, 1)


def tao_rgb_uint8(image_raw):
    """Tạo ảnh RGB uint8 từ ảnh 12 band."""
    rgb = np.stack([_stretch(image_raw[3]),
                    _stretch(image_raw[2]),
                    _stretch(image_raw[1])], axis=-1)
    return (rgb * 255).astype(np.uint8)


def ve_plotly_interactive(image_raw, mask, components, labels,
                           pixel_area_m2, show_labels, plot_height=500):
    """
    Vẽ ảnh tương tác Plotly:
    - Background: ảnh RGB
    - Overlay: mask cháy (màu đỏ trong suốt)
    - Mỗi cụm là 1 marker scatter ẩn, hover hiện info
    - Toggle show_labels: hiện/ẩn label diện tích trên cụm
    """
    rgb_uint8 = tao_rgb_uint8(image_raw)
    h, w = rgb_uint8.shape[:2]

    fig = go.Figure()

    # Layer 1: Ảnh RGB nền
    fig.add_layout_image(
        dict(
            source=image_to_b64(rgb_uint8),
            xref="x", yref="y",
            x=0, y=0, sizex=w, sizey=h,
            sizing="stretch", layer="below",
        )
    )

    # Layer 2: Mask cháy overlay (vẽ pixel cháy thành màu đỏ trong suốt)
    if mask is not None and mask.any():
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = [255, 60, 60, 130]   # đỏ semi-transparent
        fig.add_layout_image(
            dict(
                source=image_to_b64(mask_rgba, has_alpha=True),
                xref="x", yref="y",
                x=0, y=0, sizex=w, sizey=h,
                sizing="stretch", layer="above",
            )
        )

    # Layer 3: Scatter markers tại centroid của mỗi cụm để hover
    if components:
        xs = [c["centroid"][0] for c in components]
        ys = [c["centroid"][1] for c in components]
        hover_texts = [
            f"<b>Cụm #{i+1}</b><br>"
            f"Diện tích: {ts.format_area(c['area_m2'])}<br>"
            f"Số pixel: {c['n_pixels']:,}<br>"
            f"Bounding box: {c['bbox'][2]}×{c['bbox'][3]} pixel"
            for i, c in enumerate(components)
        ]
        # Marker invisible nhưng hover được
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=20,
                color="rgba(255, 255, 0, 0.0)",     # vô hình
                line=dict(color="yellow", width=2),
            ),
            text=[f"#{i+1}<br>{ts.format_area(c['area_m2'])}"
                  for i, c in enumerate(components)] if show_labels else None,
            textposition="middle center",
            textfont=dict(color="white", size=11),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            name="cumc",
        ))

        # Vẽ bounding box quanh mỗi cụm (luôn hiện, mỏng)
        for c in components:
            x, y, bw, bh = c["bbox"]
            fig.add_shape(
                type="rect",
                x0=x, y0=y, x1=x+bw, y1=y+bh,
                line=dict(color="yellow", width=1.5, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
            )

    # Cài đặt layout - khít với tỉ lệ ảnh
    fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x",
                     scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=plot_height,
        hovermode="closest",
        plot_bgcolor="white",
    )
    return fig


def image_to_b64(arr, has_alpha=False):
    """Convert numpy array thành data URL base64 cho Plotly."""
    import base64
    from PIL import Image
    mode = "RGBA" if has_alpha else "RGB"
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
def main():
    st.title("🔥 Phát hiện cháy rừng trên ảnh vệ tinh Sentinel-2")
    st.markdown(
        "Đồ án xử lý ảnh số · So sánh phương pháp truyền thống và deep learning"
    )

    # Load model
    unet, upernet, device = load_models()
    if unet is None and upernet is None:
        st.error("Không tìm thấy model checkpoint. Hãy chạy file 08, 12 trước.")
        return

    # ----- SIDEBAR -----
    with st.sidebar:
        st.header("⚙️ Cấu hình")

        # Chọn nguồn ảnh
        nguon = st.radio(
            "Nguồn ảnh",
            ["Dùng ảnh mẫu (test set)", "Upload ảnh .tif của bạn"],
            help="Ảnh mẫu là từ test set CEMS, có sẵn ground truth để so sánh.",
        )

        # Chọn model
        st.markdown("---")
        model_choice = st.selectbox(
            "Model",
            ["U-Net (best)", "UPerNet", "Baseline V3 (truyền thống)",
             "So sánh cả 3"],
        )

        # Toggle hiện diện tích
        st.markdown("---")
        show_labels = st.toggle("📐 Hiện diện tích trên hình", value=False,
                                  help="Bật để xem label diện tích trên từng "
                                       "cụm. Tắt để xem ảnh sạch (vẫn hover được).")

        st.markdown("---")
        st.caption(f"Thiết bị: {device.upper()}")

    # ----- CHỌN ẢNH -----
    image_raw = None
    mask_gt = None
    pixel_area = ts.DEFAULT_PIXEL_AREA_M2
    ten_anh = "Unknown"

    if nguon == "Dùng ảnh mẫu (test set)":
        anh_mau = liet_ke_anh_mau()
        if not anh_mau:
            st.error("Không tìm thấy ảnh mẫu. Kiểm tra thư mục wildfires-cems/")
            return

        ten_chon = st.selectbox(
            f"Chọn ảnh ({len(anh_mau)} ảnh có sẵn)",
            [a["ten"] for a in anh_mau],
        )
        chon = next(a for a in anh_mau if a["ten"] == ten_chon)
        ten_anh = chon["ten"]
        with rasterio.open(chon["s2"]) as src:
            image_raw = src.read().astype(np.float32)
        with rasterio.open(chon["del"]) as src:
            mask_gt = src.read(1).astype(np.uint8)
        pixel_area = ts.read_pixel_size_meters(chon["s2"])

    else:
        st.info(
            "📋 **Yêu cầu file upload**:\n"
            "- Đuôi `.tif` hoặc `.tiff`\n"
            "- 12 band (Sentinel-2 L2A)\n"
            "- Giá trị reflectance [0, 2]\n"
            f"- Kích thước tối đa: {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} pixel\n"
            "- Khuyến nghị: file `S2L2A.tif` từ dataset Wildfires-CEMS"
        )
        uploaded = st.file_uploader("Chọn file .tif", type=["tif", "tiff"])
        if uploaded is None:
            return
        image_raw, _, pixel_area, err = validate_upload(uploaded)
        if err:
            st.error(f"❌ Lỗi: {err}")
            return
        ten_anh = uploaded.name
        st.success(f"✅ Đọc file thành công. Kích thước: "
                   f"{image_raw.shape[1]}×{image_raw.shape[2]} pixel")

    # ----- CHẠY MODEL VÀ HIỂN THỊ -----
    st.markdown("---")
    st.subheader(f"Kết quả · {ten_anh}")
    st.caption(f"Kích thước pixel: ~{pixel_area:.1f} m² · "
               f"Ảnh: {image_raw.shape[1]}×{image_raw.shape[2]} pixel")

    if model_choice == "So sánh cả 3":
        # Bố cục 4 cột: Ảnh gốc | V3 | U-Net | UPerNet
        # Compact mode + plot nhỏ hơn vì cột hẹp
        n_cols_key = "_4col"
        compact = True
        plot_h = 380

        col_goc, col_v3, col_unet, col_uper = st.columns(4)

        with st.spinner("Đang chạy 3 model..."):
            mask_v3 = chay_model("V3", image_raw, unet, upernet, device)
            mask_unet = chay_model("U-Net", image_raw, unet, upernet, device)
            mask_uper = chay_model("UPerNet", image_raw, unet, upernet, device)

        with col_goc:
            st.markdown("### 📷 Ảnh gốc")
            hien_thi_anh_goc(image_raw, pixel_area,
                              ten_anh + "_goc" + n_cols_key,
                              compact=compact, plot_height=plot_h)

        components_all = {}
        for col, name, mask in [
            (col_v3, "Baseline V3", mask_v3),
            (col_unet, "U-Net", mask_unet),
            (col_uper, "UPerNet", mask_uper),
        ]:
            with col:
                st.markdown(f"### 🔥 {name}")
                comps = hien_thi_ket_qua(
                    image_raw, mask, pixel_area, show_labels,
                    ten_anh + "_" + name + n_cols_key,
                    compact=compact, plot_height=plot_h,
                )
                components_all[name] = comps

        # 3 expander chi tiết
        st.markdown("---")
        exp_v3, exp_unet, exp_uper = st.columns(3)
        with exp_v3:
            hien_expander_chi_tiet(components_all["Baseline V3"],
                                     "Baseline V3", "exp_v3")
        with exp_unet:
            hien_expander_chi_tiet(components_all["U-Net"],
                                     "U-Net", "exp_unet")
        with exp_uper:
            hien_expander_chi_tiet(components_all["UPerNet"],
                                     "UPerNet", "exp_uper")

    else:
        # Một model: bố cục 2 cột Ảnh gốc | Kết quả
        # Cột rộng nên dùng metric to + plot cao
        n_cols_key = "_2col"
        compact = False
        plot_h = 550

        model_name = model_choice.split(" ")[0].replace("Baseline", "V3")
        col_goc, col_kq = st.columns(2)

        with col_goc:
            st.markdown("### 📷 Ảnh gốc")
            hien_thi_anh_goc(image_raw, pixel_area,
                              ten_anh + "_goc" + n_cols_key,
                              compact=compact, plot_height=plot_h)

        with col_kq:
            st.markdown(f"### 🔥 {model_name}")
            with st.spinner(f"Đang chạy {model_name}..."):
                mask = chay_model(model_name, image_raw, unet, upernet, device)
            components = hien_thi_ket_qua(
                image_raw, mask, pixel_area, show_labels,
                ten_anh + n_cols_key,
                compact=compact, plot_height=plot_h,
            )

        # Expander full width
        st.markdown("---")
        hien_expander_chi_tiet(components, model_name, "exp_single")

    # Nếu có GT, cho phép xem để so sánh (cùng style 2 cột)
    if mask_gt is not None:
        st.markdown("---")
        with st.expander("📊 Xem Ground Truth để so sánh"):
            col_goc_gt, col_gt = st.columns(2)
            with col_goc_gt:
                st.markdown("### 📷 Ảnh gốc")
                hien_thi_anh_goc(image_raw, pixel_area, "GT_goc_2col",
                                  compact=False, plot_height=550)
            with col_gt:
                st.markdown("### ✅ Ground Truth")
                components_gt = hien_thi_ket_qua(
                    image_raw, mask_gt, pixel_area, show_labels,
                    "GT_2col", compact=False, plot_height=550,
                )
            # Expander chi tiết GT bên trong expander GT (cuối)
            hien_expander_chi_tiet(components_gt, "Ground Truth", "exp_gt")


def render_metric(label, value, compact=False):
    """Hiển thị metric. Khi compact=True thì dùng markdown nhỏ hơn st.metric."""
    if compact:
        st.markdown(
            f'<div style="line-height:1.2;margin-bottom:6px">'
            f'<div style="font-size:0.72rem;color:rgba(200,200,200,0.7)">{label}</div>'
            f'<div style="font-size:1.05rem;font-weight:600;color:#fafafa">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.metric(label, value)


def hien_thi_anh_goc(image_raw, pixel_area, key_suffix, compact=False,
                      plot_height=500):
    """Hiển thị ảnh RGB gốc + 3 metric thông tin để cân layout với panel model."""
    h, w = image_raw.shape[1], image_raw.shape[2]
    n_bands = image_raw.shape[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("Kích thước", f"{w}×{h}", compact)
    with c2:
        render_metric("Số band", f"{n_bands}", compact)
    with c3:
        render_metric("Pixel size", f"{pixel_area:.1f} m²", compact)

    rgb_uint8 = tao_rgb_uint8(image_raw)
    h, w = rgb_uint8.shape[:2]

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=image_to_b64(rgb_uint8),
            xref="x", yref="y",
            x=0, y=0, sizex=w, sizey=h,
            sizing="stretch", layer="below",
        )
    )
    fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x",
                     scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=plot_height,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", key=f"plot_{key_suffix}")


def hien_thi_ket_qua(image_raw, mask, pixel_area, show_labels, key_suffix,
                      compact=False, plot_height=500):
    """Hiển thị mask + bảng diện tích cho 1 model. Return components để
    caller có thể tạo expander ngoài column nếu cần."""
    if mask is None:
        st.warning("Model không khả dụng.")
        return None

    # Tính diện tích
    total = ts.compute_total_area(mask, pixel_area)
    components, labels = ts.compute_components(
        mask, pixel_area, min_pixels=MIN_PIXELS_CLUSTER)

    # 3 metric ở trên (compact khi 4 cột)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("Tổng diện tích cháy", ts.format_area(total["area_m2"]),
                       compact)
    with c2:
        render_metric("Số cụm cháy", f"{len(components)}", compact)
    with c3:
        render_metric("Tổng pixel cháy", f"{total['n_pixels']:,}", compact)

    # Plotly interactive
    fig = ve_plotly_interactive(
        image_raw, mask, components, labels, pixel_area, show_labels,
        plot_height=plot_height,
    )
    st.plotly_chart(fig, width="stretch", key=f"plot_{key_suffix}")

    return components


def hien_expander_chi_tiet(components, ten_hien_thi, key_suffix):
    """
    Vẽ expander chi tiết cụm cháy. Gọi NGOÀI column để chiếm full width.
    """
    if not components:
        return
    with st.expander(f"📋 Chi tiết {len(components)} cụm cháy · {ten_hien_thi}"):
        data = []
        for i, c in enumerate(components, 1):
            data.append({
                "#": i,
                "Diện tích": ts.format_area(c["area_m2"]),
                "Pixel": f"{c['n_pixels']:,}",
                "Vị trí (x, y)": f"({int(c['centroid'][0])}, "
                                 f"{int(c['centroid'][1])})",
                "Kích thước": f"{c['bbox'][2]}×{c['bbox'][3]}",
            })
        st.dataframe(data, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
