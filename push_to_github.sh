#!/bin/bash
# ============================================
# Script push code lên GitHub
# Nhóm 11 - Đồ án Xử lý Ảnh Số
# ============================================

echo ""
echo "============================================"
echo "PUSH CODE LEN GITHUB"
echo "Nhom 11 - Phat hien chay rung"
echo "============================================"
echo ""

# Di chuyen den thu muc chua script
cd "$(dirname "$0")"

# Kiem tra git
if ! command -v git &> /dev/null; then
    echo "[LOI] Khong tim thay Git. Vui long cai Git."
    exit 1
fi

# Khoi tao git repository
echo "[1/6] Khoi tao Git repository..."
git init
git config user.email "nhom11@xl.as"
git config user.name "Nhom 11 - XLAS"

# Them remote
echo "[2/6] Them remote origin..."
git remote remove origin 2>/dev/null
git remote add origin git@github.com:Tuong2608/XLAS_Nhom11.git

# Them tat ca file
echo "[3/6] Them file vao staging area..."
git add .

# Commit
echo "[4/6] Tao commit..."
git commit -m "Initial commit: Phat hien chay rung tren anh ve tinh Sentinel-2

- Hoan than 7 giai doan du an
- Baseline truyen thong: NBR + NDWI  
- Deep learning: U-Net, UPerNet
- Web app demo voi Streamlit"

# Tao branch main
echo "[5/6] Chuyen sang branch main..."
git branch -M main

# Push len GitHub
echo "[6/6] Push len GitHub..."
echo ""
echo "Vui long xac nhan push (co the can nhap mat khau GitHub)."
echo ""
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "PUSH THANH CONG!"
    echo "Link repository: https://github.com/Tuong2608/XLAS_Nhom11"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "PUSH THAT BAI!"
    echo "Vui long kiem tra:"
    echo "- Da cai dat Git chua?"
    echo "- SSH key da them vao GitHub chua?"
    echo "- Quyen truy cap repository da du chua?"
    echo "============================================"
fi
