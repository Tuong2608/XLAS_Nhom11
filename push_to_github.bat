@echo off
REM ============================================
REM Script push code lên GitHub
REM Nhóm 11 - Đồ án Xử lý Ảnh Số
REM ============================================

echo.
echo ============================================
echo PUSH CODE LEN GITHUB
echo Nhom 11 - Phat hien chay rung
echo ============================================
echo.

REM Di chuyen den thu muc project
cd /d "%~dp0"

REM Kiem tra git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Git. Vui long cai Git.
    echo Tai: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Khoi tao git repository
echo [1/6] Khoi tao Git repository...
git init
git config user.email "nhom11@xl.as"
git config user.name "Nhom 11 - XLAS"

REM Them remote
echo [2/6] Them remote origin...
git remote remove origin 2>nul
git remote add origin git@github.com:Tuong2608/XLAS_Nhom11.git

REM Them tat ca file
echo [3/6] Them file vao staging area...
git add .

REM Commit
echo [4/6] Tao commit...
git commit -m "Initial commit: Phat hien chay rung tren anh ve tinh Sentinel-2

- Hoan than 7 giai doan du an
- Baseline truyen thong: NBR + NDWI
- Deep learning: U-Net, UPerNet
- Web app demo voi Streamlit"

REM Tao branch main
echo [5/6] Chuyen sang branch main...
git branch -M main

REM Push len GitHub
echo [6/6] Push len GitHub...
echo.
echo Vui long xac nhan push (co the can nhap mat khau GitHub).
echo.
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo PUSH THANH CONG!
    echo Link repository: https://github.com/Tuong2608/XLAS_Nhom11
    echo ============================================
) else (
    echo.
    echo ============================================
    echo PUSH THAT BAI!
    echo Vui long kiem tra:
    echo - Da cai dat Git chua?
    echo - SSH key da them vao GitHub chua?
    echo - Quyen truy cap repository da du chua?
    echo ============================================
)

pause
