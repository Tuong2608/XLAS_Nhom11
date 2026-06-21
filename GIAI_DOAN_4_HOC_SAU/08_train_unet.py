"""
GIAI ĐOẠN 4 - U-NET TRAINING (Người B)
Đề tài: Phát hiện cháy rừng trên ảnh vệ tinh

Kiến trúc U-Net (Ronneberger et al., 2015) là chuẩn vàng cho image
segmentation. Đặc trưng: encoder-decoder đối xứng với skip connections,
giúp model giữ được thông tin chi tiết không gian khi upsample.

Khác biệt so với U-Net gốc:
- Input: 12 channels (Sentinel-2 đa phổ) thay vì 3 channels (RGB)
- Output: 1 channel (sigmoid) cho binary segmentation cháy/không

HYPERPARAMETERS (đưa vào báo cáo):
  Optimizer    : Adam
  Learning rate: 1e-3
  Batch size   : 4 (giới hạn bởi 3060 6GB)
  Epochs       : 40
  Loss         : BCE + Dice (tổ hợp - chuẩn cho segmentation)
  Patch size   : 256 x 256
  Augmentation : Random flip + 90° rotation (chỉ trên train)

CÁCH CHẠY:
  python 08_train_unet.py
"""

from pathlib import Path
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

# Import dataloader
from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
dataloader = import_module("06_dataloader")
tao_dataloaders = dataloader.tao_dataloaders


# ============================================================
# CẤU HÌNH (báo cáo dùng các con số này)
# ============================================================
CONFIG = {
    "in_channels": 12,       # 12 band Sentinel-2
    "num_classes": 1,        # cháy vs không (nhị phân, dùng sigmoid)
    "patch_size": 256,
    "batch_size": 4,
    "epochs": 40,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "Adam",
    "loss": "BCE + Dice",
}

OUTPUT_DIR = Path("./ket_qua_unet")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# KIẾN TRÚC U-NET
# ============================================================
class DoubleConv(nn.Module):
    """Block 2 lần Conv + BN + ReLU - đơn vị cơ bản của U-Net."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net với 4 mức encoder/decoder. Skip connection nối từ encoder
    sang decoder ở cùng mức để bảo toàn chi tiết không gian.

    Kiến trúc:
      Input [B, 12, 256, 256]
        ↓ enc1 → skip1 [64, 256, 256]
        ↓ pool + enc2 → skip2 [128, 128, 128]
        ↓ pool + enc3 → skip3 [256, 64, 64]
        ↓ pool + enc4 → skip4 [512, 32, 32]
        ↓ pool + bottleneck [1024, 16, 16]
        ↑ up + dec4 (+ skip4) [512, 32, 32]
        ↑ up + dec3 (+ skip3) [256, 64, 64]
        ↑ up + dec2 (+ skip2) [128, 128, 128]
        ↑ up + dec1 (+ skip1) [64, 256, 256]
      Output [B, 1, 256, 256] (logits)
    """
    def __init__(self, in_channels=12, num_classes=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        # Decoder (concat với skip)
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out_conv(d1)


# ============================================================
# LOSS - BCE + Dice (kết hợp 2 loss phổ biến nhất cho segmentation)
# ============================================================
class BCEDiceLoss(nn.Module):
    """
    BCE phạt sai từng pixel độc lập (tốt cho class imbalance khi gần đều).
    Dice tối đa hóa overlap (tốt khi imbalance mạnh - vùng cháy nhỏ).
    Kết hợp cả hai cho ổn định.
    """
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        target = target.float()
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        dice = 1 - (2 * intersection + 1) / (probs.sum() + target.sum() + 1)
        return (1 - self.dice_weight) * bce + self.dice_weight * dice


# ============================================================
# METRICS
# ============================================================
def dice_score(probs, target, threshold=0.5):
    """Dice cấp batch."""
    pred = (probs > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1) / (pred.sum() + target.sum() + 1)


def iou_score(probs, target, threshold=0.5):
    """IoU cấp batch."""
    pred = (probs > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1) / (union + 1)


# ============================================================
# TRAIN MỘT EPOCH
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, n_batch = 0.0, 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)  # [B, 1, H, W]
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            total_dice += dice_score(torch.sigmoid(logits), masks).item()
        n_batch += 1
    return total_loss / n_batch, total_dice / n_batch


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou, n_batch = 0.0, 0.0, 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        total_loss += loss.item()
        total_dice += dice_score(probs, masks).item()
        total_iou += iou_score(probs, masks).item()
        n_batch += 1
    return (total_loss / n_batch, total_dice / n_batch, total_iou / n_batch)


# ============================================================
# VẼ BIỂU ĐỒ TRAINING
# ============================================================
def ve_bieu_do_history(history, out_path):
    """Vẽ loss và dice qua các epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss qua các epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title("Dice qua các epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("TRAIN U-NET CHO PHÂN VÙNG CHÁY ĐA PHỔ")
    print("=" * 60)
    print(f"Thiết bị: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Lưu config ra file để báo cáo dùng
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\nConfig:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # DataLoaders
    print("\nĐang chuẩn bị dataloader...")
    loaders = tao_dataloaders(batch_size=CONFIG["batch_size"],
                              patch_size=CONFIG["patch_size"],
                              num_workers=0)
    print(f"  train: {len(loaders['train'].dataset)} samples, "
          f"{len(loaders['train'])} batches")
    print(f"  val  : {len(loaders['val'].dataset)} samples")
    print(f"  test : {len(loaders['test'].dataset)} samples")

    # Model
    model = UNet(in_channels=CONFIG["in_channels"],
                 num_classes=CONFIG["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nU-Net được khởi tạo. Tổng tham số: {n_params:.2f}M")

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    criterion = BCEDiceLoss(dice_weight=0.5)

    # Training loop
    print("\n" + "=" * 60)
    print("BẮT ĐẦU TRAINING")
    print("=" * 60)

    history = {"train_loss": [], "train_dice": [],
               "val_loss": [], "val_dice": [], "val_iou": []}
    best_val_dice = 0.0
    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        ep_start = time.time()
        train_loss, train_dice = train_epoch(model, loaders["train"],
                                              optimizer, criterion, device)
        val_loss, val_dice, val_iou = eval_epoch(model, loaders["val"],
                                                  criterion, device)

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        # Lưu model tốt nhất
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": CONFIG,
            }, OUTPUT_DIR / "best_model.pt")
            marker = " *"  # đánh dấu best
        else:
            marker = ""

        ep_time = time.time() - ep_start
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
              f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"Dice train={train_dice:.4f} val={val_dice:.4f} | "
              f"IoU val={val_iou:.4f} | {ep_time:.1f}s{marker}")

    total_time = time.time() - start
    print(f"\nTổng thời gian training: {total_time/60:.1f} phút")
    print(f"Best val Dice: {best_val_dice:.4f}")

    # Lưu history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    ve_bieu_do_history(history, OUTPUT_DIR / "training_history.png")
    print(f"Đã lưu biểu đồ training vào {OUTPUT_DIR}/training_history.png")

    # Đánh giá cuối cùng trên test set với best model
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ TRÊN TEST SET (best model)")
    print("=" * 60)
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_dice, test_iou = eval_epoch(model, loaders["test"],
                                                  criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU : {test_iou:.4f}")

    # Lưu kết quả test ra file (để báo cáo)
    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ U-NET TRÊN TEST SET\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best epoch    : {checkpoint['epoch']}\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Dice     : {test_dice:.4f}\n")
        f.write(f"Test IoU      : {test_iou:.4f}\n")
        f.write(f"Số tham số    : {n_params:.2f}M\n")
        f.write(f"Tổng thời gian: {total_time/60:.1f} phút\n")


if __name__ == "__main__":
    main()
