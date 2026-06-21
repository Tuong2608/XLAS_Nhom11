"""
GIAI ĐOẠN 4+ - TRAIN UPERNET VÀ SEGFORMER (ĐỐI CHIẾU PAPER)

File này train hai kiến trúc tiên tiến hơn U-Net để so sánh với paper gốc:
- UPerNet (ResNet50): kiến trúc CNN + Feature Pyramid Network
- SegFormer (MiT-B3): kiến trúc transformer cho segmentation

CẢ HAI DÙNG TỪ THƯ VIỆN segmentation_models_pytorch (SMP).
Lý do dùng thư viện thay vì viết tay: kiến trúc phức tạp (500+ dòng),
SMP là implementation chuẩn được kiểm thử, dùng bởi nhiều paper.

CẤU HÌNH (giống U-Net để so sánh fair):
  in_channels  : 12 (Sentinel-2 đa phổ)
  patch_size   : 256x256
  batch_size   : 4 (giảm xuống 2 nếu hết VRAM)
  epochs       : 30 (giống paper)
  pretrained   : KHÔNG (from scratch)
  optimizer    : Adam, lr=1e-3
  loss         : BCE + Dice

CÁCH CHẠY:
  python 12_train_upernet_segformer.py upernet     <- train UPerNet
  python 12_train_upernet_segformer.py segformer   <- train SegFormer

Hoặc không tham số, sẽ train UPerNet mặc định.
"""

from pathlib import Path
import time
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from importlib import import_module
sys.path.insert(0, str(Path(__file__).parent))
dataloader = import_module("06_dataloader")
unet_module = import_module("08_train_unet")
tao_dataloaders = dataloader.tao_dataloaders
BCEDiceLoss = unet_module.BCEDiceLoss
dice_score = unet_module.dice_score
iou_score = unet_module.iou_score


# ============================================================
# CHỌN MODEL TỪ COMMAND LINE
# ============================================================
ARCH = "upernet"  # default
if len(sys.argv) > 1:
    ARCH = sys.argv[1].lower()
assert ARCH in ["upernet", "segformer"], \
    "Chỉ hỗ trợ 'upernet' hoặc 'segformer'"


# ============================================================
# CẤU HÌNH
# ============================================================
CONFIG = {
    "arch": ARCH,
    "encoder_name": "resnet50" if ARCH == "upernet" else "mit_b3",
    "encoder_weights": None,    # KHÔNG dùng pretrained để so sánh fair
    "in_channels": 12,
    "classes": 1,
    "patch_size": 256,
    "batch_size": 4,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "Adam",
    "loss": "BCE + Dice",
}

OUTPUT_DIR = Path(f"./ket_qua_{ARCH}")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TẠO MODEL TỪ SMP
# ============================================================
def tao_model():
    if ARCH == "upernet":
        model = smp.UPerNet(
            encoder_name=CONFIG["encoder_name"],
            encoder_weights=CONFIG["encoder_weights"],
            in_channels=CONFIG["in_channels"],
            classes=CONFIG["classes"],
        )
    elif ARCH == "segformer":
        model = smp.Segformer(
            encoder_name=CONFIG["encoder_name"],
            encoder_weights=CONFIG["encoder_weights"],
            in_channels=CONFIG["in_channels"],
            classes=CONFIG["classes"],
        )
    return model


# ============================================================
# TRAIN/EVAL EPOCH (giống file 08)
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, n = 0.0, 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            total_dice += dice_score(torch.sigmoid(logits), masks).item()
        n += 1
    return total_loss / n, total_dice / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        total_loss += loss.item()
        total_dice += dice_score(probs, masks).item()
        total_iou += iou_score(probs, masks).item()
        n += 1
    return total_loss / n, total_dice / n, total_iou / n


def ve_bieu_do(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss qua các epoch · {ARCH.upper()}")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title(f"Dice qua các epoch · {ARCH.upper()}")
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
    print(f"TRAIN {ARCH.upper()} CHO PHÂN VÙNG CHÁY ĐA PHỔ")
    print("=" * 60)
    print(f"Thiết bị: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print("\nConfig:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # DataLoaders
    print("\nĐang chuẩn bị dataloader...")
    loaders = tao_dataloaders(batch_size=CONFIG["batch_size"],
                              patch_size=CONFIG["patch_size"],
                              num_workers=0)
    print(f"  train: {len(loaders['train'].dataset)} samples, "
          f"{len(loaders['train'])} batches")

    # Model
    print(f"\nĐang khởi tạo {ARCH.upper()}...")
    model = tao_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Tổng tham số: {n_params:.2f}M")

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
        try:
            train_loss, train_dice = train_epoch(
                model, loaders["train"], optimizer, criterion, device)
            val_loss, val_dice, val_iou = eval_epoch(
                model, loaders["val"], criterion, device)
        except torch.cuda.OutOfMemoryError:
            print(f"\n[LỖI VRAM] Hết VRAM ở epoch {epoch}.")
            print("Hãy giảm batch_size trong CONFIG (4 → 2 → 1).")
            return

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": CONFIG,
            }, OUTPUT_DIR / "best_model.pt")
            marker = " *"
        else:
            marker = ""

        ep_time = time.time() - ep_start
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
              f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"Dice train={train_dice:.4f} val={val_dice:.4f} | "
              f"IoU val={val_iou:.4f} | {ep_time:.1f}s{marker}")

    total_time = time.time() - start
    print(f"\nTổng thời gian: {total_time/60:.1f} phút")
    print(f"Best val Dice: {best_val_dice:.4f}")

    # Lưu history & biểu đồ
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    ve_bieu_do(history, OUTPUT_DIR / "training_history.png")

    # Test với best model
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ TRÊN TEST SET (best model)")
    print("=" * 60)
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt",
                              weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_dice, test_iou = eval_epoch(
        model, loaders["test"], criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU : {test_iou:.4f}")

    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write(f"KẾT QUẢ {ARCH.upper()} TRÊN TEST SET\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best epoch    : {checkpoint['epoch']}\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Dice     : {test_dice:.4f}\n")
        f.write(f"Test IoU      : {test_iou:.4f}\n")
        f.write(f"Số tham số    : {n_params:.2f}M\n")
        f.write(f"Tổng thời gian: {total_time/60:.1f} phút\n")


if __name__ == "__main__":
    main()
