"""
GIAI ĐOẠN 4 - U-NET TRAINING (Phiên bản 2 - cải tiến hội tụ)
Đề tài: Phát hiện cháy rừng trên ảnh vệ tinh

CẢI TIẾN SO VỚI v1 (08_train_unet.py):
1. Tăng epochs từ 40 -> 80 (cho hội tụ đầy đủ)
2. Giảm LR từ 1e-3 -> 5e-4 (mềm hơn, ít dao động)
3. THÊM Cosine Annealing scheduler với warmup 5 epoch
4. THÊM Gradient clipping (max_norm=1.0) - tránh "nhảy" loss
5. THÊM Early stopping (patience 15) - dừng khi không cải thiện
6. Output -> ket_qua_unet_v2 (KHÔNG đè kết quả cũ)

CÁCH CHẠY:
  python 08_train_unet_v2.py
"""

from pathlib import Path
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import matplotlib.pyplot as plt

from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
dataloader = import_module("06_dataloader")
unet_module = import_module("08_train_unet")
tao_dataloaders = dataloader.tao_dataloaders
UNet = unet_module.UNet
BCEDiceLoss = unet_module.BCEDiceLoss
dice_score = unet_module.dice_score
iou_score = unet_module.iou_score


# ============================================================
# CẤU HÌNH MỚI
# ============================================================
CONFIG = {
    "in_channels": 12,
    "num_classes": 1,
    "patch_size": 256,
    "batch_size": 4,
    "epochs": 80,                # TĂNG từ 40
    "lr": 5e-4,                  # GIẢM từ 1e-3
    "weight_decay": 1e-5,
    "warmup_epochs": 5,          # MỚI: warmup 5 epoch
    "early_stop_patience": 15,   # MỚI: early stopping
    "grad_clip": 1.0,            # MỚI: gradient clipping
    "optimizer": "Adam",
    "scheduler": "Warmup + CosineAnnealing",
    "loss": "BCE + Dice",
}

# Output dir MỚI - không đè kết quả cũ
OUTPUT_DIR = Path("./ket_qua_unet_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TRAIN EPOCH (có gradient clipping)
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, total_dice, n_batch = 0.0, 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        # GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
# VẼ BIỂU ĐỒ (thêm subplot cho learning rate)
# ============================================================
def ve_bieu_do_history(history, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
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

    axes[2].plot(epochs, history["lr"], "g-")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR schedule (Warmup + Cosine)")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("TRAIN U-NET v2 - CẢI TIẾN HỘI TỤ")
    print("=" * 60)
    print(f"Thiết bị: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print("\nConfig mới:")
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
    print(f"\nU-Net khởi tạo. Tổng tham số: {n_params:.2f}M")

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    criterion = BCEDiceLoss(dice_weight=0.5)

    # SCHEDULER: Warmup tuyến tính + Cosine Annealing
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                       total_iters=CONFIG["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer,
                                 T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"],
                                 eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[CONFIG["warmup_epochs"]])
    print(f"Scheduler: Warmup {CONFIG['warmup_epochs']} epoch + "
          f"Cosine Annealing {CONFIG['epochs'] - CONFIG['warmup_epochs']} epoch")

    # Training loop
    print("\n" + "=" * 60)
    print("BẮT ĐẦU TRAINING")
    print("=" * 60)

    history = {"train_loss": [], "train_dice": [],
               "val_loss": [], "val_dice": [], "val_iou": [], "lr": []}
    best_val_dice = 0.0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        ep_start = time.time()

        # Train với gradient clipping
        train_loss, train_dice = train_epoch(
            model, loaders["train"], optimizer, criterion, device,
            CONFIG["grad_clip"])

        # Val
        val_loss, val_dice, val_iou = eval_epoch(
            model, loaders["val"], criterion, device)

        # Scheduler step (sau val)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)

        # Lưu best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": CONFIG,
            }, OUTPUT_DIR / "best_model.pt")
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        ep_time = time.time() - ep_start
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
              f"lr={current_lr:.2e} | "
              f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"Dice train={train_dice:.4f} val={val_dice:.4f} | "
              f"IoU val={val_iou:.4f} | "
              f"pat={patience_counter}/{CONFIG['early_stop_patience']} | "
              f"{ep_time:.1f}s{marker}")

        # EARLY STOPPING
        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"\n[EARLY STOP] Val không cải thiện trong "
                  f"{CONFIG['early_stop_patience']} epoch. Dừng ở epoch {epoch}.")
            break

    total_time = time.time() - start
    print(f"\nTổng thời gian training: {total_time/60:.1f} phút")
    print(f"Best val Dice: {best_val_dice:.4f}")
    print(f"Số epoch thực tế đã chạy: {len(history['train_loss'])}")

    # Lưu history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    ve_bieu_do_history(history, OUTPUT_DIR / "training_history.png")
    print(f"Đã lưu biểu đồ: {OUTPUT_DIR}/training_history.png")

    # Đánh giá cuối cùng trên test set
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ TRÊN TEST SET (best model)")
    print("=" * 60)
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_dice, test_iou = eval_epoch(
        model, loaders["test"], criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU : {test_iou:.4f}")

    # So sánh với v1
    v1_test_dice = 0.8511
    v1_test_iou = 0.7789
    delta_dice = (test_dice - v1_test_dice) * 100
    delta_iou = (test_iou - v1_test_iou) * 100
    print(f"\nSo với v1 (U-Net cũ):")
    print(f"  Test Dice: {v1_test_dice:.4f} -> {test_dice:.4f} "
          f"({'+' if delta_dice >= 0 else ''}{delta_dice:.2f} điểm)")
    print(f"  Test IoU : {v1_test_iou:.4f} -> {test_iou:.4f} "
          f"({'+' if delta_iou >= 0 else ''}{delta_iou:.2f} điểm)")

    # Lưu kết quả
    with open(OUTPUT_DIR / "ket_qua_so.txt", "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ U-NET v2 TRÊN TEST SET\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best epoch    : {checkpoint['epoch']}\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Dice     : {test_dice:.4f}\n")
        f.write(f"Test IoU      : {test_iou:.4f}\n")
        f.write(f"Số tham số    : {n_params:.2f}M\n")
        f.write(f"Tổng thời gian: {total_time/60:.1f} phút\n")
        f.write(f"\nSo với v1:\n")
        f.write(f"  Test Dice: {v1_test_dice:.4f} -> {test_dice:.4f} "
                f"({'+' if delta_dice >= 0 else ''}{delta_dice:.2f} điểm)\n")
        f.write(f"  Test IoU : {v1_test_iou:.4f} -> {test_iou:.4f} "
                f"({'+' if delta_iou >= 0 else ''}{delta_iou:.2f} điểm)\n")


if __name__ == "__main__":
    main()
