import os
import torch
import torch.nn as nn
import argparse
from dataset import BlurCleanDataset
from unet import UNet
from utils import psnr, ssim
import matplotlib.pyplot as plt
import pandas as pd
from config import IN_CHANNELS, OUT_CHANNELS, BASE_CH, EPOCHS, BATCH_SIZE, LR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    root = "data"
    results_dir = "results/checkpoints"
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    train_set = BlurCleanDataset(root, train=True)
    val_set   = BlurCleanDataset(root, train=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base_ch=BASE_CH).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    best_psnr, no_improve = -1.0, 0
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best_psnr = 1, -1.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -1.0)

    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        running_loss = 0.0
        for blur, clean in train_loader:
            blur, clean = blur.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(blur)
                loss = criterion(out, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * blur.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_psnr_epoch, val_ssim = 0.0, 0.0, 0.0
        with torch.no_grad():
            for blur, clean in val_loader:
                blur, clean = blur.to(device), clean.to(device)
                out = model(blur)
                val_loss += criterion(out, clean).item() * blur.size(0)
                val_psnr_epoch += psnr(out, clean) * blur.size(0)
                val_ssim += ssim(out, clean) * blur.size(0)

        val_loss /= len(val_loader.dataset)
        val_psnr_epoch /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        scheduler.step(val_loss)

        improved = val_psnr_epoch > best_psnr + 0.1  # 0.1 dB margin
        if improved:
            best_psnr = val_psnr_epoch
            no_improve = 0
            torch.save(model.state_dict(), "results/checkpoints/best.pt")
        else:
            no_improve += 1

        if no_improve >= 8:  # patience
            print("Early stopping: no PSNR improvement.")
            break

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_psnr={val_psnr_epoch:.2f} dB | val_ssim={val_ssim:.3f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr_epoch)
        history["val_ssim"].append(val_ssim)

        # Save checkpoint
        ckpt_path = os.path.join(results_dir, f"epoch{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_psnr": best_psnr
        }, ckpt_path)

        if val_psnr_epoch > best_psnr:
            best_psnr = val_psnr_epoch
            torch.save(model.state_dict(), os.path.join(results_dir, "best.pt"))


    df = pd.DataFrame(history)
    df.to_csv("results/training_log.csv", index=False)

    plt.plot(df["train_loss"], label="Train loss")
    plt.plot(df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/figures/loss_curve.png")

    plt.figure()
    plt.plot(df["val_psnr"], label="Val PSNR (dB)")
    plt.plot(df["val_ssim"], label="Val SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig("results/figures/metrics_curve.png")

if __name__ == "__main__":
    main()
