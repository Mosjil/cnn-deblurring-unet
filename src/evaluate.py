import os
import torch
from torch.utils.data import DataLoader
from dataset import BlurCleanDataset
from unet import UNet
from utils import save_image_tensor, psnr, visualize_triplet
from config import IN_CHANNELS, OUT_CHANNELS, BASE_CH, CHECKPOINT_PATH, RESULTS_DIR


def main():

    data_root = "data"
    ckpt_path = CHECKPOINT_PATH
    results_dir = RESULTS_DIR
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(results_dir, exist_ok=True)

    test_set = BlurCleanDataset(root=data_root, train=False)  # suppose data/test/ existe
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base_ch=BASE_CH).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_psnr, n = 0.0, 0
    with torch.no_grad():
        for idx, (blur, clean) in enumerate(test_loader):
            blur, clean = blur.to(device), clean.to(device)
            pred = model(blur)

            total_psnr += psnr(pred, clean) * blur.size(0)
            n += blur.size(0)

            for k in range(blur.size(0)):
                base = f"sample_{idx*batch_size+k:04d}"
                save_image_tensor(blur[k], os.path.join(results_dir, f"{base}_blur.png"))
                save_image_tensor(pred[k], os.path.join(results_dir, f"{base}_pred.png"))
                save_image_tensor(clean[k], os.path.join(results_dir, f"{base}_clean.png"))

    avg_psnr = total_psnr / n
    print(f"Test PSNR = {avg_psnr:.2f} dB (avg over {n} images)")

    visualize_triplet(blur, pred, clean, save_path="results/sample.png")

if __name__ == "__main__":
    main()
