import torch
from PIL import Image
import numpy as np
import argparse
import os
from unet import UNet
from utils import save_image_tensor
from config import IN_CHANNELS, OUT_CHANNELS, BASE_CH, IMG_SIZE, CHECKPOINT_PATH


def load_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def infer(model, img_tensor, device):
    with torch.no_grad():
        pred = model(img_tensor.to(device))
    return pred


def main():
    parser = argparse.ArgumentParser(description="Run U-Net deblurring inference.")
    parser.add_argument(
        "-i", "--image", type=str, required=True,
        help="Path to the input blurred image."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str,
        default="results/checkpoints/best.pt",
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=2, default=(512, 512),
        help="Resize the input image to (width height). Default: (512 512)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base_ch=BASE_CH).to(device)
    ckpt = torch.load(args.checkpoint or CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    blur = load_image(args.image, size=tuple(args.size)).to(device)
    pred = infer(model, blur, device)

    base, ext = os.path.splitext(os.path.basename(args.image))
    output_name = f"{base}_pred{ext}"
    save_image_tensor(pred, output_name)

    print(f"Saved deblurred image: {output_name}")


if __name__ == "__main__":
    main()
