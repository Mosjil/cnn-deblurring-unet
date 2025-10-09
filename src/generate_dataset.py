import os
import numpy as np
from PIL import Image
import torch
import pado
from pado.math import mm, nm, um, fft, ifft
import random
import shutil
from sklearn.model_selection import train_test_split
from config import IMG_SIZE, NUM_BLURS_PER_IMAGE, NOISE_STD_RANGE, INPUT_DIR, OUTPUT_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT


def convolution(image, kernel, pad):
    """
    version fidèle au prof, adaptée pour [B,Ch,H,W]
    """
    image_fft = fft(image, pad_width=None)
    kernel_fft = fft(kernel, pad_width=pad)
    return ifft(image_fft * kernel_fft, pad_width=None).real

def load_and_preprocess(img_path):
    """Charge une image et la convertit en tensor torch [1,3,H,W] normalisé [0,1]."""
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor

def save_image_torch(tensor, path):
    """
    Sauvegarde un tensor torch [1,3,H,W] en PNG.
    Gère aussi les cas où il y a des dimensions parasites (5D ou 4D).
    """

    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    arr = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def generate_random_psf_torch(dim=(1, 1, 128, 128), pitch=6.4 * um, wavelength=550 * nm):
    """Génère un PSF avec une lentille réfractive + ouverture circulaire, retourne torch [1,1,h,w]."""
    focal_length = np.random.uniform(30 * mm, 100 * mm)
    aperture_diameter = np.random.uniform(0.3 * mm, 1.2 * mm)

    light = pado.light.Light(dim, pitch, wavelength)

    lens = pado.optical_element.RefractiveLens(
        dim=dim, pitch=pitch, focal_length=focal_length,
        wvl=wavelength, device="cpu"
    )
    field = lens.forward(light)

    aperture = pado.optical_element.Aperture(
        dim=dim, pitch=pitch, aperture_diameter=aperture_diameter,
        aperture_shape="circle", wvl=wavelength, device="cpu"
    )
    field = aperture.forward(field)

    prop = pado.propagator.Propagator("ASM")
    propagated = prop.forward(field, focal_length)

    psf = propagated.get_intensity(c=0).squeeze(0)
    psf = psf / psf.sum()
    return psf.unsqueeze(0).unsqueeze(0), {
        "focal_length_mm": focal_length / mm,
        "aperture_diameter_mm": aperture_diameter / mm
    }


def apply_blur_torch(image, psf):

    _, _, H, W = image.shape
    _, _, h, w = psf.shape

    kernel = psf / psf.sum()

    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    pad = (pad_w, W - w - pad_w, pad_h, H - h - pad_h)

    blurred = convolution(image, kernel, pad)  # [1,3,H,W]
    return blurred

def add_gaussian_noise_torch(image, std):
    """Ajoute bruit gaussien torch."""
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0, 1)


def main():
    img_paths = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    train_paths, temp_paths = train_test_split(
        img_paths, test_size=(1 - TRAIN_SPLIT), random_state=42
    )
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_paths, test_paths = train_test_split(temp_paths, test_size=(1 - val_ratio), random_state=42)

    splits = {"train": train_paths, "val": val_paths, "test": test_paths}

    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing dataset directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, paths in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(os.path.join(split_dir, "clean"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "blur"), exist_ok=True)

        print(f"🔹 Generating {split_name} set ({len(paths)} images)...")

        for i, path in enumerate(paths):
            clean = load_and_preprocess(path)
            save_image_torch(clean, os.path.join(split_dir, "clean", f"{i:04d}.png"))

            for j in range(NUM_BLURS_PER_IMAGE):
                psf = generate_random_psf_torch()
                blurred = apply_blur_torch(clean, psf)
                std = random.uniform(*NOISE_STD_RANGE)
                noised = add_gaussian_noise_torch(blurred, std)
                save_image_torch(noised, os.path.join(split_dir, "blur", f"{i:04d}_{j}.png"))

    print(f"\nDataset generated in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
