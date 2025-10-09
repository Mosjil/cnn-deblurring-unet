import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def mse(x, y): return torch.mean((x-y)**2).item()

def psnr(x, y, eps=1e-10):
    mse_val = torch.mean((x-y)**2).item()
    return 10.0 * np.log10(1.0/(mse_val+eps))

def save_image_tensor(x, path):
    if x.dim() == 4: x = x[0]
    arr = (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def ssim(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    ssim_val = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_val.item()

def visualize_triplet(blur, pred, clean, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Blurred", "Predicted", "Ground truth"]
    for ax, img, title in zip(axs, [blur, pred, clean], titles):
        if img.dim() == 4:
            img = img[0]
        arr = img.permute(1,2,0).cpu().numpy().clip(0,1)
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.show()