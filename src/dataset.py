import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

Image.MAX_IMAGE_PIXELS = None

class BlurCleanDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        split = "train" if train else "val"
        blur_dir  = os.path.join(root, split, "blur")
        clean_dir = os.path.join(root, split, "clean")

        self.data = []
        for fname in os.listdir(blur_dir):
            if fname.endswith(".png"):
                base = fname.split("_")[0] + ".png"
                blur_path  = os.path.join(blur_dir, fname)
                clean_path = os.path.join(clean_dir, base)
                if os.path.exists(clean_path):
                    self.data.append((blur_path, clean_path))
        self.train = train

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        blur_path, clean_path = self.data[idx]
        blur  = self._load_img(blur_path)
        clean = self._load_img(clean_path)

        if self.train:
            blur, clean = self._augment(blur, clean)

        return blur, clean

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2,0,1)

    def _augment(self, blur, clean):
        if random.random() < 0.5:
            blur  = torch.flip(blur, [2])
            clean = torch.flip(clean, [2])
        if random.random() < 0.5:
            blur  = torch.flip(blur, [1])
            clean = torch.flip(clean, [1])
        k = random.randint(0,3)
        if k:
            blur  = torch.rot90(blur, k, dims=[1,2])
            clean = torch.rot90(clean, k, dims=[1,2])
        return blur, clean
