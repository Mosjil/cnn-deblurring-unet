import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not norm),
            nn.BatchNorm2d(out_ch) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not norm),
            nn.BatchNorm2d(out_ch) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, norm=norm)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, x_ch, skip_ch, out_ch, bilinear=True, norm=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in = x_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(x_ch, x_ch//2, 2, stride=2)
            conv_in = x_ch//2 + skip_ch
        self.conv = ConvBlock(conv_in, out_ch, norm=norm)
    def forward(self, x, skip):
        x = self.up(x)
        diffY, diffX = skip.size(2)-x.size(2), skip.size(3)-x.size(3)
        if diffX or diffY:
            x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=64, depth=4, bilinear=True, norm=True):
        super().__init__()
        feats = [base_ch * (2**i) for i in range(depth+1)]
        self.inc = ConvBlock(in_channels, feats[0], norm=norm)
        self.downs = nn.ModuleList([Down(feats[i], feats[i+1], norm=norm) for i in range(depth)])
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(Up(feats[i+1], feats[i], feats[i], bilinear, norm))
        self.outc = nn.Conv2d(feats[0], out_channels, 1)
    def forward(self, x):
        x0 = self.inc(x)
        skips = [x0]
        x_curr = x0
        for down in self.downs:
            x_curr = down(x_curr)
            skips.append(x_curr)
        x_dec = x_curr
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x_dec = up(x_dec, skip)
        return torch.sigmoid(self.outc(x_dec))
