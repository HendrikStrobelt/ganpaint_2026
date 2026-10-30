"""ProGAN generator (Karras et al. 2018) in plain PyTorch.

Layer names follow the original gandissect port so that `layer4` is the 512 x 8 x 8
feature map GANPaint edits: layer1..layerN are the conv blocks, followed by
`output_256x256` (the last ToRGB). Forward returns an unclamped RGB image in [-1, 1].
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

RESOLUTION_SIZES = {
    # [z, 4x4, 8x8, ..., 256x256]: min(8192 / 2**stage, 512), as in Karras' G_paper (fmap_base=8192)
    256: [512, 512, 512, 512, 512, 256, 128, 64],
}


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ConvBlock(nn.Module):
    """pixelnorm -> [2x nearest upsample] -> conv (no bias) -> *wscale + bias -> leaky relu."""

    def __init__(self, cin, cout, kernel, padding, upsample=False, gain=None):
        super().__init__()
        self.norm = PixelNorm()
        self.upsample = upsample
        self.conv = nn.Conv2d(cin, cout, kernel, 1, padding, bias=False)
        gain = math.sqrt(2) / kernel if gain is None else gain
        self.scale = gain / math.sqrt(cin)
        self.b = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        x = self.norm(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x) * self.scale + self.b.view(1, -1, 1, 1)
        return F.leaky_relu(x, 0.2)


class OutputBlock(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.norm = PixelNorm()
        self.conv = nn.Conv2d(cin, 3, 1, bias=False)
        self.scale = 1.0 / math.sqrt(cin)
        self.b = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        return self.conv(self.norm(x)) * self.scale + self.b.view(1, -1, 1, 1)


class ProgressiveGenerator(nn.Sequential):
    def __init__(self, resolution=256, sizes=None):
        sizes = sizes or RESOLUTION_SIZES[resolution]
        blocks = [
            ("layer1", ConvBlock(sizes[0], sizes[1], 4, 3, gain=math.sqrt(2) / 4)),
            ("layer2", ConvBlock(sizes[1], sizes[1], 3, 1)),
        ]
        n = 2
        for cin, cout in zip(sizes[1:-1], sizes[2:]):
            blocks.append((f"layer{n + 1}", ConvBlock(cin, cout, 3, 1, upsample=True)))
            blocks.append((f"layer{n + 2}", ConvBlock(cout, cout, 3, 1)))
            n += 2
        blocks.append((f"output_{resolution}x{resolution}", OutputBlock(sizes[-1])))
        super().__init__(OrderedDict(blocks))

    def forward(self, z):
        return super().forward(z.view(z.shape[0], z.shape[1], 1, 1))


def load_safetensors(path, device="cpu"):
    """Build a ProgressiveGenerator from a safetensors state dict; channel sizes are read from the weights."""
    from safetensors.torch import load_file

    sd = load_file(str(path))
    n = 0
    while f"layer{n + 1}.conv.weight" in sd:
        n += 1
    sizes = [sd["layer1.conv.weight"].shape[1]] + [sd[f"layer{i}.conv.weight"].shape[0] for i in range(1, n + 1, 2)]
    resolution = 4 * 2 ** (n // 2 - 1)
    model = ProgressiveGenerator(resolution, sizes=sizes)
    model.load_state_dict(sd)
    return model.to(device).eval()
