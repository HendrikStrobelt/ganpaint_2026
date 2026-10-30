"""Stroke -> feature-map intervention, ported from gandissect's serverstate.decode_intervention(_array).

A stroke selects some units, a value per unit (the unit's high level to draw, 0 to remove) and a canvas mask.
It becomes an alpha plane (unit alpha * the mask max-pooled to the feature map) and a value plane. Several
strokes are composited with alpha-over, then applied to the layer as  x * (1 - a) + v * a.
"""
import base64
import io
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MAX_MASK_PIXELS = 512 * 512


def decode_mask_png(data_url):
    """PNG data URL -> float mask (H, W) in {0, 1}: nontransparent pixels if there is an alpha channel, else nonwhite."""
    raw = base64.b64decode(re.sub(r"^(?:data:)?image/\w+;base64,", "", data_url), validate=True)
    im = Image.open(io.BytesIO(raw))
    if im.width * im.height > MAX_MASK_PIXELS:
        raise ValueError(f"mask too large: {im.width}x{im.height}")
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        arr = np.asarray(im.convert("RGBA"))[:, :, 3] > 0
    else:
        arr = np.asarray(im.convert("RGB"))[:, :, 0] < 255
    return torch.from_numpy(arr.astype(np.float32))


def stroke_planes(units, values, mask, layer_shape):
    """(alpha, value) planes of shape (C, H, W) for one stroke.

    units: unit indices; values: {unit: activation} (0 removes); mask: (h, w) float; layer_shape: (C, H, W).
    """
    channels, height, width = layer_shape
    alpha = torch.zeros(channels, height, width)
    value = torch.zeros(channels, height, width)
    for u in units:
        alpha[u] = 1.0
        value[u] = float(values[u])
    layer_mask = F.adaptive_max_pool2d(mask[None, None], (height, width))[0]  # (1, H, W)
    return alpha * layer_mask, value


def compose(planes):
    """Alpha-over composite of [(alpha, value), ...] in stroke order; None for no strokes."""
    acc = None
    for alpha, value in planes:
        if acc is None:
            acc = (alpha.clone(), value.clone())
            continue
        acc_alpha, acc_value = acc
        new_alpha = 1 - (1 - alpha) * (1 - acc_alpha)
        new_value = (acc_value * acc_alpha * (1 - alpha) + value * alpha) / (new_alpha + 1e-40)
        acc = (new_alpha, new_value)
    return acc


def apply_edit(features, alpha, value):
    return features * (1 - alpha) + value * alpha
