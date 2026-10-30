"""A GANPaint project: generator + brush configuration, with edited image generation."""
import base64
import io
import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ganpaint_pipeline.proggan import load_safetensors
from ganpaint_pipeline.zdataset import z_sample

from .intervention import apply_edit, compose, decode_mask_png, stroke_planes


class Project:
    """config: {layer, brushes: {name: {units: [best first], ...}}, levels: [per unit], strengths, gallery}."""

    def __init__(self, name, generator, config, device="cpu", cache_size=256, z_pool=2048):
        self.name, self.config, self.device = name, config, device
        self.layer = config["layer"]
        self.generator = generator.to(device).eval()
        names = [n for n, _ in generator.named_children()]
        split = names.index(self.layer) + 1
        self.head = torch.nn.Sequential(OrderedDict((n, getattr(generator, n)) for n in names[:split]))
        self.tail = torch.nn.Sequential(OrderedDict((n, getattr(generator, n)) for n in names[split:]))
        self.levels = [float(v) for v in config["levels"]]
        self._z = z_sample(z_pool, seed=1)
        self._features = OrderedDict()  # id -> layer features, LRU
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.layer_shape = None

    @classmethod
    def load(cls, directory, device="cpu", **kw):
        directory = Path(directory)
        config = json.load(open(directory / "ganpaint.json"))
        generator = load_safetensors(directory / config["generator"])
        return cls(directory.name, generator, config, device, **kw)

    @property
    def max_id(self):
        return len(self._z) - 1

    def _head_features(self, sample_id):
        if sample_id in self._features:
            self._features.move_to_end(sample_id)
            return self._features[sample_id]
        z = self._z[sample_id : sample_id + 1].to(self.device)
        feats = self.head(z.view(1, -1, 1, 1))
        self._features[sample_id] = feats
        if len(self._features) > self._cache_size:
            self._features.popitem(last=False)
        return feats

    def units_for(self, brush, strength):
        n = self.config["strengths"][strength]
        return self.config["brushes"][brush]["units"][:n]

    def edit_planes(self, strokes, layer_shape):
        planes = []
        for s in strokes:
            units = self.units_for(s["brush"], s["strength"])
            values = self.levels if s["mode"] == "draw" else {u: 0.0 for u in units}
            planes.append(stroke_planes(units, values, decode_mask_png(s["mask"]), layer_shape))
        return compose(planes)

    @torch.no_grad()
    def generate(self, sample_id, strokes=()):
        """Image as uint8 (H, W, 3) for a sample id with the strokes applied."""
        if not 0 <= sample_id <= self.max_id:
            raise IndexError(f"sample id {sample_id} outside 0..{self.max_id}")
        with self._lock:
            feats = self._head_features(sample_id)
            if strokes:
                alpha, value = self.edit_planes(strokes, tuple(feats.shape[1:]))
                feats = apply_edit(feats, alpha.to(self.device)[None], value.to(self.device)[None])
            image = self.tail(feats)[0]
        return ((image + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def encode_image(array, fmt="WEBP", quality=92):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, fmt, quality=quality)
    return buf.getvalue()


def data_url(raw, mime="image/webp"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
