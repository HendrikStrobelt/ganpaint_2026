"""Convert a Karras TF1 ProGAN pickle (non-fused `G_paper`, 256x256) to a safetensors state dict.

Mapping (TF variable -> PyTorch parameter), with `Gs` the EMA generator:
  4x4/Dense/weight (512, 512*16)  -> layer1.conv.weight   view(in, C, 4, 4).permute(1,0,2,3).flip(2,3)
  4x4/Conv/weight  (3,3,in,out)   -> layer2.conv.weight   permute(3,2,0,1)
  {R}x{R}/Conv0|Conv1             -> layer3.. conv.weight permute(3,2,0,1)   (cross-correlation, no flip)
  ToRGB_lod0/weight (1,1,in,3)    -> output_256x256.conv.weight  permute(3,2,0,1)
  */bias                          -> *.b
"""
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file

from .proggan import ProgressiveGenerator
from .tfpickle import load_networks


def tf_variables_to_state_dict(variables):
    v = dict(variables)
    t = lambda name: torch.from_numpy(v[name].copy())
    sd = {}

    dense = t("4x4/Dense/weight")  # (z, C*16), TF reshape order is [C, 4, 4]
    sd["layer1.conv.weight"] = dense.view(dense.shape[0], -1, 4, 4).permute(1, 0, 2, 3).flip(2, 3).contiguous()
    sd["layer1.b"] = t("4x4/Dense/bias")
    sd["layer2.conv.weight"] = t("4x4/Conv/weight").permute(3, 2, 0, 1).contiguous()
    sd["layer2.b"] = t("4x4/Conv/bias")

    n, res = 3, 8
    while f"{res}x{res}/Conv0/weight" in v:
        for conv in ("Conv0", "Conv1"):
            sd[f"layer{n}.conv.weight"] = t(f"{res}x{res}/{conv}/weight").permute(3, 2, 0, 1).contiguous()
            sd[f"layer{n}.b"] = t(f"{res}x{res}/{conv}/bias")
            n += 1
        res *= 2
    out = f"output_{res // 2}x{res // 2}"
    sd[f"{out}.conv.weight"] = t("ToRGB_lod0/weight").permute(3, 2, 0, 1).contiguous()
    sd[f"{out}.b"] = t("ToRGB_lod0/bias")
    return sd


def load_old_pth(path):
    """Load a gandissect-format checkpoint (`features.N.*` / `output.*`, `wscale.scale` buffers).

    Uses weights_only=True so the pickle cannot execute code. Channel sizes are read from
    the weights. The model computes wscale itself (gain / sqrt(fan_in)).
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    n = 0
    while f"features.{n}.conv.weight" in sd:
        n += 1
    sizes = [sd["features.0.conv.weight"].shape[1]] + [sd[f"features.{i}.conv.weight"].shape[0] for i in range(0, n, 2)]
    res = 4 * 2 ** (n // 2 - 1)
    new = {}
    for i in range(n):
        new[f"layer{i + 1}.conv.weight"] = sd[f"features.{i}.conv.weight"]
        new[f"layer{i + 1}.b"] = sd[f"features.{i}.wscale.b"]
    new[f"output_{res}x{res}.conv.weight"] = sd["output.conv.weight"]
    new[f"output_{res}x{res}.b"] = sd["output.wscale.b"]
    model = ProgressiveGenerator(res, sizes=sizes)
    model.load_state_dict(new)
    stored = [sd[f"features.{i}.wscale.scale"].item() for i in range(n)] + [sd["output.wscale.scale"].item()]
    mine = [b.scale for b in model if hasattr(b, "scale")]
    # The original gandissect loader dropped the stored scales and recomputed them, so the demo ran on
    # the computed values; we do the same and only warn when the stored ones disagree noticeably.
    worst = max(abs(a / b - 1) for a, b in zip(stored, mine))
    if worst > 0.01:
        warnings.warn(f"stored wscale differs from computed by up to {worst:.1%} (output layer); using computed, like the original loader")
    return model


def convert(pkl_path, out_path, which="Gs"):
    G, D, Gs = load_networks(pkl_path)
    net = {"G": G, "Gs": Gs}[which].__dict__
    assert net["static_kwargs"].get("fused_scale") is False, "only non-fused checkpoints are supported"
    assert net["static_kwargs"]["resolution"] == 256
    sd = tf_variables_to_state_dict(net["variables"])
    model = ProgressiveGenerator(256)
    model.load_state_dict(sd)  # strict: fails on any name or shape mismatch
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_path), metadata={"source": Path(pkl_path).name, "network": which})
    return model
