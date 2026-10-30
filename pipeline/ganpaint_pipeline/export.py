"""Export everything the server needs into data/<project>/: generator.safetensors and ganpaint.json."""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import save_file

from .convert_progan import load_old_pth
from .dissect import LAYER, load_state
from .runningstats import RunningQuantile

# The original UI's brush list: (brush, ranking). Tree, grass and door use ACE rankings; the rest use IoU.
BRUSHES = [("tree", "tree-ace"), ("grass", "grass-ace"), ("door", "door-ace"),
           ("sky", "sky-iou"), ("cloud", "cloud-iou"), ("brick", "brick-iou"), ("dome", "dome-iou")]
STRENGTHS = {"low": 4, "med": 10, "high": 15}  # number of units painted
GALLERY = [104, 425, 457, 489, 495, 570, 584, 644, 700, 705, 719, 726, 816, 1085, 1146, 1362]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def top_units(scores, n):
    """Units with the lowest scores first (scores are negated: IoU, ACE weight); stable on ties."""
    return np.argsort(np.asarray(scores), kind="stable")[:n].tolist()


def export(project_dir, source_pth, dissect_dir=None, fallback_iou=False, layer=LAYER):
    """Write generator.safetensors and ganpaint.json. With fallback_iou, a missing `<c>-ace` ranking falls back to `<c>-iou`."""
    project_dir = Path(project_dir)
    dissect_dir = Path(dissect_dir or project_dir / "dissect")
    top = json.load(open(dissect_dir / "dissect.json"))
    rankings = {r["name"]: r["score"] for r in [l for l in top["layers"] if l["layer"] == layer][0]["rankings"]}

    model = load_old_pth(source_pth)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(project_dir / "generator.safetensors"),
              metadata={"source": Path(source_pth).name, "source_sha256": sha256(source_pth)})

    # The edit level of a unit is its 0.99 activation quantile (the original estimator, as the demo used).
    rq = RunningQuantile(state=load_state(dissect_dir / layer / "quantiles.npz"))
    levels = rq.quantiles([0.99])[:, 0].tolist()

    n_units = max(STRENGTHS.values())
    brushes, used = {}, {}
    for brush, ranking in BRUSHES:
        if ranking not in rankings and fallback_iou and ranking.endswith("-ace"):
            ranking = f"{brush}-iou"
        if ranking not in rankings:
            raise KeyError(f"ranking {ranking!r} missing from dissect.json (run `ace` for {brush}, or pass fallback_iou)")
        brushes[brush] = dict(label=brush, ranking=ranking, units=top_units(rankings[ranking], n_units))
        used[brush] = ranking
    config = dict(layer=layer, generator="generator.safetensors", levels=levels, strengths=STRENGTHS,
                  default_strength="high", brushes=brushes, gallery=GALLERY)
    json.dump(config, open(project_dir / "ganpaint.json", "w"), indent=1)
    return used
