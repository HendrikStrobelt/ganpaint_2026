"""GAN dissection of one layer: port of netdissect.dissection for generators.

Stages (as in `python -m netdissect --gan ... --quantile_threshold iqr`):
  1. 0.99-style quantiles and top-k of every unit over z-set A (seed 1)
  2. per-unit activation thresholds by maximising the information quality ratio (IQR) against every
     segmentation label, using the conditional quantiles of z-set B (seed 2)
  3. IoU / IQR of every (label, unit) pair on z-set A with those thresholds
  4. dissect.json with one `<label>-iou` ranking per label, in the original schema

Differences from the original, none of which change what is computed: each z-set is generated and
segmented once and cached in memory instead of once per pass; batches replace batch-size-1 loops;
`align_corners=True` is passed explicitly because the original relied on PyTorch 1.0's implicit default.
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .runningstats import RunningConditionalQuantile, RunningQuantile, RunningTopK
from .zdataset import z_sample

LAYER = "layer4"
SEG_DOWNSAMPLE = 2  # the original GeneratorSegRunner segments at half resolution


def upsample_grid(data_shape, target_shape, dtype=torch.float, device=None):
    """grid_sample grid mapping a (data_shape) feature map onto target_shape pixels (align_corners=True)."""
    scale = tuple(float(ts) / ds for ts, ds in zip(target_shape, data_shape))
    offset = tuple(0.5 * s - 0.5 for s in scale)
    ty, tx = (
        (torch.arange(ts, dtype=dtype, device=device) - o) * (2 / (s * (ss - 1))) - 1
        for ts, ss, s, o in zip(target_shape, data_shape, scale, offset)
    )
    return torch.stack((tx[None, :].expand(target_shape), ty[:, None].expand(target_shape)), 2)[None]


def upsample(feats, grid):
    return F.grid_sample(feats, grid.expand(len(feats), -1, -1, -1), padding_mode="border", align_corners=True)


class SampleSet:
    """Layer features and segmentations for a set of z samples (kept on the CPU)."""

    def __init__(self, z, feats, segs):
        self.z, self.feats, self.segs = z, feats, segs  # (N, 512), (N, C, h, w) float, (N, 5, H, W) int16

    def __len__(self):
        return len(self.z)


@torch.no_grad()
def collect_samples(model, segmenter, z, layer=LAYER, batch_size=50, desc="samples"):
    device = next(model.parameters()).device
    store = {}
    handle = getattr(model, layer).register_forward_hook(lambda m, i, o: store.__setitem__("f", o.detach()))
    feats, segs = [], []
    try:
        for i in tqdm(range(0, len(z), batch_size), desc=desc):
            images = model(z[i : i + batch_size].to(device))  # unclamped, as in the original
            feats.append(store["f"].cpu())
            segs.append(segmenter.segment_batch(images, downsample=SEG_DOWNSAMPLE).short().cpu())
    finally:
        handle.remove()
    return SampleSet(z, torch.cat(feats), torch.cat(segs))


def label_tables(segmenter):
    labelcat, categories = segmenter.get_label_and_category_names()
    label_category = [categories.index(c) if c in categories else 0 for _, c in labelcat]
    return labelcat, categories, label_category


def mutual_information(arr):
    total = 0
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            joint = arr[j, k]
            ind = arr[j, :].sum(dim=0) * arr[:, k].sum(dim=0)
            term = joint * (joint / ind).log()
            term[torch.isnan(term)] = 0
            total += term
    return total.clamp_(0)


def joint_entropy(arr):
    total = 0
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            joint = arr[j, k]
            term = joint * joint.log()
            term[torch.isnan(term)] = 0
            total += term
    return (-total).clamp_(0)


def score_tally_stats(label_category, tc, truth, cc, ic):
    pred = cc[label_category]
    total = tc[label_category][:, None]
    truth = truth[:, None]
    union = pred + truth - ic
    iou = ic.double() / (union.double() + 1e-20)
    arr = torch.empty(size=(2, 2) + ic.shape, dtype=ic.dtype, device=ic.device)
    arr[0, 0] = ic
    arr[0, 1] = pred - ic
    arr[1, 0] = truth - ic
    arr[1, 1] = total - union
    arr = arr.double() / total.double()
    iqr = mutual_information(arr) / joint_entropy(arr)
    iqr[torch.isnan(iqr)] = 0
    return iou, iqr


@torch.no_grad()
def collect_quantiles_and_topk(samples, k=100, resolution=1024, chunk=100, device="cuda"):
    """Quantiles over every spatial position and top-k images (by spatial max) for each unit."""
    quantiles, topk = RunningQuantile(resolution=resolution), RunningTopK(k)
    for i in tqdm(range(0, len(samples), chunk), desc="quantiles"):
        value = samples.feats[i : i + chunk].to(device)
        top = value.view(*value.shape[:2], -1).max(2)[0]
        quantiles.add(value.permute(0, 2, 3, 1).contiguous().view(-1, value.shape[1]))
        topk.add(top)
    quantiles.to_(torch.device("cpu"))
    topk.to_(torch.device("cpu"))
    return quantiles, topk


def label_fractions(segs, num_labels, chunk=100):
    """Fraction of pixels carrying each label (a pixel can carry several, one per seg channel)."""
    counts = torch.zeros(num_labels, dtype=torch.long)
    for i in range(0, len(segs), chunk):
        counts += torch.bincount(segs[i : i + chunk].long().view(-1), minlength=num_labels)
    return (counts.float() / (len(segs) * segs.shape[2] * segs.shape[3]))[:, None, None]


@torch.no_grad()
def collect_iqr(samples, grid, label_category, num_categories, num_labels, device="cuda"):
    """Per (label, unit): max IQR over 100 log-spaced activation fractions, and the level achieving it."""
    labelcat = torch.zeros(num_labels, num_categories, dtype=torch.long, device=device)
    labelcat.scatter_(1, torch.tensor(label_category, dtype=torch.long, device=device)[:, None], 1)
    label_fracs = label_fractions(samples.segs, num_labels)
    skip_labels = set((label_fracs.view(-1) < 1e-4).nonzero().view(-1).tolist())

    cq = RunningConditionalQuantile(resolution=2048)
    for i in tqdm(range(len(samples)), desc="conditional quantiles"):
        value = samples.feats[i : i + 1].to(device)
        seg = samples.segs[i : i + 1].long().to(device)
        upsampled = upsample(value, grid).view(value.shape[1], -1)
        cq.add(("all",), upsampled.t())
        batch_labels = torch.bincount(seg.view(-1), minlength=num_labels).nonzero()[:, 0]
        for label in batch_labels.tolist():
            if label in skip_labels:
                continue
            imask = (seg == label).max(dim=1)[0].view(-1)
            cq.add(("label", label), upsampled[:, imask].t())
        for cat in labelcat[batch_labels].max(0)[0].nonzero()[:, 0]:
            cq.add(("cat", cat.item()), upsampled.t())

    label_list = [("label", i) for i in range(num_labels)]
    category_list = [("all",)] if num_categories <= 1 else [("cat", i) for i in range(num_categories)]
    fracs = torch.logspace(-3, 0, 100)
    levels = cq.conditional(("all",)).quantiles(1 - fracs)
    truth = label_fracs.to(device)
    preds = (1 - cq.collected_normalize(category_list, levels))[label_category, :, :].to(device)
    cond_isects = 1 - cq.collected_normalize(label_list, levels).to(device)
    isects = cond_isects * truth
    unions = truth + preds - isects
    arr = torch.empty(size=(2, 2) + isects.shape, dtype=isects.dtype, device=device)
    arr[0, 0] = isects
    arr[0, 1] = preds - isects
    arr[1, 0] = truth - isects
    arr[1, 1] = 1 - unions
    arr.clamp_(0, 1)
    mi = mutual_information(arr)
    mi[:, :, -1] = 0  # at the 1.0 quantile there should be no MI
    mi[label_fracs.view(-1) < 1e-3, :, :] = 0  # too few samples to trust
    iqr = mi / joint_entropy(arr)
    iqr[torch.isnan(iqr)] = 0
    agreement = isects + arr[1, 1]
    # maximise only where the unit is positively correlated with the label and the level is positive
    iqr[agreement <= 0.8] = 0
    iqr[(levels <= 0.0)[None, :, :].expand(iqr.shape)] = 0
    max_iqr, bucket = iqr.max(2)
    units = torch.arange(levels.shape[0])
    ar = torch.arange(isects.shape[0])[:, None], torch.arange(isects.shape[1])[None, :], bucket
    matching_iou = isects[ar] / unions[ar]
    matching_iou[torch.isnan(matching_iou)] = 0
    return dict(
        max_iqr=max_iqr.cpu(),
        max_iqr_level=levels.to(device)[units[None, :], bucket].cpu(),
        max_iqr_quantile=fracs.to(device)[bucket].cpu(),
        max_iqr_iou=matching_iou.cpu(),
        label_fracs=label_fracs,
    )


@torch.no_grad()
def collect_bincounts(samples, grid, levels, label_category, num_categories, num_labels, device="cuda"):
    """Pixel counts of (thresholded unit) x (label) intersections, and the resulting IoU and IQR tables."""
    labelcat = torch.zeros(num_labels, num_categories, dtype=torch.long, device=device)
    labelcat.scatter_(1, torch.tensor(label_category, dtype=torch.long, device=device)[:, None], 1)
    units = samples.feats.shape[1]
    levels = levels.to(device)
    label_counts = torch.zeros(num_labels, dtype=torch.long, device=device)
    total_counts = torch.zeros(num_categories, dtype=torch.long, device=device)
    category_activation_counts = torch.zeros(num_categories, units, dtype=torch.long, device=device)
    intersection_counts = torch.zeros(num_labels, units, dtype=torch.long, device=device)
    for i in tqdm(range(len(samples)), desc="bincounts"):
        seg = samples.segs[i : i + 1].long().to(device)
        bc = torch.bincount(seg.view(-1), minlength=num_labels)
        label_counts += bc
        batch_labels = bc.nonzero()[:, 0]
        batch_categories = labelcat[batch_labels].max(0)[0]
        total_counts += batch_categories * (seg.shape[0] * seg.shape[2] * seg.shape[3])
        amask = upsample(samples.feats[i : i + 1].to(device), grid) > levels[None, :, None, None]
        ac = amask.int().view(units, -1).sum(1)
        for label in batch_labels.tolist():
            if label == 0:
                continue  # ignore the background label
            imask = amask * (seg == label).max(dim=1, keepdim=True)[0]
            intersection_counts[label] += imask.int().view(units, -1).sum(1)
        category_activation_counts += batch_categories[:, None] * ac[None, :]
    iou, iqr = score_tally_stats(label_category, total_counts, label_counts, category_activation_counts, intersection_counts)
    return dict(
        iou_scores=iou.cpu(), iqr_scores=iqr.cpu(), total_counts=total_counts.cpu(), label_counts=label_counts.cpu(),
        category_activation_counts=category_activation_counts.cpu(), intersection_counts=intersection_counts.cpu(),
    )


def build_report(outdir, topk, quantiles, levels, iqrdata, bins, labelnames, catnames,
                 settings=None, extra_labels=(), iou_threshold=0.05, iqr_threshold=0.01, layer=LAYER):
    """dissect.json in the original schema (units, per-label `<label>-iou` rankings)."""
    catnumber = {n: i for i, n in enumerate(catnames)}
    catnumber["-"] = 0
    labelnumber = {name: num for num, (name, _) in enumerate(labelnames)}
    iou_scores, iqr_scores = bins["iou_scores"], bins["iqr_scores"]
    max_iqr = iqrdata["max_iqr"]
    best_label = max_iqr.max(0)[1]
    ar = torch.arange(iou_scores.shape[1])
    best_score, best_qscore = iou_scores[best_label, ar], iqr_scores[best_label, ar]
    topa, topi = topk.result()
    units = []
    for u in range(len(topa)):
        label = best_label[u].item()
        name, cat = labelnames[label]
        units.append(dict(
            unit=u, interp=bool(best_qscore[u] > iqr_threshold and best_score[u] > iou_threshold),
            level=levels[u].item(),
            top=[dict(imgnum=i.item(), maxact=a.item()) for i, a in zip(topi[u], topa[u])],
            iou=best_score[u].item(), iou_iqr=best_qscore[u].item(),
            lc=bins["label_counts"][label].item(), cc=bins["category_activation_counts"][catnumber[cat], u].item(),
            ic=bins["intersection_counts"][label, u].item(),
            iou_labelnum=label, iou_label=name, iou_cat=cat,
        ))
    qualified = max_iqr.clone()
    qualified[iqrdata["max_iqr_quantile"] > 0.5] = 0
    best_iqr, best_iqr_label = qualified.max(0)
    for u, urec in enumerate(units):
        l = best_iqr_label[u].item()
        urec.update(dict(
            iqr=best_iqr[u].item(), iqr_label=labelnames[l][0], iqr_cat=labelnames[l][1],
            iqr_level=iqrdata["max_iqr_level"][l, u].item(), iqr_quantile=iqrdata["max_iqr_quantile"][l, u].item(),
            iqr_iou=iqrdata["max_iqr_iou"][l, u].item(),
        ))
    rankings = [
        dict(name="unit", score=list(range(len(units)))),
        dict(name="max iou", metric="iou", score=[-u["iou"] for u in units]),
        dict(name="max iqr", metric="iqr", score=[-u["iqr"] for u in units]),
    ]
    # One `<label>-iou` ranking per label that is some unit's best label (quadrant suffixes stripped),
    # most popular first; extra_labels are always included.
    counted = defaultdict(int)
    for u in units:
        counted[re.sub(r"-(?:t|b|l|r|tl|tr|bl|br)$", "", u["iou_label"])] += 1
    all_labels = [label for _, label in sorted((-v, k) for k, v in counted.items())]
    all_labels += [l for l in extra_labels if l not in all_labels]
    for label in all_labels:
        rankings.append(dict(name=f"{label}-iou", concept=label, metric="iou",
                             score=(-iou_scores[labelnumber[label], :]).tolist()))
    record = dict(layer=layer, units=units, rankings=rankings, interpretable=sum(u["interp"] for u in units))
    top = dict(netname="ProgressiveGenerator", default_ranking="label", quantile_threshold="iqr",
               iou_threshold=iou_threshold, iqr_threshold=iqr_threshold, layers=[record])
    if settings is not None:
        top["settings"] = settings
    Path(outdir).mkdir(parents=True, exist_ok=True)
    json.dump(top, open(Path(outdir) / "dissect.json", "w"))
    return top


def save_state(path, state):
    """Save a RunningQuantile/RunningTopK state_dict without pickle: ragged lists of arrays get one key each."""
    flat = {}
    for k, v in state.items():
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], np.ndarray):
            flat[f"{k}.__len__"] = np.array(len(v))
            flat.update({f"{k}.{i}": a for i, a in enumerate(v)})
        else:
            flat[k] = np.asarray(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)


def load_state(path):
    """Inverse of save_state; the result can be passed to RunningQuantile(state=...) / RunningTopK(state=...)."""
    with np.load(path, allow_pickle=False) as f:
        flat = {k: f[k] for k in f.files}
    state = {}
    for k, v in flat.items():
        if k.endswith(".__len__"):
            name = k[: -len(".__len__")]
            state[name] = [flat[f"{name}.{i}"] for i in range(int(v))]
        elif not (k.rsplit(".", 1)[-1].isdigit() and k.rsplit(".", 1)[0] + ".__len__" in flat):
            state[k] = v
    return state


def save_npz(path, **tensors):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tensors.items()})


def dissect(model, segmenter, outdir, size=1000, extra_labels=(), layer=LAYER, device="cuda", settings=None):
    """Run all stages and write quantiles.npz, topk.npz, iqr.npz, bincounts.npz and dissect.json to outdir."""
    outdir = Path(outdir)
    t0 = time.time()
    torch.manual_seed(0)  # RunningQuantile subsamples with torch's RNG
    labelnames, catnames, label_category = label_tables(segmenter)
    num_labels, num_categories = len(labelnames), len(catnames)

    set_a = collect_samples(model, segmenter, z_sample(size, seed=1), layer, desc="set A (seed 1)")
    grid = upsample_grid(set_a.feats.shape[2:], set_a.segs.shape[2:], device=device)
    quantiles, topk = collect_quantiles_and_topk(set_a, device=device)
    save_state(outdir / layer / "quantiles.npz", quantiles.state_dict())
    save_state(outdir / layer / "topk.npz", topk.state_dict())

    set_b = collect_samples(model, segmenter, z_sample(size, seed=2), layer, desc="set B (seed 2)")
    iqrdata = collect_iqr(set_b, grid, label_category, num_categories, num_labels, device=device)
    del set_b
    # a unit's level is the IQR-optimal level of the label it matches best
    best = iqrdata["max_iqr"].max(0)[1]
    levels = iqrdata["max_iqr_level"][best, torch.arange(best.shape[0])]
    save_npz(outdir / layer / "iqr.npz", **{k: v for k, v in iqrdata.items()}, levels=levels)

    bins = collect_bincounts(set_a, grid, levels, label_category, num_categories, num_labels, device=device)
    save_npz(outdir / layer / "bincounts.npz", **bins, levels=levels)
    report = build_report(outdir, topk, quantiles, levels, iqrdata, bins, labelnames, catnames,
                          settings=settings, extra_labels=extra_labels, layer=layer)
    print(f"dissection of {size} samples finished in {time.time() - t0:.0f}s")
    return report
