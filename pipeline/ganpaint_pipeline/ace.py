"""ACE (average causal effect) unit ranking: port of netdissect.aceoptimize for one generator layer.

For a class (tree, grass, door, ...) learn a per-unit weight a in [0, 1]^units such that
  * zeroing the units, weighted by a, at a feature-map cell where the class is present removes the class, and
  * writing the class's mean feature, weighted by a, into a cell where it is absent adds the class,
measured by the (differentiable) segmenter. The ranking score of a unit is -a.

Phases (cached in corpus.pt): 1. cells where the class is present (seed-10 search set);
2. cells where inserting the mean feature increases the class (seed-11 search set); 3. Adam on a.
Skipped relative to the original, as they do not influence the result: the 0.999-quantile phase
(used only by the `_h99` variant) and the diagnostic "discrete" evaluation experiments.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .zdataset import z_sample

SEG_DOWNSAMPLE = 2


class LayerEditor:
    """Forward hook implementing x -> x * (1 - a) + v * a on a layer's output (v = 0 when unset)."""

    def __init__(self, module):
        self.ablation = None
        self.replacement = None
        module.register_forward_hook(self._hook)

    def set(self, ablation=None, replacement=None):
        self.ablation, self.replacement = ablation, replacement

    def _hook(self, module, inputs, out):
        if self.ablation is None:
            return None
        v = 0.0 if self.replacement is None else self.replacement
        return out * (1 - self.ablation) + v * self.ablation


class Retainer:
    def __init__(self, module):
        self.value = None
        module.register_forward_hook(lambda m, i, o: setattr(self, "value", o))


def class_mask(segs, classnum):
    return (segs == classnum).max(1)[0]


def classnum_for(segmenter, classname):
    labels, _ = segmenter.get_label_and_category_names()
    return [i for i, (n, _) in enumerate(labels) if n == classname][0]


def batches(*tensors, size):
    for i in range(0, len(tensors[0]), size):
        yield [t[i : i + size] for t in tensors]


@torch.no_grad()
def compute_present_locations(model, retain, segmenter, classnum, sample, cfg, device):
    feature_shape = retain.shape
    weighted_feature_sum = torch.zeros(retain.units, device=device)
    scores = []
    for (zb,) in tqdm(list(batches(sample, size=cfg["inference_batch_size"])), desc="present locations"):
        images = model(zb.to(device))
        mask = class_mask(segmenter.segment_batch(images, downsample=SEG_DOWNSAMPLE), classnum)
        score = F.adaptive_avg_pool2d(mask.float(), feature_shape)
        scores.append(score.cpu())
        feat = retain.value
        weighted_feature_sum += (feat * score[:, None]).view(feat.shape[0], feat.shape[1], -1).sum(2).sum(0)
    presence = torch.cat(scores)
    num_locations = int(np.prod(feature_shape))
    at_image, loc_in_image = presence.view(cfg["search_size"], -1).max(1)
    best_images = torch.sort(-at_image)[1]
    n_train, n_eval = cfg["train_size"], cfg["eval_size"]
    all_idx = torch.sort(best_images[: n_train + n_eval])[0]
    train_idx, eval_idx = all_idx[:n_train], all_idx[-n_eval:]
    return dict(
        present_indices=train_idx,
        object_present_sample=sample[train_idx],
        object_present_location=loc_in_image[train_idx],
        object_location_popularity=torch.bincount(loc_in_image[train_idx], minlength=num_locations),
        weighted_mean_present_feature=(weighted_feature_sum.cpu() / (1e-20 + presence.view(-1).sum())),
        eval_present_sample=sample[eval_idx],
        eval_present_location=loc_in_image[eval_idx],
    )


@torch.no_grad()
def compute_candidate_locations(model, retain, editor, segmenter, classnum, sample, corpus, cfg, device):
    feature_shape = retain.shape
    num_locations = int(np.prod(feature_shape))
    weights = corpus["object_location_popularity"].double()
    weights += weights.mean() / 10.0
    weights = (weights / weights.sum()).numpy()
    replace_vec = corpus["weighted_mean_present_feature"][None, :, None, None].to(device)
    prng = np.random.RandomState(1)
    cand_scores, obj_scores = [], []
    for (zb,) in tqdm(list(batches(sample, size=cfg["inference_batch_size"])), desc="candidate locations"):
        zb = zb.to(device)
        batch_scores = torch.zeros((len(zb),) + feature_shape, device=device)
        flat = batch_scores.view(len(zb), -1)
        editor.set(None)
        mask = class_mask(segmenter.segment_batch(model(zb), downsample=SEG_DOWNSAMPLE), classnum)
        object_score = F.adaptive_avg_pool2d(mask.float(), feature_shape)
        baseline = mask.float().view(len(zb), -1).sum(1)
        edit_mask = torch.zeros((1, 1) + feature_shape, device=device)
        # one random set of 5 probed cells per batch, weighted by where the class is usually present
        for loc in prng.choice(np.arange(num_locations), replace=False, p=weights, size=5):
            edit_mask.zero_()
            edit_mask.view(-1)[loc] = 1
            editor.set(edit_mask, replace_vec)
            mask = class_mask(segmenter.segment_batch(model(zb), downsample=SEG_DOWNSAMPLE), classnum)
            flat[:, loc] = mask.float().view(len(zb), -1).sum(1) - baseline
        editor.set(None)
        cand_scores.append(batch_scores.cpu())
        obj_scores.append(object_score.cpu())
    obj_scores, cand_scores = torch.cat(obj_scores), torch.cat(cand_scores)
    cand_scores = cand_scores * (obj_scores == 0).float()  # only cells where the class is absent
    at_image, loc_in_image = cand_scores.view(cfg["search_size"], -1).max(1)
    best_images = torch.sort(-at_image)[1]
    n_train, n_eval = cfg["train_size"], cfg["eval_size"]
    all_idx = torch.sort(best_images[: n_train + n_eval])[0]
    train_idx, eval_idx = all_idx[:n_train], all_idx[-n_eval:]
    return dict(
        candidate_indices=train_idx,
        candidate_sample=sample[train_idx],
        candidate_location=loc_in_image[train_idx],
        candidate_score=at_image[train_idx],
        eval_candidate_sample=sample[eval_idx],
        eval_candidate_location=loc_in_image[eval_idx],
    )


def build_corpus(model, retain, editor, segmenter, classnum, cfg, cache_path, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    corpus = compute_present_locations(model, retain, segmenter, classnum, z_sample(cfg["search_size"], seed=10), cfg, device)
    corpus.update(compute_candidate_locations(model, retain, editor, segmenter, classnum, z_sample(cfg["search_size"], seed=11), corpus, cfg, device))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, cache_path)
    return corpus


def ace_loss(segmenter, classnum, model, retain, editor, high_replacement, ablation, pbatch, ploc, cbatch, cloc,
             add_at="cloc", run_backward=False, device="cuda"):
    """erase loss (class left after ablating at present cells) + add loss (-class gained by inserting at candidates).

    add_at='ploc' reproduces the repository code, which measures the insertion effect at the *present*-sample
    cell indices; add_at='cloc' measures it at the candidate cells the insertion was made at.
    """
    feature_shape = retain.shape
    p_mask = torch.zeros((len(pbatch), 1) + feature_shape)
    p_mask.view(len(pbatch), -1).scatter_(1, ploc[:, None], 1)
    editor.set(ablation * p_mask.to(device), None)
    erase_effect, _ = segmenter.predict_single_class(model(pbatch.to(device)), classnum, downsample=SEG_DOWNSAMPLE)
    erase_down = F.adaptive_avg_pool2d(erase_effect[:, None], feature_shape)[:, 0]
    erase_loss = erase_down.view(len(erase_down), -1)[torch.arange(len(erase_down)), ploc].sum()
    if run_backward:
        erase_loss.backward()

    c_mask = torch.zeros((len(cbatch), 1) + feature_shape)
    c_mask.view(len(cbatch), -1).scatter_(1, cloc[:, None], 1)
    editor.set(ablation * c_mask.to(device), high_replacement)
    add_effect, _ = segmenter.predict_single_class(model(cbatch.to(device)), classnum, downsample=SEG_DOWNSAMPLE)
    add_down = F.adaptive_avg_pool2d((-add_effect)[:, None], feature_shape)[:, 0]
    measure = {"cloc": cloc, "ploc": ploc}[add_at]
    add_loss = add_down.view(len(add_down), -1)[torch.arange(len(add_down)), measure].sum()
    if run_backward:
        add_loss.backward()
    editor.set(None)
    return erase_loss + add_loss


def initial_ablation(dissect_json, layer, classname):
    """Start from the normalised IoU scores of the class (the `<class>-iou` ranking)."""
    top = json.load(open(dissect_json))
    lrec = [l for l in top["layers"] if l["layer"] == layer][0]
    scores = -torch.tensor([r for r in lrec["rankings"] if r["name"] == f"{classname}-iou"][0]["score"])
    return scores / scores.max()


def train_ablation(model, retain, editor, segmenter, classnum, corpus, init, cfg, snapdir, device):
    snapdir = Path(snapdir)
    snapdir.mkdir(parents=True, exist_ok=True)
    high_replacement = corpus["weighted_mean_present_feature"][None, :, None, None].to(device)
    for p in model.parameters():
        p.requires_grad = False
    ablation = torch.zeros(high_replacement.shape, device=device)
    ablation.view(-1)[...] = init
    ablation.requires_grad = True
    optimizer = torch.optim.Adam([ablation], lr=0.01)
    loss_kw = dict(add_at=cfg["add_at"], device=device)

    def evaluate(epoch):
        with torch.no_grad():
            total = 0
            for pb, pl, cb, cl in batches(corpus["eval_present_sample"], corpus["eval_present_location"],
                                          corpus["eval_candidate_sample"], corpus["eval_candidate_location"],
                                          size=cfg["inference_batch_size"]):
                total = total + ace_loss(segmenter, classnum, model, retain, editor, high_replacement, ablation,
                                         pb, pl, cb, cl, **loss_kw)
            avg = (total / cfg["eval_size"]).item()
        print(f"epoch {epoch}: eval loss {avg:.4f}  reg {(cfg['l2_lambda'] * ablation.pow(2).sum()).item():.4f}", flush=True)
        return avg

    history = [evaluate(-1)]
    update_size = cfg["train_update_freq"] * cfg["train_batch_size"]
    for epoch in range(cfg["train_epochs"]):
        order = torch.randperm(len(corpus["object_present_sample"]))
        cand_order = torch.randperm(len(corpus["candidate_sample"]))
        pairs = (corpus["object_present_sample"][order], corpus["object_present_location"][order],
                 corpus["candidate_sample"][cand_order], corpus["candidate_location"][cand_order])
        train_loss = 0
        for batch_num, (pb, pl, cb, cl) in enumerate(tqdm(list(batches(*pairs, size=cfg["train_batch_size"])), desc=f"ACE epoch {epoch}")):
            if batch_num % cfg["train_update_freq"] == 0:
                optimizer.zero_grad()
            loss = ace_loss(segmenter, classnum, model, retain, editor, high_replacement, ablation,
                            pb, pl, cb, cl, run_backward=True, **loss_kw)
            train_loss = train_loss + loss.detach()
            if (batch_num + 1) % cfg["train_update_freq"] == 0:
                (cfg["l2_lambda"] * update_size * ablation.pow(2).sum()).backward()  # sparsity
                optimizer.step()
                with torch.no_grad():
                    ablation.clamp_(0, 1)
                train_loss = 0
        history.append(evaluate(epoch))
        np.save(snapdir / f"epoch-{epoch}.npy", ablation.detach().cpu().numpy())
    json.dump(history, open(snapdir / "eval_loss.json", "w"))
    return ablation.view(-1).detach().cpu().numpy()


DEFAULTS = dict(search_size=10000, train_size=1000, eval_size=200, inference_batch_size=10, train_batch_size=2,
                train_update_freq=10, train_epochs=10, l2_lambda=0.005, add_at="cloc")


def run_ace(model, segmenter, dissect_dir, classname, layer="layer4", variant="cloc", device="cuda", corpus_only=False, **overrides):
    """Learn the ACE ablation for classname; returns scores (a per unit) and saves them under <dissect_dir>/<layer>/ace/."""
    cfg = {**DEFAULTS, **overrides}
    cfg["add_at"] = cfg.get("add_at", variant)
    dissect_dir = Path(dissect_dir)
    cdir = dissect_dir / layer / "ace" / classname
    retain = Retainer(getattr(model, layer))
    editor = LayerEditor(getattr(model, layer))
    # feature map shape/units from a dummy forward
    with torch.no_grad():
        model(torch.zeros(1, 512, device=device))
    retain.shape, retain.units = tuple(retain.value.shape[2:]), retain.value.shape[1]
    classnum = classnum_for(segmenter, classname)
    torch.manual_seed(0)
    t0 = time.time()
    corpus = build_corpus(model, retain, editor, segmenter, classnum, cfg, cdir / "corpus.pt", device)
    if corpus_only:
        print(f"ACE {classname}: corpus built in {time.time() - t0:.0f}s")
        return None
    init = initial_ablation(dissect_dir / "dissect.json", layer, classname)
    scores = train_ablation(model, retain, editor, segmenter, classnum, corpus, init, cfg, cdir / variant / "snapshots", device)
    np.save(cdir / variant / "ablation.npy", scores)
    print(f"ACE {classname} ({variant}) finished in {time.time() - t0:.0f}s")
    return scores


def add_ace_ranking_to_dissection(dissect_dir, layer, classname, scores):
    """Insert `<class>-ace` (score = -a) into dissect.json at rankings[2:2], as the original does (with a .bak)."""
    path = Path(dissect_dir) / "dissect.json"
    bak = path.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy(path, bak)
    top = json.load(open(path))
    lrec = [l for l in top["layers"] if l["layer"] == layer][0]
    name = f"{classname}-ace"
    lrec["rankings"] = [r for r in lrec["rankings"] if r["name"] != name]
    lrec["rankings"][2:2] = [dict(name=name, score=(-np.asarray(scores)).flatten().tolist(), metric="ace")]
    json.dump(top, open(path, "w"))
