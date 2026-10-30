"""Rebuild the unified-parsing `labels.json` (missing from the original download host).

Layout, as consumed by netdissect's segmenter:
  object / material / scene / part: list of names indexed by label id (id 0 is '-')
  object_part: {object name: [part names]} (each list starts with '-'), ordered by object id
The CSVs come from CSAILVision/unifiedparsing meta_file/joint_dataset.
"""
import csv
import json
from pathlib import Path


def _names(path):
    rows = list(csv.DictReader(open(path)))
    assert [int(r["label"]) for r in rows] == list(range(len(rows))), f"{path}: labels not contiguous"
    return [r["name"] for r in rows]


def build_labels(meta_dir):
    meta = Path(meta_dir)
    labels = {k: _names(meta / f"{k}.csv") for k in ("object", "material", "scene", "part")}
    hier = sorted(csv.DictReader(open(meta / "object_part_hierarchy.csv")), key=lambda r: int(r["object_label"]))
    labels["object_part"] = {}
    for r in hier:
        names = r["part_names"].split(";")
        ids = [int(i) for i in r["part_labels"].split(";")]
        assert [labels["part"][i] for i in ids] == names, f"part names disagree for {r['object_name']}"
        assert labels["object"][int(r["object_label"])] == r["object_name"]
        labels["object_part"][r["object_name"]] = names
    return labels


def check_against_decoder(labels, decoder_sd):
    """Fail unless label counts equal the decoder's output channels."""
    want = {
        "object": decoder_sd["object_head.1.weight"].shape[0],
        "material": decoder_sd["material_head.1.weight"].shape[0],
        "scene": decoder_sd["scene_head.2.weight"].shape[0],
        "part": decoder_sd["part_head.1.weight"].shape[0],
    }
    got = {k: len(labels[k]) for k in ("object", "material", "scene")}
    got["part"] = sum(len(v) for v in labels["object_part"].values())
    assert got == want, f"label counts {got} != decoder heads {want}"
    return got


def write_labels(meta_dir, decoder_path, out_path):
    import torch

    labels = build_labels(meta_dir)
    counts = check_against_decoder(labels, torch.load(decoder_path, map_location="cpu", weights_only=True))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(labels, open(out_path, "w"), indent=1)
    return counts
