"""Port of netdissect.segmenter.UnifiedParsingSegmenter (unified perceptual parsing, ResNet50-UPerNet).

For dissection we only consume object and material labels, plus parts of sky/building/person,
exactly like the original. Class numbering: 0 '-', objects, [quad-divided objects], materials, parts.
"""
import json
from pathlib import Path

import numpy as np
import skimage.morphology
import torch

from .upernet import Encoder, SegmentationModule, UPerNet

# The original image preprocessing: RGB [-1, 1] -> BGR 0..255 minus the Caffe mean.
BGR_MEAN = (102.9801, 115.9465, 122.7717)
PART_OWNERS = ["sky", "building", "person"]
SYNC_BN_EXTRAS = ("_running_iter", "_tmp_running_mean", "_tmp_running_var")


def load_upp_model(model_dir, epoch=40, device="cuda"):
    """Load encoder_epoch_N.pth / decoder_epoch_N.pth / labels.json from model_dir (strict)."""
    model_dir = Path(model_dir)
    labels = json.load(open(model_dir / "labels.json"))
    n_part = sum(len(v) for v in labels["object_part"].values())
    encoder = Encoder()
    decoder = UPerNet(len(labels["object"]), len(labels["material"]), n_part)

    def load(module, name, drop=()):
        sd = torch.load(model_dir / f"{name}_epoch_{epoch}.pth", map_location="cpu", weights_only=True)
        # SynchronizedBatchNorm bookkeeping buffers and the unused scene head are not part of our modules.
        sd = {k: v for k, v in sd.items() if not k.endswith(SYNC_BN_EXTRAS) and not k.startswith(drop)}
        module.load_state_dict(sd, strict=True)

    load(encoder, "encoder")
    load(decoder, "decoder", drop=("scene_head",))
    return SegmentationModule(encoder, decoder, labels).to(device).eval()


class UnifiedParsingSegmenter:
    def __init__(self, model_dir, segsizes=None, segdiv="undivided", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segmodel = segmodel = load_upp_model(model_dir, device=self.device)
        self.segsizes = segsizes or [256]
        self.segdiv = segdiv
        mult = 5 if segdiv == "quad" else 1
        self.divmult = mult
        ld = segmodel.labeldata
        first_partnumber = (len(ld["object"]) - 1) * mult + 1 + (len(ld["material"]) - 1)
        objectnumbers = {k: v for v, k in enumerate(ld["object"])}
        # The raw part predictions are indexed by position in the (sorted) object-with-part list;
        # raw_seg_prediction below relies on sky, building, person being its first three entries.
        assert segmodel.part_owner_order[:3] == PART_OWNERS, segmodel.part_owner_order[:3]
        # Parts that are also objects (door, cloud, dome, ...) are counted as objects and share their index.
        partnumbers, partnames, self.part_index = {}, [], []
        for owner in PART_OWNERS:
            numbers = []
            for part in ld["object_part"][owner]:
                if part in objectnumbers:
                    numbers.append(objectnumbers[part])
                else:
                    if part not in partnumbers:
                        partnumbers[part] = len(partnames) + first_partnumber
                        partnames.append(part)
                    numbers.append(partnumbers[part])
            self.part_index.append(torch.tensor(numbers))
        self.objects_with_parts = [objectnumbers[o] for o in PART_OWNERS]
        self.part_names = partnames
        self.num_object_classes = len(ld["object"]) - 1
        self.num_classes = 1 + self.num_object_classes * mult + (len(ld["material"]) - 1) + len(partnames)

    def get_label_and_category_names(self):
        ld = self.segmodel.labeldata
        suffixes = ["t", "l", "b", "r"] if self.segdiv == "quad" else []
        divided = [(f"{label}-{s}", "part") for s in suffixes for label in ld["object"][1:]]
        labelcats = (
            [(label, "object") for label in ld["object"]]
            + divided
            + [(label, "material") for label in ld["material"][1:]]
            + [(label, "part") for label in self.part_names]
        )
        return labelcats, ["object", "part", "material"]

    def raw_seg_prediction(self, tensor_images, downsample=1):
        """Softmax predictions for object, material and the three part groups at (y, x) / downsample.

        Differentiable w.r.t. tensor_images (ACE backpropagates through it); wrap in no_grad otherwise.
        """
        y, x = tensor_images.shape[2:]
        img = (tensor_images + 1) / 2 * 255
        img = torch.flip(img, (1,))  # RGB -> BGR
        img = img - torch.tensor(BGR_MEAN, dtype=img.dtype, device=img.device)[None, :, None, None]
        seg_shape = (y // downsample, x // downsample)
        ld = self.segmodel.labeldata
        b, dev = len(img), img.device
        pred = {c: torch.zeros(b, len(ld[c]), *seg_shape, device=dev) for c in ("object", "material")}
        part_pred = [torch.zeros(b, len(p), *seg_shape, device=dev) for p in self.part_index]
        for s in self.segsizes:
            resized = img if (s, s) == tuple(img.shape[2:]) else torch.nn.AdaptiveAvgPool2d((s, s))(img)
            r = self.segmodel(resized, seg_size=seg_shape)
            for k in pred:
                pred[k] += r[k]
            for k in range(len(part_pred)):
                part_pred[k] += r["part"][k]
        return pred, part_pred

    @torch.no_grad()
    def segment_batch(self, tensor_images, downsample=1):
        """(n, 3 or 5, y, x) long tensor: object, material, part [, two quad-divided object channels]."""
        pred, part_pred = self.raw_seg_prediction(tensor_images, downsample)
        piece_channels = 2 if self.segdiv == "quad" else 0
        y, x = tensor_images.shape[2:]
        segs = torch.zeros(len(tensor_images), 3 + piece_channels, y // downsample, x // downsample, dtype=torch.long, device=tensor_images.device)
        segs[:, 0] = pred["object"].argmax(1)
        segs[:, 1] = pred["material"].argmax(1)
        maskout = segs[:, 1] == 0
        segs[:, 1] += (len(self.segmodel.labeldata["object"]) - 1) * self.divmult
        segs[:, 1][maskout] = 0
        for i, object_index in enumerate(self.objects_with_parts):
            trans = self.part_index[i].to(segs.device)
            seg = trans[part_pred[i].argmax(1)]
            mask = segs[:, 0] == object_index  # only trust parts where the owner object is predicted
            segs[:, 2][mask] = seg[mask]
        if self.segdiv == "quad":
            segs = self.expand_segment_quad(segs)
        return segs

    def predict_single_class(self, tensor_images, classnum, downsample=1):
        """(score map in [0, 1], boolean best-label mask) for one non-divided class.

        The score map is differentiable w.r.t. tensor_images; the mask is not.
        """
        ld = self.segmodel.labeldata
        pred, part_pred = self.raw_seg_prediction(tensor_images, downsample)
        material_offset = (len(ld["object"]) - 1) * self.divmult
        if material_offset < classnum < material_offset + len(ld["material"]):
            m = classnum - material_offset
            return pred["material"][:, m], pred["material"].argmax(1) == m
        result, mask = 0, None
        if classnum < len(ld["object"]):
            result = pred["object"][:, classnum]
            mask = pred["object"].argmax(1) == classnum
        # Objects like 'door' are also parts of other objects: add the part prediction.
        for i, object_index in enumerate(self.objects_with_parts):
            local = (self.part_index[i] == classnum).nonzero()
            if len(local) == 0:
                continue
            local = local.item()
            mask2 = (pred["object"].argmax(1) == object_index) & (part_pred[i].argmax(1) == local)
            mask = mask2 if mask is None else mask | mask2
            result = result + part_pred[i][:, local]
        assert not isinstance(result, int), f"unrecognized class {classnum}"
        return result, mask

    def expand_segment_quad(self, segs):
        segs[:, 3:] = segs[:, 0:1]  # start by copying the object channel
        n = self.num_object_classes
        for i, mask in component_masks(segs[:, 0:1]):
            top, bottom = mask.any(dim=1).nonzero()[[0, -1], 0]
            left, right = mask.any(dim=0).nonzero()[[0, -1], 0]
            vmid, hmid = (top + bottom + 1) // 2, (left + right + 1) // 2
            q = mask[None].repeat(4, 1, 1)
            q[0, vmid:, :] = 0  # top
            q[1, :, hmid:] = 0  # right
            q[2, :vmid, :] = 0  # bottom
            q[3, :, :hmid] = 0  # left
            q = q.long()
            segs[i, 3] += q[0] * n + q[2] * (3 * n)
            segs[i, 4] += q[1] * (2 * n) + q[3] * (4 * n)
        # components too small to subdivide fall back to 'no prediction'
        small = segs[:, 3:] <= n
        segs[:, 3:][small] = 0
        return segs


def component_masks(segmentation_batch):
    """Yield (batch index, boolean mask) per connected component (8-connected, on CPU).

    Faithful to the original: `range(1, num)` skips the highest-numbered component of each image.
    """
    npbatch = segmentation_batch.cpu().numpy()
    for i in range(segmentation_batch.shape[0]):
        labeled, num = skimage.morphology.label(npbatch[i][0], return_num=True)
        labeled = torch.from_numpy(labeled).to(segmentation_batch.device)
        for label in range(1, num):
            yield i, labeled == label
