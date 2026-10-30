"""ResNet50 (deep stem) encoder and UPerNet decoder from CSAILVision/unifiedparsing, inference only."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from .prroi import prroi_pool_full

POOL_SCALES = (1, 2, 3, 6)


class Encoder(nn.Module):
    """ResNet50 with a three-conv stem (stride 2 in conv1); returns the four stage outputs."""

    def __init__(self, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)

    def _make_layer(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, down)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        outs = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            outs.append(x)
        return outs


def _conv_bn_relu(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, 1, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UPerNet(nn.Module):
    """UPerNet decoder (PPM + FPN) with object, material and part heads. The scene head is not built."""

    def __init__(self, n_object, n_material, n_part, fc_dim=2048, fpn_dim=512, fpn_inplanes=(256, 512, 1024, 2048)):
        super().__init__()
        self.ppm_conv = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(fc_dim, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)) for _ in POOL_SCALES]
        )
        self.ppm_last_conv = _conv_bn_relu(fc_dim + len(POOL_SCALES) * 512, fpn_dim, 3)
        self.fpn_in = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, fpn_dim, 1, bias=False), nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)) for c in fpn_inplanes[:-1]]
        )
        self.fpn_out = nn.ModuleList([nn.Sequential(_conv_bn_relu(fpn_dim, fpn_dim, 3)) for _ in fpn_inplanes[:-1]])
        self.conv_fusion = _conv_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 3)

        def head(n):
            return nn.Sequential(_conv_bn_relu(fpn_dim, fpn_dim, 3), nn.Conv2d(fpn_dim, n, 1))

        self.object_head, self.part_head, self.material_head = head(n_object), head(n_part), head(n_material)

    def forward(self, conv_out, seg_size, part_slices):
        """part_slices: [(start, stop)] channel ranges of the part head to softmax separately."""
        conv5 = conv_out[-1]
        size5 = conv5.shape[2:]
        ppm = [conv5] + [
            conv(F.interpolate(prroi_pool_full(conv5, s), size5, mode="bilinear", align_corners=False))
            for s, conv in zip(POOL_SCALES, self.ppm_conv)
        ]
        f = self.ppm_last_conv(torch.cat(ppm, 1))
        fpn = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = self.fpn_in[i](conv_out[i])
            f = conv_x + F.interpolate(f, size=conv_x.shape[2:], mode="bilinear", align_corners=False)
            fpn.append(self.fpn_out[i](f))
        fpn.reverse()  # P2..P5

        material = self.material_head(fpn[0])
        size2 = fpn[0].shape[2:]
        fused = [fpn[0]] + [F.interpolate(p, size2, mode="bilinear", align_corners=False) for p in fpn[1:]]
        x = self.conv_fusion(torch.cat(fused, 1))
        resize = lambda t: F.interpolate(t, size=seg_size, mode="bilinear", align_corners=False)
        part = resize(self.part_head(x))
        return {
            "object": F.softmax(resize(self.object_head(x)), dim=1),
            "material": F.softmax(resize(material), dim=1),
            "part": [F.softmax(part[:, a:b], dim=1) for a, b in part_slices],
        }


class SegmentationModule(nn.Module):
    def __init__(self, encoder, decoder, labeldata):
        super().__init__()
        self.encoder, self.decoder, self.labeldata = encoder, decoder, labeldata
        # The part head concatenates the part lists of every object that has parts, ordered by object id.
        obj_id = {name: i for i, name in enumerate(labeldata["object"])}
        owners = sorted(labeldata["object_part"], key=obj_id.get)
        sizes = [len(labeldata["object_part"][o]) for o in owners]
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.part_owner_order = owners
        self.part_slices_all = [(s, s + n) for s, n in zip(starts, sizes)]

    def forward(self, img, seg_size, num_part_objects=3):
        return self.decoder(self.encoder(img), seg_size, self.part_slices_all[:num_part_objects])
