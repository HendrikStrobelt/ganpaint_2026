import base64
import io

import pytest
import torch
from PIL import Image

from ganpaint_server.intervention import apply_edit, compose, decode_mask_png, stroke_planes

SHAPE = (4, 8, 8)


def png_url(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_mask_alpha_channel_and_nonwhite():
    rgba = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    rgba.putpixel((3, 5), (255, 255, 255, 10))  # transparent-black elsewhere; one barely visible pixel
    m = decode_mask_png(png_url(rgba))
    assert m.shape == (32, 32) and m.sum() == 1 and m[5, 3] == 1
    rgb = Image.new("RGB", (32, 32), (255, 255, 255))
    rgb.putpixel((1, 2), (200, 0, 0))
    m = decode_mask_png(png_url(rgb))
    assert m.sum() == 1 and m[2, 1] == 1


def test_mask_rejects_oversized_and_garbage():
    with pytest.raises(ValueError):
        decode_mask_png(png_url(Image.new("RGBA", (600, 600))))
    with pytest.raises(Exception):
        decode_mask_png("data:image/png;base64,@@@")


def test_stroke_planes_maxpool_and_units():
    mask = torch.zeros(32, 32)
    mask[0, 0] = 1  # one canvas pixel lights the whole top-left 8x8-cell (max pooling over 4x4 blocks)
    alpha, value = stroke_planes([1, 3], {1: 5.0, 3: 7.0}, mask, SHAPE)
    assert alpha.shape == SHAPE and value.shape == SHAPE
    assert alpha[1, 0, 0] == 1 and alpha[1].sum() == 1 and alpha[0].sum() == 0 and alpha[2].sum() == 0
    assert value[1, 4, 4] == 5.0 and value[3, 4, 4] == 7.0 and value[0].sum() == 0


def test_remove_has_zero_value_but_full_alpha():
    mask = torch.ones(32, 32)
    alpha, value = stroke_planes([2], {2: 0.0}, mask, SHAPE)
    feats = torch.full((1,) + SHAPE, 3.0)
    out = apply_edit(feats, alpha[None], value[None])
    assert (out[0, 2] == 0).all() and (out[0, [0, 1, 3]] == 3).all()


def test_compose_alpha_over_matches_formula():
    half = torch.zeros(32, 32)
    half[:, :16] = 1
    a1, v1 = stroke_planes([0], {0: 10.0}, torch.ones(32, 32), SHAPE)  # draw everywhere
    a2, v2 = stroke_planes([0], {0: 0.0}, half, SHAPE)  # remove on the left half
    alpha, value = compose([(a1, v1), (a2, v2)])
    assert (alpha[0] == 1).all()  # covered by the first stroke everywhere
    assert (value[0, :, :4] == 0).all() and (value[0, :, 4:] == 10).all()  # later stroke wins where it paints


def test_compose_partial_alphas():
    a = torch.zeros(SHAPE)
    b = torch.zeros(SHAPE)
    a[0], b[0] = 0.5, 0.5
    va, vb = torch.zeros(SHAPE), torch.zeros(SHAPE)
    va[0], vb[0] = 8.0, 0.0
    alpha, value = compose([(a, va), (b, vb)])
    assert torch.allclose(alpha[0], torch.tensor(0.75))
    assert torch.allclose(value[0], torch.tensor(8.0 * 0.5 * 0.5 / 0.75))


def test_compose_empty_is_none():
    assert compose([]) is None
