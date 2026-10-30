import numpy as np
import pytest
import torch

from ganpaint_pipeline.segmenter.prroi import prroi_pool_full


def test_constant_map_closed_form():
    # Hat functions centred on 0..3 integrated over [0, 4]: the left half of hat 0 lies outside.
    out = prroi_pool_full(torch.ones(1, 1, 4, 4, dtype=torch.float64), 1)
    assert out.item() == pytest.approx((3.5 / 4) ** 2)


def _brute_force(feat, bins, n=600):
    """Average of the zero-padded bilinear surface over each bin, by fine midpoint sampling."""
    c, h, w = feat.shape

    def axis_samples(n_in, k):
        edges = np.linspace(0, n_in, bins + 1)
        t = edges[k] + (np.arange(n) + 0.5) / n * (edges[k + 1] - edges[k])
        hat = np.maximum(0, 1 - np.abs(t[:, None] - np.arange(n_in)[None, :]))  # (n, n_in)
        return hat

    out = np.zeros((c, bins, bins))
    for ky in range(bins):
        hy = axis_samples(h, ky)
        for kx in range(bins):
            hx = axis_samples(w, kx)
            out[:, ky, kx] = np.einsum("yi,cij,xj->cyx", hy, feat, hx).mean(axis=(1, 2))
    return out


@pytest.mark.parametrize("hw,bins", [((8, 8), 1), ((8, 8), 3), ((8, 8), 6), ((7, 9), 2), ((16, 16), 6), ((5, 5), 6)])
def test_matches_numerical_integration(hw, bins):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, *hw, generator=g, dtype=torch.float64)
    ours = prroi_pool_full(x, bins)[0].numpy()
    ref = _brute_force(x[0].numpy(), bins)
    np.testing.assert_allclose(ours, ref, atol=2e-3)
