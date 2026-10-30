"""Exact Precise RoI Pooling for a ROI covering a whole feature map, in plain PyTorch.

PrRoIPool (Jiang et al. 2018) averages the continuous, bilinearly interpolated feature map
f(x, y) = sum_ij F[i, j] * hat(x - i) * hat(y - j)  (F is zero outside the grid) over each
bin. The hat functions separate in x and y, so the integral over a bin is a product of two
closed-form 1-D weights: out = Wy @ F @ Wx^T / bin_area. This replaces the original CUDA
extension, which the UPerNet PPM only ever called with ROI = [0, 0, W, H].
"""
import torch


def _hat_integral(t):
    """Integral of hat(u) = max(0, 1 - |u|) from -inf to t."""
    t = t.clamp(-1.0, 1.0)
    return torch.where(t <= 0, (t + 1) ** 2 / 2, 0.5 + t - t * t / 2)


def _bin_weights(n_in, n_bins, device, dtype):
    edges = torch.linspace(0, n_in, n_bins + 1, device=device, dtype=torch.float64)
    i = torch.arange(n_in, device=device, dtype=torch.float64)
    w = _hat_integral(edges[1:, None] - i[None, :]) - _hat_integral(edges[:-1, None] - i[None, :])
    return (w / (n_in / n_bins)).to(dtype)  # (n_bins, n_in), already divided by bin length


def prroi_pool_full(x, bins):
    """x: (B, C, H, W) -> (B, C, bins, bins), pooling over the full-image ROI."""
    wy = _bin_weights(x.shape[2], bins, x.device, x.dtype)
    wx = _bin_weights(x.shape[3], bins, x.device, x.dtype)
    return torch.einsum("kh,bchw,lw->bckl", wy, x, wx)
