"""Deterministic latent samples, identical to gandissect's zdataset (sample id = row index)."""
import numpy as np
import torch


def z_sample(size, seed=1, dim=512):
    return torch.from_numpy(np.random.RandomState(seed).standard_normal(size * dim).reshape(size, dim).astype("float32"))
