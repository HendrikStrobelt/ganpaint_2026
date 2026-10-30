"""Load Karras ProGAN TF1 pickles without TensorFlow and without executing pickled code.

The pickles contain three `tfutil.Network` instances (G, D, Gs). We only need
their `variables` (name -> numpy array). The unpickler below refuses every global
except the few needed for numpy arrays and a stub for `tfutil.Network`.
"""
import pickle
from pathlib import Path

import numpy as np


class NetworkStub:
    """Receives the pickled __dict__ of tfutil.Network via __setstate__."""

    def __setstate__(self, state):
        self.__dict__.update(state)


_ALLOWED = {
    ("tfutil", "Network"): NetworkStub,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy", "ndarray"): np.ndarray,
    ("collections", "OrderedDict"): __import__("collections").OrderedDict,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "slice"): slice,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_networks(path):
    """Returns the pickled tuple (G, D, Gs) as NetworkStub objects."""
    with open(Path(path), "rb") as f:
        return RestrictedUnpickler(f).load()
