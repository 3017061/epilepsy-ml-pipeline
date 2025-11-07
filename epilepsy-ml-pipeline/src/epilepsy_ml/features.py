from __future__ import annotations
import numpy as np

def time_domain(x: np.ndarray) -> np.ndarray:
    """Simple time-domain stats per channel concatenated: mean, std, skew, kurtosis."""
    from scipy.stats import skew, kurtosis
    mu = x.mean(axis=-1)
    sd = x.std(axis=-1)
    sk = skew(x, axis=-1)
    ku = kurtosis(x, axis=-1)
    return np.stack([mu, sd, sk, ku], axis=-1).ravel().astype("float32")
