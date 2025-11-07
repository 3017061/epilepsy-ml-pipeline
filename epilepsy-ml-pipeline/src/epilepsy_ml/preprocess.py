from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.signal import butter, filtfilt, welch

def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter over last axis."""
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return (x - m) / (s + eps)

def welch_psd(x: np.ndarray, fs: float) -> np.ndarray:
    """Compute Welch PSD per-channel and concatenate across channels."""
    # x: (C, T)
    feats = []
    for c in range(x.shape[0]):
        f, p = welch(x[c], fs=fs, nperseg=min(128, x.shape[1]))
        feats.append(p.astype("float32"))
    return np.concatenate(feats, axis=0)  # shape: (C*F,)
