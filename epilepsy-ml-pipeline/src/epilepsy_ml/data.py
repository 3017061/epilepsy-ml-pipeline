from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class EEGWindow:
    """A small EEG clip and its label (0 = non-seizure, 1 = seizure)."""
    x: np.ndarray  # shape: (C, T)
    y: int

def generate_synthetic_eeg(n: int = 100, channels: int = 1, length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic EEG windows with a seizure-like burst pattern for unit tests/demos."""
    rng = np.random.default_rng(42)
    X = []
    y = []
    for i in range(n):
        base = rng.normal(0, 1, size=(channels, length))
        label = int(rng.random() < 0.3)  # 30% seizure
        if label:
            # inject a burst in the middle
            t0 = length // 2 - 16
            base[:, t0:t0+32] += rng.normal(4.0, 0.5, size=(channels, 32))
        X.append(base.astype("float32"))
        y.append(label)
    return np.stack(X), np.array(y, dtype=int)

# NOTE: Real datasets (TUH, CHB-MIT) can be loaded from EDF/CSV using libraries like mne, pyedflib, wfdb.
# To keep dependencies lightweight here, we provide a hook to plug in external loaders.
def load_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed windows saved as arrays: {'X': (N,C,T), 'y': (N,)}"""
    d = np.load(path, allow_pickle=False)
    return d["X"], d["y"]
