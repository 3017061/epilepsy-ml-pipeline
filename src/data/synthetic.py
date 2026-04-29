
import numpy as np

def generate_synthetic_eeg(n=400, channels=1, length=256, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        x = rng.normal(0, 1, (channels, length))
        label = int(rng.random() < 0.3)
        if label:
            t0 = length // 2 - 16
            x[:, t0:t0+32] += rng.normal(4, 0.5, (channels, 32))
        X.append(x.astype("float32"))
        y.append(label)
    return np.stack(X), np.array(y)
