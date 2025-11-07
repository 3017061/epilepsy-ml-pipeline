from __future__ import annotations
import numpy as np
import typer
from pathlib import Path
from typing import Optional

from .data import generate_synthetic_eeg, load_from_npz
from .preprocess import bandpass, standardize, welch_psd
from .features import time_domain
from .classic import make_svm, make_rf, train_model, evaluate_model
from .deep import CNN1D, LSTM1D, CNNLSTM, train_torch, TrainConfig

app = typer.Typer(help="End-to-end seizure prediction baselines")

@app.command()
def demo_classic(n: int = 400, fs: float = 128.0):
    """Run SVM/RF on synthetic windows with Welch PSD + time features."""
    X, y = generate_synthetic_eeg(n=n, channels=1, length=256)
    Xf = []
    for i in range(len(X)):
        xi = bandpass(X[i], fs=fs)
        xi = standardize(xi)
        feat = np.concatenate([welch_psd(xi, fs=fs), time_domain(xi)], axis=0)
        Xf.append(feat)
    Xf = np.stack(Xf)

    # split
    idx = np.arange(n)
    np.random.seed(0); np.random.shuffle(idx)
    tr, te = idx[:int(0.8*n)], idx[int(0.8*n):]
    Xtr, ytr, Xte, yte = Xf[tr], y[tr], Xf[te], y[te]

    for name, maker in [("SVM", make_svm), ("RF", make_rf)]:
        model = train_model(maker(), Xtr, ytr)
        metrics = evaluate_model(model, Xte, yte)
        typer.echo(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k,v in metrics.items()))

@app.command()
def demo_deep(n: int = 400, fs: float = 128.0, model: str = "cnn"):
    """Run CNN/LSTM/CNN+LSTM on synthetic windows."""
    X, y = generate_synthetic_eeg(n=n, channels=1, length=256)  # (N, C, T)
    # basic preprocessing
    for i in range(len(X)):
        X[i] = standardize(bandpass(X[i], fs=fs))

    if model == "cnn":
        net = CNN1D(in_ch=X.shape[1])
    elif model == "lstm":
        net = LSTM1D(in_ch=X.shape[1])
    else:
        net = CNNLSTM(in_ch=X.shape[1])

    cfg = TrainConfig(epochs=2, batch_size=64, lr=1e-3, device="cpu")
    net = train_torch(net, X, y, cfg)
    typer.echo("Training complete (demo).")

if __name__ == "__main__":
    app()
