
import numpy as np
from src.data.synthetic import generate_synthetic_eeg
from src.preprocessing.pipeline import preprocess
from src.features.extractor import extract_features
from src.models.classical import make_rf
from src.training.train import train_sklearn, evaluate

def main():
    X, y = generate_synthetic_eeg()

    feats = []
    for i in range(len(X)):
        xi = preprocess(X[i], fs=128)
        feats.append(extract_features(xi))

    X = np.stack(feats)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))

    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    model = make_rf()
    model = train_sklearn(model, Xtr, ytr)
    print(evaluate(model, Xte, yte))

if __name__ == "__main__":
    main()
