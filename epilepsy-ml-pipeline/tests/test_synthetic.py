from epilepsy_ml.data import generate_synthetic_eeg
from epilepsy_ml.preprocess import bandpass, standardize, welch_psd
from epilepsy_ml.features import time_domain
import numpy as np

def test_pipeline_shapes():
    X, y = generate_synthetic_eeg(n=16, channels=1, length=256)
    assert X.shape == (16,1,256)
    x = standardize(bandpass(X[0], fs=128.0))
    psd = welch_psd(x, fs=128.0)
    td = time_domain(x)
    assert psd.ndim == 1 and td.ndim == 1
