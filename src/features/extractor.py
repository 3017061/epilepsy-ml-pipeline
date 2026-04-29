
import numpy as np
from scipy.stats import skew, kurtosis

def time_features(x):
    mu = x.mean(axis=-1)
    sd = x.std(axis=-1)
    sk = skew(x, axis=-1)
    ku = kurtosis(x, axis=-1)
    return np.stack([mu, sd, sk, ku], axis=-1).ravel()

def extract_features(x, fs=None):
    return time_features(x)
