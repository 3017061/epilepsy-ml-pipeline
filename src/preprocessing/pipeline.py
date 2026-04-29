
import numpy as np

def bandpass(x, fs):
    return x

def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def preprocess(x, fs):
    x = bandpass(x, fs)
    x = standardize(x)
    return x
