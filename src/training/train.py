
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def train_sklearn(model, X, y):
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred)
    }
