from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_svm() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale"))])

def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)

def train_model(clf, X: np.ndarray, y: np.ndarray):
    clf.fit(X, y)
    return clf

def evaluate_model(clf, X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    proba = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X)
    pred = (proba >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, proba))
    }
