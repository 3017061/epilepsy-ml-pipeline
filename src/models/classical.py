
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def make_svm():
    return SVC(kernel="rbf", probability=True)

def make_rf():
    return RandomForestClassifier(n_estimators=100)
