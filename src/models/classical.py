"""
Classical machine learning models for biomedical classification.
Includes logistic regression, decision trees, random forests, SVM, and ensemble methods.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def make_classical_model(name="random_forest", random_state=42):
    """
    Create a classical machine learning model by name.
    
    Args:
        name: Model name
        random_state: Random seed for reproducibility
        
    Returns:
        Sklearn model instance
    """
    name = name.lower()
    
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
        ),
        "logistic_regression_saga": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver='saga',
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "svm": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=random_state),
        "svm_linear": SVC(kernel='linear', C=1.0, probability=True, random_state=random_state),
        "svm_poly": SVC(kernel='poly', degree=3, C=1.0, probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=-1),
        "knn_weighted": KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        "naive_bayes": GaussianNB(),
        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=random_state,
        ),
    }
    
    if name not in models:
        raise ValueError(
            f"Unsupported classical model: {name}. "
            f"Choose from {list(models.keys())}"
        )
    
    return models[name]


def make_ensemble_model(random_state=42):
    """
    Create an ensemble model combining multiple classifiers.
    
    Args:
        random_state: Random seed
        
    Returns:
        Voting classifier ensemble
    """
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("svm", SVC(kernel='rbf', probability=True, random_state=random_state)),
        ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
    ]
    
    return VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1,
    )


def make_stacking_model(random_state=42):
    """
    Create a stacking classifier with base learners and meta-learner.
    
    Args:
        random_state: Random seed
        
    Returns:
        Stacking classifier
    """
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=50, random_state=random_state)),
        ("gb", GradientBoostingClassifier(n_estimators=50, random_state=random_state)),
        ("svm", SVC(kernel='rbf', probability=True, random_state=random_state)),
    ]
    
    meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
    
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
    )


def make_clustering_model(n_clusters=3, random_state=42):
    """
    Create a K-means clustering model.
    
    Args:
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        KMeans model
    """
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)


def get_model_list():
    """Get list of available classical models."""
    return [
        "logistic_regression",
        "logistic_regression_saga",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svm",
        "svm_linear",
        "svm_poly",
        "knn",
        "knn_weighted",
        "naive_bayes",
        "adaboost",
    ]


def get_model_info(name):
    """Get information about a specific model."""
    info = {
        "logistic_regression": "Linear model using logistic function",
        "decision_tree": "Tree-based model with interpretable decision boundaries",
        "random_forest": "Ensemble of decision trees",
        "gradient_boosting": "Boosting ensemble method",
        "svm": "Support Vector Machine with RBF kernel",
        "svm_linear": "Support Vector Machine with linear kernel",
        "svm_poly": "Support Vector Machine with polynomial kernel",
        "knn": "K-Nearest Neighbors with uniform weights",
        "knn_weighted": "K-Nearest Neighbors with distance-based weights",
        "naive_bayes": "Probabilistic classifier based on Bayes theorem",
        "adaboost": "Adaptive boosting ensemble",
    }
    
    return info.get(name, "Unknown model")
