"""
Data loading utilities for real public biomedical and medical datasets.
All datasets are from publicly available sources with proper citations.

Datasets:
- Breast Cancer Wisconsin (UCI ML Repository)
- Wine Classification (UCI ML Repository)
- Heart Disease (UCI ML Repository)
- Diabetes (Pima Indians, UCI ML Repository)
- Iris (UCI ML Repository) - for testing
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris,
    load_diabetes, load_digits
)
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "iris": load_iris,
    "diabetes": load_diabetes,
    "digits": load_digits,
}


def load_public_dataset(name="breast_cancer", random_state=42):
    """
    Load a public biomedical dataset and return a pandas DataFrame plus metadata.
    
    All datasets are from publicly available, peer-reviewed sources:
    - Breast Cancer Wisconsin: University of Wisconsin Hospitals
    - Wine: UC Irvine ML Repository
    - Iris: UC Irvine ML Repository (classic classification)
    - Diabetes: Pima Indians Diabetes Database
    - Digits: MNIST handwritten digits (for testing ML pipelines)
    
    Args:
        name: Dataset name ('breast_cancer', 'wine', 'iris', 'diabetes', 'digits')
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with features and target, metadata dict)
        
    Citations:
        Breast Cancer: Wolberg, Street, Mangasarian (1995)
        Wine: Aeberhard, Coomans, De Vel (1992)
        Iris: Fisher (1936)
        Diabetes: Smith, Everhart, Dickson, Knowler, Johannes (1988)
        Digits: Alpaydin, Kaynak (1995)
    """
    if name not in DATASETS:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(DATASETS.keys())}\n"
            f"All datasets are real public biomedical data from UCI ML Repository."
        )

    raw = DATASETS[name]()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")
    df = pd.concat([X, y], axis=1)
    
    metadata = {
        "name": name,
        "target_names": list(raw.target_names),
        # Keep backwards-compatible key expected by some notebooks
        "class_names": list(raw.target_names),
        "feature_names": list(raw.feature_names),
        "n_samples": len(df),
        "n_features": X.shape[1],
        "n_classes": len(set(y)),
        "class_distribution": dict(y.value_counts()),
        "source": "UCI ML Repository",
        "description": raw.DESCR,
    }
    return df, metadata


def split_dataset(df, target_col="target", test_size=0.2, val_size=0.1, random_state=42):
    """
    Split a DataFrame into train, validation, and test sets.
    
    Args:
        df: Input DataFrame with features and target
        target_col: Name of target column
        test_size: Fraction for test set
        val_size: Fraction for validation set (from training data)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataset_info(name: str = None) -> dict:
    """
    Get information about available datasets.
    
    Args:
        name: Specific dataset name or None for all datasets
        
    Returns:
        Dictionary with dataset information including source and citations
    """
    dataset_info = {
        "breast_cancer": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "samples": 569,
            "features": 30,
            "classes": 2,
            "class_names": ["malignant", "benign"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
            "description": "Diagnostic features from breast cancer biopsies",
            "paper": "Wolberg, Street, Mangasarian (1995)",
        },
        "wine": {
            "name": "Wine Classification",
            "samples": 178,
            "features": 13,
            "classes": 3,
            "class_names": ["cultivar_1", "cultivar_2", "cultivar_3"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Wine",
            "description": "Chemical composition of wines from three cultivars",
            "paper": "Aeberhard, Coomans, De Vel (1992)",
        },
        "iris": {
            "name": "Iris Flower Dataset",
            "samples": 150,
            "features": 4,
            "classes": 3,
            "class_names": ["setosa", "versicolor", "virginica"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Iris",
            "description": "Measurements of iris flowers from three species",
            "paper": "Fisher (1936)",
        },
        "diabetes": {
            "name": "Pima Indians Diabetes",
            "samples": 442,
            "features": 10,
            "classes": 2,
            "class_names": ["no_diabetes", "diabetes"],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes",
            "description": "Diabetes diagnosis based on medical measurements",
            "paper": "Smith et al. (1988)",
        },
        "digits": {
            "name": "Optical Recognition of Handwritten Digits",
            "samples": 1797,
            "features": 64,
            "classes": 10,
            "class_names": [str(i) for i in range(10)],
            "source": "UCI ML Repository",
            "url": "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits",
            "description": "Handwritten digit recognition (8x8 pixel images)",
            "paper": "Alpaydin, Kaynak (1995)",
        },
    }
    
    if name and name in dataset_info:
        return dataset_info[name]
    elif name:
        return {"error": f"Unknown dataset: {name}"}
    
    return dataset_info


def load_epileptic_seizure(local_path: str = None, openml_lookup: bool = True):
    """
    Load the Epileptic Seizure Recognition dataset (UCI).

    Behavior:
      - If `local_path` is provided or `data/epileptic_seizure.csv` exists at
        the repository root, load that CSV with pandas and return (df, meta).
      - Otherwise, if `openml_lookup` is True and the `openml` package is
        available, attempt a best-effort fetch from OpenML (searching for
        dataset names containing "epileptic").

    Returns:
        (DataFrame, metadata dict)

    Notes:
      - The function prefers a local CSV file. If the dataset CSV layout
        uses a non-standard target column name, the loader will try to
        locate an appropriate target column and rename it to `target`.
      - If both local and OpenML fetch fail, a FileNotFoundError is raised
        with instructions to download the dataset manually.
    """
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "data" / "epileptic_seizure.csv"
    csv_path = Path(local_path) if local_path else default_csv

    # Try local CSV first
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Try to find a sensible target column
        target_col = None
        for c in ("target", "class", "y", "label"):
            if c in df.columns:
                target_col = c
                break

        if target_col is None:
            # If last column looks numeric, assume it's the target
            last_col = df.columns[-1]
            if pd.api.types.is_numeric_dtype(df[last_col]):
                df = df.rename(columns={last_col: "target"})
                target_col = "target"

        else:
            if target_col != "target":
                df = df.rename(columns={target_col: "target"})

        meta = {
            "name": "epileptic_seizure",
            "source": str(csv_path),
            "n_samples": len(df),
            "n_features": df.shape[1] - (1 if "target" in df.columns else 0),
            "target_col": "target" if "target" in df.columns else None,
        }
        return df, meta

    # Fallback: try OpenML (best-effort)
    if openml_lookup:
        try:
            import openml

            # Search for datasets with 'epileptic' in the name (best-effort)
            dlist = openml.datasets.list_datasets(output_format="dataframe")
            matches = dlist[dlist["name"].str.contains("epileptic", case=False, na=False)]
            if len(matches) == 0:
                raise RuntimeError("No OpenML dataset name contains 'epileptic'")

            did = int(matches.iloc[0]["did"])
            ds = openml.datasets.get_dataset(did)

            # Attempt to load as dataframe; allow ds.get_data to provide X,y
            default_target = getattr(ds, "default_target_attribute", None)
            X, y, *_ = ds.get_data(target=default_target)

            if y is None:
                # If y not provided, try using last column of X
                if isinstance(X, pd.DataFrame) and X.shape[1] > 0:
                    y = X.iloc[:, -1]
                    X = X.iloc[:, :-1]
                    y.name = "target"
                else:
                    raise RuntimeError("OpenML dataset had no target column")

            df = pd.concat([pd.DataFrame(X), pd.Series(y, name="target")], axis=1)
            meta = {
                "name": getattr(ds, "name", "epileptic_seizure"),
                "n_samples": df.shape[0],
                "n_features": df.shape[1] - 1,
                "source": "openml",
                "openml_id": did,
            }
            return df, meta

        except Exception as e:
            raise FileNotFoundError(
                "Could not find local CSV and OpenML fetch failed. "
                "Please download the UCI Epileptic Seizure Recognition CSV and "
                "save it to data/epileptic_seizure.csv. Original error: {}".format(e)
            )

    raise FileNotFoundError(
        "Could not find local CSV for Epileptic dataset and OpenML lookup is disabled. "
        f"Expected file at: {default_csv}"
    )


def load_real_dataset(name="breast_cancer", random_state=42, **kwargs):
    """Backwards-compatible alias for older notebooks that call `load_real_dataset()`.

    This simply calls `load_public_dataset` and ensures the returned metadata
    contains the `class_names` key used by the notebooks in this project.
    """
    df, meta = load_public_dataset(name=name, random_state=random_state)
    if 'class_names' not in meta and 'target_names' in meta:
        meta['class_names'] = meta['target_names']
    return df, meta
