"""
Advanced preprocessing pipelines for biomedical and EEG data.
Includes filtering, normalization, feature selection, and dimensionality reduction.
"""

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import signal
import numpy as np


def build_preprocessing_pipeline(k_best=12, scale_method="standard", add_poly=True):
    """
    Create a reusable preprocessing pipeline for numeric tabular data.
    
    Args:
        k_best: Number of features to select
        scale_method: 'standard' or 'robust' scaling
        add_poly: Whether to add polynomial features
        
    Returns:
        sklearn Pipeline object
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    
    # Add scaling
    if scale_method == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        steps.append(("scaler", StandardScaler()))
    
    # Optional polynomial features
    if add_poly:
        steps.append(
            ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False))
        )
    
    # Feature selection
    steps.append(("selector", SelectKBest(score_func=f_classif, k=k_best)))
    
    return Pipeline(steps)


def build_pca_pipeline(n_components=20, scale_method="standard"):
    """
    Create a preprocessing pipeline with PCA for dimensionality reduction.
    
    Args:
        n_components: Number of PCA components
        scale_method: 'standard' or 'robust' scaling
        
    Returns:
        sklearn Pipeline object
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    
    if scale_method == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        steps.append(("scaler", StandardScaler()))
    
    steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    
    return Pipeline(steps)


def build_eeg_preprocessing_pipeline(
    k_best=30,
    apply_filter=True,
    filter_type="bandpass",
    lowcut=1.0,
    highcut=50.0,
    order=4,
):
    """
    Create an EEG-specific preprocessing pipeline.
    
    Args:
        k_best: Number of features to select
        apply_filter: Whether to apply frequency filtering
        filter_type: 'bandpass', 'highpass', or 'lowpass'
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Function that applies EEG-specific preprocessing
    """
    from src.preprocessing.eeg_filters import apply_eeg_filter
    
    def preprocess(X, y=None):
        X_copy = X.copy()
        
        # Apply frequency filtering if needed
        if apply_filter:
            X_copy = X_copy.apply(
                lambda row: apply_eeg_filter(
                    row.values,
                    fs=256,
                    filter_type=filter_type,
                    lowcut=lowcut,
                    highcut=highcut,
                    order=order,
                ),
                axis=1,
                result_type="expand",
            )
        
        return X_copy
    
    return preprocess


def adaptive_k_best(X, y=None, percentile=0.8, min_k=5):
    """
    Adaptively select number of features based on data shape.
    
    Args:
        X: Feature matrix
        y: Target labels (for feature scoring)
        percentile: Fraction of features to keep
        min_k: Minimum number of features
        
    Returns:
        Number of features to select
    """
    n_features = X.shape[1]
    k = max(min_k, int(n_features * percentile))
    return min(k, n_features)


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing with artifact handling and filtering.
    """
    
    def __init__(self, fs=256, lowcut=1.0, highcut=50.0, order=4):
        """
        Initialize EEG preprocessor.
        
        Args:
            fs: Sampling frequency
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.scaler = StandardScaler()
    
    def apply_bandpass_filter(self, signal_data):
        """Apply bandpass filter to remove artifacts and noise."""
        nyquist = self.fs / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Ensure filter parameters are valid
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            low, high = 0.01, 0.99
        
        try:
            b, a = signal.butter(self.order, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
            return filtered
        except Exception:
            return signal_data
    
    def remove_artifacts(self, signal_data, threshold=3.0):
        """
        Simple artifact removal using amplitude thresholding.
        
        Args:
            signal_data: Input signal
            threshold: Standard deviation threshold for artifact detection
            
        Returns:
            Signal with artifacts attenuated
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        artifact_indices = np.abs(signal_data - mean) > (threshold * std)
        signal_clean = signal_data.copy()
        
        # Replace artifacts with interpolated values
        if np.any(artifact_indices):
            valid_indices = np.where(~artifact_indices)[0]
            if len(valid_indices) > 1:
                artifact_indices_idx = np.where(artifact_indices)[0]
                for idx in artifact_indices_idx:
                    closest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                    signal_clean[idx] = signal_data[closest]
        
        return signal_clean
    
    def normalize(self, signal_data):
        """Normalize signal to zero mean and unit variance."""
        return (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
    
    def preprocess(self, signal_data):
        """
        Apply full preprocessing pipeline to signal.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Preprocessed signal
        """
        signal_data = self.apply_bandpass_filter(signal_data)
        signal_data = self.remove_artifacts(signal_data)
        signal_data = self.normalize(signal_data)
        return signal_data
