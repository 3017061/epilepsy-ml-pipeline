"""
Advanced feature extraction for EEG signals and biomedical data.
Implements time-domain, frequency-domain, wavelet, and statistical features.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import periodogram, welch
from scipy.fftpack import fft
import pywt


class SignalFeatureExtractor:
    """Extract comprehensive feature set from time-series signals."""

    @staticmethod
    def time_domain_features(x: np.ndarray) -> dict:
        """Extract time-domain statistical features."""
        if len(x) == 0:
            return {}
        
        features = {
            'mean': np.mean(x),
            'std': np.std(x),
            'var': np.var(x),
            'min': np.min(x),
            'max': np.max(x),
            'median': np.median(x),
            'skewness': skew(x),
            'kurtosis': kurtosis(x),
            'peak_to_peak': np.ptp(x),
            'rms': np.sqrt(np.mean(x ** 2)),
            'energy': np.sum(x ** 2),
            'entropy': -np.sum(np.abs(x) * np.log(np.abs(x) + 1e-10)),
        }
        return features

    @staticmethod
    def hjorth_parameters(x: np.ndarray) -> dict:
        """Calculate Hjorth parameters (activity, mobility, complexity)."""
        dx = np.diff(x)
        ddx = np.diff(dx)
        
        activity = np.var(x)
        mobility = np.sqrt(np.var(dx) / activity) if activity > 0 else 0
        complexity = (
            np.sqrt(np.var(ddx) / np.var(dx)) / mobility 
            if np.var(dx) > 0 and mobility > 0 else 0
        )
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity,
        }

    @staticmethod
    def frequency_domain_features(x: np.ndarray, fs: float = 1.0) -> dict:
        """Extract frequency-domain features using FFT and Welch's method."""
        features = {}
        
        # Power spectral density using Welch's method
        freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
        
        if len(psd) > 0:
            features['psd_mean'] = np.mean(psd)
            features['psd_std'] = np.std(psd)
            features['psd_max'] = np.max(psd)
            features['psd_energy'] = np.sum(psd)
            
            # Spectral centroid
            if np.sum(psd) > 0:
                features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
                features['spectral_bandwidth'] = (
                    np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd)) / np.sum(psd)
                )
        
        # FFT magnitude spectrum
        fft_vals = np.abs(fft(x))
        features['fft_mean'] = np.mean(fft_vals)
        features['fft_std'] = np.std(fft_vals)
        features['fft_max'] = np.max(fft_vals)
        
        return features

    @staticmethod
    def wavelet_features(x: np.ndarray, wavelet: str = 'db4', level: int = 3) -> dict:
        """Extract wavelet decomposition features."""
        features = {}
        
        try:
            coeffs = pywt.wavedec(x, wavelet, level=min(level, int(np.log2(len(x)))))
            
            for i, coeff in enumerate(coeffs):
                prefix = f'wavelet_d{i}' if i > 0 else 'wavelet_a'
                features[f'{prefix}_mean'] = np.mean(coeff)
                features[f'{prefix}_std'] = np.std(coeff)
                features[f'{prefix}_energy'] = np.sum(coeff ** 2)
                features[f'{prefix}_entropy'] = -np.sum(np.abs(coeff) * np.log(np.abs(coeff) + 1e-10))
        except Exception:
            pass
        
        return features

    @staticmethod
    def zero_crossings(x: np.ndarray) -> dict:
        """Calculate zero crossing rate and other crossing features."""
        zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
        
        return {
            'zero_crossing_rate': zero_crossings / len(x) if len(x) > 0 else 0,
            'zero_crossings': zero_crossings,
        }

    @classmethod
    def extract_all_features(
        cls, 
        x: np.ndarray, 
        fs: float = 1.0,
        include_wavelets: bool = True
    ) -> np.ndarray:
        """
        Extract all available features from signal.
        
        Args:
            x: Input signal (1D array)
            fs: Sampling frequency
            include_wavelets: Whether to include wavelet features
            
        Returns:
            1D feature vector
        """
        all_features = {}
        
        # Time domain
        all_features.update(cls.time_domain_features(x))
        all_features.update(cls.hjorth_parameters(x))
        all_features.update(cls.zero_crossings(x))
        
        # Frequency domain
        all_features.update(cls.frequency_domain_features(x, fs))
        
        # Wavelets (optional due to computational cost)
        if include_wavelets:
            all_features.update(cls.wavelet_features(x))
        
        # Convert to ordered array
        feature_vector = np.array([all_features[k] for k in sorted(all_features.keys())])
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector

    @staticmethod
    def get_feature_names(include_wavelets: bool = True) -> list:
        """Get names of all extracted features in order."""
        extractor = SignalFeatureExtractor()
        dummy_signal = np.random.randn(256)
        
        all_features = {}
        all_features.update(extractor.time_domain_features(dummy_signal))
        all_features.update(extractor.hjorth_parameters(dummy_signal))
        all_features.update(extractor.zero_crossings(dummy_signal))
        all_features.update(extractor.frequency_domain_features(dummy_signal))
        
        if include_wavelets:
            all_features.update(extractor.wavelet_features(dummy_signal))
        
        return sorted(all_features.keys())


# Legacy functions for backward compatibility
def time_features(x):
    """Extract time-domain features."""
    features = SignalFeatureExtractor.time_domain_features(x)
    return np.array([features['mean'], features['std'], features['skewness'], features['kurtosis']])


def extract_features(x, fs=None):
    """Extract comprehensive features from signal."""
    fs = fs or 1.0
    return SignalFeatureExtractor.extract_all_features(x, fs=fs, include_wavelets=False)
