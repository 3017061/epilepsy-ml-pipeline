"""
EEG signal filtering utilities for preprocessing.
Implements various frequency domain filtering techniques.
"""

import numpy as np
from scipy import signal


def apply_eeg_filter(
    signal_data,
    fs=256,
    filter_type="bandpass",
    lowcut=1.0,
    highcut=50.0,
    order=4,
):
    """
    Apply frequency domain filter to EEG signal.
    
    Args:
        signal_data: Input signal array
        fs: Sampling frequency
        filter_type: 'bandpass', 'highpass', 'lowpass', 'notch'
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered signal
    """
    try:
        nyquist = fs / 2.0
        
        if filter_type == "bandpass":
            low = lowcut / nyquist
            high = highcut / nyquist
            low = np.clip(low, 0.001, 0.999)
            high = np.clip(high, 0.001, 0.999)
            if low >= high:
                low, high = 0.01, 0.99
            b, a = signal.butter(order, [low, high], btype='band')
        
        elif filter_type == "highpass":
            high = lowcut / nyquist
            high = np.clip(high, 0.001, 0.999)
            b, a = signal.butter(order, high, btype='high')
        
        elif filter_type == "lowpass":
            low = highcut / nyquist
            low = np.clip(low, 0.001, 0.999)
            b, a = signal.butter(order, low, btype='low')
        
        elif filter_type == "notch":
            # 50/60 Hz notch filter
            freq = 50.0 / nyquist
            b, a = signal.iirnotch(freq, Q=30.0, fs=fs)
        
        else:
            return signal_data
        
        # Apply filter twice (forward-backward) for zero phase distortion
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    
    except Exception as e:
        print(f"Warning: Filter application failed ({e}), returning original signal")
        return signal_data


def apply_notch_filter(signal_data, fs=256, freq=50.0, Q=30.0):
    """
    Apply notch filter to remove power line interference.
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        freq: Frequency to notch (50 or 60 Hz)
        Q: Quality factor
        
    Returns:
        Filtered signal
    """
    try:
        b, a = signal.iirnotch(freq, Q, fs)
        return signal.filtfilt(b, a, signal_data)
    except Exception:
        return signal_data


def apply_artifact_removal(signal_data, method="simple_threshold", threshold=3.0):
    """
    Remove artifacts from EEG signal.
    
    Args:
        signal_data: Input signal
        method: 'simple_threshold' or 'iqr'
        threshold: Threshold value
        
    Returns:
        Signal with artifacts attenuated
    """
    signal_clean = signal_data.copy()
    
    if method == "simple_threshold":
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        artifact_indices = np.abs(signal_data - mean) > (threshold * std)
    
    elif method == "iqr":
        q1 = np.percentile(signal_data, 25)
        q3 = np.percentile(signal_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        artifact_indices = (signal_data < lower_bound) | (signal_data > upper_bound)
    
    else:
        return signal_clean
    
    # Replace artifacts with median of nearby valid values
    if np.any(artifact_indices):
        window_size = 5
        for idx in np.where(artifact_indices)[0]:
            start = max(0, idx - window_size)
            end = min(len(signal_data), idx + window_size + 1)
            valid_vals = signal_clean[start:end][~artifact_indices[start:end]]
            if len(valid_vals) > 0:
                signal_clean[idx] = np.median(valid_vals)
    
    return signal_clean


def common_average_reference(eeg_data):
    """
    Apply Common Average Reference (CAR) to multi-channel EEG.
    
    Args:
        eeg_data: EEG data (n_samples, n_channels) or (n_channels, n_timepoints)
        
    Returns:
        Referenced EEG data with same shape
    """
    if eeg_data.ndim == 2:
        # Assuming (n_channels, n_timepoints) or (n_samples, n_features)
        avg = np.mean(eeg_data, axis=0, keepdims=True)
        return eeg_data - avg
    else:
        return eeg_data
