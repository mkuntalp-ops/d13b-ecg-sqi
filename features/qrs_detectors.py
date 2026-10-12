"""
Two Complementary QRS Detectors & Inter-Detector Agreement (bSQI)
Used for robust Signal Quality Indexing.
"""

import numpy as np
from scipy import signal

def detect_qrs_pan_tompkins(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan-Tompkins QRS Detection Algorithm.
    Returns array of R-peak sample indices.
    """
    if len(ecg) < int(fs * 0.5):
        return np.array([], dtype=int)
    
    # 1. Bandpass Filter (5 - 15 Hz)
    nyq = 0.5 * fs
    low = max(0.01, 5.0 / nyq)
    high = min(0.99, 15.0 / nyq)
    b, a = signal.butter(2, [low, high], btype='bandpass')
    filtered = signal.filtfilt(b, a, ecg)
    
    # 2. Derivative Filter
    diff_kernel = np.array([-1, -2, 0, 2, 1]) / 8.0
    derivative = np.convolve(filtered, diff_kernel, mode='same')
    
    # 3. Squaring
    squared = derivative ** 2
    
    # 4. Moving Window Integration (approx 150 ms window)
    window_len = max(3, int(0.15 * fs))
    kernel = np.ones(window_len) / window_len
    mwi = np.convolve(squared, kernel, mode='same')
    
    # 5. Adaptive Thresholding
    refractory_samples = int(0.20 * fs) # 200 ms refractory period
    peaks, _ = signal.find_peaks(mwi, distance=refractory_samples)
    
    if len(peaks) == 0:
        return np.array([], dtype=int)
    
    # Threshold based on signal mean and std
    mwi_mean = np.mean(mwi)
    mwi_std = np.std(mwi)
    threshold = mwi_mean + 0.3 * mwi_std
    
    valid_peaks = peaks[mwi[peaks] > threshold]
    
    # Refine peak locations to maximum in raw ECG within [-50ms, +50ms]
    search_window = int(0.06 * fs)
    refined_peaks = []
    for p in valid_peaks:
        start = max(0, p - search_window)
        end = min(len(ecg), p + search_window)
        if end > start:
            refined_peaks.append(start + np.argmax(np.abs(ecg[start:end])))
            
    return np.unique(np.array(refined_peaks, dtype=int))


def detect_qrs_difference(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Second-Order Difference / Absolute Derivative QRS Detector.
    Acts as an independent complementary detector.
    """
    if len(ecg) < int(fs * 0.5):
        return np.array([], dtype=int)
        
    # Baseline suppression via highpass filter (0.5 Hz)
    nyq = 0.5 * fs
    b, a = signal.butter(2, min(0.99, 0.5 / nyq), btype='highpass')
    hp_ecg = signal.filtfilt(b, a, ecg)
    
    # Second-order difference: d2[n] = |x[n+2] - 2*x[n] + x[n-2]|
    d2 = np.abs(np.diff(hp_ecg, n=2))
    d2 = np.pad(d2, (1, 1), mode='edge')
    
    # Smooth with short moving average (~80 ms)
    smooth_len = max(3, int(0.08 * fs))
    kernel = np.ones(smooth_len) / smooth_len
    energy = np.convolve(d2, kernel, mode='same')
    
    # Dynamic peak thresholding
    refractory_samples = int(0.22 * fs)
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    
    peaks, _ = signal.find_peaks(energy, height=threshold, distance=refractory_samples)
    
    # Refine peak positions on raw ECG
    search_window = int(0.05 * fs)
    refined_peaks = []
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(ecg), p + search_window)
        if end > start:
            refined_peaks.append(start + np.argmax(np.abs(ecg[start:end])))
            
    return np.unique(np.array(refined_peaks, dtype=int))


def calculate_bsqi(peaks_a: np.ndarray, peaks_b: np.ndarray, fs: float, tolerance_ms: float = 50.0) -> float:
    """
    Computes bSQI (QRS detector agreement F1-score) between two detectors.
    tolerance_ms: maximum time difference to count as matched peak.
    Returns:
        f1_agreement in [0.0, 1.0]
    """
    if len(peaks_a) == 0 and len(peaks_b) == 0:
        return 0.0 # No detections from either detector
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0.0 # One found peaks, one found none -> total disagreement
        
    tol_samples = int(tolerance_ms * fs / 1000.0)
    
    # Count matching peaks
    matched_a = 0
    used_b = set()
    
    for pa in peaks_a:
        # Match to the nearest peak in b that is still unused (one-to-one matching);
        # matching to the globally nearest peak could block a valid second match
        diffs = np.abs(peaks_b.astype(np.int64) - int(pa)).astype(float)
        if used_b:
            diffs[list(used_b)] = np.inf
        closest_idx = int(np.argmin(diffs))
        if diffs[closest_idx] <= tol_samples:
            matched_a += 1
            used_b.add(closest_idx)
            
    tp = matched_a
    precision = tp / len(peaks_b) if len(peaks_b) > 0 else 0.0
    recall = tp / len(peaks_a) if len(peaks_a) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(np.clip(f1, 0.0, 1.0))
