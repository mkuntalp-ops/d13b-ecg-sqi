"""
Signal Quality Indicators (SQI) Extractor
Extracts interpretable numerical signal quality metrics and provides structured representations
for multimodal decision models.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

from features.qrs_detectors import detect_qrs_pan_tompkins, detect_qrs_difference, calculate_bsqi

SQI_FEATURE_NAMES = [
    "bsqi_detector_agreement",
    "baseline_wander_ratio",
    "high_frequency_noise_ratio",
    "saturation_fraction",
    "flatline_fraction",
    "qrs_energy_ratio",
    "kurtosis_ksqi",
    "skewness_ssqi",
    "zero_crossing_rate",
    "powerline_noise_ratio",
    "signal_std",
    "duration_sec",
    "sampling_rate"
]

def compute_spectral_powers(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Computes Welch power spectral density and relative power bands:
    - baseline (< 0.5 Hz)
    - ecg_band (0.5 - 40 Hz)
    - qrs_band (5 - 15 Hz)
    - hf_noise (> 40 Hz up to min(fs/2, 100 Hz))
    - powerline (49-51 Hz and 59-61 Hz)
    """
    nperseg = min(len(ecg), int(2.0 * fs))
    if nperseg < 8:
        return {"bw_ratio": 1.0, "hf_ratio": 1.0, "qrs_ratio": 0.0, "pl_ratio": 0.0}
        
    freqs, psd = signal.welch(ecg, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd) + 1e-12
    
    # Frequency masks
    mask_bw = (freqs >= 0.0) & (freqs < 0.5)
    mask_ecg = (freqs >= 0.5) & (freqs <= 40.0)
    mask_qrs = (freqs >= 5.0) & (freqs <= 15.0)
    mask_hf = (freqs > 40.0) & (freqs <= min(fs * 0.49, 100.0))
    mask_pl = ((freqs >= 49.0) & (freqs <= 51.0)) | ((freqs >= 59.0) & (freqs <= 61.0))
    
    power_bw = np.sum(psd[mask_bw])
    power_ecg = np.sum(psd[mask_ecg]) + 1e-12
    power_qrs = np.sum(psd[mask_qrs])
    power_hf = np.sum(psd[mask_hf])
    power_pl = np.sum(psd[mask_pl])
    
    bw_ratio = float(power_bw / power_ecg)
    hf_ratio = float(power_hf / power_ecg)
    qrs_ratio = float(power_qrs / power_ecg)
    pl_ratio = float(power_pl / power_ecg)
    
    return {
        "bw_ratio": bw_ratio,
        "hf_ratio": hf_ratio,
        "qrs_ratio": qrs_ratio,
        "pl_ratio": pl_ratio
    }


def compute_flatline_and_saturation(ecg: np.ndarray) -> Tuple[float, float]:
    """
    Detects flatline (repeated identical values) and ADC saturation (clipping at extrema).
    """
    if len(ecg) < 2:
        return 1.0, 1.0
        
    # Flatline: consecutive sample differences == 0
    diffs = np.diff(ecg)
    flat_samples = np.sum(np.abs(diffs) < 1e-6)
    flatline_fraction = float(flat_samples / len(ecg))
    
    # Saturation: samples sitting exactly at min or max
    s_min = np.min(ecg)
    s_max = np.max(ecg)
    span = s_max - s_min
    if span < 1e-6:
        return 1.0, 1.0 # Completely flat
        
    # Threshold at top/bottom 0.5% margin
    sat_samples = np.sum(ecg <= (s_min + 0.005 * span)) + np.sum(ecg >= (s_max - 0.005 * span))
    # Consider only if concentrated at identical boundary values
    min_count = np.sum(np.isclose(ecg, s_min, atol=1e-5))
    max_count = np.sum(np.isclose(ecg, s_max, atol=1e-5))
    sat_fraction = float((min_count + max_count) / len(ecg))
    
    return flatline_fraction, sat_fraction


def extract_sqi_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extracts all signal quality metrics for a given ECG waveform segment.
    """
    ecg = np.asarray(ecg, dtype=np.float64).flatten()
    duration_sec = float(len(ecg) / fs)
    
    if len(ecg) == 0 or np.all(np.isnan(ecg)):
        return {k: 0.0 for k in SQI_FEATURE_NAMES}
        
    # Clean NaNs if any
    if np.any(np.isnan(ecg)):
        ecg = np.nan_to_num(ecg, nan=np.nanmedian(ecg))
        
    # 1. QRS detector agreement (bSQI)
    peaks_pt = detect_qrs_pan_tompkins(ecg, fs)
    peaks_diff = detect_qrs_difference(ecg, fs)
    bsqi = calculate_bsqi(peaks_pt, peaks_diff, fs, tolerance_ms=50.0)
    
    # 2. Spectral power metrics
    spec = compute_spectral_powers(ecg, fs)
    
    # 3. Flatline and saturation
    flat_frac, sat_frac = compute_flatline_and_saturation(ecg)
    
    # 4. Statistical moments
    k_val = float(kurtosis(ecg, fisher=True)) if len(ecg) > 4 else 0.0
    s_val = float(skew(ecg)) if len(ecg) > 4 else 0.0
    std_val = float(np.std(ecg))
    
    # 5. Zero-crossing rate (after mean centering)
    centered = ecg - np.median(ecg)
    zero_crossings = np.sum(np.diff(np.signbit(centered)) != 0)
    zcr = float(zero_crossings / len(ecg))
    
    sqi_dict = {
        "bsqi_detector_agreement": float(bsqi),
        "baseline_wander_ratio": float(np.clip(spec["bw_ratio"], 0.0, 100.0)),
        "high_frequency_noise_ratio": float(np.clip(spec["hf_ratio"], 0.0, 100.0)),
        "saturation_fraction": float(sat_frac),
        "flatline_fraction": float(flat_frac),
        "qrs_energy_ratio": float(np.clip(spec["qrs_ratio"], 0.0, 1.0)),
        "kurtosis_ksqi": float(np.clip(k_val, -5.0, 50.0)),
        "skewness_ssqi": float(np.clip(s_val, -10.0, 10.0)),
        "zero_crossing_rate": float(zcr),
        "powerline_noise_ratio": float(np.clip(spec["pl_ratio"], 0.0, 100.0)),
        "signal_std": float(std_val),
        "duration_sec": duration_sec,
        "sampling_rate": float(fs)
    }
    
    # Sanitize NaNs and Infs
    for k, v in sqi_dict.items():
        if np.isnan(v) or np.isinf(v):
            sqi_dict[k] = 0.0
            
    return sqi_dict


def sqi_dict_to_vector(sqi_dict: Dict[str, float]) -> np.ndarray:
    """Converts SQI dictionary to ordered feature vector."""
    vec = np.array([sqi_dict.get(k, 0.0) for k in SQI_FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=100.0, neginf=-100.0)


def format_sqi_for_prompt(sqi_dict: Dict[str, float]) -> str:
    """
    Formats the SQI metrics into a clean, structured textual summary
    for the multimodal / textual decision model.
    """
    bsqi = sqi_dict.get("bsqi_detector_agreement", 0.0)
    bw = sqi_dict.get("baseline_wander_ratio", 0.0)
    hf = sqi_dict.get("high_frequency_noise_ratio", 0.0)
    sat = sqi_dict.get("saturation_fraction", 0.0)
    flat = sqi_dict.get("flatline_fraction", 0.0)
    qrs_en = sqi_dict.get("qrs_energy_ratio", 0.0)
    kurt = sqi_dict.get("kurtosis_ksqi", 0.0)
    skew_val = sqi_dict.get("skewness_ssqi", 0.0)
    dur = sqi_dict.get("duration_sec", 10.0)
    fs = sqi_dict.get("sampling_rate", 500.0)
    
    text = (
        f"ECG Signal Quality Indicators (SQIs):\n"
        f"- QRS Detector Agreement (bSQI): {bsqi:.3f} (1.0 = perfect consensus, 0.0 = total disagreement)\n"
        f"- Baseline Wander Ratio: {bw:.3f} (higher indicates severe low-frequency baseline drift)\n"
        f"- High-Frequency Noise Ratio: {hf:.3f} (higher indicates severe EMG/electromyographic artifact)\n"
        f"- Saturation Fraction: {sat*100:.2f}% (ADC clipping)\n"
        f"- Flatline Fraction: {flat*100:.2f}% (lead disconnection or lost signal)\n"
        f"- QRS Band Energy Ratio: {qrs_en:.3f} (relative power in 5-15 Hz range)\n"
        f"- Kurtosis (kSQI): {kurt:.2f}, Skewness (sSQI): {skew_val:.2f}\n"
        f"- Window Duration: {dur:.1f} s, Sampling Rate: {fs:.0f} Hz."
    )
    return text
