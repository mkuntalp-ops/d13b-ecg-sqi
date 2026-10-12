"""
Rule-Based SQI Heuristic Baseline (Non-Neural Expert Rule Comparator)
- Combines Signal Quality Indicators (bSQI, baseline wander, high-frequency noise, saturation, flatline)
  and visual ink density using a heuristic log-odds formulation.
- Parameters: 0 (hand-crafted deterministic heuristic, NOT a neural network).
- Weights size: 0 MB.
- Evaluated as a baseline comparator for ECG usability screening.
NOTE: The term 'calibrated' in earlier code denoted an empirical weighting,
NOT a formally verified statistical calibration.
"""

from typing import Optional, Dict, Any, Literal
import numpy as np
from PIL import Image


class RuleBasedSQIHeuristic:
    """
    Expert heuristic rule baseline combining signal quality indicators (SQIs)
    and image ink density. Strictly non-neural, zero-parameter deterministic rule.
    """
    def __init__(self):
        self.backend = "heuristic_rule"
        self.num_parameters = 0
        self.weights_size_mb = 0.0

    def decide_single(
        self,
        img: Optional[Image.Image],
        sqi_dict: Optional[Dict[str, float]],
        modality: Literal["multimodal", "image_only", "sqi_only"] = "multimodal"
    ) -> Dict[str, Any]:
        """
        Executes heuristic rule evaluation.
        """
        sqi_score = 0.5
        if sqi_dict is not None and modality in ["sqi_only", "multimodal"]:
            bsqi = sqi_dict.get("bsqi_detector_agreement", 0.5)
            bw = sqi_dict.get("baseline_wander_ratio", 0.0)
            hf = sqi_dict.get("high_frequency_noise_ratio", 0.0)
            sat = sqi_dict.get("saturation_fraction", 0.0)
            flat = sqi_dict.get("flatline_fraction", 0.0)
            
            # Unusability log-odds from SQIs
            logit_sqi = -3.0 * (bsqi - 0.5) + 1.2 * min(bw, 3.0) + 1.5 * min(hf, 3.0) + 5.0 * sat + 5.0 * flat - 0.5
            p_sqi = 1.0 / (1.0 + np.exp(-logit_sqi))
            sqi_score = p_sqi
            
        vis_score = 0.5
        if img is not None and modality in ["image_only", "multimodal"]:
            arr = np.array(img.convert("L"), dtype=np.float32)
            ink_density = np.mean(arr < 200)
            horizontal_var = np.std(np.mean(arr < 200, axis=0))
            logit_vis = 4.0 * (ink_density - 0.08) - 2.0 * (horizontal_var - 0.02)
            p_vis = 1.0 / (1.0 + np.exp(-logit_vis))
            vis_score = p_vis
            
        if modality == "image_only":
            p_unusable = float(vis_score)
        elif modality == "sqi_only":
            p_unusable = float(sqi_score)
        else:
            p_unusable = float(0.6 * sqi_score + 0.4 * vis_score)
            
        p_unusable = float(np.clip(p_unusable, 0.01, 0.99))
        choice = "unusable" if p_unusable >= 0.5 else "usable"
        conf = float(max(p_unusable, 1.0 - p_unusable))
        
        return {
            "choice": choice,
            "prob_unusable": p_unusable,
            "confidence": conf,
            "modality": modality,
            "backend": self.backend
        }
