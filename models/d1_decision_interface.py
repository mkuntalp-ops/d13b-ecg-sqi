"""
Liquid AI d1-3B Single-Pass Multimodal Decision Model Wrapper (fail-closed).
Follows the Decision Index schema and single-pass system_one interface:
- model.system_one(state, questions, images=...)
- Supports three input modalities:
    1. 'image_only': state = None, images = [img]
    2. 'sqi_only': state = formatted_sqi_text, images = None
    3. 'multimodal': state = formatted_sqi_text, images = [img]

Fail-closed policy: there is NO heuristic fallback. If the weights are not loaded or
the forward pass fails, a RuntimeError is raised. (The earlier silent fallback produced
results labelled as LiquidAI/d1-3B that were actually a hand-set rule; see
manuscript/REVIEW notes. The rule baseline lives in models/rule_heuristic_baseline.py.)
For the production wrapper used in the benchmarks, see models/d1_real_neural_model.py.
"""

from typing import Optional, Dict, Any, Literal
from PIL import Image
import torch

from features.sqi_extractor import format_sqi_for_prompt

DECISION_QUESTIONS = {
    "usability": {
        "type": "choice",
        "instructions": (
            "Determine whether this electrocardiogram (ECG) recording segment is usable for reliable "
            "QRS-based rhythm analysis, or unusable due to severe artifacts, baseline wander, high-frequency noise, "
            "or saturation/flatline."
        ),
        "criteria": {
            "usable": "QRS complexes are distinguishable and suitable for rhythm analysis. Artifacts are manageable.",
            "unusable": "ECG is severely corrupted by noise, baseline drift, saturation, or absent complexes; requires human review or re-recording."
        }
    }
}


class D1DecisionModelWrapper:
    def __init__(
        self,
        model_name: str = "LiquidAI/d1-3B",
        load_weights: bool = True,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.load_weights = load_weights

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        if self.load_weights:
            self._init_model()

    def _init_model(self):
        from transformers import AutoModel
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        print(f"[d1-3B] Loading {self.model_name} on {self.device}...")
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=dtype
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"[d1-3B] Could not load {self.model_name}; fallback is forbidden. Error: {e}") from e
        print(f"[d1-3B] Successfully loaded {self.model_name}.")

    def decide_single(
        self,
        img: Optional[Image.Image],
        sqi_dict: Optional[Dict[str, float]],
        modality: Literal["multimodal", "image_only", "sqi_only"] = "multimodal"
    ) -> Dict[str, Any]:
        """
        Executes a single-pass decision for one ECG sample.
        Returns:
            {
                "choice": "usable" | "unusable",
                "prob_unusable": float,
                "confidence": float,
                "modality": modality,
                "raw_response": dict
            }
        """
        if self.model is None:
            raise RuntimeError("[d1-3B] Weights are not loaded (load_weights=False). Refusing to fall back to a heuristic.")

        state = None
        images = None

        if modality in ["sqi_only", "multimodal"]:
            state = format_sqi_for_prompt(sqi_dict) if sqi_dict is not None else "No auxiliary signal quality indicators provided."

        if modality in ["image_only", "multimodal"] and img is not None:
            images = [img.convert("RGB")]

        res = self.model.system_one(state, DECISION_QUESTIONS, images=images)
        ans = res["answers"]["usability"]
        probs = ans.get("probabilities", {})
        if "unusable" not in probs:
            raise RuntimeError(f"[d1-3B] Response has no 'unusable' probability: {res}")
        p_unusable = float(probs["unusable"])
        return {
            "choice": str(ans.get("choice")),
            "prob_unusable": p_unusable,
            "confidence": float(ans.get("confidence", max(p_unusable, 1.0 - p_unusable))),
            "modality": modality,
            "raw_response": res
        }
