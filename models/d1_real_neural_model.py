"""
Real LiquidAI/d1-3B Neural Decision Model Wrapper (Strictly Fail-Closed)
- Official architecture: 3.12B parameter multimodal decision model (LFM2.5-VL-3B backbone + SigLIP2 NaFlex).
- Official inference: AutoModel.system_one(state, questions, images=...)
- Zero output tokens (output_tokens: 0), single-pass decision classification.
- STRICT FAIL-CLOSED POLICY:
  * No fallback heuristics allowed.
  * If weights fail to load or inference errors, immediately raises RuntimeError.
  * Failed samples (if any) are recorded to an error log file.
  * Saves raw system_one response dictionary for every evaluated sample.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from PIL import Image
import torch
from transformers import AutoModel

from features.sqi_extractor import format_sqi_for_prompt

# Standardized Decision Index Schema for ECG Usability
D1_ECG_USABILITY_QUESTION = {
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

DEFAULT_D1_PATH = "/Volumes/KIOXIA/AI Models/liquid-ai/d1-3B"


class RealD1DecisionModel:
    """
    Fail-closed wrapper for the genuine 3.12B LiquidAI/d1-3B neural decision model.
    """
    def __init__(
        self,
        model_path: str = DEFAULT_D1_PATH,
        device: Optional[str] = None,
        error_log_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.backend = "neural"
        self.model_name = "LiquidAI/d1-3B"
        self.num_parameters = 3_120_000_000
        self.weights_file = os.path.join(model_path, "model.safetensors")
        self.weights_size_bytes = os.path.getsize(self.weights_file) if os.path.exists(self.weights_file) else 0
        self.weights_size_mb = self.weights_size_bytes / (1024 * 1024)
        
        if error_log_path is None:
            self.error_log_path = Path("results/d1_neural_inference_errors.jsonl")
        else:
            self.error_log_path = Path(error_log_path)
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Device selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        
        # Load weights - FAIL CLOSED (Strictly no fallback)
        print(f"[RealD1Neural] Loading 3.12B weights from '{self.model_path}' onto {self.device} (dtype={self.dtype})...")
        t0 = time.time()
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=self.dtype
            ).to(self.device)
            load_sec = time.time() - t0
            print(f"[RealD1Neural] Successfully loaded weights in {load_sec:.2f}s. Model type: {type(self.model)}")
        except Exception as e:
            raise RuntimeError(
                f"[RealD1Neural] CRITICAL: Failed to load real d1-3B neural weights from '{self.model_path}'. "
                f"Automatic fallback is forbidden. Error: {e}"
            ) from e

    def decide_single(
        self,
        img: Optional[Image.Image],
        sqi_dict: Optional[Dict[str, float]],
        modality: Literal["multimodal", "image_only", "sqi_only"] = "multimodal",
        segment_id: Optional[str] = None,
        questions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes genuine single-pass neural decision for one ECG sample.
        `questions` overrides the primary usability question (used only by the prompt-sensitivity analysis).
        FAIL-CLOSED: If neural forward pass fails, raises RuntimeError and records error log.
        """
        state = None
        images = None

        if modality in ["sqi_only", "multimodal"]:
            if sqi_dict is not None:
                state = format_sqi_for_prompt(sqi_dict)
            else:
                state = "No auxiliary signal quality indicators provided."

        if modality in ["image_only", "multimodal"]:
            if img is not None:
                images = [img.convert("RGB")]
            else:
                if modality == "image_only":
                    raise ValueError(f"[RealD1Neural] image_only modality requested for segment '{segment_id}' but img is None.")

        t_start = time.perf_counter()
        try:
            # Genuine system_one forward pass
            raw_response = self.model.system_one(
                state=state,
                questions=questions or D1_ECG_USABILITY_QUESTION,
                images=images
            )
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            # Parse Decision Index structure
            ans = raw_response["answers"]["usability"]
            probs = ans.get("probabilities", {})
            prob_unusable = float(probs.get("unusable"))
            prob_usable = float(probs.get("usable"))
            choice = str(ans.get("choice"))
            confidence = float(ans.get("confidence"))
            usage = raw_response.get("usage", {})
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))

            if output_tokens != 0:
                print(f"[RealD1Neural] Warning: unexpected output_tokens={output_tokens} (expected 0 for decision model).")

            return {
                "choice": choice,
                "prob_unusable": prob_unusable,
                "prob_usable": prob_usable,
                "confidence": confidence,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": lat_ms,
                "modality": modality,
                "backend": self.backend,
                "model_name": self.model_name,
                "device": self.device,
                "dtype": str(self.dtype),
                "raw_response": raw_response
            }

        except Exception as e:
            # Fail closed - record error and raise
            err_entry = {
                "segment_id": segment_id,
                "modality": modality,
                "timestamp": time.time(),
                "error": str(e),
                "device": self.device,
                "dtype": str(self.dtype)
            }
            with open(self.error_log_path, "a") as f:
                f.write(json.dumps(err_entry) + "\n")
            raise RuntimeError(
                f"[RealD1Neural] Forward pass failed on segment '{segment_id}' with error: {e}. "
                f"Fail-closed policy active; automatic fallback is forbidden."
            ) from e
