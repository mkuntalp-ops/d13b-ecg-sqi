"""
Table 2c: scaling and feature sensitivity of the SQI models under BUT QDB -> CinC 2011 transfer.

Train on the 1,047-window BUT QDB cohort (fit scalers on BUT QDB only), evaluate frozen on the
150 CinC 2011 Set A records (Lead II). Reports AUROC / AP for:
- Logistic Regression under three scalers (StandardScaler, RobustScaler, z-score clipped to [-3, 3])
- Logistic Regression and Independent SQI MLP under feature-subset variants

Output: results/cinc2011_scaling_sensitivity.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.sqi_extractor import SQI_FEATURE_NAMES as F
from models.multimodal_fusion import IndependentSQIClassifier

RESULTS_DIR = PROJECT_ROOT / "results"
COHORT = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
CINC = PROJECT_ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_segments_lead1_max150.csv"
OUT = RESULTS_DIR / "cinc2011_scaling_sensitivity.csv"

SAT_FLAT_CONST = ["saturation_fraction", "flatline_fraction", "duration_sec", "sampling_rate"]
VARIANTS = {
    "All 13 features": F,
    "Drop signal_std": [f for f in F if f != "signal_std"],
    "Drop saturation, flatline, constants": [f for f in F if f not in SAT_FLAT_CONST],
    "Drop signal_std, saturation, flatline, constants": [f for f in F if f not in SAT_FLAT_CONST + ["signal_std"]],
}


class _ClipScaler:
    """z-score with train statistics, then clip to [-3, 3]."""

    def fit(self, X):
        self.s = StandardScaler().fit(X)
        return self

    def transform(self, X):
        return np.clip(self.s.transform(X), -3.0, 3.0)


def _impute(train: np.ndarray, test: np.ndarray):
    imp = SimpleImputer(strategy="median").fit(train)
    return imp.transform(train), imp.transform(test)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    tr = pd.read_csv(COHORT)
    te = pd.read_csv(CINC)
    y_tr, y_te = tr["label"].values, te["label"].values
    rows = []

    for vname, feats in VARIANTS.items():
        A, B = _impute(tr[feats].values.astype(float), te[feats].values.astype(float))
        for sname, scaler in [
            ("StandardScaler", StandardScaler()),
            ("RobustScaler", RobustScaler()),
            ("Z-clip [-3,3]", _ClipScaler()),
        ]:
            scaler.fit(A)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(scaler.transform(A), y_tr)
            p = clf.predict_proba(scaler.transform(B))[:, 1]
            rows.append({"model": "LR", "features": vname, "scaler": sname,
                         "transfer_auroc": roc_auc_score(y_te, p),
                         "transfer_ap": average_precision_score(y_te, p)})

        # MLP: IndependentSQIClassifier standardizes internally (train statistics), so only its default is reported
        mlp = IndependentSQIClassifier(in_features=len(feats), epochs=35, lr=1e-3)
        mlp.fit(A, y_tr)
        p = mlp.predict_proba(B)
        rows.append({"model": "Independent_SQI_MLP", "features": vname, "scaler": "internal StandardScaler",
                     "transfer_auroc": roc_auc_score(y_te, p),
                     "transfer_ap": average_precision_score(y_te, p)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
