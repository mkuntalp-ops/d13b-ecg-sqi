"""
Tabular Baseline Models for ECG Usability Assessment.
Trained on numerical Signal Quality Indicators (SQIs):
1. Logistic Regression (with StandardScaler & regularization)
2. Gradient Boosting (LightGBM Classifier)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from sklearn.impute import SimpleImputer
from features.sqi_extractor import SQI_FEATURE_NAMES

class SQILogisticRegression:
    def __init__(self, c_param: float = 1.0, random_state: int = 42):
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c_param,
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000
            ))
        ])
        self.feature_names = SQI_FEATURE_NAMES
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.nan_to_num(X, nan=0.0)
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(unusable), i.e. column 1."""
        X = np.nan_to_num(X, nan=0.0)
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            return np.zeros(len(X))
        return probs[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class SQIGradientBoosting:
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.feature_names = SQI_FEATURE_NAMES
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.nan_to_num(X, nan=0.0)
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0)
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            return np.zeros(len(X))
        return probs[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, [float(x) for x in importances]))
