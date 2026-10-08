import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from config.config import CLASSIFIER_PARAMS
from src.utils.logger import get_logger

logger = get_logger("Stage1Classifier")

class Stage1Classifier:
    """
    Stage 1 XGBoost Classifier for predicting On-Time (0) vs Late (1).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 42):
        self.params = params or CLASSIFIER_PARAMS.copy()
        self.params["random_state"] = seed
        self.model = xgb.XGBClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Stage1Classifier":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba!")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
