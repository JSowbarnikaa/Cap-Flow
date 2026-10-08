import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from config.config import REGRESSOR_PARAMS
from src.utils.logger import get_logger

logger = get_logger("Stage2Regressor")

class Stage2Regressor:
    """
    Stage 2 XGBoost Regressor for predicting delay days (Days_Overdue_Delay)
    trained strictly on delayed instances (y_delay > 0).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 42):
        self.params = params or REGRESSOR_PARAMS.copy()
        self.params["random_state"] = seed
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y_reg: pd.Series, y_cls: Optional[pd.Series] = None) -> "Stage2Regressor":
        # Filter strictly for delayed invoices if classification labels provided
        if y_cls is not None:
            mask = (y_cls == 1) & (y_reg > 0)
        else:
            mask = y_reg > 0
            
        X_train = X[mask]
        y_train = y_reg[mask]
        
        if len(X_train) == 0:
            raise ValueError("No delayed instances found for Stage 2 training!")
            
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f"Fitted Stage 2 Regressor on {len(X_train)} positive delay records.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Regressor must be fitted before predict!")
        preds = self.model.predict(X)
        return np.maximum(0.0, preds)
