import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    
    try:
        auc_roc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc_roc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold)
    }

def evaluate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0
        
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
