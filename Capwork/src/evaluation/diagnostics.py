import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any
from config.config import COL_TARGET_CLS, COL_TARGET_REG

def compute_feature_correlations(
    df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """
    Computes Pearson correlation coefficient and p-value for every feature
    against both classification and regression targets. Handles numeric and categorical columns.
    """
    records = []
    y_cls = df[COL_TARGET_CLS].values
    y_reg = df[COL_TARGET_REG].values
    
    for feat in features:
        if feat in df.columns:
            s = df[feat]
            if pd.api.types.is_numeric_dtype(s):
                med = float(s.median()) if not s.isna().all() else 0.0
                vals = s.fillna(med).values.astype(float)
            else:
                # Categorical column: factorize for correlation check
                vals = pd.factorize(s.fillna("Unknown"))[0].astype(float)
                
            # Correlation with Classification Target
            if np.std(vals) > 0 and np.std(y_cls) > 0:
                r_cls, p_cls = stats.pearsonr(vals, y_cls)
            else:
                r_cls, p_cls = 0.0, 1.0
                
            # Correlation with Regression Target
            if np.std(vals) > 0 and np.std(y_reg) > 0:
                r_reg, p_reg = stats.pearsonr(vals, y_reg)
            else:
                r_reg, p_reg = 0.0, 1.0
                
            records.append({
                "feature": feat,
                "r_classification": float(r_cls),
                "p_classification": float(p_cls),
                "r_regression": float(r_reg),
                "p_regression": float(p_reg)
            })
            
    return pd.DataFrame(records).sort_values(by="r_classification", ascending=False)

def audit_train_test_generalization(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Compares Train vs Test performance to identify potential overfitting.
    """
    rows = []
    for metric in ["accuracy", "f1", "auc_roc", "mae", "rmse"]:
        if metric in train_metrics and metric in test_metrics:
            tr = train_metrics[metric]
            te = test_metrics[metric]
            gap = te - tr
            pct_gap = (gap / (abs(tr) + 1e-9)) * 100.0
            rows.append({
                "metric": metric.upper(),
                "train_score": round(tr, 4),
                "test_score": round(te, 4),
                "gap (test - train)": round(gap, 4),
                "relative_gap_%": round(pct_gap, 2)
            })
    return pd.DataFrame(rows)
