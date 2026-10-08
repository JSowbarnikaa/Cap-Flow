import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any
from config.config import COL_CUSTOMER_ID, COL_TARGET_CLS, COL_TARGET_REG
from src.utils.logger import get_logger

logger = get_logger("SignificanceTesting")

def paired_t_test(values_cond: List[float], values_base: List[float]) -> Tuple[float, float]:
    """
    Computes standard two-tailed paired t-test across model runs / seeds.
    """
    arr_c = np.array(values_cond)
    arr_b = np.array(values_base)
    diff = arr_c - arr_b
    if np.all(diff == 0) or np.std(diff, ddof=1) == 0:
        return 0.0, 1.0
    res = stats.ttest_rel(arr_c, arr_b)
    return float(res.statistic), float(res.pvalue)

def compute_cluster_level_metrics(
    df_eval: pd.DataFrame,
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_proba_cls: np.ndarray,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray
) -> pd.DataFrame:
    """
    Aggregates invoice-level predictions into customer-level metric contributions (N_cust = 28).
    """
    eval_df = df_eval.copy()
    eval_df["y_true_cls"] = y_true_cls
    eval_df["y_pred_cls"] = y_pred_cls
    eval_df["y_proba_cls"] = y_proba_cls
    eval_df["y_true_reg"] = y_true_reg
    eval_df["y_pred_reg"] = y_pred_reg
    
    # Add instance losses
    eval_df["acc_contrib"] = (eval_df["y_true_cls"] == eval_df["y_pred_cls"]).astype(float)
    eval_df["brier_contrib"] = (eval_df["y_proba_cls"] - eval_df["y_true_cls"]) ** 2
    
    # For regression, evaluate on positive delays
    eval_df["is_late"] = (eval_df["y_true_cls"] == 1) & (eval_df["y_true_reg"] > 0)
    eval_df["mae_contrib"] = np.abs(eval_df["y_pred_reg"] - eval_df["y_true_reg"])
    
    cust_groups = eval_df.groupby(COL_CUSTOMER_ID)
    
    cust_records = []
    for cid, group in cust_groups:
        n_invoices = len(group)
        cust_acc = group["acc_contrib"].mean()
        cust_brier = group["brier_contrib"].mean()
        
        late_group = group[group["is_late"]]
        if len(late_group) > 0:
            cust_mae = late_group["mae_contrib"].mean()
        else:
            cust_mae = 0.0
            
        cust_records.append({
            COL_CUSTOMER_ID: cid,
            "invoice_count": n_invoices,
            "cust_accuracy": cust_acc,
            "cust_brier_score": cust_brier,
            "cust_mae": cust_mae,
            "has_late_invoices": len(late_group) > 0
        })
        
    return pd.DataFrame(cust_records)

def evaluate_customer_cluster_significance(
    df_eval: pd.DataFrame,
    cond_preds: Dict[str, np.ndarray],
    base_preds: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Computes primary authoritative p_cluster test across N_cust customer clusters.
    """
    df_cluster_cond = compute_cluster_level_metrics(
        df_eval,
        cond_preds["y_true_cls"], cond_preds["y_pred_cls"], cond_preds["y_proba_cls"],
        cond_preds["y_true_reg"], cond_preds["y_pred_reg"]
    )
    df_cluster_base = compute_cluster_level_metrics(
        df_eval,
        base_preds["y_true_cls"], base_preds["y_pred_cls"], base_preds["y_proba_cls"],
        base_preds["y_true_reg"], base_preds["y_pred_reg"]
    )
    
    merged = pd.merge(df_cluster_cond, df_cluster_base, on=COL_CUSTOMER_ID, suffixes=("_cond", "_base"))
    
    # 1. Classification Accuracy across clusters
    t_acc, p_acc = stats.ttest_rel(merged["cust_accuracy_cond"], merged["cust_accuracy_base"])
    
    # 2. Classification Brier Score (lower is better)
    t_brier, p_brier = stats.ttest_rel(merged["cust_brier_score_cond"], merged["cust_brier_score_base"])
    
    # 3. Regression MAE across clusters with late invoices
    late_mask = merged["has_late_invoices_cond"] & merged["has_late_invoices_base"]
    late_merged = merged[late_mask]
    if len(late_merged) > 1:
        t_mae, p_mae = stats.ttest_rel(late_merged["cust_mae_cond"], late_merged["cust_mae_base"])
    else:
        t_mae, p_mae = 0.0, 1.0
        
    return {
        "accuracy": {"t_stat": float(t_acc), "p_value": float(p_acc), "n_clusters": len(merged)},
        "brier_score": {"t_stat": float(t_brier), "p_value": float(p_brier), "n_clusters": len(merged)},
        "mae": {"t_stat": float(t_mae), "p_value": float(p_mae), "n_clusters": len(late_merged)}
    }
