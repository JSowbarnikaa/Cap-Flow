import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from config.config import THRESHOLD_GRID_POINTS
from src.utils.logger import get_logger

logger = get_logger("ThresholdOptimizer")

def optimize_classification_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_grid_points: int = THRESHOLD_GRID_POINTS
) -> float:
    """
    Finds the optimal probability threshold tau* on the VALIDATION set by maximizing F1 score.
    Applied identically to test set evaluation.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    
    thresholds = np.linspace(0.05, 0.95, num_grid_points)
    
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
            
    logger.info(f"Optimized validation threshold: tau* = {best_threshold:.4f} (Val F1: {best_f1:.4f})")
    return best_threshold
