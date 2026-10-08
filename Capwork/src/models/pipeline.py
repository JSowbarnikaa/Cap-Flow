import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.models.classifier import Stage1Classifier
from src.models.regressor import Stage2Regressor

class TwoStagePipeline:
    """
    End-to-end inference wrapper combining Stage 1 Classifier and Stage 2 Regressor.
    """
    def __init__(self, classifier: Stage1Classifier, regressor: Stage2Regressor, threshold: float = 0.5):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            (cls_proba, cls_pred, reg_pred_conditional)
        """
        cls_proba = self.classifier.predict_proba(X)
        cls_pred = (cls_proba >= self.threshold).astype(int)
        raw_reg = self.regressor.predict(X)
        
        # In full 2-stage inference, if predicted on-time (0), predicted delay is 0.0
        final_delay_pred = np.where(cls_pred == 1, raw_reg, 0.0)
        return cls_proba, cls_pred, final_delay_pred
