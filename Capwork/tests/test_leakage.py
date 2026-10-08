import unittest
import pandas as pd
import numpy as np
from src.features.text_generator import compute_pre_outcome_risk_score, generate_controlled_communications
from config.config import (
    COL_CUSTOMER_ID,
    COL_AMOUNT,
    COL_PAYMENT_TERMS_DAYS,
    COL_NO_OF_ORDERS,
    COL_HIST_DELAY_MEAN,
    COL_HIST_LATE_RATE,
    COL_TARGET_CLS,
    COL_TARGET_REG,
    COL_CLEAR_DATE
)

class TestLeakage(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            COL_CUSTOMER_ID: ["C101", "C102", "C103"],
            COL_AMOUNT: [5000.0, 15000.0, 50000.0],
            COL_PAYMENT_TERMS_DAYS: [30.0, 45.0, 60.0],
            COL_NO_OF_ORDERS: [5, 20, 100],
            COL_HIST_DELAY_MEAN: [2.5, 0.0, 12.0],
            COL_HIST_LATE_RATE: [0.3, 0.0, 0.8],
            COL_TARGET_CLS: [0, 1, 1],
            COL_TARGET_REG: [0.0, 5.0, 20.0],
            COL_CLEAR_DATE: ["2024-02-15", "2024-03-20", "2024-04-10"]
        })

    def test_target_leakage_invariance(self):
        # 1. Compute text and scores with target columns present
        scores_with_targets = compute_pre_outcome_risk_score(self.sample_df)
        texts_with_targets = generate_controlled_communications(self.sample_df, signal_strength=1.0, seed=42)

        # 2. Drop all target and clearing date columns
        df_no_targets = self.sample_df.drop(columns=[COL_TARGET_CLS, COL_TARGET_REG, COL_CLEAR_DATE])
        
        scores_no_targets = compute_pre_outcome_risk_score(df_no_targets)
        texts_no_targets = generate_controlled_communications(df_no_targets, signal_strength=1.0, seed=42)

        # 3. Assert strict mathematical identity
        np.testing.assert_allclose(scores_with_targets, scores_no_targets, rtol=1e-7, atol=1e-7)
        self.assertEqual(texts_with_targets, texts_no_targets)

if __name__ == "__main__":
    unittest.main()
