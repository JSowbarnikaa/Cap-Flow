import unittest
import pandas as pd
import numpy as np
from config.config import (
    COL_POSTING_DATE,
    COL_DUE_DATE,
    COL_AMOUNT,
    COL_PAYMENT_TERM,
    COL_PAYMENT_METHOD,
    COL_REGION,
    COL_CITY,
    COL_CUSTOMER_AGE_MONTHS,
    COL_NO_OF_ORDERS,
    COL_HIST_DELAY_MEAN,
    COL_HIST_DELAY_STD,
    COL_HIST_LATE_RATE
)
from src.features.tabular_pipeline import TabularPipeline, engineer_base_tabular_columns

class TestSingleInvoiceInference(unittest.TestCase):
    def setUp(self):
        # Create a batch of training records
        np.random.seed(42)
        n = 100
        self.df_train = pd.DataFrame({
            COL_POSTING_DATE: pd.date_range("2020-01-01", periods=n, freq="D"),
            COL_DUE_DATE: pd.date_range("2020-01-31", periods=n, freq="D"),
            COL_AMOUNT: np.random.uniform(1000, 50000, size=n),
            COL_PAYMENT_TERM: np.random.choice(["Net30", "Net60", "Net15"], size=n),
            COL_PAYMENT_METHOD: np.random.choice(["ACH", "Wire", "Check"], size=n),
            COL_REGION: np.random.choice(["North", "South", "East"], size=n),
            COL_CITY: np.random.choice(["CityA", "CityB", "CityC"], size=n),
            COL_CUSTOMER_AGE_MONTHS: np.random.uniform(12, 120, size=n),
            COL_NO_OF_ORDERS: np.random.randint(1, 50, size=n),
            COL_HIST_DELAY_MEAN: np.random.uniform(0, 15, size=n),
            COL_HIST_DELAY_STD: np.random.uniform(0, 5, size=n),
            COL_HIST_LATE_RATE: np.random.uniform(0, 1, size=n),
        })

    def test_single_invoice_invariance(self):
        # 1. Feature engineer and fit pipeline on training batch
        df_train_eng = engineer_base_tabular_columns(self.df_train)
        pipeline = TabularPipeline()
        pipeline.fit(df_train_eng)

        # 2. Select a single row as batch evaluation
        single_row_batch = df_train_eng.iloc[[42]].copy()
        batch_transformed = pipeline.transform(single_row_batch)

        # 3. Simulate real single-invoice standalone inference
        raw_single_invoice = self.df_train.iloc[[42]].copy()
        single_eng = engineer_base_tabular_columns(raw_single_invoice)
        single_transformed = pipeline.transform(single_eng)

        # 4. Assert exact equality
        pd.testing.assert_frame_equal(batch_transformed, single_transformed, check_exact=True)

if __name__ == "__main__":
    unittest.main()
