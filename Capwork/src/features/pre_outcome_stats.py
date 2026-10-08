import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from config.config import (
    COL_CUSTOMER_ID,
    COL_POSTING_DATE,
    COL_TARGET_CLS,
    COL_TARGET_REG,
    COL_HIST_DELAY_MEAN,
    COL_HIST_DELAY_STD,
    COL_HIST_LATE_RATE
)
from src.utils.logger import get_logger

logger = get_logger("CustomerHistoryTracker")

class CustomerHistoryTracker:
    """
    Computes customer historical payment delay metrics strictly on the training partition
    with expanding window (.shift(1)) to prevent self-inclusion and partition leakage.
    Provides frozen snapshot lookups with training-median fallbacks for validation and test.
    """
    def __init__(self):
        self.customer_stats_: Dict[str, Dict[str, float]] = {}
        self.global_fallbacks_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit_transform_train(self, df_train: pd.DataFrame) -> pd.DataFrame:
        df = df_train.copy()
        df[COL_CUSTOMER_ID] = df[COL_CUSTOMER_ID].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        
        # Sort chronologically by customer and posting date
        df = df.sort_values(by=[COL_CUSTOMER_ID, COL_POSTING_DATE]).reset_index(drop=True)
        
        # 1. Compute expanding historical statistics EXCLUDING current row (.shift(1))
        grouped = df.groupby(COL_CUSTOMER_ID)
        
        df[COL_HIST_DELAY_MEAN] = grouped[COL_TARGET_REG].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df[COL_HIST_DELAY_STD] = grouped[COL_TARGET_REG].transform(
            lambda x: x.shift(1).expanding().std()
        )
        df[COL_HIST_LATE_RATE] = grouped[COL_TARGET_CLS].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        
        # 2. Global training medians for cold-start / initial invoice fallback
        self.global_fallbacks_ = {
            COL_HIST_DELAY_MEAN: float(df[COL_TARGET_REG].median()),
            COL_HIST_DELAY_STD: 0.0,
            COL_HIST_LATE_RATE: float(df[COL_TARGET_CLS].median()),
        }
        
        # Fill first invoice per customer with global fallbacks
        df[COL_HIST_DELAY_MEAN] = df[COL_HIST_DELAY_MEAN].fillna(self.global_fallbacks_[COL_HIST_DELAY_MEAN])
        df[COL_HIST_DELAY_STD] = df[COL_HIST_DELAY_STD].fillna(self.global_fallbacks_[COL_HIST_DELAY_STD])
        df[COL_HIST_LATE_RATE] = df[COL_HIST_LATE_RATE].fillna(self.global_fallbacks_[COL_HIST_LATE_RATE])
        
        # 3. Create frozen snapshot summary for test / validation lookups
        cust_summary = df.groupby(COL_CUSTOMER_ID).agg(
            mean_delay=(COL_TARGET_REG, "mean"),
            std_delay=(COL_TARGET_REG, "std"),
            late_rate=(COL_TARGET_CLS, "mean")
        ).reset_index()
        
        for _, row in cust_summary.iterrows():
            cid = str(row[COL_CUSTOMER_ID]).strip()
            self.customer_stats_[cid] = {
                COL_HIST_DELAY_MEAN: float(row["mean_delay"]) if not pd.isna(row["mean_delay"]) else self.global_fallbacks_[COL_HIST_DELAY_MEAN],
                COL_HIST_DELAY_STD: float(row["std_delay"]) if not pd.isna(row["std_delay"]) else self.global_fallbacks_[COL_HIST_DELAY_STD],
                COL_HIST_LATE_RATE: float(row["late_rate"]) if not pd.isna(row["late_rate"]) else self.global_fallbacks_[COL_HIST_LATE_RATE],
            }
            
        self.is_fitted_ = True
        logger.info(f"Fitted CustomerHistoryTracker on {len(df)} training records across {len(self.customer_stats_)} customers.")
        return df

    def transform(self, df_eval: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("CustomerHistoryTracker must be fitted on training data before transform!")
            
        df = df_eval.copy()
        df[COL_CUSTOMER_ID] = df[COL_CUSTOMER_ID].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        
        hist_means, hist_stds, hist_rates = [], [], []
        cold_start_count = 0
        
        for cid in df[COL_CUSTOMER_ID]:
            cid_str = str(cid).strip()
            if cid_str in self.customer_stats_:
                stats = self.customer_stats_[cid_str]
            else:
                stats = self.global_fallbacks_
                cold_start_count += 1
                
            hist_means.append(stats[COL_HIST_DELAY_MEAN])
            hist_stds.append(stats[COL_HIST_DELAY_STD])
            hist_rates.append(stats[COL_HIST_LATE_RATE])
            
        df[COL_HIST_DELAY_MEAN] = hist_means
        df[COL_HIST_DELAY_STD] = hist_stds
        df[COL_HIST_LATE_RATE] = hist_rates
        
        logger.info(f"Transformed {len(df)} eval records (Cold-start count: {cold_start_count} / {len(df)}).")
        return df
