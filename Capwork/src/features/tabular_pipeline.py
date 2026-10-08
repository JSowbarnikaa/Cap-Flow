import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.config import (
    COL_POSTING_DATE,
    COL_DUE_DATE,
    COL_AMOUNT,
    COL_LOG_AMOUNT,
    COL_PAYMENT_TERMS_DAYS,
    COL_POSTING_MONTH_SIN,
    COL_POSTING_MONTH_COS,
    COL_POSTING_DOW,
    BASE_TABULAR_NUMERICAL,
    BASE_TABULAR_CATEGORICAL,
    TABULAR_FEATURE_SET
)
from src.utils.logger import get_logger

logger = get_logger("TabularPipeline")

def engineer_base_tabular_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes deterministic, fixed-reference tabular features.
    Guaranteed to produce identical results for single-invoice inference vs batch training.
    """
    df = df.copy()
    
    # 1. Log Amount (deterministic transformation)
    if COL_AMOUNT in df.columns:
        df[COL_LOG_AMOUNT] = np.log1p(np.maximum(0.0, df[COL_AMOUNT].fillna(0.0)))
    else:
        df[COL_LOG_AMOUNT] = 0.0
        
    # 2. Payment Terms Duration in Days (Net_Due_Date - Posting_Date)
    if COL_DUE_DATE in df.columns and COL_POSTING_DATE in df.columns:
        due = pd.to_datetime(df[COL_DUE_DATE], errors="coerce")
        post = pd.to_datetime(df[COL_POSTING_DATE], errors="coerce")
        diff_days = (due - post).dt.days
        df[COL_PAYMENT_TERMS_DAYS] = diff_days.fillna(30.0).clip(lower=0.0)
    else:
        df[COL_PAYMENT_TERMS_DAYS] = 30.0
        
    # 3. Cyclical Temporal Features (Fixed Calendar Reference, NO batch max/min)
    if COL_POSTING_DATE in df.columns:
        post = pd.to_datetime(df[COL_POSTING_DATE], errors="coerce")
        month = post.dt.month.fillna(1).astype(int)
        df[COL_POSTING_MONTH_SIN] = np.sin(2.0 * np.pi * month / 12.0)
        df[COL_POSTING_MONTH_COS] = np.cos(2.0 * np.pi * month / 12.0)
        df[COL_POSTING_DOW] = post.dt.dayofweek.fillna(0).astype(int)
    else:
        df[COL_POSTING_MONTH_SIN] = 0.0
        df[COL_POSTING_MONTH_COS] = 1.0
        df[COL_POSTING_DOW] = 0
        
    return df

class TabularPipeline:
    """
    Scalers and encoders fit strictly on the training partition and transformed on val/test.
    """
    def __init__(self, numerical_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None):
        self.num_features = numerical_features or BASE_TABULAR_NUMERICAL
        self.cat_features = categorical_features or BASE_TABULAR_CATEGORICAL
        self.scaler = StandardScaler()
        self.cat_encoders: Dict[str, Dict[str, int]] = {}
        self.num_impute_values: Dict[str, float] = {}
        self.cat_impute_values: Dict[str, str] = {}
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "TabularPipeline":
        df = df_train.copy()
        
        # 1. Compute and store numerical median imputation values & fit scaler
        for col in self.num_features:
            val = float(df[col].median()) if col in df.columns and not df[col].isna().all() else 0.0
            self.num_impute_values[col] = val
            df[col] = df[col].fillna(val)
            
        self.scaler.fit(df[self.num_features])
        
        # 2. Compute categorical mode imputation values & fit label mappings
        for col in self.cat_features:
            mode_val = str(df[col].mode()[0]) if col in df.columns and not df[col].isna().all() else "Unknown"
            self.cat_impute_values[col] = mode_val
            series_clean = df[col].fillna(mode_val).astype(str)
            unique_cats = sorted(series_clean.unique())
            # Mapping from category to integer code (0 to K-1), with -1 reserved for unseen
            mapping = {c: i for i, c in enumerate(unique_cats)}
            self.cat_encoders[col] = mapping
            
        self.is_fitted = True
        logger.info(f"Fitted TabularPipeline with {len(self.num_features)} numerical and {len(self.cat_features)} categorical features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("TabularPipeline must be fitted on training data before calling transform()!")
            
        df_out = pd.DataFrame(index=df.index)
        
        # 1. Process numerical features
        num_data = pd.DataFrame(index=df.index)
        for col in self.num_features:
            if col in df.columns:
                num_data[col] = df[col].fillna(self.num_impute_values[col])
            else:
                num_data[col] = self.num_impute_values[col]
                
        scaled_nums = self.scaler.transform(num_data[self.num_features])
        for i, col in enumerate(self.num_features):
            df_out[col] = scaled_nums[:, i]
            
        # 2. Process categorical features
        for col in self.cat_features:
            mapping = self.cat_encoders[col]
            if col in df.columns:
                raw_series = df[col].fillna(self.cat_impute_values[col]).astype(str)
            else:
                raw_series = pd.Series(self.cat_impute_values[col], index=df.index)
                
            # Map known categories, default to -1 for novel/unseen categories
            encoded = raw_series.map(mapping).fillna(-1).astype(int)
            df_out[col] = encoded
            
        return df_out[self.num_features + self.cat_features]

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)
