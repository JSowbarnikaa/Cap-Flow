import pandas as pd
from typing import Tuple
from config.config import (
    COL_POSTING_DATE,
    COL_INVOICE_ID,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)
from src.utils.logger import get_logger

logger = get_logger("TemporalSplit")

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sorts chronologically by Posting_Date and splits into Train, Val, Test partitions.
    Guarantees no temporal leakage across splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    sort_cols = [COL_POSTING_DATE]
    if COL_INVOICE_ID in df.columns:
        sort_cols.append(COL_INVOICE_ID)
    
    df_sorted = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    logger.info(
        f"Chronological split completed (N={n}): "
        f"Train={len(train_df)} ({len(train_df)/n:.1%}, {train_df[COL_POSTING_DATE].min().strftime('%Y-%m-%d')} to {train_df[COL_POSTING_DATE].max().strftime('%Y-%m-%d')}), "
        f"Val={len(val_df)} ({len(val_df)/n:.1%}, {val_df[COL_POSTING_DATE].min().strftime('%Y-%m-%d')} to {val_df[COL_POSTING_DATE].max().strftime('%Y-%m-%d')}), "
        f"Test={len(test_df)} ({len(test_df)/n:.1%}, {test_df[COL_POSTING_DATE].min().strftime('%Y-%m-%d')} to {test_df[COL_POSTING_DATE].max().strftime('%Y-%m-%d')})"
    )
    
    return train_df, val_df, test_df
