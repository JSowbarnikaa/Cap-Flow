import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config.config import (
    RAW_DATA_PATH,
    COL_CUSTOMER_ID,
    COL_INVOICE_ID,
    COL_POSTING_DATE,
    COL_DUE_DATE,
    COL_CLEAR_DATE,
    COL_AMOUNT,
    COL_PAYMENT_TERM,
    COL_PAYMENT_METHOD,
    COL_REGION,
    COL_CITY,
    COL_CUSTOMER_AGE_MONTHS,
    COL_NO_OF_ORDERS,
    COL_TARGET_CLS,
    COL_TARGET_REG
)
from src.utils.logger import get_logger

logger = get_logger("DataLoader")

def load_raw_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    path = file_path or RAW_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    
    logger.info(f"Loading raw B2B dataset from {path}...")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Raw records loaded: {len(df)}")
    
    # Ensure customer ID is string
    if COL_CUSTOMER_ID in df.columns:
        df[COL_CUSTOMER_ID] = df[COL_CUSTOMER_ID].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        
    if COL_INVOICE_ID in df.columns:
        df[COL_INVOICE_ID] = df[COL_INVOICE_ID].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    
    # 1. Parse dates robustly using format='mixed'
    for col in [COL_POSTING_DATE, COL_DUE_DATE, COL_CLEAR_DATE]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    
    # 2. Clean numeric columns
    num_cols = [COL_AMOUNT, COL_CUSTOMER_AGE_MONTHS, COL_NO_OF_ORDERS, COL_TARGET_REG]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 3. Clean string / categorical columns
    cat_cols = [COL_PAYMENT_TERM, COL_PAYMENT_METHOD, COL_REGION, COL_CITY]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown"})
    
    # 4. Derive targets if missing or verify
    if COL_TARGET_REG not in df.columns or df[COL_TARGET_REG].isna().any():
        df[COL_TARGET_REG] = (df[COL_CLEAR_DATE] - df[COL_DUE_DATE]).dt.days.fillna(0).clip(lower=0)
    
    if COL_TARGET_CLS not in df.columns or df[COL_TARGET_CLS].isna().any():
        df[COL_TARGET_CLS] = (df[COL_TARGET_REG] > 0).astype(int)
    else:
        df[COL_TARGET_CLS] = df[COL_TARGET_CLS].astype(int)
        
    logger.info(f"Sanitized dataset: {len(df)} records, {len(df.columns)} columns. Late rate: {df[COL_TARGET_CLS].mean():.2%}")
    return df
