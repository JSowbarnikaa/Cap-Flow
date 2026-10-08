from pathlib import Path
from typing import List, Dict, Any

# =====================================================================
# 1. FILE PATHS & DIRECTORIES
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = Path(r"C:\Users\sowba\OneDrive\Desktop\CIT Chennai\SEM 3\ML\Project\capFlow\data\raw\Dataset.csv")
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "saved_models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. RAW COLUMN DEFINITIONS
# =====================================================================
COL_CUSTOMER_ID = "Cust_Num"
COL_INVOICE_ID = "Document_No"
COL_POSTING_DATE = "Posting_Date"
COL_DUE_DATE = "Net_Due_Date"
COL_CLEAR_DATE = "Clearing_date"
COL_AMOUNT = "Amount"
COL_PAYMENT_TERM = "Payment_Term"
COL_PAYMENT_METHOD = "Payment_Method_description"
COL_REGION = "Region"
COL_CITY = "City"
COL_CUSTOMER_AGE_MONTHS = "Age_Of_Customer_Months"
COL_NO_OF_ORDERS = "No_of_orders_by_customer"

# Target Columns
COL_TARGET_CLS = "DelayFlag"           # 1 if late, 0 if on time
COL_TARGET_REG = "Days_Overdue_Delay"  # Days overdue (>= 0)

# =====================================================================
# 3. ENGINEERED TABULAR FEATURES
# =====================================================================
COL_HIST_DELAY_MEAN = "cust_hist_delay_mean"
COL_HIST_DELAY_STD = "cust_hist_delay_std"
COL_HIST_LATE_RATE = "cust_hist_late_rate"
COL_PAYMENT_TERMS_DAYS = "payment_terms_days"
COL_INVOICE_AGE_DAYS = "invoice_age_days"
COL_POSTING_MONTH_SIN = "posting_month_sin"
COL_POSTING_MONTH_COS = "posting_month_cos"
COL_POSTING_DOW = "posting_day_of_week"
COL_LOG_AMOUNT = "log_amount"

# NLP / Communication Features
COL_COMM_TEXT = "customer_communication_text"
COL_DISPUTE_SCORE = "comm_dispute_score"
COL_DELAY_RISK_SCORE = "comm_delay_risk_score"
COL_SENTIMENT_NEG = "comm_sentiment_negative"
COL_URGENCY_SCORE = "comm_urgency_score"

# =====================================================================
# 4. UNIFIED FEATURE SETS
# =====================================================================
BASE_TABULAR_NUMERICAL: List[str] = [
    COL_AMOUNT,
    COL_LOG_AMOUNT,
    COL_PAYMENT_TERMS_DAYS,
    COL_CUSTOMER_AGE_MONTHS,
    COL_NO_OF_ORDERS,
    COL_POSTING_MONTH_SIN,
    COL_POSTING_MONTH_COS,
    COL_POSTING_DOW,
    COL_HIST_DELAY_MEAN,
    COL_HIST_DELAY_STD,
    COL_HIST_LATE_RATE,
]

BASE_TABULAR_CATEGORICAL: List[str] = [
    COL_PAYMENT_TERM,
    COL_PAYMENT_METHOD,
    COL_REGION,
    COL_CITY,
]

TABULAR_FEATURE_SET: List[str] = BASE_TABULAR_NUMERICAL + BASE_TABULAR_CATEGORICAL

SENTIMENT_FEATURE_SET: List[str] = [
    COL_DISPUTE_SCORE,
    COL_DELAY_RISK_SCORE,
    COL_SENTIMENT_NEG,
    COL_URGENCY_SCORE,
]

FULL_FEATURE_SET: List[str] = TABULAR_FEATURE_SET + SENTIMENT_FEATURE_SET

# =====================================================================
# 5. EXPERIMENTAL CONDITIONS & REPRODUCIBILITY SEEDS
# =====================================================================
SEEDS: List[int] = [42, 123, 456, 789, 2026]

SIGNAL_STRENGTH_LEVELS: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]

EXPERIMENTAL_CONDITIONS: List[str] = [
    "Tabular-Only",
    "Signal_0.00",
    "Signal_0.25",
    "Signal_0.50",
    "Signal_0.75",
    "Signal_1.00",
]

# Partition Split Fractions (Strict Chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =====================================================================
# 6. MODEL HYPERPARAMETERS (EXPLICIT FIXED CONSTANTS)
# =====================================================================
CLASSIFIER_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

REGRESSOR_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
}

# =====================================================================
# 7. EVALUATION & STATISTICAL PROTOCOLS
# =====================================================================
THRESHOLD_GRID_POINTS = 100
PRIMARY_ALPHA = 0.05
