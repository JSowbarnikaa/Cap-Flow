import numpy as np
import pandas as pd
from typing import List, Optional
from config.config import (
    COL_HIST_DELAY_MEAN,
    COL_HIST_LATE_RATE,
    COL_AMOUNT,
    COL_PAYMENT_TERMS_DAYS,
    COL_NO_OF_ORDERS,
    COL_COMM_TEXT
)
from src.utils.logger import get_logger

logger = get_logger("ControlledTextGenerator")

# Domain-grounded B2B enterprise communications
HIGH_RISK_COMMUNICATIONS = [
    "Invoice placed on hold pending vendor audit review of itemized charges.",
    "Disputing line-item pricing calculations. Halting outgoing disbursement until resolved.",
    "Critical purchase order mismatch detected; payment delayed pending departmental approval.",
    "Experiencing temporary cash flow constraints, formal request submitted for payment deferral.",
    "Refusing payment release until short-shipment inventory discrepancy is reconciled.",
    "Invoice payment suspended due to billing dispute regarding contractual rate schedule.",
    "Payment workflow blocked due to missing compliance documentation and audit review."
]

MEDIUM_RISK_COMMUNICATIONS = [
    "Invoice received and queued in standard accounts payable processing cycle.",
    "Payment logged for routine batch verification in upcoming monthly treasury run.",
    "Acknowledging receipt of billing statement; invoice currently under administrative review.",
    "Scheduled for disbursement in accordance with standard Net 30 corporate payment terms.",
    "Awaiting routine dual-signoff authorization prior to release of funds.",
    "Invoice recorded into financial ERP system; payment execution pending weekly queue.",
    "Processing payment per established supplier reconciliation timetable."
]

LOW_RISK_COMMUNICATIONS = [
    "Invoice approved for prompt settlement; funds scheduled for early disbursement.",
    "Payment advice generated and transmitted; remittance scheduled ahead of due date.",
    "Confirmed: invoice fully verified and authorized for immediate priority payment.",
    "Working capital allocation confirmed; payment cleared for swift release.",
    "Procurement and accounting signoffs completed; expedited remittance initiated.",
    "Early payment discount applied; disbursement finalized in current cycle.",
    "All audit checks passed with zero exceptions; payment in transit to vendor."
]

def compute_pre_outcome_risk_score(df: pd.DataFrame) -> np.ndarray:
    """
    Computes S_pre purely from pre-outcome, available-at-posting tabular features.
    Guaranteed zero target leakage.
    """
    # 1. Historical customer delay & late rate
    hist_mean = df[COL_HIST_DELAY_MEAN].fillna(0.0).values if COL_HIST_DELAY_MEAN in df.columns else np.zeros(len(df))
    hist_rate = df[COL_HIST_LATE_RATE].fillna(0.5).values if COL_HIST_LATE_RATE in df.columns else np.full(len(df), 0.5)
    
    # 2. Invoice amount magnitude (log scale)
    amt = df[COL_AMOUNT].fillna(0.0).values if COL_AMOUNT in df.columns else np.zeros(len(df))
    log_amt = np.log1p(np.maximum(0.0, amt))
    
    # 3. Payment terms duration
    terms = df[COL_PAYMENT_TERMS_DAYS].fillna(30.0).values if COL_PAYMENT_TERMS_DAYS in df.columns else np.full(len(df), 30.0)
    
    # 4. Order frequency
    orders = df[COL_NO_OF_ORDERS].fillna(10.0).values if COL_NO_OF_ORDERS in df.columns else np.full(len(df), 10.0)
    log_orders = np.log1p(np.maximum(0.0, orders))
    
    # Normalize components using robust clipping and standard scaling heuristics
    z_hist_mean = np.clip(hist_mean / 15.0, 0.0, 3.0)
    z_hist_rate = np.clip(hist_rate, 0.0, 1.0)
    z_amt = np.clip((log_amt - 8.0) / 3.0, -2.0, 2.0)
    z_terms = np.clip((terms - 30.0) / 30.0, -1.0, 2.0)
    z_orders = np.clip((log_orders - 3.0) / 2.0, -2.0, 2.0)
    
    # Composite linear combination
    raw_composite = (
        0.40 * z_hist_mean +
        0.35 * z_hist_rate +
        0.15 * z_amt +
        0.15 * z_terms -
        0.15 * z_orders
    )
    
    # Sigmoid mapping to ensure bounds in [0.05, 0.95]
    s_pre = 1.0 / (1.0 + np.exp(-1.8 * (raw_composite - 0.5)))
    return np.clip(s_pre, 0.01, 0.99)

def generate_controlled_communications(
    df: pd.DataFrame,
    signal_strength: float = 1.0,
    seed: int = 42
) -> List[str]:
    """
    Generates synthetic communication text with controlled signal injection (lambda).
    lambda = 0.0 -> Pure noise (uncorrelated random communications)
    lambda = 1.0 -> Pure pre-outcome risk signal
    """
    assert 0.0 <= signal_strength <= 1.0, "signal_strength must be between 0.0 and 1.0"
    rng = np.random.RandomState(seed)
    n = len(df)
    
    # 1. Compute pre-outcome risk score
    s_pre = compute_pre_outcome_risk_score(df)
    
    # 2. Generate stochastic noise
    noise = rng.uniform(0.0, 1.0, size=n)
    
    # 3. Controlled signal blend: S_injected = lambda * S_pre + (1 - lambda) * noise
    s_injected = signal_strength * s_pre + (1.0 - signal_strength) * noise
    
    texts = []
    for i in range(n):
        score = s_injected[i]
        # Stochastic category selection weighted by score
        p_high = np.clip(score ** 1.8, 0.02, 0.90)
        p_low = np.clip((1.0 - score) ** 1.8, 0.02, 0.90)
        p_med = max(0.01, 1.0 - p_high - p_low)
        total_p = p_high + p_med + p_low
        p_vec = [p_low / total_p, p_med / total_p, p_high / total_p]
        
        category = rng.choice(["low", "medium", "high"], p=p_vec)
        if category == "high":
            texts.append(rng.choice(HIGH_RISK_COMMUNICATIONS))
        elif category == "medium":
            texts.append(rng.choice(MEDIUM_RISK_COMMUNICATIONS))
        else:
            texts.append(rng.choice(LOW_RISK_COMMUNICATIONS))
            
    return texts

def enrich_with_controlled_text(
    df: pd.DataFrame,
    signal_strength: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    df_out = df.copy()
    df_out[COL_COMM_TEXT] = generate_controlled_communications(df, signal_strength=signal_strength, seed=seed)
    return df_out
