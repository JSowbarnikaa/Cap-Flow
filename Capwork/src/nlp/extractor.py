import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config.config import (
    COL_COMM_TEXT,
    COL_DISPUTE_SCORE,
    COL_DELAY_RISK_SCORE,
    COL_SENTIMENT_NEG,
    COL_URGENCY_SCORE,
    SENTIMENT_FEATURE_SET
)
from src.utils.logger import get_logger

logger = get_logger("NLPExtractor")

# Domain lexicons for B2B financial communication risk extraction
DISPUTE_KEYWORDS = {
    "dispute": 0.40, "disputing": 0.40, "pricing": 0.30, "mismatch": 0.35,
    "hold": 0.30, "audit": 0.25, "reconciled": 0.20, "discrepancy": 0.35,
    "refusing": 0.45, "suspended": 0.35, "blocked": 0.40, "exceptions": 0.25
}

DELAY_RISK_KEYWORDS = {
    "constraints": 0.35, "deferral": 0.35, "delayed": 0.35, "liquidity": 0.40,
    "cash flow": 0.35, "routine": 0.15, "queued": 0.20, "scheduled": 0.15,
    "timetable": 0.15, "upcoming": 0.15, "pending": 0.25, "batch": 0.20
}

NEGATIVE_SENTIMENT_KEYWORDS = {
    "halt": 0.35, "halting": 0.35, "refusing": 0.40, "mismatch": 0.30,
    "blocked": 0.35, "suspended": 0.35, "dispute": 0.35, "constraints": 0.30,
    "delayed": 0.30, "short-shipment": 0.35, "hold": 0.25
}

POSITIVE_SENTIMENT_KEYWORDS = {
    "prompt": 0.30, "approved": 0.30, "settlement": 0.25, "early": 0.35,
    "priority": 0.30, "swift": 0.30, "confirmed": 0.25, "expedited": 0.30,
    "discount": 0.25, "passed": 0.25, "cleared": 0.30, "verified": 0.25
}

URGENCY_KEYWORDS = {
    "critical": 0.45, "immediate": 0.40, "priority": 0.35, "swift": 0.30,
    "formal": 0.25, "expedited": 0.35, "halt": 0.35, "refusing": 0.35
}

def score_text_by_lexicon(text: str, keywords: Dict[str, float], base_score: float = 0.05) -> float:
    t_low = text.lower()
    score = base_score
    for kw, weight in keywords.items():
        if kw in t_low:
            score += weight
    return float(np.clip(score, 0.01, 0.99))

class NLPExtractor:
    """
    Extracts structured sentiment and dispute-risk features from B2B communication text.
    """
    def __init__(self):
        self.feature_names = SENTIMENT_FEATURE_SET

    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        dispute_scores = []
        delay_scores = []
        sentiment_neg_scores = []
        urgency_scores = []

        for text in texts:
            t = str(text) if text is not None else ""
            
            d_score = score_text_by_lexicon(t, DISPUTE_KEYWORDS, base_score=0.08)
            del_score = score_text_by_lexicon(t, DELAY_RISK_KEYWORDS, base_score=0.10)
            neg_score = score_text_by_lexicon(t, NEGATIVE_SENTIMENT_KEYWORDS, base_score=0.05)
            pos_score = score_text_by_lexicon(t, POSITIVE_SENTIMENT_KEYWORDS, base_score=0.05)
            
            # Net negative sentiment
            net_neg = np.clip(neg_score - (0.5 * pos_score) + 0.20, 0.01, 0.99)
            
            urg_score = score_text_by_lexicon(t, URGENCY_KEYWORDS, base_score=0.05)

            dispute_scores.append(d_score)
            delay_scores.append(del_score)
            sentiment_neg_scores.append(net_neg)
            urgency_scores.append(urg_score)

        df_feats = pd.DataFrame({
            COL_DISPUTE_SCORE: dispute_scores,
            COL_DELAY_RISK_SCORE: delay_scores,
            COL_SENTIMENT_NEG: sentiment_neg_scores,
            COL_URGENCY_SCORE: urgency_scores
        })
        return df_feats

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_COMM_TEXT not in df.columns:
            raise ValueError(f"Column '{COL_COMM_TEXT}' not found in dataframe!")
        df_out = df.copy()
        extracted = self.extract_features(df[COL_COMM_TEXT].tolist())
        for col in extracted.columns:
            df_out[col] = extracted[col].values
        return df_out
