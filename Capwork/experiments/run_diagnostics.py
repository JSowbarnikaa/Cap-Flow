import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    RAW_DATA_PATH,
    TABULAR_FEATURE_SET,
    SENTIMENT_FEATURE_SET,
    FULL_FEATURE_SET,
    COL_TARGET_CLS,
    COL_TARGET_REG,
    REPORTS_DIR
)
from src.data.loader import load_raw_dataset
from src.data.temporal_split import chronological_split
from src.features.pre_outcome_stats import CustomerHistoryTracker
from src.features.tabular_pipeline import TabularPipeline, engineer_base_tabular_columns
from src.features.text_generator import enrich_with_controlled_text
from src.nlp.extractor import NLPExtractor
from src.models.classifier import Stage1Classifier
from src.models.regressor import Stage2Regressor
from src.evaluation.metrics import evaluate_classification_metrics, evaluate_regression_metrics
from src.evaluation.threshold import optimize_classification_threshold
from src.evaluation.diagnostics import compute_feature_correlations, audit_train_test_generalization
from src.utils.logger import get_logger

logger = get_logger("RunDiagnostics")

def run_pre_training_diagnostics():
    logger.info("=" * 70)
    logger.info("  CAPFLOW: PRE-TRAINING DIAGNOSTICS & AUDIT SUITE")
    logger.info("=" * 70)

    # 1. Ingest Data
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    train_raw, val_raw, test_raw = chronological_split(raw_df)

    # 2. Extract Customer History (Strictly on Train)
    tracker = CustomerHistoryTracker()
    train_eng = tracker.fit_transform_train(train_raw)
    val_eng = tracker.transform(val_raw)
    test_eng = tracker.transform(test_raw)

    # 3. Base Tabular Feature Engineering
    train_eng = engineer_base_tabular_columns(train_eng)
    val_eng = engineer_base_tabular_columns(val_eng)
    test_eng = engineer_base_tabular_columns(test_eng)

    # 4. Controlled Signal-Injection & NLP Feature Extraction (Signal Strength = 1.0)
    logger.info("Generating text and extracting NLP features at lambda = 1.0 for diagnostics...")
    extractor = NLPExtractor()
    
    train_txt = enrich_with_controlled_text(train_eng, signal_strength=1.0, seed=42)
    val_txt = enrich_with_controlled_text(val_eng, signal_strength=1.0, seed=42)
    test_txt = enrich_with_controlled_text(test_eng, signal_strength=1.0, seed=42)

    train_full = extractor.transform_df(train_txt)
    val_full = extractor.transform_df(val_txt)
    test_full = extractor.transform_df(test_txt)

    # 5. Fit Tabular Pipeline strictly on train
    pipe = TabularPipeline(
        numerical_features=TABULAR_FEATURE_SET[:11],
        categorical_features=TABULAR_FEATURE_SET[11:]
    )
    X_train_tab = pipe.fit_transform(train_full)
    X_val_tab = pipe.transform(val_full)
    X_test_tab = pipe.transform(test_full)

    # Combine with NLP features
    for col in SENTIMENT_FEATURE_SET:
        X_train_tab[col] = train_full[col].values
        X_val_tab[col] = val_full[col].values
        X_test_tab[col] = test_full[col].values

    # 6. Feature-Target Pearson Correlation Analysis
    logger.info("[AUDIT 1] Computing Pearson Correlations with Targets...")
    train_corr_df = compute_feature_correlations(train_full, FULL_FEATURE_SET)
    test_corr_df = compute_feature_correlations(test_full, FULL_FEATURE_SET)

    merged_corr = pd.merge(
        train_corr_df, test_corr_df, on="feature",
        suffixes=("_train", "_test")
    )
    print("\n" + "=" * 90)
    print("  FEATURE-TARGET PEARSON CORRELATION MATRIX (TRAIN vs TEST)")
    print("=" * 90)
    print(merged_corr.to_string(index=False))

    # 7. Train-vs-Test Overfitting Diagnostic Check
    logger.info("\n[AUDIT 2] Running Overfitting Audit on Canonical Model...")
    clf = Stage1Classifier(seed=42)
    clf.fit(X_train_tab[FULL_FEATURE_SET], train_full[COL_TARGET_CLS])

    val_proba = clf.predict_proba(X_val_tab[FULL_FEATURE_SET])
    tau_opt = optimize_classification_threshold(val_full[COL_TARGET_CLS].values, val_proba)

    train_proba = clf.predict_proba(X_train_tab[FULL_FEATURE_SET])
    test_proba = clf.predict_proba(X_test_tab[FULL_FEATURE_SET])

    train_cls_metrics = evaluate_classification_metrics(train_full[COL_TARGET_CLS].values, train_proba, threshold=tau_opt)
    test_cls_metrics = evaluate_classification_metrics(test_full[COL_TARGET_CLS].values, test_proba, threshold=tau_opt)

    reg = Stage2Regressor(seed=42)
    reg.fit(X_train_tab[FULL_FEATURE_SET], train_full[COL_TARGET_REG], train_full[COL_TARGET_CLS])

    # Regressor evaluation on positive delays
    train_late_mask = train_full[COL_TARGET_REG] > 0
    test_late_mask = test_full[COL_TARGET_REG] > 0

    train_reg_pred = reg.predict(X_train_tab[FULL_FEATURE_SET][train_late_mask])
    test_reg_pred = reg.predict(X_test_tab[FULL_FEATURE_SET][test_late_mask])

    train_reg_metrics = evaluate_regression_metrics(train_full[COL_TARGET_REG][train_late_mask].values, train_reg_pred)
    test_reg_metrics = evaluate_regression_metrics(test_full[COL_TARGET_REG][test_late_mask].values, test_reg_pred)

    combined_train = {**train_cls_metrics, **train_reg_metrics}
    combined_test = {**test_cls_metrics, **test_reg_metrics}

    gen_audit_df = audit_train_test_generalization(combined_train, combined_test)
    print("\n" + "=" * 90)
    print("  TRAIN vs. TEST GENERALIZATION & OVERFITTING AUDIT")
    print("=" * 90)
    print(gen_audit_df.to_string(index=False))

    # Save to JSON report
    out_dict = {
        "correlations": merged_corr.to_dict(orient="records"),
        "generalization_audit": gen_audit_df.to_dict(orient="records"),
        "validation_threshold": tau_opt
    }
    with open(REPORTS_DIR / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    logger.info(f"\nSaved diagnostics summary to {REPORTS_DIR / 'diagnostics_summary.json'}")

if __name__ == "__main__":
    run_pre_training_diagnostics()
