import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    RAW_DATA_PATH,
    SEEDS,
    EXPERIMENTAL_CONDITIONS,
    TABULAR_FEATURE_SET,
    SENTIMENT_FEATURE_SET,
    FULL_FEATURE_SET,
    COL_TARGET_CLS,
    COL_TARGET_REG,
    COL_CUSTOMER_ID,
    REPORTS_DIR
)
from src.data.loader import load_raw_dataset
from src.features.pre_outcome_stats import CustomerHistoryTracker
from src.features.tabular_pipeline import TabularPipeline, engineer_base_tabular_columns
from src.features.text_generator import enrich_with_controlled_text
from src.nlp.extractor import NLPExtractor
from src.models.classifier import Stage1Classifier
from src.models.regressor import Stage2Regressor
from src.evaluation.metrics import evaluate_classification_metrics, evaluate_regression_metrics
from src.evaluation.threshold import optimize_classification_threshold
from src.utils.logger import get_logger, set_seed

logger = get_logger("CustomerHoldoutSweep")

def customer_group_split(df: pd.DataFrame, seed: int = 42):
    """
    Splits dataset strictly by customer ID: 70% train customers, 15% val customers, 15% test customers.
    Guarantees 0 customer overlap between train, val, and test partitions (Cold-Start evaluation).
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df[COL_CUSTOMER_ID]))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[COL_CUSTOMER_ID]))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def run_customer_holdout_sweep():
    logger.info("=" * 80)
    logger.info("  CAPFLOW: CUSTOMER-STRATIFIED HOLDOUT EVALUATION (COLD-START BENCHMARK)")
    logger.info("=" * 80)

    raw_df = load_raw_dataset(RAW_DATA_PATH)
    
    results = {
        cond: {
            "seeds": {},
            "mean_metrics": {},
            "std_metrics": {}
        }
        for cond in EXPERIMENTAL_CONDITIONS
    }

    extractor = NLPExtractor()

    for cond in EXPERIMENTAL_CONDITIONS:
        logger.info(f"\n>>> Running Condition: {cond} across {len(SEEDS)} seeds (Customer-Holdout)...")
        is_tabular_only = (cond == "Tabular-Only")
        sig_level = 0.0 if is_tabular_only else float(cond.split("_")[1])
        cond_seed_metrics = []

        for seed in SEEDS:
            set_seed(seed)
            # Split strictly by customer accounts for this seed
            train_raw, val_raw, test_raw = customer_group_split(raw_df, seed=seed)

            # Customer History Tracker strictly on train (Validation & Test will receive global train median prior)
            tracker = CustomerHistoryTracker()
            train_eng = tracker.fit_transform_train(train_raw)
            val_eng = tracker.transform(val_raw)
            test_eng = tracker.transform(test_raw)

            train_base = engineer_base_tabular_columns(train_eng)
            val_base = engineer_base_tabular_columns(val_eng)
            test_base = engineer_base_tabular_columns(test_eng)

            if is_tabular_only:
                train_df = train_base.copy()
                val_df = val_base.copy()
                test_df = test_base.copy()
                active_features = TABULAR_FEATURE_SET
            else:
                train_txt = enrich_with_controlled_text(train_base, signal_strength=sig_level, seed=seed)
                val_txt = enrich_with_controlled_text(val_base, signal_strength=sig_level, seed=seed)
                test_txt = enrich_with_controlled_text(test_base, signal_strength=sig_level, seed=seed)

                train_df = extractor.transform_df(train_txt)
                val_df = extractor.transform_df(val_txt)
                test_df = extractor.transform_df(test_txt)
                active_features = FULL_FEATURE_SET

            # Fit Tabular Pipeline strictly on train
            pipe = TabularPipeline()
            X_train = pipe.fit_transform(train_df)
            X_val = pipe.transform(val_df)
            X_test = pipe.transform(test_df)

            if not is_tabular_only:
                for col in SENTIMENT_FEATURE_SET:
                    X_train[col] = train_df[col].values
                    X_val[col] = val_df[col].values
                    X_test[col] = test_df[col].values

            # Fit Classifier
            clf = Stage1Classifier(seed=seed)
            clf.fit(X_train[active_features], train_df[COL_TARGET_CLS])

            val_proba = clf.predict_proba(X_val[active_features])
            tau_opt = optimize_classification_threshold(val_df[COL_TARGET_CLS].values, val_proba)

            test_proba = clf.predict_proba(X_test[active_features])
            test_cls_metrics = evaluate_classification_metrics(
                test_df[COL_TARGET_CLS].values, test_proba, threshold=tau_opt
            )

            # Fit Regressor
            reg = Stage2Regressor(seed=seed)
            reg.fit(X_train[active_features], train_df[COL_TARGET_REG], train_df[COL_TARGET_CLS])

            test_late_mask = test_df[COL_TARGET_REG] > 0
            test_reg_pred = np.zeros(len(test_df))
            if test_late_mask.sum() > 0:
                test_reg_pred[test_late_mask] = reg.predict(X_test[active_features][test_late_mask])

            test_reg_metrics = evaluate_regression_metrics(
                test_df[COL_TARGET_REG][test_late_mask].values,
                test_reg_pred[test_late_mask]
            )

            combined_metrics = {**test_cls_metrics, **test_reg_metrics}
            results[cond]["seeds"][str(seed)] = combined_metrics
            cond_seed_metrics.append(combined_metrics)

        # Aggregate metrics across seeds
        metric_keys = list(cond_seed_metrics[0].keys())
        for k in metric_keys:
            vals = [m[k] for m in cond_seed_metrics]
            results[cond]["mean_metrics"][k] = float(np.mean(vals))
            results[cond]["std_metrics"][k] = float(np.std(vals, ddof=1))

    # Formatted Summary
    summary_rows = []
    for cond in EXPERIMENTAL_CONDITIONS:
        m = results[cond]["mean_metrics"]
        s = results[cond]["std_metrics"]
        summary_rows.append({
            "Condition": cond,
            "AUC-ROC": f"{m['auc_roc']:.4f} ± {s['auc_roc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f} ± {s['pr_auc']:.4f}",
            "Accuracy": f"{m['accuracy']:.4f} ± {s['accuracy']:.4f}",
            "F1-Score": f"{m['f1']:.4f} ± {s['f1']:.4f}",
            "Precision": f"{m['precision']:.4f} ± {s['precision']:.4f}",
            "Recall": f"{m['recall']:.4f} ± {s['recall']:.4f}",
            "MAE (Days)": f"{m['mae']:.4f} ± {s['mae']:.4f}",
            "RMSE (Days)": f"{m['rmse']:.4f} ± {s['rmse']:.4f}",
            "R2": f"{m['r2']:.4f} ± {s['r2']:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 110)
    print("  CUSTOMER-STRATIFIED HOLDOUT EVALUATION (COLD-START) BENCHMARK SUMMARY")
    print("=" * 110)
    print(summary_df.to_string(index=False))

    with open(REPORTS_DIR / "customer_holdout_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary_table": summary_df.to_dict(orient="records"), "full_results": results}, f, indent=2)

if __name__ == "__main__":
    run_customer_holdout_sweep()
