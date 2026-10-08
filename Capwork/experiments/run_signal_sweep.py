import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    RAW_DATA_PATH,
    SEEDS,
    SIGNAL_STRENGTH_LEVELS,
    EXPERIMENTAL_CONDITIONS,
    TABULAR_FEATURE_SET,
    SENTIMENT_FEATURE_SET,
    FULL_FEATURE_SET,
    COL_TARGET_CLS,
    COL_TARGET_REG,
    COL_CUSTOMER_ID,
    REPORTS_DIR,
    FIGURES_DIR
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
from src.evaluation.significance import paired_t_test, evaluate_customer_cluster_significance
from src.utils.logger import get_logger, set_seed

logger = get_logger("SignalSweepExperiment")

def run_experiment_sweep():
    logger.info("=" * 80)
    logger.info("  CAPFLOW: CONTROLLED SIGNAL-INJECTION MULTI-SEED EXPERIMENTAL SUITE")
    logger.info(f"  Conditions: {EXPERIMENTAL_CONDITIONS}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info("=" * 80)

    # 1. Ingest Raw Dataset & Chronological Split
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    train_raw, val_raw, test_raw = chronological_split(raw_df)

    # 2. Extract Customer Historical Stats strictly on Train
    tracker = CustomerHistoryTracker()
    train_eng = tracker.fit_transform_train(train_raw)
    val_eng = tracker.transform(val_raw)
    test_eng = tracker.transform(test_raw)

    # 3. Base Tabular Feature Engineering (deterministic & single-invoice invariant)
    train_base = engineer_base_tabular_columns(train_eng)
    val_base = engineer_base_tabular_columns(val_eng)
    test_base = engineer_base_tabular_columns(test_eng)

    extractor = NLPExtractor()

    # Results dictionary structure
    results = {
        cond: {
            "seeds": {},
            "mean_metrics": {},
            "std_metrics": {}
        }
        for cond in EXPERIMENTAL_CONDITIONS
    }

    # Store aggregated prediction vectors per condition for cluster testing
    pooled_test_preds = {}

    for cond in EXPERIMENTAL_CONDITIONS:
        logger.info(f"\n>>> Running Condition: {cond} across {len(SEEDS)} seeds...")
        
        cond_seed_metrics = []
        cond_preds_list = []

        is_tabular_only = (cond == "Tabular-Only")
        if not is_tabular_only:
            sig_level = float(cond.split("_")[1])
        else:
            sig_level = 0.0

        for seed in SEEDS:
            set_seed(seed)
            
            # Step A: Text Generation & NLP Extraction (if applicable)
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

            # Step B: Fit Tabular Scaler & Categorical Encoders strictly on Train
            pipe = TabularPipeline(
                numerical_features=TABULAR_FEATURE_SET[:11],
                categorical_features=TABULAR_FEATURE_SET[11:]
            )
            X_train = pipe.fit_transform(train_df)
            X_val = pipe.transform(val_df)
            X_test = pipe.transform(test_df)

            # Append NLP features if active
            if not is_tabular_only:
                for col in SENTIMENT_FEATURE_SET:
                    X_train[col] = train_df[col].values
                    X_val[col] = val_df[col].values
                    X_test[col] = test_df[col].values

            # Step C: Train Stage 1 Classifier
            clf = Stage1Classifier(seed=seed)
            clf.fit(X_train[active_features], train_df[COL_TARGET_CLS])

            # Optimize threshold on Validation set
            val_proba = clf.predict_proba(X_val[active_features])
            tau_opt = optimize_classification_threshold(val_df[COL_TARGET_CLS].values, val_proba)

            # Evaluate Stage 1 on Test set
            test_proba = clf.predict_proba(X_test[active_features])
            test_cls_metrics = evaluate_classification_metrics(
                test_df[COL_TARGET_CLS].values, test_proba, threshold=tau_opt
            )

            # Step D: Train Stage 2 Regressor (on positive delays)
            reg = Stage2Regressor(seed=seed)
            reg.fit(X_train[active_features], train_df[COL_TARGET_REG], train_df[COL_TARGET_CLS])

            # Evaluate Stage 2 on Test late records
            test_late_mask = test_df[COL_TARGET_REG] > 0
            test_reg_pred = np.zeros(len(test_df))
            test_reg_pred[test_late_mask] = reg.predict(X_test[active_features][test_late_mask])

            test_reg_metrics = evaluate_regression_metrics(
                test_df[COL_TARGET_REG][test_late_mask].values,
                test_reg_pred[test_late_mask]
            )

            combined_metrics = {**test_cls_metrics, **test_reg_metrics}
            results[cond]["seeds"][str(seed)] = combined_metrics
            cond_seed_metrics.append(combined_metrics)

            # Save prediction vectors
            test_cls_pred = (test_proba >= tau_opt).astype(int)
            cond_preds_list.append({
                "y_true_cls": test_df[COL_TARGET_CLS].values,
                "y_pred_cls": test_cls_pred,
                "y_proba_cls": test_proba,
                "y_true_reg": test_df[COL_TARGET_REG].values,
                "y_pred_reg": test_reg_pred
            })

        # Aggregate across seeds
        metric_keys = list(cond_seed_metrics[0].keys())
        for k in metric_keys:
            vals = [m[k] for m in cond_seed_metrics]
            results[cond]["mean_metrics"][k] = float(np.mean(vals))
            results[cond]["std_metrics"][k] = float(np.std(vals, ddof=1))

        # Store representative / mean prediction dict
        pooled_test_preds[cond] = cond_preds_list[0]

    # 4. Statistical Significance Testing
    logger.info("\n" + "=" * 80)
    logger.info("  STATISTICAL SIGNIFICANCE TESTING (p_cluster & p_seed)")
    logger.info("=" * 80)

    sig_tests = {}
    base_cond_noise = "Signal_0.00"
    base_cond_tab = "Tabular-Only"

    for cond in EXPERIMENTAL_CONDITIONS:
        if cond in [base_cond_noise, base_cond_tab]:
            continue

        sig_tests[cond] = {
            "vs_Signal_0.00": {},
            "vs_Tabular_Only": {}
        }

        # Comparisons vs Signal_0.00 and vs Tabular-Only
        for base_name, base_key in [("vs_Signal_0.00", base_cond_noise), ("vs_Tabular_Only", base_cond_tab)]:
            # 1. Multi-seed paired t-tests
            seed_p_vals = {}
            for metric in ["auc_roc", "pr_auc", "accuracy", "f1", "mae", "rmse"]:
                cond_vals = [results[cond]["seeds"][str(s)][metric] for s in SEEDS]
                base_vals = [results[base_key]["seeds"][str(s)][metric] for s in SEEDS]
                t_stat, p_val = paired_t_test(cond_vals, base_vals)
                seed_p_vals[metric] = {"t_stat": t_stat, "p_seed": p_val}

            # 2. Customer-cluster aggregated paired test (authoritative p_cluster)
            cluster_p_vals = evaluate_customer_cluster_significance(
                test_base, pooled_test_preds[cond], pooled_test_preds[base_key]
            )

            sig_tests[cond][base_name] = {
                "seed_tests": seed_p_vals,
                "cluster_tests": cluster_p_vals
            }

    # 5. Generate Formatted Output Tables
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
    print("  CAPFLOW CONTROLLED SIGNAL-INJECTION EXPERIMENT: 5-SEED BENCHMARK SUMMARY")
    print("=" * 110)
    print(summary_df.to_string(index=False))

    # Save outputs
    final_output = {
        "summary_table": summary_df.to_dict(orient="records"),
        "full_results": results,
        "significance_testing": sig_tests,
        "n_test_customers": 28,
        "n_test_invoices": len(test_base)
    }

    with open(REPORTS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    # 6. Generate Figures
    generate_summary_figures(results)
    logger.info(f"\nExperiment complete! Saved reports to {REPORTS_DIR}")

def generate_summary_figures(results: dict):
    sig_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    sig_conds = [f"Signal_{s:.2f}" for s in sig_levels]
    
    auc_means = [results[c]["mean_metrics"]["auc_roc"] for c in sig_conds]
    auc_stds = [results[c]["std_metrics"]["auc_roc"] for c in sig_conds]
    f1_means = [results[c]["mean_metrics"]["f1"] for c in sig_conds]
    f1_stds = [results[c]["std_metrics"]["f1"] for c in sig_conds]
    mae_means = [results[c]["mean_metrics"]["mae"] for c in sig_conds]
    mae_stds = [results[c]["std_metrics"]["mae"] for c in sig_conds]

    tab_auc = results["Tabular-Only"]["mean_metrics"]["auc_roc"]
    tab_f1 = results["Tabular-Only"]["mean_metrics"]["f1"]
    tab_mae = results["Tabular-Only"]["mean_metrics"]["mae"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. AUC-ROC Curve
    axes[0].errorbar(sig_levels, auc_means, yerr=auc_stds, fmt="-o", color="#1f77b4", label="Signal Sweep", capsize=4, lw=2)
    axes[0].axhline(tab_auc, color="red", linestyle="--", label=f"Tabular-Only ({tab_auc:.4f})")
    axes[0].set_title("AUC-ROC vs. Signal Strength (λ)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Signal Strength (λ)", fontsize=11)
    axes[0].set_ylabel("AUC-ROC", fontsize=11)
    axes[0].legend(loc="lower right")

    # 2. F1 Curve
    axes[1].errorbar(sig_levels, f1_means, yerr=f1_stds, fmt="-s", color="#2ca02c", label="Signal Sweep", capsize=4, lw=2)
    axes[1].axhline(tab_f1, color="red", linestyle="--", label=f"Tabular-Only ({tab_f1:.4f})")
    axes[1].set_title("F1-Score vs. Signal Strength (λ)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Signal Strength (λ)", fontsize=11)
    axes[1].set_ylabel("F1-Score", fontsize=11)
    axes[1].legend(loc="lower right")

    # 3. MAE Curve
    axes[2].errorbar(sig_levels, mae_means, yerr=mae_stds, fmt="-^", color="#d62728", label="Signal Sweep", capsize=4, lw=2)
    axes[2].axhline(tab_mae, color="black", linestyle="--", label=f"Tabular-Only ({tab_mae:.4f})")
    axes[2].set_title("MAE (Days Late) vs. Signal Strength (λ)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Signal Strength (λ)", fontsize=11)
    axes[2].set_ylabel("MAE (Days)", fontsize=11)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "signal_strength_performance_curves.png", dpi=300)
    plt.close()
    logger.info(f"Saved performance figure to {FIGURES_DIR / 'signal_strength_performance_curves.png'}")

if __name__ == "__main__":
    run_experiment_sweep()
