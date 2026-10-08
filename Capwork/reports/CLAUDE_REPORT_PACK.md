# CapFlow: Controlled Signal-Injection Study & Dual-Protocol B2B Payment Risk Engine
## Comprehensive Project Dossier & Research Report Source

> **Purpose of this Document:** This single document contains the complete context, methodology, experimental findings, statistical significance analyses, multi-algorithm benchmarks, and epistemic limitations of the **CapFlow** project. Upload or paste this file directly into Claude to generate an academic research paper, technical whitepaper, or executive presentation.

---

## 1. Executive Summary & Core Research Questions

### Core Motivation
In B2B Accounts Receivable (AR), credit managers seek to predict:
1. **Stage 1 (Classification)**: Will an invoice be paid on-time or late?
2. **Stage 2 (Regression)**: If late, how many days overdue will the payment arrive?
3. **Multimodal Signal Value**: Does extracting pre-outcome communication sentiment and dispute risk from customer correspondence provide actionable predictive uplift over tabular payment history alone?

---

## 2. Headline Dataset Limitation & Concentration Analysis

> **CRITICAL CONCENTRATION METRIC:**
> - **Total Records**: 45,839 enterprise invoices (spanning Nov 2011 to Feb 2016).
> - **Total Unique Customers**: 69 enterprise accounts.
> - **Top Account (`Cust_Num` `5039221094`)**: Accounts for **10,219 invoices (22.29% of the entire dataset)**.
> - **Top 5 Mega-Accounts**: Comprise **36,400 invoices (79.41% of total dataset volume)**.
> - **Remaining 64 Sparse Accounts**: Comprise only **9,439 invoices (20.59% of dataset volume)**.

### Why Concentration Dictates Evaluation:
1. **Chronological Split (Known Customers Over Time)**:
   - 91.86% of test invoices belong to the top 5 repetitive accounts.
   - High test performance (**94.98% Accuracy, 0.957 AUC-ROC**) measures *behavioral payment continuity of known accounts*, not generalized B2B commercial intelligence.
2. **Customer-Stratified Holdout (Cold-Start Generalization)**:
   - When evaluating on genuinely new customer accounts (zero overlap between train and test), Accuracy drops to **69.27%** and RMSE increases to **35.65 days** (with high seed variance +/-29.46d driven by where the 10,219-invoice account lands).

---

## 3. Methodological Design & Anti-Leakage Protocol

### A. Chronological Partitioning (70% Train / 15% Val / 15% Test)
- **Train**: 32,087 records (Nov 2011 - Nov 2015)
- **Validation**: 6,875 records (Nov 2015 - Jan 2016)
- **Test**: 6,877 records (Jan 2016 - Feb 2016)

### B. Leak-Free Expanding Customer History Tracker
Historical statistics (`cust_hist_late_rate`, `cust_hist_delay_mean`, `cust_hist_delay_std`) are computed strictly sequentially:
- Only preceding closed invoices for that customer are aggregated.
- Future invoices and current invoice clearing dates are strictly excluded.
- Cold-start fallback: global median from training data is imputed when evaluating unseen accounts.

### C. Controlled Signal-Injection NLP Framework (lambda in [0.0, 1.0])
To rigorously test multimodal utility without confounding leaks:
S = lambda * S_pre + (1 - lambda) * epsilon,  epsilon ~ N(0, 1)
Where S_pre is derived strictly from pre-outcome posting metadata, and lambda=0.00 represents pure random noise control.

---

## 4. Multi-Algorithm Benchmark (Tabular-Only, Chronological Test Set)

| Algorithm | Train Accuracy | Test Accuracy | Generalization Gap | Test F1 | Test AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8124 | 0.7915 | -0.0209 | 0.8354 | 0.8521 |
| **Decision Tree (depth=6)** | 0.9240 | 0.9085 | -0.0155 | 0.9280 | 0.9210 |
| **Random Forest (n=200)** | 0.9785 | 0.9380 | -0.0405 | 0.9501 | 0.9485 |
| **CapFlow XGBoost (Ours)** | **0.9868** | **0.9498** | **-0.0370** | **0.9595** | **0.9569** |

---

## 5. Dual Evaluation Protocol: Known Accounts vs. Cold-Start Generalization

| Evaluation Protocol | Target Population | Accuracy | F1-Score | AUC-ROC | MAE (Days) | RMSE (Days) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chronological Split** | Known accounts over time (N_test=6,877) | **94.98%** +/- 0.06% | **0.9595** +/- 0.0005 | **0.9569** +/- 0.0106 | **6.05** +/- 1.03 d | **10.44** +/- 2.10 d |
| **Customer Holdout** | Genuinely new accounts (N_cust=11 held out) | **69.27%** +/- 9.35% | **0.7545** +/- 0.1206 | **0.7903** +/- 0.1238 | **10.04** +/- 9.37 d | **35.65** +/- 29.46 d |
| **Performance Penalty** | *Generalization Penalty* | **-25.71 pp** | **-0.2050** | **-0.1666** | **+3.99 d** | **+25.21 d (+241%)** |

---

## 6. Customer Segment Breakdown (Mega-Accounts vs. Sparse Accounts)

| Customer Segment | Invoices (N) | Vol Share (%) | Accounts (N) | Accuracy | F1-Score | AUC-ROC | MAE (Days) | RMSE (Days) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 5 Mega-Accounts** | 6,317 | 91.86% | 5 | 94.73% | 0.9594 | 0.9641 | **5.61 d** | **8.87 d** |
| **22 Sparse Accounts** | 560 | 8.14% | 22 | 97.14% | 0.9484 | **0.9989** | **15.70 d** | **20.30 d** |
| **Full Test Set** | 6,877 | 100.0% | 28 | 94.98% | 0.9595 | 0.9569 | **6.05 d** | **10.44 d** |

### Key Segmental Insights:
1. **Sparse Accuracy (97.14%) vs. Naive Baseline (71.96%)**:
   - The sparse segment has a 28.0% late rate. A naive majority-class baseline achieves 71.96%.
   - The model delivers a genuine **+25.18 pp uplift** by accurately identifying low-risk vs high-risk accounts.
2. **AUC-ROC (0.9989)**:
   - Reflects *between-customer separation* across 22 distinct accounts, not within-customer invoice ranking.

---

## 7. Multimodal / Sentiment Signal Findings & Mechanistic Audit

| Metric | Tabular-Only (5-Seed Mean) | CapFlow Multimodal (lambda=1.00) | Statistical Significance (p-value) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.9498 +/- 0.0006 | 0.9484 +/- 0.0031 | p_cluster = 0.4326 (Not Significant) |
| **F1-Score** | 0.9595 +/- 0.0005 | 0.9583 +/- 0.0026 | Not Significant |
| **AUC-ROC** | 0.9569 +/- 0.0106 | 0.9554 +/- 0.0039 | Not Significant |
| **PR-AUC** | 0.9565 +/- 0.0134 | 0.9543 +/- 0.0055 | Not Significant |
| **MAE (Days)** | 6.05 +/- 1.03 d | 5.29 +/- 0.58 d | Artifact (Subsampling & Tree Distraction) |
| **RMSE (Days)** | 10.44 +/- 2.10 d | 8.91 +/- 1.18 d | Artifact (Subsampling & Tree Distraction) |

### Mechanistic Explanation of the Apparent MAE Gain:
- **73% of the delta** is explained by `colsample_bytree = 0.8` pool-dilution (adding features dilutes selection of dominant terms). At `colsample_bytree = 1.0`, the delta drops from -2.68d to -0.73d.
- **27% residual delta** is tree split-distraction implicit regularization (p = 0.0021, Wilcoxon test).
- **Cluster-level Wilcoxon Test**: No signal condition outperforms the pure noise baseline (p > 0.32).

---

## 8. Prescriptive Decisioning Matrix

| Risk Tier | Stage 1 P(Late) | Stage 2 Estimated Delay (Days) | Recommended Commercial Directive |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Discount Eligible)** | < 0.20 | 0 | EXPAND_TERMS: Offer early-payment cash discounts (e.g. 2/10 Net 30). |
| **Tier 2 (Standard Flow)** | 0.20 - 0.50 | 0 | STANDARD_MONITOR: Routine automated billing delivery. |
| **Tier 3 (Operational Delay)** | >= 0.50 | 1 - 7 d | REMINDER_SCHEDULED: Proactive automated dunning reminders prior to due date. |
| **Tier 4 (High Delay / Default)** | >= 0.50 | > 7 d | HIGH_RISK_FACTORING: Expedited collections, factoring, or credit hold. |

---

## 9. Recommended Prompts for Claude

When pasting this pack into Claude, you can use any of these prompts:

1. **For an IEEE/ACM Academic Research Paper**:
   > "Using the attached comprehensive CapFlow project pack, write a full academic research paper following standard IEEE/ACM format (Abstract, Introduction, Related Work, Problem Formulation, Leak-Free Methodology, Dual-Protocol Experimental Results, Mechanistic Artifact Audit, Epistemic Limitations, and Conclusion). Maintain rigorous scientific tone and emphasize the dual-protocol evaluation and concentration limitations."

2. **For an Executive Technical Whitepaper**:
   > "Generate a structured, professional technical whitepaper for enterprise CFOs and AR leaders summarizing the CapFlow machine learning framework, key benchmark results, the prescriptive credit decision matrix, and lessons learned regarding NLP features and customer concentration."

3. **For a Project Presentation / Defense**:
   > "Create an 8-slide presentation script and slide deck outline with talking points defending this CapFlow machine learning project for a university capstone committee."
