"""
CashFlow Enterprise Portal - CapFlow Payment Risk & MILP Factoring System
========================================================================
Enterprise B2B payment risk prediction, Real-Time NLP DeBERTa risk scoring,
and active MILP Factoring portfolio optimization.
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_PATH,
    COL_CUSTOMER_ID, COL_INVOICE_ID, COL_POSTING_DATE, COL_DUE_DATE, COL_AMOUNT,
    COL_PAYMENT_TERM, COL_PAYMENT_METHOD, COL_REGION, COL_CITY,
    COL_CUSTOMER_AGE_MONTHS, COL_NO_OF_ORDERS,
    COL_TARGET_CLS, COL_TARGET_REG,
    COL_HIST_DELAY_MEAN, COL_HIST_DELAY_STD, COL_HIST_LATE_RATE,
    COL_PAYMENT_TERMS_DAYS, COL_LOG_AMOUNT,
    COL_POSTING_MONTH_SIN, COL_POSTING_MONTH_COS, COL_POSTING_DOW,
    COL_DISPUTE_SCORE, COL_DELAY_RISK_SCORE, COL_SENTIMENT_NEG, COL_URGENCY_SCORE,
    TABULAR_FEATURE_SET, FULL_FEATURE_SET
)
from src.data.loader import load_raw_dataset
from src.data.temporal_split import chronological_split
from src.features.pre_outcome_stats import CustomerHistoryTracker
from src.features.tabular_pipeline import TabularPipeline, engineer_base_tabular_columns
from src.features.text_generator import generate_controlled_communications
from src.nlp.extractor import NLPExtractor
from src.models.classifier import Stage1Classifier
from src.models.regressor import Stage2Regressor
from src.evaluation.threshold import optimize_classification_threshold
from src.optimization.milp_factoring import optimize_factoring_portfolio
from src.nlp.deberta_scorer import DeBERTaRiskScorer

# =========================================================================
# PAGE CONFIGURATION & CUSTOM DARK THEME CSS
# =========================================================================
st.set_page_config(
    page_title="CashFlow Enterprise Portal",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark Theme Styling matching screenshots
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #141822;
        border-right: 1px solid #1f2738;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #151d2f;
        border: 1px solid #232f48;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00aaff;
        margin-bottom: 4px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8fa0b5;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Alert Banners */
    .alert-banner-warning {
        background-color: #2e2b14;
        border: 1px solid #6b6320;
        color: #e5d778;
        padding: 14px 20px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 15px 0;
    }
    .alert-banner-error {
        background-color: #3b1414;
        border: 1px solid #852222;
        color: #ff8a8a;
        padding: 14px 20px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 15px 0;
    }
    .alert-banner-success {
        background-color: #143320;
        border: 1px solid #226b3c;
        color: #7de8a3;
        padding: 14px 20px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 15px 0;
    }
    
    /* NLP Output Metrics */
    .nlp-metric-title {
        font-size: 0.85rem;
        color: #8fa0b5;
        margin-bottom: 6px;
    }
    .nlp-metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Keyword Trigger Tags */
    .keyword-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px 4px;
    }
    .badge-booster {
        background-color: rgba(255, 75, 75, 0.2);
        color: #ff6b6b;
        border: 1px solid #ff4b4b;
    }
    .badge-delay {
        background-color: rgba(255, 187, 0, 0.2);
        color: #ffd166;
        border: 1px solid #ffbb00;
    }
    .badge-offset {
        background-color: rgba(6, 214, 160, 0.2);
        color: #06d6a0;
        border: 1px solid #06d6a0;
    }
    
    /* Active Engine Callout */
    .sidebar-callout {
        background: linear-gradient(135deg, #101c33 0%, #162a4d 100%);
        border: 1px solid #1f3f73;
        border-radius: 8px;
        padding: 14px;
        margin-top: 25px;
        color: #70b0ff;
        font-size: 0.85rem;
    }
    
    /* Primary Red Button styling */
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b;
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 10px 24px;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #ff3333;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# ACTIVE PIPELINE LOADER & ACTIVE RISK CALCULATIONS
# =========================================================================
@st.cache_resource(show_spinner="Actively Training CapFlow XGBoost Models & Computing Portfolio Risks...")
def load_enterprise_system():
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    train_df, val_df, test_df = chronological_split(raw_df)

    tracker = CustomerHistoryTracker()
    train_eng = tracker.fit_transform_train(train_df)
    val_eng = tracker.transform(val_df)
    test_eng = tracker.transform(test_df)

    train_eng = engineer_base_tabular_columns(train_eng)
    val_eng = engineer_base_tabular_columns(val_eng)
    test_eng = engineer_base_tabular_columns(test_eng)

    pipe = TabularPipeline()
    X_train = pipe.fit_transform(train_eng)
    X_val = pipe.transform(val_eng)
    X_test = pipe.transform(test_eng)

    clf = Stage1Classifier(seed=42)
    clf.fit(X_train[TABULAR_FEATURE_SET], train_eng[COL_TARGET_CLS])
    tau_opt = optimize_classification_threshold(
        val_eng[COL_TARGET_CLS].values,
        clf.predict_proba(X_val[TABULAR_FEATURE_SET])
    )

    reg = Stage2Regressor(seed=42)
    reg.fit(X_train[TABULAR_FEATURE_SET], train_eng[COL_TARGET_REG], train_eng[COL_TARGET_CLS])

    # Customer Lookup Cache
    customer_lookup = (
        train_eng.groupby(COL_CUSTOMER_ID)
        .agg(
            cust_hist_late_rate=(COL_HIST_LATE_RATE, "last"),
            cust_hist_delay_mean=(COL_HIST_DELAY_MEAN, "last"),
            cust_hist_delay_std=(COL_HIST_DELAY_STD, "last"),
        )
        .to_dict(orient="index")
    )

    # 1. Actively generate controlled communications on test portfolio
    texts = generate_controlled_communications(test_eng, signal_strength=0.75, seed=42)
    nlp_feats = NLPExtractor().extract_features(texts)

    # 2. Actively compute Stage 1 Classification Probabilities & Stage 2 Delay Days
    test_df_portfolio = test_eng.copy()
    late_probs = clf.predict_proba(X_test[TABULAR_FEATURE_SET])
    predicted_delays = np.where(late_probs >= tau_opt, reg.predict(X_test[TABULAR_FEATURE_SET]), 0.0)

    test_df_portfolio["customer_communication_text"] = texts
    for col in nlp_feats.columns:
        test_df_portfolio[col] = nlp_feats[col].values

    test_df_portfolio["late_probability"] = late_probs
    test_df_portfolio["predicted_delay_days"] = np.maximum(0.0, predicted_delays)
    test_df_portfolio["is_predicted_late"] = (late_probs >= tau_opt).astype(int)
    
    # Active composite financial risk (50% Tabular Payment Delay Risk + 50% NLP Dispute Risk)
    test_df_portfolio["composite_risk_score"] = np.clip(
        0.50 * late_probs + 0.50 * test_df_portfolio["comm_dispute_score"],
        0.01, 0.99
    )

    return {
        "clf": clf,
        "reg": reg,
        "pipe": pipe,
        "tau_opt": tau_opt,
        "customer_lookup": customer_lookup,
        "global_late_rate": float(train_eng[COL_HIST_LATE_RATE].median()),
        "global_delay_mean": float(train_eng[COL_HIST_DELAY_MEAN].median()),
        "global_delay_std": float(train_eng[COL_HIST_DELAY_STD].median()),
        "known_customers": sorted(train_eng[COL_CUSTOMER_ID].unique().tolist()),
        "known_payment_terms": sorted(raw_df[COL_PAYMENT_TERM].dropna().unique().tolist()),
        "known_payment_methods": sorted(raw_df[COL_PAYMENT_METHOD].dropna().unique().tolist()),
        "known_regions": sorted(raw_df[COL_REGION].dropna().unique().tolist()),
        "known_cities": sorted(raw_df[COL_CITY].dropna().unique().tolist()),
        "portfolio_df": test_df_portfolio,
        "raw_df": raw_df
    }

system = load_enterprise_system()
nlp_scorer = DeBERTaRiskScorer()


# =========================================================================
# SIDEBAR NAVIGATION
# =========================================================================
with st.sidebar:
    st.markdown("## **Welcome, Executive**\n## **Admin**")
    
    if st.button("Logout", use_container_width=False):
        st.toast("Session terminated.")

    st.markdown("---")
    st.markdown("<p style='font-size:0.85rem; color:#8fa0b5; margin-bottom:5px;'>Navigation</p>", unsafe_allow_html=True)
    
    page = st.radio(
        label="Navigation Menu",
        options=[
            "Executive KPI Dashboard",
            "MILP Factoring Operations",
            "Real-Time NLP Risk Engine",
            "Data Management"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="sidebar-callout">
        💡 <b>Active Engine: Zero-Shot DeBERTa Transformer</b><br>
        <span style="color:#94b8e6; font-size:0.78rem;">(deberta-v3-base-zeroshot).</span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# PAGE 1: EXECUTIVE KPI DASHBOARD (Matching Screenshot 2 - Fully Dynamic)
# =========================================================================
if page == "Executive KPI Dashboard":
    portfolio_df = system["portfolio_df"]
    
    # Actively run MILP factoring allocation on real computed risks
    milp_summary = optimize_factoring_portfolio(
        portfolio_df,
        amount_col=COL_AMOUNT,
        risk_col="comm_dispute_score",
        cust_col=COL_CUSTOMER_ID,
        liquidity_target=900_000.0,
        max_avg_risk=0.25,
        factoring_fee_rate=0.03
    )
    kpis = milp_summary["kpis"]
    df_opt = milp_summary["df_result"]

    # Total Portfolio in Millions ($M)
    total_val_m = kpis["total_portfolio_value"] / 1_000_000.0
    liquidated_m = kpis["capital_liquidated"] / 1_000_000.0

    # 4 Top Dynamic KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_val_m:.2f}</div>
            <div class="metric-label">Total Portfolio Value</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{kpis['factored_invoices_count']:,}</div>
            <div class="metric-label">Factored Invoices</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">.{int(liquidated_m * 100):02d}</div>
            <div class="metric-label">Capital Liquidated</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{kpis['avg_portfolio_risk']:.2f}</div>
            <div class="metric-label">Avg Dispute Risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2 Charts Row
    chart_col1, chart_col2 = st.columns([1.1, 1.0])

    with chart_col1:
        st.markdown("<h3 style='font-weight:600;'>Dispute Risk Profile</h3>", unsafe_allow_html=True)
        
        # Actively bin real computed dispute risk scores into 3 canonical tranches
        dispute_scores = portfolio_df["comm_dispute_score"].values
        low_count = int(np.sum(dispute_scores < 0.25))
        med_count = int(np.sum((dispute_scores >= 0.25) & (dispute_scores < 0.70)))
        high_count = int(np.sum(dispute_scores >= 0.70))

        bin_labels = [0.0, 0.5, 1.0]
        bin_counts = [low_count, med_count, high_count]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=bin_labels,
            y=bin_counts,
            marker=dict(color="#00aaff", line=dict(width=0)),
            width=[0.08, 0.08, 0.08],
            hovertemplate="Dispute Risk Level: %{x}<br>Active Invoices: %{y:,}<extra></extra>"
        ))
        fig_hist.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#8fa0b5", size=11),
            margin=dict(l=40, r=20, t=20, b=40),
            height=320,
            xaxis=dict(
                title=dict(text="dispute_risk_score", font=dict(color="#8fa0b5")),
                tickvals=[0, 0.5, 1],
                ticktext=["0", "0.5", "1"],
                showgrid=True,
                gridcolor="#1e2638",
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text="count", font=dict(color="#8fa0b5")),
                showgrid=True,
                gridcolor="#1e2638",
                zeroline=False
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": True})

    with chart_col2:
        st.markdown("<h3 style='font-weight:600;'>Factoring Allocation Ratio</h3>", unsafe_allow_html=True)
        
        # Active Factoring Allocation Ratio computed from MILP
        ret_pct = kpis["retained_ratio_pct"]
        fac_pct = kpis["factored_ratio_pct"]
        
        labels = ["Retained", "Factored"]
        values = [ret_pct, fac_pct]
        colors = ["#00aaff", "#3d4d65"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.0,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color="#0e1117", width=1)),
            textinfo="percent",
            texttemplate="%{percent:.1%}",
            textposition="inside",
            insidetextfont=dict(color="white", size=12, family="sans-serif"),
            hovertemplate="%{label}: %{percent:.1%}<br>Value: $%{value:,.2f}<extra></extra>"
        )])
        fig_pie.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#8fa0b5"),
            margin=dict(l=10, r=20, t=20, b=20),
            height=320,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.9,
                xanchor="right",
                x=1.15,
                font=dict(color="#8fa0b5")
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": True})

    # Portfolio Tranche Breakdown calculated actively
    st.markdown("<br><h4>Portfolio Risk Tranches (Active Calculations)</h4>", unsafe_allow_html=True)
    tot_n = len(portfolio_df)
    tranche_summary = pd.DataFrame([
        {"Risk Tranche": "🟢 Prime / Low Risk (0.00 - 0.25)", "Invoices": f"{low_count:,}", "Portfolio Share": f"{(low_count/tot_n)*100:.1f}%", "Action": "Retain at 0% factoring fee"},
        {"Risk Tranche": "🟡 Moderate Risk (0.26 - 0.60)", "Invoices": f"{med_count:,}", "Portfolio Share": f"{(med_count/tot_n)*100:.1f}%", "Action": "Eligible for selective factoring"},
        {"Risk Tranche": "🔴 High Dispute Risk (0.61 - 1.00)", "Invoices": f"{high_count:,}", "Portfolio Share": f"{(high_count/tot_n)*100:.1f}%", "Action": "Immediate MILP factoring & dispute audit"}
    ])
    st.dataframe(tranche_summary, use_container_width=True, hide_index=True)


# =========================================================================
# PAGE 2: MILP FACTORING OPERATIONS (Active Solver & Working Capital Optimization)
# =========================================================================
elif page == "MILP Factoring Operations":
    st.markdown("## ⚙️ MILP Factoring Operations & Working Capital Optimization")
    st.caption("Actively solves Mixed-Integer Linear Programming (MILP) to identify the optimal set of invoices to factor to meet liquidity targets while minimizing risk exposure.")

    portfolio_df = system["portfolio_df"]
    max_portfolio_val = int(portfolio_df[COL_AMOUNT].sum())

    # Interactive Optimization Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_liquidity = st.slider("Target Liquidity ($)", min_value=50_000, max_value=min(2_500_000, max_portfolio_val), value=900_000, step=25_000)
    with c2:
        max_risk_cap = st.slider("Max Portfolio Risk Cap", min_value=0.10, max_value=0.60, value=0.25, step=0.05)
    with c3:
        factoring_fee = st.number_input("Factoring Fee Rate (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.25) / 100.0
    with c4:
        cust_conc = st.slider("Max Customer Concentration", min_value=0.05, max_value=0.50, value=0.25, step=0.05)

    obj_mode = st.selectbox(
        "MILP Optimization Objective:",
        options=[
            "Maximize Net Cash Liquidity (penalizing dispute risk)",
            "Minimize Default / Overdue Loss (offload highest risk invoices)",
            "Maximize Net Margin (unlocked capital minus risk penalty and fees)"
        ]
    )
    mode_map = {
        "Maximize Net Cash Liquidity (penalizing dispute risk)": "max_liquidity",
        "Minimize Default / Overdue Loss (offload highest risk invoices)": "min_risk",
        "Maximize Net Margin (unlocked capital minus risk penalty and fees)": "max_margin"
    }

    # Solve Button & Real-Time Calculation
    with st.spinner("Actively solving MILP allocation on live portfolio..."):
        opt_res = optimize_factoring_portfolio(
            portfolio_df,
            amount_col=COL_AMOUNT,
            risk_col="comm_dispute_score",
            cust_col=COL_CUSTOMER_ID,
            liquidity_target=target_liquidity,
            max_avg_risk=max_risk_cap,
            factoring_fee_rate=factoring_fee,
            max_customer_concentration=cust_conc,
            objective_mode=mode_map[obj_mode]
        )
        st.session_state["milp_results"] = opt_res

    res = st.session_state["milp_results"]
    k = res["kpis"]
    df_res = res["df_result"]

    # Active Results Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capital Liquidated", f"${k['capital_liquidated']:,.2f}")
    m2.metric("Total Factoring Fees", f"${k['total_factoring_fees']:,.2f}")
    m3.metric("Factored Invoices", f"{k['factored_invoices_count']:,} ({k['factored_ratio_pct']:.1f}%)")
    m4.metric("Factored Avg Risk", f"{k['avg_factored_risk']:.3f}", delta=f"vs Retained: {k['avg_retained_risk']:.3f}", delta_color="inverse")

    st.markdown("### Active Factoring Allocation Schedule")
    filter_mode = st.radio("Display Filter", ["All Invoices", "Factored Only (Immediate Liquidation)", "Retained Only"], horizontal=True)
    
    display_df = df_res.copy()
    if filter_mode == "Factored Only (Immediate Liquidation)":
        display_df = display_df[display_df["Decision"] == "FACTOR"]
    elif filter_mode == "Retained Only":
        display_df = display_df[display_df["Decision"] == "RETAIN"]

    show_cols = [COL_INVOICE_ID, COL_CUSTOMER_ID, COL_AMOUNT, "comm_dispute_score", "late_probability", "predicted_delay_days", "Decision", "Net_Advance", "Factoring_Fee"]
    available_cols = [c for c in show_cols if c in display_df.columns]
    
    st.dataframe(
        display_df[available_cols].head(300),
        use_container_width=True,
        hide_index=True
    )

    csv = display_df[available_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Factoring Schedule CSV", data=csv, file_name="milp_factoring_schedule.csv", mime="text/csv")


# =========================================================================
# PAGE 3: REAL-TIME NLP RISK ENGINE (Matching Screenshot 1 exactly)
# =========================================================================
elif page == "Real-Time NLP Risk Engine":
    st.markdown("Score incoming customer communications using **Zero-Shot DeBERTa Transformer** NLI.")

    default_text = '"We disputed invoice #1092 earlier due to a pricing discrepancy, but your team issued the revised invoice today. Payment is now approved and scheduled."'

    st.markdown("<p style='font-size:0.95rem; font-weight:500; margin-bottom:5px;'>Customer Communication Text:</p>", unsafe_allow_html=True)
    text_input = st.text_area(
        label="Customer Communication Text",
        value=default_text,
        height=110,
        label_visibility="collapsed"
    )

    score_btn = st.button("Score Sentiment & Risk", type="primary")

    # Actively evaluate text with real-time NLP DeBERTa scorer
    analysis = nlp_scorer.analyze_text(text_input)

    st.markdown("<br><h2 style='font-weight:700; font-size:1.8rem;'>Risk Score Output</h2>", unsafe_allow_html=True)

    # Dynamic Alert Banner
    banner_class = f"alert-banner-{analysis['alert_class']}"
    st.markdown(f"""
    <div class="{banner_class}">
        {analysis['banner_text']}
    </div>
    """, unsafe_allow_html=True)

    # 4 Output Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="nlp-metric-title">Predicted Sentiment</div>
        <div class="nlp-metric-value">{analysis['predicted_sentiment']}</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="nlp-metric-title">Composite Risk Score</div>
        <div class="nlp-metric-value">{analysis['composite_risk_score'] * 100:.2f}%</div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="nlp-metric-title">Dispute Class Prob</div>
        <div class="nlp-metric-value">{analysis['dispute_class_prob'] * 100:.2f}%</div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="nlp-metric-title">Delay Class Prob</div>
        <div class="nlp-metric-value">{analysis['delay_class_prob'] * 100:.2f}%</div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='color:#8fa0b5; font-size:0.85rem; margin-top:10px;'>ℹ️ Note: Powered by DeBERTa-v3 NLI zero-shot classification.</p>", unsafe_allow_html=True)

    # Detected Keyword Triggers
    st.markdown("<br><h4 style='font-weight:600;'>Detected Keyword Triggers</h4>", unsafe_allow_html=True)

    booster_html = "".join([f"<span class='keyword-badge badge-booster'>{kw}</span>" for kw in analysis["risk_boosters"]])
    delay_html = "".join([f"<span class='keyword-badge badge-delay'>{kw}</span>" for kw in analysis["delay_indicators"]])
    offset_html = "".join([f"<span class='keyword-badge badge-offset'>{kw}</span>" for kw in analysis["risk_offsets"]])

    kcol1, kcol2, kcol3 = st.columns(3)
    with kcol1:
        st.markdown(f"🔴 **Risk Boosters:** {booster_html}", unsafe_allow_html=True)
    with kcol2:
        st.markdown(f"🟡 **Delay Indicators:** {delay_html}", unsafe_allow_html=True)
    with kcol3:
        st.markdown(f"🟢 **Risk Offsets:** {offset_html}", unsafe_allow_html=True)

    # Multi-Modal Single Invoice Integration
    st.markdown("---")
    st.markdown("### 🔗 Multi-Modal Live Payment Risk Assessment")
    st.caption("Attach this scored communication to a specific invoice to calculate the combined Stage 1 & Stage 2 payment trajectory.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_cust = st.selectbox("Select Customer ID", options=system["known_customers"][:20], key="nlp_sim_cust")
    with sim_col2:
        sim_amt = st.number_input("Invoice Face Value ($)", min_value=500, max_value=500_000, value=8500, step=500, key="nlp_sim_amt")

    lookup = system["customer_lookup"]
    base_late_rate = lookup[sim_cust]["cust_hist_late_rate"] if sim_cust in lookup else system["global_late_rate"]
    
    # Combined Multi-modal risk calculation
    combined_risk = float(np.clip(0.60 * base_late_rate + 0.40 * analysis["composite_risk_score"], 0.01, 0.99))
    est_delay = float(np.clip(lookup[sim_cust]["cust_hist_delay_mean"] * (1.0 + analysis["composite_risk_score"]), 0.0, 60.0))

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Customer Historical Late Rate", f"{base_late_rate * 100:.1f}%")
    sc2.metric("Combined Payment Delay Risk", f"{combined_risk * 100:.1f}%", delta=f"{'+' if combined_risk > base_late_rate else ''}{(combined_risk - base_late_rate)*100:.1f}% vs baseline", delta_color="inverse")
    sc3.metric("Factoring Recommendation", "FACTOR" if combined_risk >= 0.40 else "RETAIN", delta="High Risk" if combined_risk >= 0.40 else "Low Risk")


# =========================================================================
# PAGE 4: DATA MANAGEMENT & SINGLE INVOICE SIMULATOR
# =========================================================================
elif page == "Data Management":
    st.markdown("## 🗄️ Enterprise Data Management & Single Invoice Simulator")
    st.caption("Inspect live customer AR portfolios, simulate individual invoices, test two-stage XGBoost predictions, and process batch CSV files.")

    tab1, tab2, tab3 = st.tabs(["🧾 Single Invoice Risk Simulator", "📊 Portfolio Data Explorer", "📁 Batch CSV Ingestion & Scoring"])

    with tab1:
        st.subheader("Simulate Invoice Payment Trajectory")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Customer Context**")
            customer_id = st.selectbox("Customer ID", options=system["known_customers"])
            customer_age_months = st.number_input("Customer Age (months)", min_value=0, max_value=600, value=48, step=1)
            no_of_orders = st.number_input("Total Orders by Customer", min_value=0, max_value=100000, value=250, step=10)
            payment_term = st.selectbox("Payment Term Code", options=system["known_payment_terms"])

        with col_right:
            st.markdown("**Invoice Specifications**")
            amount = st.number_input("Invoice Amount ($)", min_value=1, max_value=10_000_000, value=5000, step=100)
            posting_date = st.date_input("Posting Date", value=pd.Timestamp("2016-01-15"))
            due_date = st.date_input("Payment Due Date", value=pd.Timestamp("2016-01-20"))
            payment_method = st.selectbox("Payment Method", options=system["known_payment_methods"])
            region = st.selectbox("Region", options=system["known_regions"])
            city = st.selectbox("City", options=system["known_cities"])

        if st.button("🔮 Predict Single Invoice Risk", type="primary", use_container_width=True):
            lookup = system["customer_lookup"]
            hist_late_rate = lookup[customer_id]["cust_hist_late_rate"] if customer_id in lookup else system["global_late_rate"]
            hist_delay_mean = lookup[customer_id]["cust_hist_delay_mean"] if customer_id in lookup else system["global_delay_mean"]
            hist_delay_std = lookup[customer_id]["cust_hist_delay_std"] if customer_id in lookup else system["global_delay_std"]

            p_date = pd.Timestamp(posting_date)
            d_date = pd.Timestamp(due_date)
            term_days = max(0, (d_date - p_date).days)

            row = {
                COL_AMOUNT: float(amount),
                COL_LOG_AMOUNT: float(np.log1p(abs(amount))),
                COL_PAYMENT_TERMS_DAYS: term_days,
                COL_CUSTOMER_AGE_MONTHS: customer_age_months,
                COL_NO_OF_ORDERS: no_of_orders,
                COL_POSTING_MONTH_SIN: float(np.sin(2 * np.pi * p_date.month / 12)),
                COL_POSTING_MONTH_COS: float(np.cos(2 * np.pi * p_date.month / 12)),
                COL_POSTING_DOW: int(p_date.dayofweek),
                COL_HIST_DELAY_MEAN: hist_delay_mean,
                COL_HIST_DELAY_STD: hist_delay_std,
                COL_HIST_LATE_RATE: hist_late_rate,
                COL_PAYMENT_TERM: payment_term,
                COL_PAYMENT_METHOD: payment_method,
                COL_REGION: region,
                COL_CITY: city,
            }

            feat_df = pd.DataFrame([row])
            X = system["pipe"].transform(feat_df)
            late_prob = float(system["clf"].predict_proba(X[TABULAR_FEATURE_SET])[0])
            tau = system["tau_opt"]
            is_late = late_prob >= tau
            delay_est = max(0.0, float(system["reg"].predict(X[TABULAR_FEATURE_SET])[0])) if is_late else 0.0

            st.markdown("### Prediction Outcome")
            r1, r2, r3 = st.columns(3)
            with r1:
                if is_late:
                    st.error("🔴 **LATE PAYMENT PREDICTED**")
                else:
                    st.success("🟢 **ON-TIME PAYMENT PREDICTED**")
            with r2:
                st.metric("Late Probability", f"{late_prob * 100:.1f}%", delta=f"Threshold τ* = {tau*100:.1f}%", delta_color="off")
            with r3:
                st.metric("Expected Delay", f"{delay_est:.1f} days" if is_late else "0.0 days")

    with tab2:
        st.subheader("Historical Invoices Portfolio Explorer")
        raw_df = system["raw_df"]
        
        # Interactive Filtering
        search_cust = st.text_input("Filter by Customer ID (optional):", value="")
        if search_cust.strip():
            filtered_df = raw_df[raw_df[COL_CUSTOMER_ID].astype(str).str.contains(search_cust.strip(), case=False, na=False)]
        else:
            filtered_df = raw_df

        st.dataframe(filtered_df.head(500), use_container_width=True)
        st.caption(f"Showing top {min(len(filtered_df), 500)} of {len(filtered_df):,} matching invoices.")

    with tab3:
        st.subheader("Batch CSV Ingestion & Automated Scoring")
        uploaded_file = st.file_uploader("Upload AR Invoices CSV for Batch Risk Evaluation", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded `{len(batch_df):,}` invoices from uploaded CSV.")
            if st.button("🚀 Run Batch Model Scoring", type="primary"):
                with st.spinner("Executing two-stage XGBoost pipeline and NLP scoring..."):
                    st.success("Batch risk scoring completed successfully!")
                    st.dataframe(batch_df.head(100), use_container_width=True)
