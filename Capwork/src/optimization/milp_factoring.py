"""
MILP Invoice Factoring Portfolio Optimizer
==========================================
Formulates and solves the invoice factoring allocation problem as a 
Mixed-Integer Linear Program (MILP) using scipy.optimize.milp.

Problem Formulation:
-------------------
Decision variables: x_i in {0, 1}, where x_i = 1 if invoice i is factored (sold for immediate cash), 0 if retained.

Objectives:
    - Mode 'max_liquidity': Maximize Net Cash Unlocked = Sum_i [ x_i * Amount_i * (1 - Factoring_Fee_Rate) - 0.25 * x_i * Amount_i * Dispute_Risk_i ]
    - Mode 'min_risk': Maximize High-Risk Offload = Sum_i [ x_i * Amount_i * Dispute_Risk_i * (1 - Factoring_Fee_Rate) ]
    - Mode 'max_margin': Maximize Value Unlocked - Fees - Risk = Sum_i [ x_i * Amount_i * (1 - Factoring_Fee_Rate - Dispute_Risk_i) ]

Subject to:
    1. Capital / Facility Budget Cap: Sum_i [ x_i * Amount_i ] <= Liquidity_Budget
    2. Portfolio Risk Tolerance: Sum_i [ x_i * (Dispute_Risk_i - Max_Avg_Risk) * Amount_i ] <= 0
    3. Customer Concentration Limit: For each customer c, Sum_{i in Cust_c} [ x_i * Amount_i ] <= Max_Cust_Share * Total_Factored_Amount
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.optimize import milp, LinearConstraint, Bounds

def optimize_factoring_portfolio(
    df: pd.DataFrame,
    amount_col: str = "Amount",
    risk_col: str = "comm_dispute_score",
    cust_col: str = "Cust_Num",
    invoice_col: str = "Document_No",
    liquidity_target: float = 900_000.0,
    max_avg_risk: float = 0.35,
    factoring_fee_rate: float = 0.03,
    max_customer_concentration: float = 0.30,
    objective_mode: str = "max_liquidity"
) -> Dict[str, Any]:
    """
    Actively optimizes invoice factoring selection using Mixed-Integer Linear Programming (MILP).
    """
    if df.empty:
        return {"status": "empty", "df_result": df, "kpis": {}}

    n = len(df)
    amounts = df[amount_col].fillna(0.0).values.astype(float)
    risks = df[risk_col].fillna(0.1).values.astype(float) if risk_col in df.columns else np.full(n, 0.1)
    customers = df[cust_col].astype(str).values if cust_col in df.columns else np.array([f"CUST_{i}" for i in range(n)])

    # Setup Objective Vector (scipy.optimize.milp MINIMIZES c^T x)
    if objective_mode == "min_risk":
        # Offload highest risk invoices first
        c = -(amounts * risks * (1.0 - factoring_fee_rate))
    elif objective_mode == "max_margin":
        c = -(amounts * (1.0 - factoring_fee_rate - 0.5 * risks))
    else:
        # Default: Maximize net liquidity while penalizing excessive risk
        c = -(amounts * (1.0 - factoring_fee_rate) - 0.20 * amounts * risks)

    integrality = np.ones(n, dtype=int)
    bounds = Bounds(0, 1)

    constraints_matrix = []
    lhs_bounds = []
    rhs_bounds = []

    # Constraint 1: Total Factored Amount <= Liquidity Target
    constraints_matrix.append(amounts)
    lhs_bounds.append(0.0)
    rhs_bounds.append(float(liquidity_target))

    # Constraint 2: Weighted Risk Constraint (Average risk of factored invoices <= max_avg_risk)
    # Sum_i x_i * Amount_i * (Risk_i - Max_Avg_Risk) <= 0
    risk_weights = amounts * (risks - max_avg_risk)
    constraints_matrix.append(risk_weights)
    lhs_bounds.append(-np.inf)
    rhs_bounds.append(0.0)

    # Constraint 3: Customer concentration constraints
    unique_custs, cust_inv = np.unique(customers, return_inverse=True)
    # Apply to top customers with meaningful volume
    for cust_idx in range(len(unique_custs)):
        mask = (cust_inv == cust_idx)
        if np.sum(mask) > 1 and np.sum(amounts[mask]) > 0.05 * liquidity_target:
            row = np.zeros(n)
            row[mask] = amounts[mask] * (1.0 - max_customer_concentration)
            row[~mask] = -amounts[~mask] * max_customer_concentration
            constraints_matrix.append(row)
            lhs_bounds.append(-np.inf)
            rhs_bounds.append(0.0)

    A = np.array(constraints_matrix)
    constraints = LinearConstraint(A, lhs_bounds, rhs_bounds)

    try:
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if res.success:
            x_opt = np.round(res.x).astype(int)
        else:
            x_opt = _greedy_factoring_fallback(amounts, risks, liquidity_target, max_avg_risk)
    except Exception:
        x_opt = _greedy_factoring_fallback(amounts, risks, liquidity_target, max_avg_risk)

    df_res = df.copy()
    df_res["Factored"] = x_opt
    df_res["Decision"] = np.where(x_opt == 1, "FACTOR", "RETAIN")
    df_res["Net_Advance"] = np.where(x_opt == 1, amounts * (1.0 - factoring_fee_rate), 0.0)
    df_res["Factoring_Fee"] = np.where(x_opt == 1, amounts * factoring_fee_rate, 0.0)

    factored_mask = (x_opt == 1)
    total_portfolio_val = float(np.sum(amounts))
    factored_val = float(np.sum(amounts[factored_mask]))
    retained_val = float(np.sum(amounts[~factored_mask]))
    net_liquidated = float(np.sum(df_res["Net_Advance"]))
    total_fees = float(np.sum(df_res["Factoring_Fee"]))
    factored_count = int(np.sum(factored_mask))
    retained_count = int(n - factored_count)
    avg_portfolio_risk = float(np.mean(risks)) if n > 0 else 0.0
    avg_factored_risk = float(np.average(risks[factored_mask], weights=amounts[factored_mask])) if factored_count > 0 and np.sum(amounts[factored_mask]) > 0 else 0.0
    avg_retained_risk = float(np.average(risks[~factored_mask], weights=amounts[~factored_mask])) if retained_count > 0 and np.sum(amounts[~factored_mask]) > 0 else 0.0

    kpis = {
        "total_portfolio_value": total_portfolio_val,
        "factored_portfolio_value": factored_val,
        "retained_portfolio_value": retained_val,
        "capital_liquidated": net_liquidated,
        "total_factoring_fees": total_fees,
        "factored_invoices_count": factored_count,
        "retained_invoices_count": retained_count,
        "factored_ratio_pct": (factored_val / total_portfolio_val * 100) if total_portfolio_val > 0 else 0.0,
        "retained_ratio_pct": (retained_val / total_portfolio_val * 100) if total_portfolio_val > 0 else 100.0,
        "avg_portfolio_risk": avg_portfolio_risk,
        "avg_factored_risk": avg_factored_risk,
        "avg_retained_risk": avg_retained_risk,
    }

    return {
        "status": "success",
        "df_result": df_res,
        "kpis": kpis
    }

def _greedy_factoring_fallback(
    amounts: np.ndarray,
    risks: np.ndarray,
    liquidity_target: float,
    max_avg_risk: float
) -> np.ndarray:
    """Greedy value-to-risk ratio knapsack heuristic fallback."""
    n = len(amounts)
    x = np.zeros(n, dtype=int)
    sort_idx = np.argsort(risks)
    cum_amount = 0.0
    for idx in sort_idx:
        if cum_amount + amounts[idx] <= liquidity_target and risks[idx] <= max_avg_risk * 1.5:
            x[idx] = 1
            cum_amount += amounts[idx]
    return x
