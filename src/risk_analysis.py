"""Pure financial mathematics for LoanIQ risk analysis.

All calculations use standard financial formulas.
Assumptions are documented per function.
Results are for decision-support only and must not be used as sole lending criteria.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RISK_HIGH_MIN, RISK_LOW_MAX


# ── DTI ───────────────────────────────────────────────────────────────────────

def calculate_dti(monthly_debt_payments: float, gross_monthly_income: float) -> float:
    """
    Debt-to-Income Ratio (DTI).
    Formula: DTI = (Monthly Debt Payments / Gross Monthly Income) × 100
    Returns percentage (0–100+). Returns 0 if income is zero or negative.
    """
    if gross_monthly_income <= 0:
        return 0.0
    return round((monthly_debt_payments / gross_monthly_income) * 100, 2)


# ── EMI ───────────────────────────────────────────────────────────────────────

def calculate_emi(principal: float, annual_interest_rate: float, tenure_months: int) -> float:
    """
    Equated Monthly Instalment (EMI) using the standard amortisation formula.
    Formula: EMI = P × r × (1+r)^n / ((1+r)^n – 1)
    Where:
        P = Principal loan amount
        r = Monthly interest rate = Annual rate / 12 / 100
        n = Number of monthly instalments

    Returns 0 for zero/negative principal or non-positive tenure.
    For 0% interest returns simple division P/n.
    """
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    if annual_interest_rate <= 0:
        return round(principal / tenure_months, 2)
    r = annual_interest_rate / 12 / 100
    n = tenure_months
    emi = principal * r * math.pow(1 + r, n) / (math.pow(1 + r, n) - 1)
    return round(emi, 2)


# ── LTI ───────────────────────────────────────────────────────────────────────

def calculate_lti(loan_amount: float, annual_income: float) -> float:
    """
    Loan-to-Income Ratio (LTI).
    Formula: LTI = Loan Amount / Annual Income
    Returns 0 if annual income is zero or negative.
    """
    if annual_income <= 0:
        return 0.0
    return round(loan_amount / annual_income, 3)


# ── Payment Burden ────────────────────────────────────────────────────────────

def calculate_payment_burden(
    existing_monthly_emi: float,
    proposed_emi: float,
    monthly_income: float,
) -> float:
    """
    Payment Burden (% of income going to EMIs).
    Formula: (Existing EMI + Proposed EMI) / Monthly Income × 100
    Returns 0 if monthly income is zero or negative.
    """
    if monthly_income <= 0:
        return 0.0
    total_emi = existing_monthly_emi + proposed_emi
    return round((total_emi / monthly_income) * 100, 2)


# ── Disposable Income ─────────────────────────────────────────────────────────

def calculate_disposable_income(
    monthly_income: float,
    monthly_expenses: float,
    total_monthly_emi: float,
) -> float:
    """
    Disposable Income after expenses and EMIs.
    Formula: Monthly Income - Monthly Expenses - Total Monthly EMI
    Can be negative (indicating financial stress).
    """
    return round(monthly_income - monthly_expenses - total_monthly_emi, 2)


# ── Composite Risk Score ──────────────────────────────────────────────────────

def calculate_risk_score(
    dti: float,
    lti: float,
    credit_score: float,
    payment_burden: float,
    previous_defaults: float,
    bankruptcy_history: float,
    credit_card_utilization: float,
) -> float:
    """
    Composite financial risk score (0–100; higher = riskier).

    Scoring methodology (documented, not a calibrated probability):
      - DTI component        (weight 25): DTI > 50% → high risk
      - LTI component        (weight 20): LTI > 5×  → high risk
      - Credit score         (weight 25): lower score → higher risk
      - Payment burden       (weight 15): burden > 60% → high risk
      - Default history      (weight 10): any defaults or bankruptcy → penalty
      - Credit utilisation   (weight  5): > 80% utilisation → penalty

    Returns a clamped score [0, 100].
    """
    score = 0.0

    # DTI (25 pts): 0% → 0 pts, ≥ 60% → 25 pts
    dti_score = min(dti / 60.0, 1.0) * 25
    score += dti_score

    # LTI (20 pts): 0× → 0 pts, ≥ 6× → 20 pts
    lti_score = min(lti / 6.0, 1.0) * 20
    score += lti_score

    # Credit score (25 pts): 850 → 0 pts, 300 → 25 pts
    credit_norm = max(0, min((850 - credit_score) / (850 - 300), 1.0))
    score += credit_norm * 25

    # Payment burden (15 pts): 0% → 0 pts, ≥ 70% → 15 pts
    burden_score = min(payment_burden / 70.0, 1.0) * 15
    score += burden_score

    # Defaults + bankruptcy (10 pts)
    default_penalty = min(previous_defaults * 3, 7) + (3 if bankruptcy_history >= 1 else 0)
    score += min(default_penalty, 10)

    # Credit utilisation (5 pts): > 80% → 5 pts
    util_score = min(credit_card_utilization / 80.0, 1.0) * 5
    score += util_score

    return round(min(max(score, 0.0), 100.0), 1)


# ── Risk Category ─────────────────────────────────────────────────────────────

def classify_risk(risk_score: float) -> str:
    """Return 'Low', 'Medium', or 'High' based on risk score thresholds."""
    if risk_score < RISK_LOW_MAX:
        return "Low"
    if risk_score < RISK_HIGH_MIN:
        return "Medium"
    return "High"


# ── Affordability helpers ────────────────────────────────────────────────────

def loan_affordability_check(
    monthly_income: float,
    proposed_emi: float,
    existing_emi: float = 0.0,
    monthly_expenses: float = 0.0,
) -> dict:
    """
    Returns a quick affordability summary dict:
      - affordable: bool — disposable income after all commitments is positive
      - recommended_max_emi: float — 40% of monthly income guideline
      - headroom: float — remaining budget after proposed EMI & expenses
    """
    recommended_max = monthly_income * 0.40
    total_emi = proposed_emi + existing_emi
    headroom = monthly_income - monthly_expenses - total_emi
    return {
        "affordable": headroom >= 0,
        "recommended_max_emi": round(recommended_max, 2),
        "headroom": round(headroom, 2),
        "total_monthly_commitment": round(total_emi + monthly_expenses, 2),
    }
