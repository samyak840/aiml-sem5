"""Prediction and preprocessing helpers for LoanIQ inference."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.risk_analysis import (
    calculate_dti,
    calculate_emi,
    calculate_lti,
    calculate_payment_burden,
    calculate_risk_score,
    classify_risk,
)


# ── Application dict → DataFrame ─────────────────────────────────────────────

def application_to_df(application: dict[str, Any]) -> pd.DataFrame:
    """Convert a flat application dict to a single-row DataFrame with correct dtypes."""
    row = {}
    for col in NUMERIC_FEATURES:
        row[col] = float(application.get(col, 0.0))
    for col in CATEGORICAL_FEATURES:
        row[col] = str(application.get(col, ""))
    return pd.DataFrame([row])[ALL_FEATURES]


# ── Core prediction ───────────────────────────────────────────────────────────

def predict_application(bundle: dict[str, Any], application: dict[str, Any]) -> dict[str, Any]:
    """
    Run the trained best model on a single application dict.

    Returns:
        approved             bool
        approval_probability float  [0, 1]
        rejection_probability float [0, 1]
        risk_score           float  [0, 100]  (rule-based financial risk score)
        risk_band            str    Low / Medium / High
        model_name           str
    """
    errors = validate_application(application)
    if errors:
        raise ValueError("Invalid application: " + " ".join(errors))
    pipeline = bundle.get("best_pipeline", bundle.get("model"))
    if pipeline is None:
        raise KeyError("Model bundle must contain 'best_pipeline' or 'model'.")
    row = application_to_df(application)
    prob_approval = float(pipeline.predict_proba(row)[0, 1])
    prob_approval = round(prob_approval, 4)
    monthly_income = float(application["MonthlyIncome"])
    existing_debt = float(application["MonthlyDebtPayments"])
    annual_rate = float(application["InterestRate"])
    proposed_emi = calculate_emi(
        float(application["LoanAmount"]),
        annual_rate * 100 if annual_rate <= 1 else annual_rate,
        max(1, round(float(application["LoanDuration"]))),
    )
    dti = calculate_dti(existing_debt, monthly_income)
    lti = calculate_lti(
        float(application["LoanAmount"]),
        float(application["AnnualIncome"]),
    )
    payment_burden = calculate_payment_burden(
        existing_debt, proposed_emi, monthly_income
    )
    risk_score = calculate_risk_score(
        dti=dti,
        lti=lti,
        credit_score=float(application["CreditScore"]),
        payment_burden=payment_burden,
        previous_defaults=float(application["PreviousLoanDefaults"]),
        bankruptcy_history=float(application["BankruptcyHistory"]),
        credit_card_utilization=float(application["CreditCardUtilizationRate"]),
    )

    return {
        "approved":              prob_approval >= 0.5,
        "approval_probability":  prob_approval,
        "rejection_probability": round(1.0 - prob_approval, 4),
        "risk_score":            risk_score,
        "risk_band":             classify_risk(risk_score),
        "model_name":            bundle.get("best_model_name", "ML Model"),
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_application(application: dict[str, Any]) -> list[str]:
    """
    Return a list of validation error messages.
    Empty list means the application passes all checks.
    """
    errors = []

    def _num(key: str) -> float:
        try:
            value = float(application.get(key, float("nan")))
            return value if np.isfinite(value) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    for key in NUMERIC_FEATURES:
        value = _num(key)
        if not np.isfinite(value):
            errors.append(f"{key.replace('_', ' ')} is required and must be numeric.")
        elif value < 0 and key != "NetWorth":
            errors.append(f"{key.replace('_', ' ')} cannot be negative.")

    if _num("AnnualIncome") <= 0:
        errors.append("Annual income must be greater than zero.")
    if _num("MonthlyIncome") <= 0:
        errors.append("Monthly income must be greater than zero.")
    if _num("LoanAmount") <= 0:
        errors.append("Loan amount must be greater than zero.")
    if _num("LoanDuration") <= 0:
        errors.append("Loan duration must be at least 1 month.")
    if not (300 <= _num("CreditScore") <= 850):
        errors.append("Credit score must be between 300 and 850.")
    if not (0 <= _num("CreditCardUtilizationRate") <= 100):
        errors.append("Credit card utilisation must be between 0% and 100%.")
    if not (0 <= _num("BaseInterestRate") <= 1):
        errors.append("Base interest rate must be a decimal between 0 and 1.")
    if not (0 <= _num("InterestRate") <= 1):
        errors.append("Interest rate must be a decimal between 0 and 1.")
    if not (0 <= _num("UtilityBillsPaymentHistory") <= 1):
        errors.append("Utility payment history must be between 0 and 1.")
    if not (0 <= _num("PaymentHistory") <= 100):
        errors.append("Payment history score must be between 0 and 100.")
    for key in CATEGORICAL_FEATURES:
        value = application.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key.replace('_', ' ')} is required.")

    return errors
