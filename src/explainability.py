"""Feature importance and explainability for LoanIQ.

Uses Random Forest feature importances (not SHAP) to rank contributing factors.
Counterfactual comparison is used to generate positive / negative factor lists.
NOTE: Feature importance shows correlation, not causation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


# ── Plain-English feature labels ──────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "Age":                        "Applicant Age",
    "AnnualIncome":               "Annual Income",
    "CreditScore":                "Credit Score",
    "Experience":                 "Work Experience",
    "LoanAmount":                 "Requested Loan Amount",
    "LoanDuration":               "Loan Duration (months)",
    "NumberOfDependents":         "Number of Dependents",
    "MonthlyDebtPayments":        "Monthly Debt Payments",
    "CreditCardUtilizationRate":  "Credit Card Utilisation",
    "NumberOfOpenCreditLines":    "Open Credit Lines",
    "NumberOfCreditInquiries":    "Recent Credit Inquiries",
    "DebtToIncomeRatio":          "Debt-to-Income Ratio",
    "BankruptcyHistory":          "Bankruptcy History",
    "PreviousLoanDefaults":       "Previous Loan Defaults",
    "PaymentHistory":             "Payment History Score",
    "LengthOfCreditHistory":      "Length of Credit History",
    "SavingsAccountBalance":      "Savings Account Balance",
    "CheckingAccountBalance":     "Checking Account Balance",
    "TotalAssets":                "Total Assets",
    "TotalLiabilities":           "Total Liabilities",
    "MonthlyIncome":              "Monthly Income",
    "UtilityBillsPaymentHistory": "Utility Bills Payment History",
    "JobTenure":                  "Job Tenure (years)",
    "NetWorth":                   "Net Worth",
    "BaseInterestRate":           "Base Interest Rate",
    "InterestRate":               "Applied Interest Rate",
    "MonthlyLoanPayment":         "Monthly Loan Payment",
    "TotalDebtToIncomeRatio":     "Total Debt-to-Income Ratio",
    "EmploymentStatus":           "Employment Status",
    "EducationLevel":             "Education Level",
    "MaritalStatus":              "Marital Status",
    "HomeOwnershipStatus":        "Home Ownership Status",
    "LoanPurpose":                "Loan Purpose",
}

# ── Positive / negative phrasing templates ────────────────────────────────────

POSITIVE_TEMPLATES: dict[str, str] = {
    "CreditScore":               "Strong credit score",
    "PaymentHistory":            "Consistent payment history",
    "AnnualIncome":              "Stable annual income",
    "MonthlyIncome":             "Adequate monthly income",
    "SavingsAccountBalance":     "Healthy savings balance",
    "CheckingAccountBalance":    "Active checking account",
    "NetWorth":                  "Positive net worth",
    "TotalAssets":               "Substantial total assets",
    "LengthOfCreditHistory":     "Long credit history",
    "JobTenure":                 "Stable employment tenure",
    "Experience":                "Significant work experience",
    "UtilityBillsPaymentHistory":"Good utility payment record",
}

NEGATIVE_TEMPLATES: dict[str, str] = {
    "DebtToIncomeRatio":          "High debt-to-income ratio",
    "TotalDebtToIncomeRatio":     "High total debt burden",
    "MonthlyDebtPayments":        "Significant monthly debt obligations",
    "LoanAmount":                 "Large loan amount requested",
    "CreditCardUtilizationRate":  "High credit card utilisation",
    "NumberOfCreditInquiries":    "Multiple recent credit inquiries",
    "BankruptcyHistory":          "History of bankruptcy",
    "PreviousLoanDefaults":       "Previous loan defaults on record",
    "TotalLiabilities":           "High total liabilities",
    "NumberOfDependents":         "Large number of dependents",
    "InterestRate":               "High interest rate on this loan",
}


# ── RF Feature importance ─────────────────────────────────────────────────────

def get_rf_feature_importance(bundle: dict[str, Any], top_n: int = 15) -> pd.DataFrame:
    """
    Extract global feature importances from the Random Forest pipeline.
    Returns a DataFrame with columns: feature, label, importance.
    """
    rf_pipeline = bundle.get("rf_pipeline")
    if rf_pipeline is None:
        return pd.DataFrame(columns=["feature", "label", "importance"])

    rf_model = rf_pipeline.named_steps["model"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]

    # Reconstruct feature names after one-hot encoding
    try:
        ohe = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    except Exception:
        cat_names = []

    all_names = NUMERIC_FEATURES + cat_names
    importances = rf_model.feature_importances_

    n = min(len(all_names), len(importances))
    df = pd.DataFrame({
        "feature":    all_names[:n],
        "importance": importances[:n],
    })
    # Map to plain labels (only for original features; OHE variants keep raw name)
    df["label"] = df["feature"].apply(
        lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title())
    )
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


# ── Counterfactual factor explanation ─────────────────────────────────────────

def explain_application(bundle: dict[str, Any], application: dict[str, Any]) -> dict:
    """
    Generate top positive and negative factors for a single application.

    Strategy: compare each feature's value against the training-set median
    (numeric) or mode (categorical).  Features whose values look 'favourable'
    are listed as positive; unfavourable values are listed as negative.

    Returns:
        {
          "positive": [{"factor": str, "detail": str}, ...],
          "negative": [{"factor": str, "detail": str}, ...],
        }
    """
    df = bundle.get("data")
    if df is None:
        return {"positive": [], "negative": []}

    positive = []
    negative = []

    # --- Positive signals ------------------------------------------------
    for feat, label in POSITIVE_TEMPLATES.items():
        if feat not in application or feat not in df.columns:
            continue
        val = application[feat]
        median = float(pd.to_numeric(df[feat], errors="coerce").median())
        if float(val) >= median * 1.05:
            positive.append({"factor": label, "detail": f"Value: {_fmt(feat, val)} (above median)"})

    # --- Negative signals ------------------------------------------------
    for feat, label in NEGATIVE_TEMPLATES.items():
        if feat not in application or feat not in df.columns:
            continue
        val = float(application[feat])
        median = float(pd.to_numeric(df[feat], errors="coerce").median())
        # For risk indicators: high values are bad
        if feat in ("BankruptcyHistory", "PreviousLoanDefaults") and val > 0:
            negative.append({"factor": label, "detail": f"Value: {_fmt(feat, val)}"})
        elif feat not in ("BankruptcyHistory", "PreviousLoanDefaults") and val >= median * 1.1:
            negative.append({"factor": label, "detail": f"Value: {_fmt(feat, val)} (above median)"})

    # Limit to top 5 each
    return {"positive": positive[:5], "negative": negative[:5]}


def _fmt(feat: str, val) -> str:
    """Format a feature value for display."""
    monetary = {
        "AnnualIncome", "MonthlyIncome", "LoanAmount", "SavingsAccountBalance",
        "CheckingAccountBalance", "TotalAssets", "TotalLiabilities",
        "NetWorth", "MonthlyDebtPayments", "MonthlyLoanPayment",
    }
    pct = {"CreditCardUtilizationRate", "DebtToIncomeRatio", "TotalDebtToIncomeRatio",
           "InterestRate", "BaseInterestRate"}
    if feat in monetary:
        return f"${float(val):,.0f}"
    if feat in pct:
        return f"{float(val):.1f}%"
    return str(round(float(val), 2))
