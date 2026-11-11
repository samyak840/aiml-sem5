import pytest

from src.risk_analysis import (
    calculate_disposable_income,
    calculate_dti,
    calculate_emi,
    calculate_lti,
    calculate_payment_burden,
    calculate_risk_score,
    classify_risk,
    loan_affordability_check,
)


def test_financial_ratios_and_payment_calculations():
    assert calculate_dti(1200, 6000) == 20.0
    assert calculate_lti(24000, 80000) == 0.3
    assert calculate_payment_burden(500, 1000, 5000) == 30.0
    assert calculate_emi(12000, 0, 12) == 1000.0
    assert calculate_emi(12000, 12, 12) == pytest.approx(1066.19, abs=0.01)


def test_zero_income_and_invalid_loan_amount_have_safe_results():
    assert calculate_dti(500, 0) == 0.0
    assert calculate_lti(10000, 0) == 0.0
    assert calculate_payment_burden(500, 1000, 0) == 0.0
    assert calculate_emi(0, 12, 12) == 0.0
    assert calculate_emi(12000, 12, 0) == 0.0


def test_disposable_income_and_affordability_account_for_all_commitments():
    assert calculate_disposable_income(6000, 1500, 1800) == 2700.0
    result = loan_affordability_check(
        monthly_income=6000,
        proposed_emi=1000,
        existing_emi=800,
        monthly_expenses=1500,
    )
    assert result == {
        "affordable": True,
        "recommended_max_emi": 2400.0,
        "headroom": 2700.0,
        "total_monthly_commitment": 3300.0,
    }

    over_budget = loan_affordability_check(
        monthly_income=2000,
        proposed_emi=1000,
        existing_emi=500,
        monthly_expenses=600,
    )
    assert over_budget["affordable"] is False
    assert over_budget["headroom"] == -100.0


def test_risk_score_is_bounded_and_classified_by_configured_thresholds():
    low_score = calculate_risk_score(0, 0, 850, 0, 0, 0, 0)
    high_score = calculate_risk_score(100, 10, 300, 100, 10, 1, 100)

    assert low_score == 0.0
    assert high_score == 100.0
    assert classify_risk(low_score) == "Low"
    assert classify_risk(50) == "Medium"
    assert classify_risk(high_score) == "High"
