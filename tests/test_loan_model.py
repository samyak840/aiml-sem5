import pytest

from loan_model import load_dataset, predict_application, score_anomaly, train_model
from src.prediction import validate_application


def test_supplied_dataset_has_target_and_expected_scale():
    df = load_dataset()
    assert len(df) >= 1000
    assert "LoanApproved" in df.columns
    assert set(df["LoanApproved"].unique()).issubset({0, 1})


def test_model_trains_and_returns_bounded_prediction():
    bundle = train_model(load_dataset().sample(800, random_state=11))
    assert "Age" not in bundle["features"]
    assert "MaritalStatus" not in bundle["features"]
    application = bundle["data"].iloc[0][bundle["features"]].to_dict()
    result = predict_application(bundle, application)
    assert 0 <= result["approval_probability"] <= 1
    assert result["approval_probability"] + result["rejection_probability"] == 1
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert bundle["metrics"]["roc_auc"] >= 0.5
    anomaly = score_anomaly(bundle, application)
    assert 0 <= anomaly["score"] <= 100
    with pytest.raises(ValueError, match="Invalid application"):
        predict_application(bundle, {"AnnualIncome": -1})


def test_application_validation_rejects_missing_and_invalid_values():
    application = load_dataset().iloc[0].to_dict()
    errors = validate_application(application)
    assert errors == []

    missing_income = application.copy()
    missing_income.pop("AnnualIncome")
    assert any("AnnualIncome is required" in error
               for error in validate_application(missing_income))

    invalid_application = application.copy()
    invalid_application["MonthlyDebtPayments"] = -1
    invalid_application["LoanAmount"] = -100
    errors = validate_application(invalid_application)
    assert any("Loan amount must be greater" in error for error in errors)
    assert any("MonthlyDebtPayments" in error for error in errors)
