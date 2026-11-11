import json

from src import database


def _record():
    return {
        "application": {"CreditScore": 680, "EmploymentStatus": "Employed"},
        "monthly_expenses": 1250.0,
        "approved": 1,
        "approval_probability": 0.73,
        "rejection_probability": 0.27,
        "model_used": "Logistic Regression",
        "risk_score": 31.5,
        "risk_category": "Low",
        "dti": 18.0,
        "emi": 400.0,
        "lti": 0.5,
        "payment_burden": 30.0,
        "disposable_income": 900.0,
        "anomaly_score": 44.0,
        "anomaly_status": "Typical",
    }


def test_assessment_can_be_saved_retrieved_exported_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.sqlite")

    assessment_id = database.save_assessment(_record())

    retrieved = database.get_assessment_by_id(assessment_id)
    assert retrieved is not None
    assert json.loads(retrieved["application_json"]) == _record()["application"]
    assert retrieved["approval_probability"] == 0.73
    assert retrieved["monthly_expenses"] == 1250.0
    assert database.get_assessment_stats()["total"] == 1
    exported = database.export_assessments_to_csv()
    assert "approval_probability" in exported
    assert str(assessment_id) in exported

    database.delete_assessment(assessment_id)
    assert database.get_assessment_by_id(assessment_id) is None
    assert database.get_assessment_stats()["total"] == 0

