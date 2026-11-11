from src import train_model
from loan_model import load_dataset


def test_model_comparison_trains_and_selects_by_roc_auc(tmp_path, monkeypatch):
    monkeypatch.setattr(train_model, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_model, "BEST_MODEL_PATH", tmp_path / "best.joblib")
    monkeypatch.setattr(train_model, "RF_MODEL_PATH", tmp_path / "random_forest.joblib")
    monkeypatch.setattr(
        train_model, "PREPROCESSOR_PATH", tmp_path / "preprocessor.joblib"
    )
    monkeypatch.setattr(
        train_model, "MODEL_COMPARISON_PATH", tmp_path / "comparison.json"
    )

    sample = load_dataset().sample(600, random_state=23)
    bundle = train_model.train_all_models(sample)

    assert set(bundle["comparison"]) == set(train_model.MODELS)
    assert bundle["best_model_name"] == max(
        bundle["comparison"],
        key=lambda name: bundle["comparison"][name]["roc_auc"],
    )
    assert bundle["best_metrics"]["roc_auc"] >= 0.5
    assert bundle["best_pipeline"] is not None
    assert train_model.BEST_MODEL_PATH.exists()

