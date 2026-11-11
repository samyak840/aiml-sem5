"""Multi-model training, comparison, and serialisation for LoanIQ.

Models trained:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Gradient Boosting Classifier

Best model is selected by ROC-AUC on the held-out test set.
Random Forest is always saved separately for feature importance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALL_FEATURES, BEST_MODEL_PATH, CATEGORICAL_FEATURES, MODEL_COMPARISON_PATH,
    MODELS_DIR, NUMERIC_FEATURES, PREPROCESSOR_PATH, RANDOM_SEED, RF_MODEL_PATH,
)
from src.data_preprocessing import build_preprocessor, load_data, prepare_data


# ── Model definitions ─────────────────────────────────────────────────────────

MODELS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=RANDOM_SEED,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_SEED,
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        ),
        "y_test": y_test.tolist(),
        "y_proba": proba.tolist(),
    }


# ── Main training routine ─────────────────────────────────────────────────────

def train_all_models(df=None) -> dict[str, Any]:
    """
    Train all 4 models, compare on test set, save best + RF.
    Returns a bundle dict with everything needed for inference and reporting.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    preprocessor = build_preprocessor()
    comparison = {}
    trained_pipelines = {}

    for name, clf in MODELS.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor if name == list(MODELS.keys())[0] else build_preprocessor()),
            ("model", clf),
        ])
        pipeline.fit(X_train, y_train)
        metrics = _evaluate(pipeline, X_test, y_test)
        comparison[name] = metrics
        trained_pipelines[name] = pipeline

    # Select best by ROC-AUC
    best_name = max(comparison, key=lambda n: comparison[n]["roc_auc"])
    best_pipeline = trained_pipelines[best_name]
    rf_pipeline = trained_pipelines["Random Forest"]

    # Persist models
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(rf_pipeline, RF_MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)

    # Persist comparison (strip non-serialisable arrays for JSON)
    comparison_json = {
        name: {k: v for k, v in m.items() if k not in ("y_test", "y_proba")}
        for name, m in comparison.items()
    }
    with open(MODEL_COMPARISON_PATH, "w") as fh:
        json.dump({"models": comparison_json, "best": best_name}, fh, indent=2)

    return {
        "best_pipeline":    best_pipeline,
        "rf_pipeline":      rf_pipeline,
        "best_model_name":  best_name,
        "comparison":       comparison,
        "best_metrics":     comparison[best_name],
        "feature_names":    ALL_FEATURES,
        "numeric":          NUMERIC_FEATURES,
        "categorical":      CATEGORICAL_FEATURES,
        "data":             df,
        "X_test":           X_test,
        "y_test":           y_test,
        "train_rows":       len(X_train),
        "test_rows":        len(X_test),
    }


# ── Load or train ─────────────────────────────────────────────────────────────

def load_or_train() -> dict[str, Any]:
    """
    Load cached models if they exist and are newer than the dataset.
    Otherwise retrain from scratch.
    """
    from config import DATA_PATH  # local import avoids circular at module level

    models_exist = BEST_MODEL_PATH.exists() and RF_MODEL_PATH.exists()
    if models_exist:
        data_mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0
        model_mtime = BEST_MODEL_PATH.stat().st_mtime
        if model_mtime >= data_mtime and MODEL_COMPARISON_PATH.exists():
            best_pipeline = joblib.load(BEST_MODEL_PATH)
            rf_pipeline = joblib.load(RF_MODEL_PATH)
            with open(MODEL_COMPARISON_PATH) as fh:
                comp_meta = json.load(fh)
            if list(best_pipeline.feature_names_in_) != list(ALL_FEATURES):
                return train_all_models()
            df = load_data()
            _, X_test, _, y_test = prepare_data(df)
            best_metrics = _evaluate(best_pipeline, X_test, y_test)
            comparison = comp_meta.get("models", {})
            comparison[comp_meta.get("best", "Unknown")] = best_metrics
            return {
                "best_pipeline":    best_pipeline,
                "rf_pipeline":      rf_pipeline,
                "best_model_name":  comp_meta.get("best", "Unknown"),
                "comparison":       comparison,
                "best_metrics":     best_metrics,
                "feature_names":    ALL_FEATURES,
                "numeric":          NUMERIC_FEATURES,
                "categorical":      CATEGORICAL_FEATURES,
                "data":             df,
                "X_test":           X_test,
                "y_test":           y_test,
                "train_rows":       int(len(df) * 0.8),
                "test_rows":        int(len(df) * 0.2),
            }

    return train_all_models()
