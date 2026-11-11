"""Single-dataset loan approval and risk analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.prediction import predict_application as predict_bundle_application

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Loan.csv"
MODEL_PATH = ROOT / "loan_model.joblib"
TARGET = "LoanApproved"
EXCLUDED_COLUMNS = {
    TARGET, "RiskScore", "ApplicationDate", "Age", "MaritalStatus",
}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/Loan.csv is required for this project.")
    return pd.read_csv(DATA_PATH)


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()
    df.columns = df.columns.str.strip()
    feature_names = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    numeric = df[feature_names].select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in feature_names if c not in numeric]
    return df.dropna(subset=[TARGET]), numeric, categorical


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    return Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])


def fit_anomaly_monitor(df: pd.DataFrame, numeric: list[str]) -> dict[str, Any]:
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    matrix = prep.fit_transform(df[numeric])
    detector = IsolationForest(n_estimators=150, contamination=0.03, random_state=42)
    detector.fit(matrix)
    decisions = detector.decision_function(matrix)
    anomaly_table = df.copy()
    anomaly_table["AnomalyScore"] = np.round(np.clip(50 - 35 * (decisions - decisions.mean()) / (decisions.std() + 1e-9), 0, 100), 1)
    anomaly_table["AnomalyStatus"] = np.where(detector.predict(matrix) == -1, "Review", "Typical")
    return {"anomaly_preprocessor": prep, "anomaly_detector": detector, "anomaly_table": anomaly_table}


def train_model(df: pd.DataFrame | None = None) -> dict[str, Any]:
    raw = load_dataset() if df is None else df.copy()
    df, numeric, categorical = prepare_dataset(raw)
    features = numeric + categorical
    x_train, x_test, y_train, y_test = train_test_split(df[features], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET])
    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    bundle = {"model": pipeline, "metrics": metrics, "data": df, "numeric": numeric, "categorical": categorical, "features": features, "data_path": str(DATA_PATH)}
    bundle.update(fit_anomaly_monitor(df, numeric))
    return bundle


def save_model(bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, MODEL_PATH)


def load_or_train() -> dict[str, Any]:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime >= DATA_PATH.stat().st_mtime:
        cached = joblib.load(MODEL_PATH)
        _, numeric_features, categorical_features = prepare_dataset(load_dataset())
        expected_features = numeric_features + categorical_features
        if (
            "model" in cached
            and "anomaly_detector" in cached
            and cached.get("features") == expected_features
        ):
            return cached
    bundle = train_model()
    save_model(bundle)
    return bundle


def predict_application(bundle: dict[str, Any], application: dict[str, Any]) -> dict[str, Any]:
    return predict_bundle_application(bundle, application)


def explain_application(bundle: dict[str, Any], application: dict[str, Any]) -> pd.DataFrame:
    base = predict_application(bundle, application)["approval_probability"]
    typical = {feature: float(bundle["data"][feature].median()) if feature in bundle["numeric"] else bundle["data"][feature].mode(dropna=True).iloc[0] for feature in bundle["features"]}
    rows = []
    for feature in bundle["features"]:
        counterfactual = application.copy()
        counterfactual[feature] = typical[feature]
        delta = base - predict_application(bundle, counterfactual)["approval_probability"]
        rows.append({"factor": feature.replace("_", " ").title(), "impact": delta, "direction": "Positive" if delta >= 0 else "Negative"})
    return pd.DataFrame(rows).sort_values("impact", key=np.abs, ascending=False).head(8)


def score_anomaly(bundle: dict[str, Any], application: dict[str, Any]) -> dict[str, Any]:
    row = pd.DataFrame([application])[bundle["numeric"]]
    matrix = bundle["anomaly_preprocessor"].transform(row)
    decision = float(bundle["anomaly_detector"].decision_function(matrix)[0])
    detector = bundle["anomaly_detector"]
    reference = bundle["anomaly_table"]["AnomalyScore"].mean()
    score = float(np.clip(reference + 35 * (detector.offset_ - decision), 0, 100))
    flagged = bool(detector.predict(matrix)[0] == -1)
    return {"score": round(score, 1), "status": "Review" if flagged else "Typical", "flagged": flagged}
