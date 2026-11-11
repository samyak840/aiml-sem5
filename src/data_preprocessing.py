"""Data loading, cleaning, and preprocessing pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALL_FEATURES, CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES,
    RANDOM_SEED, TARGET, TEST_SIZE,
)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load and lightly clean Loan.csv."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()
    return df


# ── Preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric + categorical features."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


# ── Train/test split ─────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame | None = None):
    """Return X_train, X_test, y_train, y_test with stratified split."""
    if df is None:
        df = load_data()
    # Ensure only known features are present (handle missing cols gracefully)
    available = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available]
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )


# ── Feature value helpers ─────────────────────────────────────────────────────

def get_feature_ranges(df: pd.DataFrame) -> dict:
    """Return min/max/median/options per feature for form defaults."""
    ranges = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            ranges[col] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            ranges[col] = {
                "options": sorted(df[col].dropna().astype(str).unique().tolist()),
            }
    return ranges
