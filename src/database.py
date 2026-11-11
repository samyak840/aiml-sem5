"""SQLite persistence layer for LoanIQ application history."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS applications (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at               TEXT    NOT NULL,

    -- Applicant meta (not used in model)
    applicant_name           TEXT,

    -- Numeric model features
    age                      REAL,
    annual_income            REAL,
    credit_score             REAL,
    experience               REAL,
    loan_amount              REAL,
    loan_duration            REAL,
    number_of_dependents     REAL,
    monthly_debt_payments    REAL,
    credit_card_utilization  REAL,
    open_credit_lines        REAL,
    credit_inquiries         REAL,
    debt_to_income_ratio     REAL,
    bankruptcy_history       REAL,
    previous_loan_defaults   REAL,
    payment_history          REAL,
    length_credit_history    REAL,
    savings_balance          REAL,
    checking_balance         REAL,
    total_assets             REAL,
    total_liabilities        REAL,
    monthly_income           REAL,
    utility_payment_history  REAL,
    job_tenure               REAL,
    net_worth                REAL,
    base_interest_rate       REAL,
    interest_rate            REAL,
    monthly_loan_payment     REAL,
    total_dti                REAL,

    -- Categorical model features
    employment_status        TEXT,
    education_level          TEXT,
    marital_status           TEXT,
    home_ownership           TEXT,
    loan_purpose             TEXT,

    -- Prediction output
    approval_probability     REAL,
    approved                 INTEGER,  -- 1 / 0
    model_used               TEXT,

    -- Financial indicators (computed)
    risk_score               REAL,
    risk_category            TEXT,
    dti                      REAL,
    emi                      REAL,
    lti                      REAL,
    payment_burden           REAL,
    disposable_income        REAL
);
"""

CREATE_ASSESSMENTS_SQL = """
CREATE TABLE IF NOT EXISTS assessments (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at               TEXT NOT NULL,
    application_json         TEXT NOT NULL,
    monthly_expenses         REAL NOT NULL DEFAULT 0,
    approved                 INTEGER NOT NULL,
    approval_probability     REAL NOT NULL,
    rejection_probability    REAL NOT NULL,
    model_used               TEXT NOT NULL,
    risk_score               REAL NOT NULL,
    risk_category            TEXT NOT NULL,
    dti                      REAL NOT NULL,
    emi                      REAL NOT NULL,
    lti                      REAL NOT NULL,
    payment_burden           REAL NOT NULL,
    disposable_income        REAL NOT NULL,
    anomaly_score            REAL NOT NULL,
    anomaly_status           TEXT NOT NULL
);
"""

ASSESSMENT_FIELDS = (
    "monthly_expenses",
    "approved",
    "approval_probability",
    "rejection_probability",
    "model_used",
    "risk_score",
    "risk_category",
    "dti",
    "emi",
    "lti",
    "payment_burden",
    "disposable_income",
    "anomaly_score",
    "anomaly_status",
)


# ── Connection helper ─────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the application and assessment tables if they don't exist."""
    with _connect() as conn:
        conn.execute(CREATE_SQL)
        conn.execute(CREATE_ASSESSMENTS_SQL)
        conn.commit()


def save_assessment(record: dict[str, Any]) -> int:
    """Persist one model assessment and its application inputs."""
    init_db()
    values = {field: record[field] for field in ASSESSMENT_FIELDS}
    values["application_json"] = json.dumps(record["application"], allow_nan=False)
    values["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = ("created_at", "application_json", *ASSESSMENT_FIELDS)
    placeholders = ", ".join(f":{column}" for column in columns)
    with _connect() as conn:
        cursor = conn.execute(
            f"INSERT INTO assessments ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        return int(cursor.lastrowid)


def get_all_assessments() -> pd.DataFrame:
    """Return saved assessments, newest first."""
    init_db()
    with _connect() as conn:
        return pd.read_sql_query("SELECT * FROM assessments ORDER BY id DESC", conn)


def get_assessment_by_id(assessment_id: int) -> dict[str, Any] | None:
    """Return a single saved assessment, or None if it does not exist."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE id = ?", (assessment_id,)
        ).fetchone()
    return dict(row) if row else None


def get_assessment_stats() -> dict[str, Any]:
    """Return dashboard aggregates for saved model assessments."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(approved), 0) AS total_approved,
                COALESCE(SUM(1 - approved), 0) AS total_rejected,
                ROUND(AVG(risk_score), 1) AS avg_risk,
                COALESCE(SUM(CASE WHEN risk_category = 'Low' THEN 1 ELSE 0 END), 0) AS low_risk,
                COALESCE(SUM(CASE WHEN risk_category = 'Medium' THEN 1 ELSE 0 END), 0) AS med_risk,
                COALESCE(SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END), 0) AS high_risk
            FROM assessments
            """
        ).fetchone()
    return dict(row) if row else {
        "total": 0, "total_approved": 0, "total_rejected": 0,
        "avg_risk": None, "low_risk": 0, "med_risk": 0, "high_risk": 0,
    }


def delete_assessment(assessment_id: int) -> None:
    """Delete one saved assessment by ID."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,))


def export_assessments_to_csv() -> str:
    """Export saved assessments to CSV content."""
    return get_all_assessments().to_csv(index=False)


# ── Write ─────────────────────────────────────────────────────────────────────

def save_application(record: dict[str, Any]) -> int:
    """
    Insert a new application record.
    ``record`` should be a flat dict with keys matching the column names.
    Returns the new row's id.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO applications (
                created_at, applicant_name,
                age, annual_income, credit_score, experience, loan_amount,
                loan_duration, number_of_dependents, monthly_debt_payments,
                credit_card_utilization, open_credit_lines, credit_inquiries,
                debt_to_income_ratio, bankruptcy_history, previous_loan_defaults,
                payment_history, length_credit_history, savings_balance,
                checking_balance, total_assets, total_liabilities, monthly_income,
                utility_payment_history, job_tenure, net_worth, base_interest_rate,
                interest_rate, monthly_loan_payment, total_dti,
                employment_status, education_level, marital_status,
                home_ownership, loan_purpose,
                approval_probability, approved, model_used,
                risk_score, risk_category, dti, emi, lti,
                payment_burden, disposable_income
            ) VALUES (
                :created_at, :applicant_name,
                :age, :annual_income, :credit_score, :experience, :loan_amount,
                :loan_duration, :number_of_dependents, :monthly_debt_payments,
                :credit_card_utilization, :open_credit_lines, :credit_inquiries,
                :debt_to_income_ratio, :bankruptcy_history, :previous_loan_defaults,
                :payment_history, :length_credit_history, :savings_balance,
                :checking_balance, :total_assets, :total_liabilities, :monthly_income,
                :utility_payment_history, :job_tenure, :net_worth, :base_interest_rate,
                :interest_rate, :monthly_loan_payment, :total_dti,
                :employment_status, :education_level, :marital_status,
                :home_ownership, :loan_purpose,
                :approval_probability, :approved, :model_used,
                :risk_score, :risk_category, :dti, :emi, :lti,
                :payment_burden, :disposable_income
            )
            """,
            {**record, "created_at": now},
        )
        conn.commit()
        return cur.lastrowid


# ── Read ──────────────────────────────────────────────────────────────────────

def get_all_applications() -> pd.DataFrame:
    """Return all saved applications as a DataFrame."""
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM applications ORDER BY id DESC", conn)
    return df


def get_application_by_id(app_id: int) -> dict | None:
    """Return a single application as a dict, or None if not found."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,)
        ).fetchone()
    return dict(row) if row else None


def get_summary_stats() -> dict:
    """Return aggregate stats for the dashboard KPI cards."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                    AS total,
                SUM(approved)                               AS total_approved,
                SUM(1 - approved)                           AS total_rejected,
                ROUND(AVG(risk_score), 1)                   AS avg_risk,
                SUM(CASE WHEN risk_category='Low'    THEN 1 ELSE 0 END) AS low_risk,
                SUM(CASE WHEN risk_category='Medium' THEN 1 ELSE 0 END) AS med_risk,
                SUM(CASE WHEN risk_category='High'   THEN 1 ELSE 0 END) AS high_risk
            FROM applications
            """
        ).fetchone()
    return dict(row) if row else {
        "total": 0, "total_approved": 0, "total_rejected": 0,
        "avg_risk": None, "low_risk": 0, "med_risk": 0, "high_risk": 0,
    }


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_application(app_id: int) -> None:
    """Hard-delete a single application by id."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
        conn.commit()


def delete_all_applications() -> None:
    """Delete every row (useful for demo reset)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM applications")
        conn.commit()


# ── Export ────────────────────────────────────────────────────────────────────

def export_to_csv(filepath: str | Path | None = None) -> str:
    """Export all applications to CSV. Returns the csv content as a string."""
    df = get_all_applications()
    if filepath:
        df.to_csv(filepath, index=False)
    return df.to_csv(index=False)
