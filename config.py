"""LoanIQ – global configuration constants."""
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

DATA_PATH = DATA_DIR / "Loan.csv"
DB_PATH = DATA_DIR / "loaniq.db"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

# ── Dataset ─────────────────────────────────────────────────────────────────
TARGET = "LoanApproved"
EXCLUDED_COLUMNS = {TARGET, "RiskScore", "ApplicationDate"}
RANDOM_SEED = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "AnnualIncome", "CreditScore", "Experience", "LoanAmount",
    "LoanDuration", "NumberOfDependents", "MonthlyDebtPayments",
    "CreditCardUtilizationRate", "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries", "DebtToIncomeRatio", "BankruptcyHistory",
    "PreviousLoanDefaults", "PaymentHistory", "LengthOfCreditHistory",
    "SavingsAccountBalance", "CheckingAccountBalance", "TotalAssets",
    "TotalLiabilities", "MonthlyIncome", "UtilityBillsPaymentHistory",
    "JobTenure", "NetWorth", "BaseInterestRate", "InterestRate",
    "MonthlyLoanPayment", "TotalDebtToIncomeRatio",
]

CATEGORICAL_FEATURES = [
    "EmploymentStatus", "EducationLevel", "HomeOwnershipStatus", "LoanPurpose",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_LOW_MAX = 35       # 0–34  → Low
RISK_HIGH_MIN = 65      # 65–100 → High
# 35–64 → Medium

# ── Default assumptions ──────────────────────────────────────────────────────
DEFAULT_INTEREST_RATE = 10.5   # % per annum (shown in form as suggestion)

# ── Colour palette ───────────────────────────────────────────────────────────
COLOR_PRIMARY   = "#0a1628"
COLOR_SECONDARY = "#1e3a5f"
COLOR_ACCENT    = "#00d4aa"
COLOR_SUCCESS   = "#10b981"
COLOR_WARNING   = "#f59e0b"
COLOR_DANGER    = "#ef4444"
COLOR_BG        = "#f0f4f8"
COLOR_CARD      = "#ffffff"
COLOR_TEXT      = "#1e293b"
COLOR_MUTED     = "#64748b"

# chart colours
COLOR_APPROVED  = "#10b981"
COLOR_REJECTED  = "#ef4444"
COLOR_LOW_RISK  = "#10b981"
COLOR_MED_RISK  = "#f59e0b"
COLOR_HIGH_RISK = "#ef4444"
