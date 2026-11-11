# LoanLens — loan approval and financial risk analysis

LoanLens is a local Streamlit prototype for loan approval prediction and financial risk analysis. It provides a dashboard, application assessment, prediction explanation, risk review, SQLite application history, model comparison, and project information. The application assessment includes an indicative monthly payment, debt ratios, and cash-flow affordability summary; entered non-debt expenses are used only for that summary and are not sent to the prediction model.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown by Streamlit. The first launch trains and caches four classification models from `data/Loan.csv`, selects the model with the highest ROC-AUC on a stratified holdout set, and saves artifacts under `models/`. New assessments are stored in the local SQLite database at `data/loaniq.db`.

## Dataset and model

The application uses the supplied `data/Loan.csv` dataset: 20,000 rows and 36 columns. The exact 16,000-row training split and 4,000-row testing split are included as `data/train.csv` and `data/test.csv`. Full column, target, split, and evaluation details are documented in `MODEL_DETAILS.md`.

The application compares logistic regression, a decision tree, a random forest, and gradient boosting. Preprocessing is fitted within each training pipeline; numeric values are median-imputed and standardized, categorical values are most-frequent-imputed and one-hot encoded. The stratified 80/20 test set is used to report accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, ROC curve, and classification report. Model selection uses ROC-AUC, not accuracy alone. The comparison and chosen model are saved under `models/`.

## Features

- Dashboard with saved assessment KPIs and historical dataset charts.
- New loan assessment with input checks, approval/rejection probabilities, local factor comparisons, anomaly review, and affordability calculations.
- Separate prediction and financial-risk screens.
- SQLite assessment history with decision/risk filters, ID lookup, details, CSV export, and confirmed deletion.
- Model performance comparison and held-out diagnostic charts.
- About page describing assumptions and responsible use.

The affordability summary estimates the proposed monthly payment from the entered loan amount, annual interest rate, and term. The supplied data records interest rates as decimal fractions. The calculation combines the estimate with existing monthly debt payments, monthly income, and the optional monthly expense input. These calculations are separate from the model's approval probability and do not determine an approval decision. Financial risk scores are rule-based indicators, not calibrated probabilities.

Age and marital status are not used as model inputs. The local history database retains the model input values and generated assessment to support application review. Do not enter real applicant or financial data: this prototype has no authentication or production data-security controls.

## Financial formulas

- DTI (%) = existing monthly debt payments ÷ gross monthly income × 100.
- EMI = `P × r × (1+r)^n / ((1+r)^n - 1)`, where `P` is requested principal, `r` is monthly interest rate, and `n` is number of monthly payments. At zero interest, EMI is `P ÷ n`.
- Loan-to-income = requested loan amount ÷ annual income.
- Payment burden (%) = (existing monthly debt payments + proposed EMI) ÷ monthly income × 100.
- Disposable income = monthly income − other monthly expenses − existing monthly debt payments − proposed EMI.

The form interprets rate inputs as annual decimal fractions (`0.24` means `24%`) and loan duration as months. The 40% of monthly income payment guideline is illustrative. The 0–100 risk score is an uncalibrated sum of capped financial indicators and is not a probability.

## Project layout

- `app.py` — Streamlit navigation, application form, dashboard, and result views.
- `loan_model.py` — dataset loading, model inference, anomaly monitor, and prediction explanations.
- `src/train_model.py` — multi-model training, held-out evaluation, model selection, and serialization.
- `src/prediction.py` — model input conversion and validation helpers.
- `src/risk_analysis.py` — financial formulas and transparent risk scoring.
- `src/database.py` — local SQLite persistence, summary, and export helpers.
- `src/evaluation.py` — Plotly model evaluation charts.
- `data/Loan.csv` — supplied dataset; `data/loaniq.db` — local assessment history created at runtime.
- `tests/` — prediction, risk, model comparison, and database tests.

Model metric values are generated from the current dataset and training split and are shown in the Model Performance page; they are not hard-coded into this README.

## Limitations and future work

The supplied sample data and holdout split do not establish real-world lending performance, calibration, fairness, or regulatory compliance. The prototype has no authentication, encryption-at-rest, or production access controls. Verify dataset provenance and license before redistribution; do not enter real applicant data or use output as the sole basis for a lending decision. Future work could include independently sourced and licensed data, fairness and calibration studies, temporal validation, SHAP explanations, authentication, and secure deployment.

Run the tests with:

```bash
python -m pytest -q
```

## Important limitation

These files are assumed to be public/sample Kaggle data and should be checked against the original Kaggle license and attribution requirements before redistribution. This is a demonstration and decision-support prototype. It does not establish fairness, legal compliance, accuracy for real lending, or production readiness. Never use it as the sole basis for a credit or lending decision.
