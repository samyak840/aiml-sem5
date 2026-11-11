# LoanLens model and data details

## Purpose

LoanLens provides an indicative loan approval probability and a separate financial risk view for human review. It is a demonstration project, not a lending policy or production credit system.

## Dataset

- Source file: `data/Loan.csv`
- Rows: 20,000
- Target: `LoanApproved` (`1` = approved, `0` = rejected)
- Training rows: 16,000
- Testing rows: 4,000
- Split: 80% training and 20% testing
- Split method: stratified random split
- Random seed: `42`

The project uses only this dataset in the application. The older `loan_approval_dataset.csv` and `loan_applications_synthetic.csv` remain in the folder but are not used by the website.

## Columns used for training and testing

The same input columns are used for training and testing. The target column is kept separate from the inputs.

### Numeric inputs

`AnnualIncome`, `CreditScore`, `Experience`, `LoanAmount`, `LoanDuration`, `NumberOfDependents`, `MonthlyDebtPayments`, `CreditCardUtilizationRate`, `NumberOfOpenCreditLines`, `NumberOfCreditInquiries`, `DebtToIncomeRatio`, `BankruptcyHistory`, `PreviousLoanDefaults`, `PaymentHistory`, `LengthOfCreditHistory`, `SavingsAccountBalance`, `CheckingAccountBalance`, `TotalAssets`, `TotalLiabilities`, `MonthlyIncome`, `UtilityBillsPaymentHistory`, `JobTenure`, `NetWorth`, `BaseInterestRate`, `InterestRate`, `MonthlyLoanPayment`, `TotalDebtToIncomeRatio`

### Categorical inputs

`EmploymentStatus`, `EducationLevel`, `HomeOwnershipStatus`, `LoanPurpose`

## Columns excluded from model inputs

- `LoanApproved`: the prediction target
- `RiskScore`: excluded to reduce target leakage because it is outcome-like and closely related to risk
- `ApplicationDate`: excluded because the current MVP does not model time-based drift
- `Age` and `MaritalStatus`: excluded from model inputs to avoid using sensitive applicant attributes for automated eligibility predictions

## Preparation

- Numeric missing values use the training-set median.
- Categorical missing values use the most common training-set category.
- Numeric fields are standardised.
- Categorical fields are one-hot encoded.
- Class balancing is enabled during training.

## Evaluation

The test set is held out during training. Four candidate models are compared using accuracy, precision, recall, F1 score, and ROC-AUC. The highest-ROC-AUC model is used for inference. The Model Performance screen also shows its confusion matrix, ROC curve, and classification report. All results are dataset- and split-specific benchmarks.

## Risk analysis

The financial risk score is a separate, rule-based score from 0 to 100. It combines DTI (25 points), loan-to-income (20), credit score (25), payment burden (15), defaults and bankruptcy (10), and credit utilisation (5); component scores are capped before summing. The displayed approval and rejection probabilities come from the selected classifier and are not calibrated repayment probabilities. The risk monitor also highlights statistically unusual application profiles for human review. These flags do not prove fraud because the supplied files do not contain transaction-level fraud labels.

The Loan assessment screen separately estimates the proposed monthly payment using the standard amortisation formula and shows existing debt-to-income, total payment burden, loan-to-income, and monthly headroom after entered expenses and debt payments. Other monthly expenses are user-entered and are not model features. The 40% suggested payment guideline is illustrative only; these calculations do not alter the model output or represent lending policy.

The form treats interest-rate inputs as annual decimal fractions (for example, `0.24` means 24%) and loan duration as months. DTI is existing monthly debt divided by gross monthly income; payment burden is existing monthly debt plus proposed EMI divided by monthly income; loan-to-income is requested principal divided by annual income; disposable income subtracts entered non-debt expenses and all monthly EMIs from monthly income. For zero interest, EMI is principal divided by the number of payments.

The Model Performance screen compares logistic regression, a decision tree, a random forest, and gradient boosting using a stratified 80/20 split. The highest-ROC-AUC model is used for application predictions. Accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, ROC curve, and a classification report are displayed from held-out predictions. Results are specific to this dataset and split.

## Responsible-use note

The dataset license and provenance should be checked against the original Kaggle page. The project has not established fairness, legal compliance, calibration, or production readiness. Do not use it as the sole basis for a real lending decision.
