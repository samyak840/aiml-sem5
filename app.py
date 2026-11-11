from __future__ import annotations

import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from loan_model import (
    explain_application,
    load_or_train,
    predict_application,
    score_anomaly,
)
from src.database import (
    delete_assessment,
    export_assessments_to_csv,
    get_all_assessments,
    get_assessment_by_id,
    get_assessment_stats,
    save_assessment,
)
from src.evaluation import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curve,
)
from src.explainability import get_rf_feature_importance
from src.prediction import validate_application
from src.risk_analysis import (
    calculate_disposable_income,
    calculate_dti,
    calculate_emi,
    calculate_lti,
    calculate_payment_burden,
    loan_affordability_check,
)
from src.train_model import load_or_train as load_or_train_comparison

st.set_page_config(
    page_title="LoanLens",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { max-width: 1240px; padding: 1.4rem 2.4rem 3rem; }
[data-testid="stSidebar"] { background: #102a43; }
[data-testid="stSidebar"] * { color: #e6f4f1 !important; }
.brand { padding: .8rem 0 1.5rem; border-bottom: 1px solid rgba(255,255,255,.2); margin-bottom: 1.2rem; }
.brand h1 { color: white; margin: 0; font-size: 1.9rem; letter-spacing: -.05em; }
.brand p { color: #a5ded6; margin: .2rem 0 0; font-size: .82rem; }
.hero { padding: 2.4rem 2.5rem; border-radius: 22px; background: linear-gradient(120deg,#102a43,#176b70); color: white; margin-bottom: 1.5rem; box-shadow: 0 15px 35px rgba(16,42,67,.16); }
.hero h1 { margin: .3rem 0; font-size: 2.8rem; letter-spacing: -.05em; }
.hero p { color: #d8f2ef; margin: 0; font-size: 1.08rem; max-width: 760px; }
.eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; color: #72d2c5; font-weight: 750; }
.metric-card { background: white; border: 1px solid #d9e7eb; border-radius: 15px; padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(16,42,67,.05); }
.metric-card .label { color: #627d98; font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; }
.metric-card .value { color: #102a43; font-size: 1.65rem; font-weight: 750; margin-top: .25rem; }
.result { padding: 1.4rem 1.5rem; border-radius: 17px; color: white; box-shadow: 0 10px 25px rgba(16,42,67,.12); }
.approved { background: linear-gradient(135deg,#087f70,#19a786); }
.rejected { background: linear-gradient(135deg,#9e3f49,#d66063); }
.review { background: linear-gradient(135deg,#8a5b12,#d28b23); }
.note { color: #627d98; font-size: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle():
    bundle = load_or_train()
    comparison = load_or_train_comparison()
    bundle["model"] = comparison["best_pipeline"]
    bundle["metrics"] = comparison["best_metrics"]
    bundle["model_comparison"] = comparison["comparison"]
    bundle["rf_pipeline"] = comparison["rf_pipeline"]
    bundle["best_model_name"] = comparison["best_model_name"]
    bundle["train_rows"] = comparison["train_rows"]
    bundle["test_rows"] = comparison["test_rows"]
    bundle["evaluation_y_test"] = comparison["best_metrics"]["y_test"]
    bundle["evaluation_y_proba"] = comparison["best_metrics"]["y_proba"]
    return bundle


bundle = get_bundle()
df = bundle["data"]
metrics = bundle["metrics"]
history = get_all_assessments()
history_stats = get_assessment_stats()


def cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def header(eyebrow, title, description):
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h1 style="color:#102a43;margin:.15rem 0 .35rem">{title}</h1>'
        f'<p class="note">{description}</p>',
        unsafe_allow_html=True,
    )


def application_form():
    application = {}
    numeric = set(bundle["numeric"])
    sections = {
        "Personal and employment": [
            "NumberOfDependents", "EducationLevel",
            "EmploymentStatus", "Experience", "JobTenure",
        ],
        "Income and assets": [
            "AnnualIncome", "MonthlyIncome", "SavingsAccountBalance",
            "CheckingAccountBalance", "TotalAssets", "TotalLiabilities", "NetWorth",
        ],
        "Loan request": [
            "LoanAmount", "LoanDuration", "LoanPurpose", "InterestRate",
            "BaseInterestRate", "MonthlyLoanPayment",
        ],
        "Existing obligations": [
            "MonthlyDebtPayments", "DebtToIncomeRatio", "TotalDebtToIncomeRatio",
            "CreditCardUtilizationRate", "NumberOfOpenCreditLines",
            "NumberOfCreditInquiries",
        ],
        "Credit history and housing": [
            "CreditScore", "PreviousLoanDefaults", "BankruptcyHistory",
            "PaymentHistory", "LengthOfCreditHistory",
            "UtilityBillsPaymentHistory", "HomeOwnershipStatus",
        ],
    }
    for section, features in sections.items():
        st.subheader(section)
        cols = st.columns(2)
        for index, feature in enumerate(features):
            with cols[index % 2]:
                label = re.sub(r"(?<!^)(?=[A-Z])", " ", feature).replace(
                    "_", " "
                ).title()
                if feature in numeric:
                    if feature == "MonthlyLoanPayment":
                        annual_rate = float(application["InterestRate"])
                        estimated_emi = calculate_emi(
                            float(application["LoanAmount"]),
                            annual_rate * 100 if annual_rate <= 1 else annual_rate,
                            max(1, round(float(application["LoanDuration"]))),
                        )
                        application[feature] = estimated_emi
                        st.metric(label, f"${estimated_emi:,.2f}")
                        continue
                    series = pd.to_numeric(df[feature], errors="coerce").dropna()
                    low, high, default = (
                        float(series.min()),
                        float(series.max()),
                        float(series.median()),
                    )
                    default = min(max(default, low), high)
                    step = max((high - low) / 100, 0.01)
                    if feature in {"InterestRate", "BaseInterestRate"}:
                        label += " (annual decimal; 0.24 = 24%)"
                    application[feature] = st.number_input(
                        label,
                        min_value=low,
                        max_value=high,
                        value=default,
                        step=step,
                        key=f"input_{feature}_{low:.6g}_{high:.6g}",
                    )
                else:
                    options = sorted(df[feature].dropna().astype(str).unique().tolist())
                    application[feature] = st.selectbox(
                        label, options, key=f"input_{feature}"
                    )
    st.subheader("Monthly budget")
    monthly_expenses = st.number_input(
        "Other monthly expenses (excluding debt payments)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help="Used only for the affordability estimate; this value is not sent to the prediction model.",
    )
    return application, monthly_expenses


def calculate_financials(application, monthly_expenses):
    monthly_income = float(application["MonthlyIncome"])
    existing_emi = float(application["MonthlyDebtPayments"])
    interest_rate = float(application["InterestRate"])
    annual_rate_percent = interest_rate * 100 if interest_rate <= 1 else interest_rate
    proposed_emi = calculate_emi(
        float(application["LoanAmount"]),
        annual_rate_percent,
        max(1, round(float(application["LoanDuration"]))),
    )
    dti = calculate_dti(existing_emi, monthly_income)
    lti = calculate_lti(
        float(application["LoanAmount"]), float(application["AnnualIncome"])
    )
    payment_burden = calculate_payment_burden(
        existing_emi, proposed_emi, monthly_income
    )
    disposable_income = calculate_disposable_income(
        monthly_income, monthly_expenses, existing_emi + proposed_emi
    )
    affordability = loan_affordability_check(
        monthly_income,
        proposed_emi,
        existing_emi=existing_emi,
        monthly_expenses=monthly_expenses,
    )
    return {
        "emi": proposed_emi,
        "dti": dti,
        "lti": lti,
        "payment_burden": payment_burden,
        "disposable_income": disposable_income,
        "affordability": affordability,
    }


def show_prediction(result, application):
    title = "Indicative approval" if result["approved"] else "Indicative rejection"
    result_class = "approved" if result["approved"] else "rejected"
    st.markdown(
        f'<div class="result {result_class}"><div class="eyebrow">Model result</div>'
        f"<h2>{title}</h2><p>Approval probability: "
        f"<b>{result['approval_probability']:.1%}</b> · Rejection probability: "
        f"<b>{result['rejection_probability']:.1%}</b> · Financial risk score: "
        f"<b>{result['risk_score']:.1f}/100 ({result['risk_band']})</b></p></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Model probabilities reflect this dataset and model; they are not calibrated "
        "repayment probabilities or guaranteed real-world outcomes."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Positive indicators")
        positive = explain_application(bundle, application)
        positive_rows = positive.loc[positive["impact"] > 0].head(5)
        if positive_rows.empty:
            st.write("No positive local comparison signals were identified.")
        else:
            st.dataframe(
                positive_rows[["factor", "impact"]].assign(
                    impact=lambda values: values["impact"].map(
                        lambda value: f"{value * 100:+.1f} percentage points"
                    )
                ),
                hide_index=True,
                width="stretch",
            )
    with right:
        st.subheader("Potential concerns")
        negative_rows = positive.loc[positive["impact"] < 0].head(5)
        if negative_rows.empty:
            st.write("No negative local comparison signals were identified.")
        else:
            st.dataframe(
                negative_rows[["factor", "impact"]].assign(
                    impact=lambda values: values["impact"].map(
                        lambda value: f"{value * 100:+.1f} percentage points"
                    )
                ),
                hide_index=True,
                width="stretch",
            )
    st.caption(
        "Factors are local model comparisons, not causal explanations. "
        "Feature comparisons can be correlated with other inputs."
    )


with st.sidebar:
    st.markdown(
        '<div class="brand"><h1>LoanLens</h1>'
        "<p>Loan approval and risk analysis</p></div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "New Loan Application",
            "Loan Prediction",
            "Financial Risk Analysis",
            "Application History",
            "Model Performance",
            "About Project",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset used")
    st.write("Loan.csv")
    st.caption(f"{len(df):,} applications · {len(bundle['features'])} model inputs")
    st.caption(f"{history_stats['total']:,} saved assessments")
    st.warning("Decision support only. Not an actual lending decision.")


if page == "Dashboard":
    st.markdown(
        '<div class="hero"><div class="eyebrow">Financial decision support</div>'
        "<h1>Understand every loan application</h1>"
        "<p>Review model results, affordability indicators, and saved assessment trends "
        "in one place.</p></div>",
        unsafe_allow_html=True,
    )
    header(
        "Dashboard",
        "Portfolio overview",
        "Prediction counts below come from saved demo assessments. Dataset outcome charts "
        "are historical labels, not live portfolio outcomes.",
    )
    cards([
        ("Applications evaluated", f"{history_stats['total']:,}"),
        ("Predicted approvals", f"{history_stats['total_approved']:,}"),
        ("Predicted rejections", f"{history_stats['total_rejected']:,}"),
        (
            "Average financial risk",
            f"{history_stats['avg_risk']:.1f}/100"
            if history_stats["avg_risk"] is not None else "—",
        ),
        ("Low-risk assessments", f"{history_stats['low_risk']:,}"),
        ("High-risk assessments", f"{history_stats['high_risk']:,}"),
    ])
    st.write("")
    left, right = st.columns(2)
    with left:
        st.subheader("Saved prediction outcomes")
        if history.empty:
            st.info("Save an assessment to populate live application-history charts.")
        else:
            outcome = (
                history["approved"]
                .map({0: "Predicted rejection", 1: "Predicted approval"})
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Applications")
            )
            st.plotly_chart(
                px.pie(
                    outcome,
                    names="Prediction",
                    values="Applications",
                    hole=0.6,
                    color="Prediction",
                    color_discrete_map={
                        "Predicted approval": "#0d9488",
                        "Predicted rejection": "#d66063",
                    },
                ),
                width="stretch",
            )
    with right:
        st.subheader("Saved financial risk categories")
        if history.empty:
            st.info("Risk distribution appears after the first saved assessment.")
        else:
            risk_counts = (
                history["risk_category"]
                .value_counts()
                .reindex(["Low", "Medium", "High"], fill_value=0)
                .rename_axis("Risk category")
                .reset_index(name="Applications")
            )
            st.plotly_chart(
                px.bar(
                    risk_counts,
                    x="Risk category",
                    y="Applications",
                    color="Risk category",
                    color_discrete_map={
                        "Low": "#0d9488",
                        "Medium": "#d28b23",
                        "High": "#d66063",
                    },
                ),
                width="stretch",
            )
    st.divider()
    st.subheader("Historical dataset outcomes")
    outcome = (
        df["LoanApproved"]
        .map({0: "Rejected", 1: "Approved"})
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Applications")
    )
    chart_df = df.sample(min(1000, len(df)), random_state=42).copy()
    chart_df["Outcome"] = chart_df["LoanApproved"].map({0: "Rejected", 1: "Approved"})
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.pie(
                outcome,
                names="Outcome",
                values="Applications",
                hole=0.6,
                color="Outcome",
                color_discrete_map={"Approved": "#0d9488", "Rejected": "#d66063"},
                title="Dataset approval outcomes",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.histogram(
                chart_df,
                x="LoanAmount",
                color="Outcome",
                nbins=30,
                color_discrete_map={"Approved": "#0d9488", "Rejected": "#d66063"},
                title="Historical loan amounts by outcome",
            ),
            width="stretch",
        )
    st.plotly_chart(
        px.histogram(
            chart_df,
            x="AnnualIncome",
            color="Outcome",
            nbins=30,
            color_discrete_map={"Approved": "#0d9488", "Rejected": "#d66063"},
            title="Historical annual income by outcome",
        ),
        width="stretch",
    )
    st.subheader("Credit and employment patterns in the source dataset")
    left, right = st.columns(2)
    with left:
        credit_df = df.copy()
        credit_df["Outcome"] = credit_df["LoanApproved"].map(
            {0: "Rejected", 1: "Approved"}
        )
        st.plotly_chart(
            px.histogram(
                credit_df,
                x="CreditScore",
                color="Outcome",
                nbins=25,
                barmode="overlay",
                opacity=0.75,
                title="Credit score distribution by historical outcome",
                color_discrete_map={"Approved": "#0d9488", "Rejected": "#d66063"},
            ),
            width="stretch",
        )
    with right:
        employment = (
            df.groupby("EmploymentStatus", as_index=False)["LoanApproved"]
            .mean()
            .assign(LoanApproved=lambda values: values["LoanApproved"] * 100)
        )
        st.plotly_chart(
            px.bar(
                employment,
                x="EmploymentStatus",
                y="LoanApproved",
                title="Historical approval rate by employment status",
                labels={"LoanApproved": "Approval rate (%)"},
                color="LoanApproved",
                color_continuous_scale="Teal",
            ),
            width="stretch",
        )
    st.caption(
        f"Current selected model: {bundle['best_model_name']} · "
        f"held-out ROC-AUC: {metrics['roc_auc']:.3f}. "
        "Dataset results are not evidence of real-world lending performance."
    )

elif page == "New Loan Application":
    header(
        "New loan application",
        "Assess a hypothetical applicant",
        "Provide the applicant and loan details for a model prediction and separate "
        "financial-risk calculation. No applicant name is collected.",
    )
    with st.form("loan_form"):
        application, monthly_expenses = application_form()
        submitted = st.form_submit_button(
            "Assess application", type="primary", width="stretch"
        )
    if submitted:
        errors = validate_application(application)
        if monthly_expenses < 0:
            errors.append("Other monthly expenses cannot be negative.")
        if errors:
            st.session_state.pop("last_assessment", None)
            st.error("Please correct the following inputs:\n\n" + "\n".join(
                f"- {message}" for message in errors
            ))
        else:
            result = predict_application(bundle, application)
            financials = calculate_financials(application, monthly_expenses)
            anomaly = score_anomaly(bundle, application)
            st.session_state.last_assessment = {
                "application": application,
                "monthly_expenses": float(monthly_expenses),
                "result": result,
                "financials": financials,
                "anomaly": anomaly,
            }
            st.session_state.pop("saved_assessment_id", None)

    assessment = st.session_state.get("last_assessment")
    if assessment:
        application = assessment["application"]
        result = assessment["result"]
        financials = assessment["financials"]
        anomaly = assessment["anomaly"]
        st.divider()
        show_prediction(result, application)
        st.subheader("Financial indicators")
        cards([
            ("Estimated proposed EMI", f"${financials['emi']:,.2f}"),
            ("Existing debt-to-income", f"{financials['dti']:.1f}%"),
            ("Payment burden", f"{financials['payment_burden']:.1f}%"),
            ("Loan-to-income", f"{financials['lti']:.2f}×"),
            ("Disposable monthly income", f"${financials['disposable_income']:,.2f}"),
            ("Profile anomaly monitor", anomaly["status"]),
        ])
        affordability = financials["affordability"]
        st.caption(
            f"Illustrative new-payment guideline (40% of monthly income): "
            f"${affordability['recommended_max_emi']:,.2f}."
        )
        if affordability["affordable"]:
            st.success("Entered monthly income covers the entered expenses and debt payments.")
        else:
            st.warning("Estimated monthly commitments exceed the entered monthly income.")
        if st.session_state.get("saved_assessment_id"):
            st.success(
                f"Assessment saved as ID #{st.session_state.saved_assessment_id}."
            )
        elif st.button("Save assessment to history", type="primary"):
            saved_id = save_assessment({
                "application": application,
                "monthly_expenses": assessment["monthly_expenses"],
                "approved": int(result["approved"]),
                "approval_probability": result["approval_probability"],
                "rejection_probability": result["rejection_probability"],
                "model_used": bundle["best_model_name"],
                "risk_score": result["risk_score"],
                "risk_category": result["risk_band"],
                "dti": financials["dti"],
                "emi": financials["emi"],
                "lti": financials["lti"],
                "payment_burden": financials["payment_burden"],
                "disposable_income": financials["disposable_income"],
                "anomaly_score": anomaly["score"],
                "anomaly_status": anomaly["status"],
            })
            st.session_state.saved_assessment_id = saved_id
            st.rerun()
        st.caption(
            "EMI assumes the entered annual rate and duration are fixed for the full term. "
            "The dataset stores rates as decimal fractions; the interface displays "
            "the source value. The illustrative 40% guideline is not a lending policy."
        )

elif page == "Loan Prediction":
    header(
        "Loan prediction",
        "Prediction result and explanation",
        "The model prediction is displayed separately from the arithmetic financial-risk score.",
    )
    assessment = st.session_state.get("last_assessment")
    if not assessment:
        st.info("Assess a new loan application to view its prediction here.")
    else:
        show_prediction(assessment["result"], assessment["application"])
        st.write(
            f"Selected model: **{bundle['best_model_name']}**. "
            "The prediction is based on the submitted model features."
        )
        st.dataframe(
            pd.DataFrame([{
                "Approval probability": assessment["result"]["approval_probability"],
                "Rejection probability": assessment["result"]["rejection_probability"],
                "Financial risk score": assessment["result"]["risk_score"],
                "Financial risk category": assessment["result"]["risk_band"],
                "Saved assessment ID": st.session_state.get("saved_assessment_id"),
            }]),
            hide_index=True,
            width="stretch",
        )

elif page == "Financial Risk Analysis":
    header(
        "Financial risk analysis",
        "Understand affordability and risk signals",
        "Financial scores are transparent, rule-based indicators, not calibrated "
        "probabilities or lending policy.",
    )
    assessment = st.session_state.get("last_assessment")
    if assessment:
        financials = assessment["financials"]
        result = assessment["result"]
        cards([
            ("Financial risk score", f"{result['risk_score']:.1f}/100"),
            ("Risk category", result["risk_band"]),
            ("Proposed EMI", f"${financials['emi']:,.2f}"),
            ("Payment burden", f"{financials['payment_burden']:.1f}%"),
        ])
        st.write(
            f"Existing DTI: **{financials['dti']:.1f}%** · Loan-to-income: "
            f"**{financials['lti']:.2f}×** · Monthly disposable income after entered "
            f"expenses and all EMIs: **${financials['disposable_income']:,.2f}**."
        )
        st.write(
            f"Profile anomaly monitor: **{assessment['anomaly']['status']}** "
            f"({assessment['anomaly']['score']:.1f}/100). Anomaly flags identify "
            "unusual records for human review; they do not prove fraud."
        )
        st.caption(
            "Financial risk score uses DTI (25 points), loan-to-income (20), "
            "credit score (25), payment burden (15), defaults and bankruptcy (10), "
            "and credit utilisation (5). Indicators are capped and summed to 0–100."
        )
    else:
        st.info("Assess an application to see its financial-risk breakdown.")
    st.divider()
    if history.empty:
        st.info("No saved assessments are available yet.")
    else:
        st.subheader("Saved assessment risk profile")
        st.plotly_chart(
            px.histogram(
                history,
                x="risk_score",
                color="risk_category",
                nbins=20,
                color_discrete_map={
                    "Low": "#0d9488",
                    "Medium": "#d28b23",
                    "High": "#d66063",
                },
                labels={"risk_score": "Financial risk score"},
            ),
            width="stretch",
        )

elif page == "Application History":
    header(
        "Application history",
        "Saved assessment records",
        "Records are stored locally in SQLite. Applicant names are not collected; "
        "application features are retained to support review.",
    )
    if history.empty:
        st.info("No assessments have been saved.")
    else:
        decision_filter = st.selectbox(
            "Prediction filter", ["All", "Predicted approval", "Predicted rejection"]
        )
        risk_filter = st.selectbox(
            "Risk category filter", ["All", "Low", "Medium", "High"]
        )
        filtered = history.copy()
        if decision_filter != "All":
            filtered = filtered[
                filtered["approved"] == (decision_filter == "Predicted approval")
            ]
        if risk_filter != "All":
            filtered = filtered[filtered["risk_category"] == risk_filter]
        display = filtered[
            [
                "id", "created_at", "approved", "approval_probability",
                "risk_score", "risk_category", "emi", "disposable_income",
            ]
        ].rename(columns={
            "id": "Application ID",
            "created_at": "Created (UTC)",
            "approved": "Predicted approval (1/0)",
            "approval_probability": "Approval probability",
            "risk_score": "Financial risk score",
            "risk_category": "Risk category",
            "emi": "Proposed EMI",
            "disposable_income": "Disposable income",
        })
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button(
            "Export all saved records as CSV",
            data=export_assessments_to_csv(),
            file_name="loanlens_assessments.csv",
            mime="text/csv",
            width="stretch",
        )
        st.subheader("Find assessment by ID")
        assessment_id = st.number_input(
            "Application ID",
            min_value=1,
            step=1,
            value=int(
                filtered.iloc[0]["id"] if not filtered.empty else history.iloc[0]["id"]
            ),
        )
        selected = get_assessment_by_id(int(assessment_id))
        if selected:
            st.json({
                "id": selected["id"],
                "created_at": selected["created_at"],
                "approved": bool(selected["approved"]),
                "approval_probability": selected["approval_probability"],
                "rejection_probability": selected["rejection_probability"],
                "risk_score": selected["risk_score"],
                "risk_category": selected["risk_category"],
                "financial_indicators": {
                    key: selected[key]
                    for key in (
                        "dti", "emi", "lti", "payment_burden",
                        "disposable_income", "monthly_expenses",
                    )
                },
                "application": json.loads(selected["application_json"]),
            })
            confirm_delete = st.checkbox(
                f"Confirm deletion of demo assessment #{int(assessment_id)}"
            )
            if st.button("Delete selected assessment", disabled=not confirm_delete):
                delete_assessment(int(assessment_id))
                if st.session_state.get("saved_assessment_id") == int(assessment_id):
                    st.session_state.pop("saved_assessment_id")
                st.success("Assessment deleted.")
                st.rerun()
        elif int(assessment_id) > 0:
            st.info("No assessment with that ID was found.")

elif page == "Model Performance":
    header(
        "Model performance",
        "Compare held-out model results",
        "Metrics are calculated on the stratified test split. The selected model is "
        "the highest-ROC-AUC candidate and powers new predictions.",
    )
    st.caption(
        f"Best model: {bundle['best_model_name']} · "
        f"{bundle['train_rows']:,} training rows · "
        f"{bundle['test_rows']:,} held-out test rows."
    )
    comparison = bundle["model_comparison"]
    st.plotly_chart(plot_model_comparison(comparison), width="stretch")
    st.dataframe(
        pd.DataFrame(comparison).T[
            ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ].rename(columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
        }).style.format("{:.3f}"),
        width="stretch",
    )
    left, right = st.columns(2)
    with left:
        predictions = (pd.Series(bundle["evaluation_y_proba"]) >= 0.5).astype(int)
        st.plotly_chart(
            plot_confusion_matrix(
                bundle["evaluation_y_test"],
                predictions,
                bundle["best_model_name"],
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            plot_roc_curve(
                bundle["evaluation_y_test"],
                bundle["evaluation_y_proba"],
                bundle["best_model_name"],
            ),
            width="stretch",
        )
    st.subheader("Selected model classification report")
    st.dataframe(
        pd.DataFrame(metrics["classification_report"]).T.style.format("{:.3f}"),
        width="stretch",
    )
    st.subheader("Random Forest global feature importance")
    importance = get_rf_feature_importance(bundle)
    if importance.empty:
        st.info("Feature importance is unavailable for the loaded Random Forest.")
    else:
        st.plotly_chart(
            plot_feature_importance(importance),
            width="stretch",
        )
    st.caption(
        "Feature importance describes associations used by the Random Forest. "
        "It does not explain an individual prediction or establish causation."
    )
    approval_rate = float(df["LoanApproved"].mean())
    st.info(
        f"The source dataset has an approval rate of {approval_rate:.1%}; accuracy "
        "alone can hide poor performance on a smaller class. Compare precision, "
        "recall, F1, ROC-AUC, and the confusion matrix together. These benchmark "
        "results do not establish production performance, calibration, or fairness."
    )

else:
    header(
        "About LoanLens",
        "A transparent decision-support prototype",
        "LoanLens demonstrates a machine-learning workflow and formula-based "
        "financial indicators; it is not a credit-scoring service.",
    )
    st.subheader("Dataset and model")
    st.write(
        "The application uses the supplied `data/Loan.csv` sample dataset. "
        "Four scikit-learn classifiers are compared using a fixed stratified "
        "train/test split; the model with the highest held-out ROC-AUC is selected. "
        "The target is `LoanApproved`. Dataset provenance and license conditions "
        "should be checked with the original provider before redistribution. Age "
        "and marital status are not used as prediction inputs."
    )
    st.subheader("Risk calculations and assumptions")
    st.write(
        "EMI uses fixed-rate monthly amortisation. DTI divides existing monthly debt "
        "payments by gross monthly income; payment burden includes existing debt and "
        "the proposed EMI; loan-to-income divides requested principal by annual "
        "income. Disposable income subtracts user-entered non-debt expenses and all "
        "monthly EMIs from monthly income. The rule-based risk score is distinct "
        "from model probabilities and is not statistically calibrated."
    )
    st.subheader("Responsible use and privacy")
    st.warning(
        "Predictions are academic decision-support outputs, not lending decisions. "
        "The model has not been validated for real applications, fairness, legal "
        "compliance, or probability calibration. Require qualified human review. "
        "Saved assessment inputs are stored locally in SQLite; this prototype has "
        "no authentication, encryption-at-rest, or production access controls. "
        "Do not enter real personal or financial data."
    )
    st.subheader("Run locally")
    st.code(
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "python -m pip install -r requirements.txt\n"
        "python -m streamlit run app.py",
        language="bash",
    )
