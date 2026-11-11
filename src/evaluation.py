"""Model evaluation metrics and Plotly visualisations for LoanIQ."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    auc, confusion_matrix, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLOR_APPROVED, COLOR_DANGER, COLOR_HIGH_RISK, COLOR_LOW_RISK, COLOR_MED_RISK, COLOR_PRIMARY, COLOR_REJECTED


PLOTLY_CONFIG = {"displayModeBar": False}
LAYOUT_BASE = dict(
    font_family="Inter, sans-serif",
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=20),
)


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str = "") -> go.Figure:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Rejected", "Approved"]
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        color_continuous_scale=[[0, "#f0f4f8"], [0.5, "#0d4f4a"], [1, COLOR_PRIMARY]],
        text_auto=True,
        title=f"Confusion Matrix — {model_name}",
        labels=dict(x="Predicted", y="Actual"),
        aspect="equal",
    )
    fig.update_layout(**LAYOUT_BASE, height=320, coloraxis_showscale=False,
                      title_font_size=14)
    return fig


# ── ROC Curve ────────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_proba, model_name: str = "") -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC = {roc_auc:.3f})",
        line=dict(color="#00d4aa", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="#94a3b8", dash="dash", width=1.5),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"ROC Curve — {model_name}",
        title_font_size=14,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=340,
        showlegend=True,
        legend=dict(x=0.55, y=0.1),
    )
    return fig


# ── Model Comparison Bar ──────────────────────────────────────────────────────

def plot_model_comparison(comparison: dict[str, dict]) -> go.Figure:
    """Grouped bar chart of Accuracy / Precision / Recall / F1 / ROC-AUC."""
    models = list(comparison.keys())
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#00d4aa"]

    fig = go.Figure()
    for key, label, color in zip(metrics_keys, metric_labels, colors):
        vals = [comparison[m].get(key, 0) for m in models]
        fig.add_trace(go.Bar(name=label, x=models, y=vals, marker_color=color))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Model Performance Comparison",
        title_font_size=14,
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis_title="Model",
        height=380,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.sort_values("importance", ascending=True).tail(12)
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["label"],
        orientation="h",
        marker_color="#00d4aa",
        marker_line_width=0,
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Top Feature Importances (Random Forest)",
        title_font_size=14,
        xaxis_title="Importance",
        yaxis_title="",
        height=420,
    )
    return fig


# ── Risk Distribution ─────────────────────────────────────────────────────────

def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    colors = [COLOR_LOW_RISK, COLOR_MED_RISK, COLOR_HIGH_RISK]
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=colors,
        text=counts.values.tolist(),
        textposition="outside",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Risk Category Distribution",
        title_font_size=14,
        xaxis_title="Risk Category",
        yaxis_title="Applications",
        height=320,
    )
    return fig


# ── Approval Pie ──────────────────────────────────────────────────────────────

def plot_approval_pie(total_approved: int, total_rejected: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Approved", "Rejected"],
        values=[total_approved, total_rejected],
        hole=0.6,
        marker_colors=[COLOR_APPROVED, COLOR_REJECTED],
        textinfo="percent+label",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Approval vs Rejection",
        title_font_size=14,
        showlegend=False,
        height=300,
    )
    return fig


# ── Income Distribution ───────────────────────────────────────────────────────

def plot_income_distribution(apps_df: pd.DataFrame) -> go.Figure:
    if "annual_income" not in apps_df.columns or apps_df.empty:
        return go.Figure()
    fig = px.histogram(
        apps_df, x="annual_income", nbins=20,
        color="risk_category",
        color_discrete_map={"Low": COLOR_LOW_RISK, "Medium": COLOR_MED_RISK, "High": COLOR_HIGH_RISK},
        title="Income Distribution by Risk",
        labels={"annual_income": "Annual Income ($)", "risk_category": "Risk"},
    )
    fig.update_layout(**LAYOUT_BASE, height=320, title_font_size=14)
    return fig


# ── Loan Amount Distribution ──────────────────────────────────────────────────

def plot_loan_distribution(apps_df: pd.DataFrame) -> go.Figure:
    if "loan_amount" not in apps_df.columns or apps_df.empty:
        return go.Figure()
    fig = px.histogram(
        apps_df, x="loan_amount", nbins=20,
        color_discrete_sequence=["#3b82f6"],
        title="Loan Amount Distribution",
        labels={"loan_amount": "Loan Amount ($)"},
    )
    fig.update_layout(**LAYOUT_BASE, height=320, title_font_size=14)
    return fig


# ── Approval by Employment ────────────────────────────────────────────────────

def plot_approval_by_employment(apps_df: pd.DataFrame) -> go.Figure:
    if "employment_status" not in apps_df.columns or apps_df.empty:
        return go.Figure()
    group = (
        apps_df.groupby("employment_status")["approved"]
        .mean()
        .reset_index()
        .rename(columns={"approved": "approval_rate"})
        .sort_values("approval_rate", ascending=False)
    )
    group["approval_rate"] = (group["approval_rate"] * 100).round(1)
    fig = px.bar(
        group, x="employment_status", y="approval_rate",
        color="approval_rate",
        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        title="Approval Rate by Employment Status (%)",
        labels={"employment_status": "Employment", "approval_rate": "Approval Rate (%)"},
        text="approval_rate",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(**LAYOUT_BASE, height=320, title_font_size=14,
                      coloraxis_showscale=False)
    return fig
