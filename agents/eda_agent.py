"""
agents/eda_agent.py
────────────────────
EDA: summary statistics, correlation, distributions, trend detection,
and automatic Plotly chart generation with LLM-powered narrative.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

from agents.supervisor import AgentState
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE


# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=np.number)
    cat = df.select_dtypes(include=["object", "category", "bool"])

    return {
        "shape"        : list(df.shape),
        "numeric_stats": num.describe().round(4).to_dict() if not num.empty else {},
        "categorical"  : {
            col: {"unique": int(df[col].nunique()), "top": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A"}
            for col in cat.columns
        },
        "missing"      : df.isnull().sum().to_dict(),
        "dtypes"       : df.dtypes.astype(str).to_dict(),
    }


def compute_correlation(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return {}
    corr = num.corr().round(4)
    # Find top correlations
    pairs = []
    cols  = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"col1": cols[i], "col2": cols[j], "corr": round(corr.iloc[i, j], 4)})
    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return {"matrix": corr.to_dict(), "top_pairs": pairs[:10]}


def detect_trends(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    num = df.select_dtypes(include=np.number)
    for col in num.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        # Simple linear trend
        x    = np.arange(len(s))
        coef = np.polyfit(x, s.values, 1)[0]
        pct  = abs(coef) / (s.mean() + 1e-9) * 100
        if pct > 5:
            direction = "increasing" if coef > 0 else "decreasing"
            insights.append(f"**{col}** shows a {direction} trend (slope={coef:.4f})")
    return insights


# ── Chart generators ─────────────────────────────────────────────────────────

def auto_charts(df: pd.DataFrame) -> list[go.Figure]:
    figs: list[go.Figure] = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Correlation heatmap
    if len(num_cols) >= 2:
        corr  = df[num_cols].corr()
        fig   = px.imshow(
            corr, text_auto=".2f", title="Correlation Heatmap",
            color_continuous_scale="RdBu_r", aspect="auto",
        )
        fig.update_layout(template="plotly_white")
        figs.append(fig)

    # 2. Histograms for top-4 numeric cols
    for col in num_cols[:4]:
        fig = px.histogram(df, x=col, title=f"Distribution: {col}",
                           marginal="box", template="plotly_white",
                           color_discrete_sequence=["#1B4FD8"])
        figs.append(fig)

    # 3. Bar charts for top-2 categorical cols
    for col in cat_cols[:2]:
        counts = df[col].value_counts().head(15).reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Value Counts: {col}",
                     template="plotly_white", color_discrete_sequence=["#7C3AED"])
        figs.append(fig)

    # 4. Scatter matrix (top-4 numeric)
    if len(num_cols) >= 3:
        fig = px.scatter_matrix(df[num_cols[:4]], title="Scatter Matrix",
                                template="plotly_white",
                                color_continuous_scale="Viridis")
        figs.append(fig)

    # 5. Box plots
    if num_cols:
        fig = px.box(df[num_cols[:6]], title="Box Plots (Numeric Features)",
                     template="plotly_white")
        figs.append(fig)

    return figs


# ── LLM narrative ─────────────────────────────────────────────────────────────

def _llm_narrative(summary: dict, trends: list[str]) -> str:
    """Ask Groq to produce an EDA narrative."""
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set — narrative unavailable."
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=GROQ_TEMPERATURE)
        prompt = (
            "You are a senior data analyst. Based on the following dataset summary, "
            "write a concise (5-8 bullet) EDA narrative highlighting key patterns, "
            "risks, and recommendations. Use markdown.\n\n"
            f"Summary:\n{json.dumps(summary, default=str)[:3000]}\n\n"
            f"Trend signals:\n" + "\n".join(trends)
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content
    except Exception as e:
        logger.warning(f"[EDA] LLM narrative failed: {e}")
        return f"LLM narrative unavailable: {e}"


# ── LangGraph Node ────────────────────────────────────────────────────────────

def eda_node(state: AgentState) -> AgentState:
    df: Optional[pd.DataFrame] = state.get("_df")
    if df is None:
        return {**state, "error": "No dataframe for EDA."}

    summary = compute_summary(df)
    corr    = compute_correlation(df)
    trends  = detect_trends(df)
    charts  = auto_charts(df)
    narr    = _llm_narrative(summary, trends)

    result = {
        "summary"  : summary,
        "correlation": corr,
        "trends"   : trends,
        "narrative": narr,
        "_charts"  : charts,   # Plotly figures — handled by UI
    }

    answer = narr or "EDA complete. See charts above."

    return {**state, "eda_result": result, "final_answer": answer}
