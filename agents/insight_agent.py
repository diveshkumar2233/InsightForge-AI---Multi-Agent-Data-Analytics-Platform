"""
agents/insight_agent.py
────────────────────────
Business Insight agent: KPI analysis, root-cause analysis,
and strategic recommendations powered by Groq LLM.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from agents.supervisor import AgentState


# ── KPI Computation ───────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> list[dict]:
    """
    Derive a set of generic KPIs from any numeric dataframe.
    Returns list of {"label": str, "value": str, "delta": str}.
    """
    kpis: list[dict] = []
    num = df.select_dtypes(include=np.number)

    kpis.append({"label": "Total Rows",    "value": f"{len(df):,}", "delta": ""})
    kpis.append({"label": "Total Columns", "value": f"{len(df.columns)}", "delta": ""})

    missing_pct = round(df.isnull().sum().sum() / max(df.size, 1) * 100, 1)
    kpis.append({"label": "Missing Data %", "value": f"{missing_pct}%", "delta": ""})

    for col in num.columns[:5]:
        s = df[col].dropna()
        kpis.append({
            "label": f"Mean {col}",
            "value": f"{s.mean():.2f}",
            "delta": f"std {s.std():.2f}",
        })
        kpis.append({
            "label": f"Max {col}",
            "value": f"{s.max():.2f}",
            "delta": "",
        })

    return kpis[:12]   # cap at 12


# ── Root Cause Detection ──────────────────────────────────────────────────────

def root_cause_signals(df: pd.DataFrame) -> list[str]:
    """Heuristic root-cause signals."""
    signals: list[str] = []
    num = df.select_dtypes(include=np.number)

    for col in num.columns:
        s   = df[col].dropna()
        cv  = s.std() / (s.mean() + 1e-9)
        if cv > 1.0:
            signals.append(f"**{col}** has very high variability (CV={cv:.2f}) — investigate outliers")
        miss = df[col].isnull().mean() * 100
        if miss > 20:
            signals.append(f"**{col}** has {miss:.1f}% missing — data quality concern")

    # Correlation-based signals
    if num.shape[1] >= 2:
        corr  = num.corr().abs()
        pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                 for i in range(len(corr.columns))
                 for j in range(i+1, len(corr.columns))
                 if corr.iloc[i, j] > 0.85]
        for c1, c2, v in pairs[:3]:
            signals.append(f"**{c1}** and **{c2}** are highly correlated ({v:.2f}) — potential multicollinearity")

    return signals or ["No strong root-cause signals detected."]


# ── LLM Business Insights ──────────────────────────────────────────────────────

def generate_insights(
    df: pd.DataFrame,
    kpis: list[dict],
    signals: list[str],
    ml_summary: Optional[str] = None,
    user_context: str = "",
) -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set — LLM insights unavailable."

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.3)

        summary_stats = df.describe(include="all").iloc[:3].to_string()
        kpi_text      = "\n".join(f"• {k['label']}: {k['value']}" for k in kpis)
        signals_text  = "\n".join(signals)

        prompt = f"""You are a senior business intelligence analyst.
Analyze this dataset and generate actionable business insights.

**KPIs:**
{kpi_text}

**Root-Cause Signals:**
{signals_text}

**Statistical Summary (excerpt):**
{summary_stats[:1500]}

{"**ML Model Findings:** " + ml_summary if ml_summary else ""}

{"**Business Context:** " + user_context if user_context else ""}

Generate a structured report with:
1. Executive Summary (3-4 sentences)
2. Key Findings (5 bullet points)
3. Risk Factors (3 bullet points)
4. Strategic Recommendations (5 actionable items)
5. Next Steps

Use markdown formatting. Be specific and data-driven."""

        resp = llm.invoke([
            SystemMessage(content="You are a world-class business intelligence analyst."),
            HumanMessage(content=prompt),
        ])
        return resp.content
    except Exception as e:
        logger.warning(f"[Insight] LLM failed: {e}")
        return f"LLM insight generation failed: {e}"


# ── LangGraph Node ────────────────────────────────────────────────────────────

def insight_node(state: AgentState) -> AgentState:
    df: Optional[pd.DataFrame] = state.get("_df")
    if df is None:
        return {**state, "error": "No dataframe for insight generation."}

    ml_summary = None
    if state.get("ml_result"):
        ml_res     = state["ml_result"]
        best       = ml_res.get("best_model", "")
        metrics    = ml_res.get("results", {}).get(best, {}).get("metrics", {})
        ml_summary = f"Best model: {best}. Metrics: {metrics}"

    kpis    = compute_kpis(df)
    signals = root_cause_signals(df)
    text    = generate_insights(df, kpis, signals, ml_summary, user_context=state.get("user_query",""))

    return {
        **state,
        "insight_result": {"kpis": kpis, "signals": signals, "narrative": text},
        "final_answer"  : text,
    }
