"""
agents/report_agent.py
───────────────────────
Report Generation agent:
  • Aggregates results from all other agents
  • Exports a professional PDF report via utils/pdf_utils
  • Stores report path in SQLite
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from config import REPORTS_DIR
from utils.pdf_utils import generate_report
from agents.supervisor import AgentState


def build_report(
    state: AgentState,
    title: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Assemble a full PDF from whatever agent results exist in state.
    Returns the path to the generated PDF.
    """
    title = title or f"InsightForge AI Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = REPORTS_DIR / f"report_{timestamp}.pdf"

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis: list[dict] = []
    if state.get("insight_result"):
        kpis = state["insight_result"].get("kpis", [])[:6]
    elif df is not None:
        from agents.insight_agent import compute_kpis
        kpis = compute_kpis(df)[:6]

    # Convert KPIs to {"label": ..., "value": ...}
    kpi_cards = [{"label": k.get("label", ""), "value": str(k.get("value", ""))} for k in kpis]

    # ── Sections ──────────────────────────────────────────────────────────────
    sections: list[dict] = []

    # 1. Ingestion summary
    if state.get("ingestion_result"):
        inp = state["ingestion_result"]
        sections.append({
            "heading": "Data Ingestion Summary",
            "content": (
                f"File: {inp.get('file_name','N/A')}  |  "
                f"Rows: {inp.get('rows','?')}  |  "
                f"Columns: {inp.get('columns','?')}  |  "
                f"Missing: {inp.get('missing_pct','?')}%\n\n"
                + ("\n".join(inp.get("warnings", [])) or "No warnings.")
            ),
        })

    # 2. Cleaning summary
    if state.get("cleaning_result"):
        cl = state["cleaning_result"]
        sections.append({
            "heading": "Data Cleaning",
            "content": (
                f"Shape before: {cl.get('shape_before')}  →  after: {cl.get('shape_after')}\n\n"
                "Changes:\n" + "\n".join(f"• {c}" for c in cl.get("changes", []))
            ),
        })

    # 3. EDA narrative
    if state.get("eda_result"):
        narr = state["eda_result"].get("narrative", "")
        if narr:
            sections.append({"heading": "Exploratory Data Analysis", "content": narr})

    # 4. ML results
    if state.get("ml_result"):
        ml  = state["ml_result"]
        rows = []
        for name, data in ml.get("results", {}).items():
            m = data.get("metrics", {})
            rows.append(f"**{name}**: " + ", ".join(f"{k}={v}" for k, v in m.items()))
        sections.append({
            "heading": "Machine Learning Results",
            "content": (
                f"Task: {ml.get('task','?')}  |  Best Model: {ml.get('best_model','?')}\n\n" +
                "\n".join(rows)
            ),
        })

    # 5. DL results
    if state.get("dl_result"):
        dl = state["dl_result"]
        sections.append({
            "heading": "Deep Learning Results",
            "content": (
                f"Model: {dl.get('model','?')}  |  Task: {dl.get('task','?')}\n\n" +
                "\n".join(f"• {k}: {v}" for k, v in dl.get("metrics", {}).items())
            ),
        })

    # 6. Business insights
    if state.get("insight_result"):
        narr = state["insight_result"].get("narrative", "")
        if narr:
            sections.append({"heading": "Business Insights & Recommendations", "content": narr})

    # Fallback
    if not sections:
        sections.append({"heading": "Report", "content": "No analysis results available yet."})

    # ── Generate PDF ──────────────────────────────────────────────────────────
    pdf_path = generate_report(
        output_path=out_path,
        report_title=title,
        sections=sections,
        kpis=kpi_cards,
        dataframe=df,
    )

    # Persist to SQLite
    try:
        from utils.db_utils import get_connection
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO reports (dataset_id, report_type, file_path) VALUES (?,?,?)",
                (state.get("dataset_id"), "full", str(pdf_path)),
            )
    except Exception as e:
        logger.warning(f"[Report] SQLite persist failed: {e}")

    logger.success(f"[Report] PDF written → {pdf_path}")
    return pdf_path


# ── LangGraph Node ────────────────────────────────────────────────────────────

def report_node(state: AgentState) -> AgentState:
    df: Optional[pd.DataFrame] = state.get("_df")
    try:
        path = build_report(state, df=df)
        return {
            **state,
            "report_result": {"path": str(path)},
            "final_answer" : f"📄 Report generated: `{path.name}`",
        }
    except Exception as e:
        logger.exception(f"[Report] node error: {e}")
        return {**state, "error": str(e)}
