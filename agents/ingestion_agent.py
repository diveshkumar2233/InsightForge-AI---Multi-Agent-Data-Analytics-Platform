"""
agents/ingestion_agent.py
──────────────────────────
Handles CSV / Excel / PDF / SQLite ingestion, validation, and registration.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pdfplumber
from loguru import logger

from config import UPLOADS_DIR, SQLITE_DB_PATH
from utils.data_utils import (
    load_bytes_csv,
    load_bytes_excel,
    validate_dataframe,
    file_hash,
)
from utils.db_utils import insert_dataset
from agents.supervisor import AgentState


# ── Core loader ───────────────────────────────────────────────────────────────

def ingest_file(uploaded_file: Any) -> tuple[pd.DataFrame, dict]:
    """
    Accept a Streamlit UploadedFile (or any object with .name and .read()).
    Returns (dataframe, validation_report).
    """
    name = uploaded_file.name
    ext  = Path(name).suffix.lower()
    data = uploaded_file.read()

    logger.info(f"[Ingestion] ingesting {name} ({len(data)/1024:.1f} KB)")

    # Save raw file
    dest = UPLOADS_DIR / name
    dest.write_bytes(data)

    if ext == ".csv":
        df = load_bytes_csv(data)
        ftype = "csv"
    elif ext in {".xlsx", ".xls"}:
        df = load_bytes_excel(data)
        ftype = "excel"
    elif ext == ".pdf":
        df = _pdf_to_df(data)
        ftype = "pdf"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    report = validate_dataframe(df)
    report["file_name"] = name
    report["file_type"] = ftype

    # Register in SQLite
    h = file_hash(dest)
    dataset_id = insert_dataset(
        name=name, file_type=ftype,
        rows=report["rows"], columns=report["columns"],
        file_path=str(dest), hash_=h,
    )
    report["dataset_id"] = dataset_id
    logger.success(f"[Ingestion] dataset_id={dataset_id}  shape={df.shape}")
    return df, report


def ingest_sqlite_table(db_path: str, table_name: str) -> tuple[pd.DataFrame, dict]:
    """Load a table from an external SQLite database file."""
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    report = validate_dataframe(df)
    report.update({"file_name": f"{db_path}::{table_name}", "file_type": "sqlite"})
    dataset_id = insert_dataset(
        name=f"{db_path}::{table_name}", file_type="sqlite",
        rows=report["rows"], columns=report["columns"],
    )
    report["dataset_id"] = dataset_id
    return df, report


def list_sqlite_tables(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur  = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return tables


# ── PDF → DataFrame ───────────────────────────────────────────────────────────

def _pdf_to_df(data: bytes) -> pd.DataFrame:
    """Extract tables from PDF; fallback to text-per-page frame."""
    frames: list[pd.DataFrame] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables:
                if tbl and len(tbl) > 1:
                    headers = tbl[0]
                    rows    = tbl[1:]
                    frames.append(pd.DataFrame(rows, columns=headers))

        if not frames:
            # Fallback: one row per page with raw text
            text_rows = []
            for i, page in enumerate(pdf.pages):
                text_rows.append({"page": i + 1, "text": page.extract_text() or ""})
            frames = [pd.DataFrame(text_rows)]

    return pd.concat(frames, ignore_index=True)


# ── LangGraph Node ────────────────────────────────────────────────────────────

def ingestion_node(state: AgentState) -> AgentState:
    """
    LangGraph node wrapper.
    Expects state["ingestion_input"] to be set by the UI before graph invocation.
    """
    inp = state.get("ingestion_result", {})
    # In the Streamlit flow, ingestion is done before graph run (via ingest_file).
    # This node simply echoes back the already-computed result.
    return {
        **state,
        "final_answer": (
            f"✅ File **{inp.get('file_name','unknown')}** ingested successfully.\n\n"
            f"- Rows: {inp.get('rows', '?')}\n"
            f"- Columns: {inp.get('columns', '?')}\n"
            f"- Missing cells: {inp.get('missing_cells', '?')} ({inp.get('missing_pct','?')}%)\n"
            f"- Duplicates: {inp.get('duplicate_rows', '?')}\n"
            + ("\n⚠️ Warnings:\n" + "\n".join(f"- {w}" for w in inp.get("warnings", [])) if inp.get("warnings") else "")
        ),
    }
