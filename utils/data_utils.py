"""
utils/data_utils.py
────────────────────
Shared helpers for data loading, validation, and profiling.
"""

from __future__ import annotations

import io
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file with sensible defaults."""
    logger.debug(f"Loading CSV: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_excel(path: str | Path, sheet_name: int | str = 0, **kwargs) -> pd.DataFrame:
    """Load an Excel file."""
    logger.debug(f"Loading Excel: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", **kwargs)


def load_bytes_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def load_bytes_excel(data: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")


# ── Validation ───────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> dict:
    """Return a validation report dict."""
    report: dict = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / max(df.size, 1) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "warnings": [],
    }
    if report["missing_pct"] > 30:
        report["warnings"].append(f"High missing data: {report['missing_pct']}%")
    if report["duplicate_rows"] > 0:
        report["warnings"].append(f"{report['duplicate_rows']} duplicate rows detected")
    if report["rows"] < 10:
        report["warnings"].append("Very small dataset — model reliability may be low")
    return report


# ── Profiling ────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict:
    """Lightweight statistical profile."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

    profile: dict = {
        "shape": df.shape,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": df.select_dtypes(include=["datetime"]).columns.tolist(),
        "describe": df.describe(include="all").to_dict(),
        "nulls_per_column": df.isnull().sum().to_dict(),
        "cardinality": {c: df[c].nunique() for c in df.columns},
    }
    return profile


# ── Misc ─────────────────────────────────────────────────────────────────────

def file_hash(path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:12]


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: last column or column named 'target'/'label'/'y'."""
    candidates = [c for c in df.columns if c.lower() in {"target", "label", "y", "class", "output"}]
    return candidates[0] if candidates else df.columns[-1]
