"""
agents/cleaning_agent.py
─────────────────────────
Data cleaning pipeline: missing values, duplicates, outliers, type inference,
and basic feature engineering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from agents.supervisor import AgentState


# ── Core Cleaning Pipeline ───────────────────────────────────────────────────

class DataCleaner:
    """Stateful cleaner; call .run(df) to get clean_df + change_log."""

    def __init__(
        self,
        missing_strategy: str = "auto",   # auto | mean | median | mode | drop
        outlier_method  : str = "iqr",    # iqr | zscore | none
        outlier_action  : str = "cap",    # cap | drop | none
        iqr_factor      : float = 1.5,
        zscore_thresh   : float = 3.0,
        encode_categoricals: bool = True,
    ):
        self.missing_strategy    = missing_strategy
        self.outlier_method      = outlier_method
        self.outlier_action      = outlier_action
        self.iqr_factor          = iqr_factor
        self.zscore_thresh       = zscore_thresh
        self.encode_categoricals = encode_categoricals
        self.change_log: list[str] = []

    # ── Public entry point ────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.change_log = []
        df = df.copy()
        df = self._drop_constant_columns(df)
        df = self._handle_duplicates(df)
        df = self._infer_types(df)
        df = self._handle_missing(df)
        df = self._handle_outliers(df)
        if self.encode_categoricals:
            df = self._encode_low_cardinality(df)
        logger.info(f"[Cleaning] done — {len(self.change_log)} changes")
        return df

    # ── Steps ──────────────────────────────────────────────────────────────────
    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        before = df.shape[1]
        df = df.loc[:, df.nunique() > 1]
        dropped = before - df.shape[1]
        if dropped:
            self.change_log.append(f"Dropped {dropped} constant column(s)")
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = df.duplicated().sum()
        if n:
            df = df.drop_duplicates()
            self.change_log.append(f"Removed {n} duplicate rows")
        return df

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        converted = 0
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col])
                converted += 1
                continue
            except (ValueError, TypeError):
                pass
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                converted += 1
            except Exception:
                pass
        if converted:
            self.change_log.append(f"Auto-converted types for {converted} column(s)")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        total_missing = df.isnull().sum().sum()
        if not total_missing:
            return df

        strat = self.missing_strategy
        # Numeric
        for col in num_cols:
            miss = df[col].isnull().sum()
            if not miss:
                continue
            if strat in ("auto", "median"):
                df[col].fillna(df[col].median(), inplace=True)
            elif strat == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strat == "drop":
                df.dropna(subset=[col], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        # Categorical
        for col in cat_cols:
            miss = df[col].isnull().sum()
            if miss:
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown", inplace=True)

        self.change_log.append(f"Imputed {total_missing} missing value(s)")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_method == "none" or self.outlier_action == "none":
            return df
        num_cols = df.select_dtypes(include=np.number).columns
        flagged  = 0
        for col in num_cols:
            s = df[col].dropna()
            if self.outlier_method == "iqr":
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR    = Q3 - Q1
                lo, hi = Q1 - self.iqr_factor * IQR, Q3 + self.iqr_factor * IQR
            else:  # zscore
                z  = np.abs(stats.zscore(s))
                lo = s[z < self.zscore_thresh].min()
                hi = s[z < self.zscore_thresh].max()

            mask = (df[col] < lo) | (df[col] > hi)
            n    = mask.sum()
            if not n:
                continue
            flagged += n
            if self.outlier_action == "cap":
                df[col] = df[col].clip(lower=lo, upper=hi)
            elif self.outlier_action == "drop":
                df = df[~mask]

        if flagged:
            self.change_log.append(f"Outlier handling ({self.outlier_action}): {flagged} value(s)")
        return df

    def _encode_low_cardinality(self, df: pd.DataFrame, thresh: int = 15) -> pd.DataFrame:
        encoded = 0
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() <= thresh:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df      = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                encoded += 1
        if encoded:
            self.change_log.append(f"One-hot encoded {encoded} categorical column(s)")
        return df


# ── LangGraph Node ────────────────────────────────────────────────────────────

def cleaning_node(state: AgentState) -> AgentState:
    df_raw = state.get("_df")   # injected by Streamlit UI bridge
    if df_raw is None:
        return {**state, "error": "No dataframe available for cleaning."}

    cleaner  = DataCleaner()
    df_clean = cleaner.run(df_raw)

    result = {
        "shape_before": df_raw.shape,
        "shape_after" : df_clean.shape,
        "changes"     : cleaner.change_log,
        "_df_clean"   : df_clean,
    }

    answer = (
        f"🧹 **Data Cleaning Complete**\n\n"
        f"- Rows: {df_raw.shape[0]} → {df_clean.shape[0]}\n"
        f"- Columns: {df_raw.shape[1]} → {df_clean.shape[1]}\n\n"
        f"**Changes applied:**\n" +
        "\n".join(f"- {c}" for c in cleaner.change_log)
    )

    return {**state, "cleaning_result": result, "final_answer": answer, "_df": df_clean}
