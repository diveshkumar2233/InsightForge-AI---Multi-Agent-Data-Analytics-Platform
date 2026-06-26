"""
agents/ml_agent.py
───────────────────
Machine Learning agent: auto-task detection, multi-model comparison,
cross-validation, feature importance, and Plotly visualisations.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from agents.supervisor import AgentState
from utils.db_utils import save_model_run

warnings.filterwarnings("ignore")


# ── Task detection ────────────────────────────────────────────────────────────

def detect_task(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    if y.dtype == object or y.nunique() <= 20:
        return "classification"
    return "regression"


# ── Model zoo ────────────────────────────────────────────────────────────────

_CLF_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

_REG_MODELS = {
    "Linear Regression"     : LinearRegression(),
    "Random Forest"         : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"     : GradientBoostingRegressor(n_estimators=100, random_state=42),
}


# ── Preparation ──────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, target: str):
    df = df.copy()

    # Encode target if categorical
    le = None
    if df[target].dtype == object or str(df[target].dtype) == "bool":
        le         = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))

    # Drop datetime columns (can't use directly)
    dt_cols = df.select_dtypes(include=["datetime64"]).columns
    df.drop(columns=dt_cols, inplace=True)

    # Encode remaining categoricals
    for col in df.select_dtypes(include="object").columns:
        if col == target:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df      = pd.concat([df.drop(col, axis=1), dummies], axis=1)

    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y, le


# ── Train & evaluate ─────────────────────────────────────────────────────────

def train_and_compare(
    df: pd.DataFrame,
    target: str,
    task: Optional[str] = None,
    dataset_id: int = 0,
) -> dict:
    X, y, le = prepare_data(df, target)
    task      = task or detect_task(y)
    models    = _CLF_MODELS if task == "classification" else _REG_MODELS

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results    : dict[str, dict] = {}
    cv_scores  : dict[str, list] = {}
    best_name  : Optional[str]   = None
    best_score : float            = -np.inf

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task == "classification" else 5

    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if task == "classification":
                acc  = round(accuracy_score(y_test, y_pred), 4)
                f1   = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
                try:
                    auc  = round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]
                                               if y.nunique() == 2 else
                                               pipe.predict_proba(X_test), multi_class="ovr"), 4)
                except Exception:
                    auc  = None
                metrics = {"accuracy": acc, "f1_weighted": f1, "roc_auc": auc}
                cv_s    = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").tolist()
                score   = acc
            else:
                rmse    = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
                mae     = round(mean_absolute_error(y_test, y_pred), 4)
                r2      = round(r2_score(y_test, y_pred), 4)
                metrics = {"rmse": rmse, "mae": mae, "r2": r2}
                cv_s    = cross_val_score(pipe, X, y, cv=5, scoring="r2").tolist()
                score   = r2

            results[name]   = {"metrics": metrics, "model": pipe}
            cv_scores[name] = [round(s, 4) for s in cv_s]

            if score > best_score:
                best_score, best_name = score, name

            save_model_run(
                dataset_id=dataset_id, model_type=name, task=task,
                target_column=target, metrics=metrics, params={},
            )
        except Exception as e:
            logger.warning(f"[ML] {name} failed: {e}")
            results[name] = {"metrics": {}, "error": str(e)}

    # Feature importance from best model
    feat_imp: dict = {}
    if best_name and "error" not in results[best_name]:
        try:
            est = results[best_name]["model"].named_steps["model"]
            if hasattr(est, "feature_importances_"):
                imp = dict(zip(X.columns, est.feature_importances_))
                feat_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20])
            elif hasattr(est, "coef_"):
                coef = np.abs(est.coef_).mean(axis=0) if est.coef_.ndim > 1 else np.abs(est.coef_)
                imp  = dict(zip(X.columns, coef))
                feat_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20])
        except Exception:
            pass

    return {
        "task"            : task,
        "target"          : target,
        "best_model"      : best_name,
        "results"         : {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                             for k, v in results.items()},
        "cv_scores"       : cv_scores,
        "feature_importance": feat_imp,
        "_pipelines"      : {k: v["model"] for k, v in results.items() if "model" in v},
    }


# ── Charts ───────────────────────────────────────────────────────────────────

def ml_charts(ml_result: dict) -> list[go.Figure]:
    figs = []
    results = ml_result.get("results", {})
    task    = ml_result.get("task", "classification")

    # 1. Model comparison bar chart
    if results:
        metric_key = "accuracy" if task == "classification" else "r2"
        rows = [
            {"Model": name, metric_key: data["metrics"].get(metric_key, 0)}
            for name, data in results.items()
            if data.get("metrics")
        ]
        if rows:
            fig = px.bar(
                pd.DataFrame(rows), x="Model", y=metric_key,
                title=f"Model Comparison — {metric_key.upper()}",
                color="Model", template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            figs.append(fig)

    # 2. Feature importance
    fi = ml_result.get("feature_importance", {})
    if fi:
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance")
        fig   = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                       title="Feature Importance", template="plotly_white",
                       color="Importance", color_continuous_scale="Blues")
        figs.append(fig)

    # 3. CV score distribution
    cv = ml_result.get("cv_scores", {})
    if cv:
        rows2 = []
        for model, scores in cv.items():
            for s in scores:
                rows2.append({"Model": model, "CV Score": s})
        if rows2:
            fig = px.box(pd.DataFrame(rows2), x="Model", y="CV Score",
                         title="Cross-Validation Score Distribution",
                         template="plotly_white", color="Model")
            figs.append(fig)

    return figs


# ── LangGraph Node ────────────────────────────────────────────────────────────

def ml_node(state: AgentState) -> AgentState:
    df         = state.get("_df")
    target_col = state.get("_target_col")
    dataset_id = state.get("dataset_id", 0)

    if df is None:
        return {**state, "error": "No dataframe for ML."}
    if not target_col:
        from utils.data_utils import infer_target_column
        target_col = infer_target_column(df)

    try:
        result  = train_and_compare(df, target=target_col, dataset_id=dataset_id or 0)
        charts  = ml_charts(result)
        task    = result["task"]
        best    = result["best_model"]
        metrics = result["results"].get(best, {}).get("metrics", {})

        answer = (
            f"🤖 **ML Training Complete** — Task: `{task}`\n\n"
            f"**Best Model:** {best}\n\n"
            "**Metrics:**\n" +
            "\n".join(f"- {k}: `{v}`" for k, v in metrics.items())
        )
        return {**state, "ml_result": {**result, "_charts": charts}, "final_answer": answer}
    except Exception as e:
        logger.exception(f"[ML] node error: {e}")
        return {**state, "error": str(e)}
