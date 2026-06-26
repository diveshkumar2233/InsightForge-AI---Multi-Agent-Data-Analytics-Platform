"""
tests/test_agents.py
─────────────────────
Unit tests for InsightForge AI agents.
Run with:  pytest tests/ -v
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age"    : np.random.randint(18, 70, 200),
        "salary" : np.random.normal(55000, 15000, 200),
        "dept"   : np.random.choice(["Sales","Eng","HR","Finance"], 200),
        "churn"  : np.random.choice([0, 1], 200, p=[0.75, 0.25]),
        "score"  : np.random.uniform(0, 100, 200),
    })


@pytest.fixture
def dirty_df():
    df = pd.DataFrame({
        "A": [1, 2, None, 4, 5, 5, 1000],
        "B": ["cat", "dog", None, "cat", "bird", "dog", "cat"],
        "C": [10, 20, 30, 40, 50, 20, 10],
    })
    return df


# ── Data Utils ────────────────────────────────────────────────────────────────

def test_validate_dataframe(sample_df):
    from utils.data_utils import validate_dataframe
    report = validate_dataframe(sample_df)
    assert report["rows"] == 200
    assert report["columns"] == 5
    assert "missing_pct" in report
    assert isinstance(report["warnings"], list)


def test_profile_dataframe(sample_df):
    from utils.data_utils import profile_dataframe
    profile = profile_dataframe(sample_df)
    assert "numeric_columns" in profile
    assert "age" in profile["numeric_columns"]
    assert "categorical_columns" in profile


def test_infer_target_column(sample_df):
    from utils.data_utils import infer_target_column
    target = infer_target_column(sample_df)
    assert target in sample_df.columns


# ── Cleaning Agent ────────────────────────────────────────────────────────────

def test_cleaning_removes_duplicates(dirty_df):
    from agents.cleaning_agent import DataCleaner
    cleaner  = DataCleaner(outlier_action="none", encode_categoricals=False)
    df_clean = cleaner.run(dirty_df)
    # Original has duplicates; cleaned should have fewer rows
    assert df_clean.duplicated().sum() == 0


def test_cleaning_handles_missing(dirty_df):
    from agents.cleaning_agent import DataCleaner
    assert dirty_df.isnull().sum().sum() > 0
    cleaner  = DataCleaner(outlier_action="none", encode_categoricals=False)
    df_clean = cleaner.run(dirty_df)
    assert df_clean.isnull().sum().sum() == 0


def test_cleaning_caps_outliers():
    from agents.cleaning_agent import DataCleaner
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    cleaner  = DataCleaner(outlier_method="iqr", outlier_action="cap",
                           encode_categoricals=False, missing_strategy="none")
    df_clean = cleaner.run(df)
    assert df_clean["x"].max() < 1000


# ── EDA Agent ─────────────────────────────────────────────────────────────────

def test_compute_summary(sample_df):
    from agents.eda_agent import compute_summary
    s = compute_summary(sample_df)
    assert s["shape"] == [200, 5]
    assert "age" in s["numeric_stats"]


def test_compute_correlation(sample_df):
    from agents.eda_agent import compute_correlation
    corr = compute_correlation(sample_df)
    assert "top_pairs" in corr
    assert "matrix" in corr


def test_detect_trends(sample_df):
    from agents.eda_agent import detect_trends
    trends = detect_trends(sample_df)
    assert isinstance(trends, list)


def test_auto_charts_returns_figures(sample_df):
    from agents.eda_agent import auto_charts
    figs = auto_charts(sample_df)
    assert len(figs) > 0


# ── ML Agent ──────────────────────────────────────────────────────────────────

def test_detect_task_classification(sample_df):
    from agents.ml_agent import detect_task
    assert detect_task(sample_df["churn"]) == "classification"


def test_detect_task_regression(sample_df):
    from agents.ml_agent import detect_task
    assert detect_task(sample_df["salary"]) == "regression"


def test_prepare_data(sample_df):
    from agents.ml_agent import prepare_data
    X, y, le = prepare_data(sample_df, "churn")
    assert X.shape[0] == 200
    assert len(y) == 200


def test_train_and_compare_classification(sample_df):
    from agents.ml_agent import train_and_compare
    result = train_and_compare(sample_df, target="churn", task="classification")
    assert result["task"] == "classification"
    assert result["best_model"] is not None
    assert "accuracy" in result["results"][result["best_model"]]["metrics"]


def test_train_and_compare_regression(sample_df):
    from agents.ml_agent import train_and_compare
    result = train_and_compare(sample_df, target="salary", task="regression")
    assert result["task"] == "regression"
    assert "r2" in result["results"][result["best_model"]]["metrics"]


# ── Insight Agent ─────────────────────────────────────────────────────────────

def test_compute_kpis(sample_df):
    from agents.insight_agent import compute_kpis
    kpis = compute_kpis(sample_df)
    assert len(kpis) > 0
    assert all("label" in k and "value" in k for k in kpis)


def test_root_cause_signals(sample_df):
    from agents.insight_agent import root_cause_signals
    signals = root_cause_signals(sample_df)
    assert isinstance(signals, list)
    assert len(signals) > 0


# ── Supervisor Routing ────────────────────────────────────────────────────────

def test_supervisor_routes_eda():
    from agents.supervisor import route_query
    assert route_query("show me the correlation heatmap") == "eda"


def test_supervisor_routes_ml():
    from agents.supervisor import route_query
    assert route_query("train a classification model") == "ml"


def test_supervisor_routes_rag():
    from agents.supervisor import route_query
    assert route_query("search documents for revenue information") == "rag"


def test_supervisor_routes_report():
    from agents.supervisor import route_query
    assert route_query("generate a PDF report") == "report"


def test_supervisor_routes_cleaning():
    from agents.supervisor import route_query
    assert route_query("clean my data and remove duplicates") == "cleaning"
