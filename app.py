"""
app.py
───────
InsightForge AI — Main Streamlit Application
Professional multi-agent data analytics platform.
Beautified with Tailwind CDN + Bootstrap Icons + Glassmorphism design system.
"""

from __future__ import annotations

import io
import uuid
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "InsightForge AI — Multi-Agent Data Analytics Platform"},
)

# ── Bootstrap DB ──────────────────────────────────────────────────────────────
from config import validate_env
from utils.db_utils import bootstrap_db
bootstrap_db()

# ── Session State init ────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "session_id"   : str(uuid.uuid4()),
        "df"           : None,
        "df_raw"       : None,
        "dataset_id"   : None,
        "dataset_name" : None,
        "ingestion_result" : None,
        "cleaning_result"  : None,
        "eda_result"       : None,
        "ml_result"        : None,
        "dl_result"        : None,
        "insight_result"   : None,
        "report_path"      : None,
        "chat_messages"    : [],
        "rag_messages"     : [],
        "rag_docs_count"   : 0,
        "active_page"      : "🏠 Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ── Tailwind + Bootstrap Icons + Custom Design System ─────────────────────────
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">

<style>
:root{
  --primary:#1B4FD8; --secondary:#7C3AED; --bg:#F8FAFC;
  --card:#ffffff; --text:#0F172A; --muted:#64748B;
}
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.main { background: var(--bg); }

/* ===== Animated gradient sidebar ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0F172A 0%,#1B4FD8 60%,#7C3AED 100%);
    background-size: 200% 200%;
    animation: sidebarGlow 12s ease infinite;
}
@keyframes sidebarGlow{0%{background-position:0% 0%}50%{background-position:0% 100%}100%{background-position:0% 0%}}
[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
[data-testid="stSidebar"] .stRadio label { color:white !important; font-size:15px; }

/* ===== Glassmorphism cards ===== */
.metric-card{
    background: rgba(255,255,255,.85);
    backdrop-filter: blur(12px);
    border-radius:16px; padding:22px;
    box-shadow:0 4px 6px rgba(0,0,0,.04),0 10px 25px rgba(27,79,216,.08);
    border:1px solid rgba(255,255,255,.6);
    border-left:4px solid var(--primary);
    transition:transform .25s ease, box-shadow .25s ease;
    margin-bottom:16px;
}
.metric-card:hover{
    transform:translateY(-6px) scale(1.02);
    box-shadow:0 12px 30px rgba(27,79,216,.18);
}
.metric-val{font-size:2.1rem;font-weight:700;
    background:linear-gradient(135deg,#1B4FD8,#7C3AED);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.metric-label{font-size:.85rem;color:var(--muted);font-weight:500;margin-top:4px;}

/* ===== Animated hero banner ===== */
.hero-banner{
    background:linear-gradient(135deg,#1B4FD8 0%,#7C3AED 50%,#DB2777 100%);
    background-size:200% 200%;
    animation:heroFlow 8s ease infinite;
    padding:48px 36px;border-radius:20px;margin-bottom:28px;color:white;
    position:relative;overflow:hidden;
    box-shadow:0 20px 40px rgba(124,58,237,.25);
}
@keyframes heroFlow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero-banner::after{
    content:'';position:absolute;top:-50%;right:-10%;width:300px;height:300px;
    background:radial-gradient(circle,rgba(255,255,255,.2),transparent 70%);
    border-radius:50%;
}
.hero-title{font-family:'Poppins',sans-serif;font-size:2.4rem;font-weight:800;margin:0;
    display:flex;align-items:center;gap:12px;}
.hero-sub{font-size:1.08rem;opacity:.9;margin-top:10px;}

/* ===== Streamlit buttons ===== */
.stButton>button{
    border-radius:10px;font-weight:600;border:none;
    background:linear-gradient(135deg,#1B4FD8,#7C3AED);color:white;
    transition:all .2s ease;box-shadow:0 4px 12px rgba(27,79,216,.3);
}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(124,58,237,.4);}

/* ===== Chips ===== */
.chip-green{background:#DCFCE7;color:#166534;padding:3px 12px;border-radius:99px;font-size:12px;font-weight:600;}
.chip-blue {background:#DBEAFE;color:#1E40AF;padding:3px 12px;border-radius:99px;font-size:12px;font-weight:600;}
.chip-red  {background:#FEE2E2;color:#991B1B;padding:3px 12px;border-radius:99px;font-size:12px;font-weight:600;}

/* ===== Section header ===== */
.section-header{
    font-family:'Poppins',sans-serif;font-size:1.45rem;font-weight:700;color:var(--text);
    margin:26px 0 14px;padding-bottom:8px;
    border-bottom:3px solid transparent;
    border-image:linear-gradient(90deg,#1B4FD8,#7C3AED) 1;
    display:flex;align-items:center;gap:10px;
}

/* ===== Chat bubbles ===== */
.chat-user{background:linear-gradient(135deg,#1B4FD8,#7C3AED);color:white;
    border-radius:18px 18px 4px 18px;padding:12px 16px;margin:6px 0;max-width:78%;
    margin-left:auto;font-size:14px;box-shadow:0 2px 8px rgba(27,79,216,.2);}
.chat-ai{background:white;color:#0F172A;border-radius:18px 18px 18px 4px;
    padding:12px 16px;margin:6px 0;max-width:85%;box-shadow:0 2px 8px rgba(0,0,0,.08);font-size:14px;}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;font-weight:600;padding:8px 18px;transition:all .2s ease;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1B4FD8,#7C3AED);color:white !important;}

/* Fade-in */
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.metric-card,.hero-banner{animation:fadeUp .5s ease both;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _df() -> Optional[pd.DataFrame]:
    return st.session_state.get("df")

def _show_df_metrics(df: pd.DataFrame):
    metrics = [
        ("bi-table",                "Rows",    f"{len(df):,}",                                "#1B4FD8"),
        ("bi-layout-three-columns", "Columns", f"{len(df.columns)}",                          "#7C3AED"),
        ("bi-question-octagon",     "Missing", f"{df.isnull().sum().sum():,}",                "#DB2777"),
        ("bi-hdd",                  "Memory",  f"{df.memory_usage(deep=True).sum()/1e6:.2f} MB", "#059669"),
    ]
    cols = st.columns(4)
    for col, (icon, label, val, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{color}'>
              <div style='font-size:1.4rem;color:{color}'><i class="bi {icon}"></i></div>
              <div class='metric-val'>{val}</div>
              <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

def _run_graph(query: str, extra: dict = {}) -> str:
    """Invoke the LangGraph pipeline and return the final answer."""
    from graph.pipeline import run_graph
    from agents.supervisor import AgentState
    from langchain_core.messages import HumanMessage

    state: AgentState = {
        "messages"    : [HumanMessage(content=query)],
        "user_query"  : query,
        "session_id"  : st.session_state["session_id"],
        "dataset_id"  : st.session_state.get("dataset_id"),
        "dataset_name": st.session_state.get("dataset_name"),
        "_df"         : st.session_state.get("df"),
        **extra,
    }

    result = run_graph(state)

    for key in ["ingestion_result","cleaning_result","eda_result",
                "ml_result","dl_result","rag_result","insight_result","report_result"]:
        if result.get(key):
            st.session_state[key] = result[key]

    if result.get("_df") is not None:
        st.session_state["df"] = result["_df"]

    return result.get("final_answer") or result.get("error") or "Done."

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <span style='font-size:2.6rem'>🔮</span>
      <div style='font-family:Poppins,sans-serif;font-size:1.3rem;font-weight:800;color:white;margin-top:4px'>
        InsightForge AI
      </div>
      <div style='font-size:.75rem;color:#94A3B8;margin-top:2px'>Multi-Agent Analytics Platform</div>
    </div>
    <hr style='border-color:#334155;margin:12px 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📁 Data Upload",
            "🧹 Data Cleaning",
            "📊 EDA Dashboard",
            "🤖 ML Training",
            "🧠 Deep Learning",
            "📚 RAG Document Chat",
            "💡 Business Insights",
            "📄 Report Generator",
            "💬 AI Chat",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )
    st.session_state["active_page"] = page

    st.markdown("<hr style='border-color:#334155'>", unsafe_allow_html=True)

    df = _df()
    if df is not None:
        st.markdown(f"""
        <div style='background:rgba(30,58,95,.6);backdrop-filter:blur(8px);border-radius:10px;padding:12px;'>
          <div style='color:#60A5FA;font-size:11px;font-weight:600'><i class="bi bi-database-fill"></i> ACTIVE DATASET</div>
          <div style='color:white;font-size:13px;font-weight:600;margin-top:4px'>
            {st.session_state.get('dataset_name','Unknown')}
          </div>
          <div style='color:#94A3B8;font-size:11px;margin-top:2px'>
            {len(df):,} rows × {len(df.columns)} cols
          </div>
        </div>
        """, unsafe_allow_html=True)

    missing = validate_env()
    st.markdown("<br>", unsafe_allow_html=True)
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
    else:
        st.success("✅ API keys configured")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown("""
    <div class='hero-banner'>
      <div class='hero-title'><i class="bi bi-magic"></i> InsightForge AI</div>
      <div class='hero-sub'>Multi-Agent Data Analytics Platform — powered by LangGraph + Groq + ChromaDB</div>
      <div style="display:flex;gap:8px;margin-top:18px;flex-wrap:wrap">
        <span style="background:rgba(255,255,255,.2);padding:5px 14px;border-radius:99px;font-size:12px;font-weight:600;backdrop-filter:blur(6px)">
          <i class="bi bi-cpu"></i> 9 AI Agents</span>
        <span style="background:rgba(255,255,255,.2);padding:5px 14px;border-radius:99px;font-size:12px;font-weight:600;backdrop-filter:blur(6px)">
          <i class="bi bi-database"></i> 4 Data Sources</span>
        <span style="background:rgba(255,255,255,.2);padding:5px 14px;border-radius:99px;font-size:12px;font-weight:600;backdrop-filter:blur(6px)">
          <i class="bi bi-search"></i> RAG Enabled</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    home_metrics = [
        (c1, "bi-cpu-fill", "9", "Specialized AI Agents", "#1B4FD8"),
        (c2, "bi-hdd-stack-fill", "4", "Data Source Types", "#7C3AED"),
        (c3, "bi-infinity", "∞", "Insight Possibilities", "#DB2777"),
        (c4, "bi-journal-text", "RAG", "Document Intelligence", "#059669"),
    ]
    for col, icon, val, label, color in home_metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{color}'>
              <div style='font-size:1.4rem;color:{color}'><i class="bi {icon}"></i></div>
              <div class='metric-val'>{val}</div>
              <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><i class='bi bi-grid-3x3-gap-fill'></i> Platform Capabilities</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    capabilities = [
        ("bi-folder-fill",          "Data Ingestion",   "CSV · Excel · PDF · SQLite with automatic validation"),
        ("bi-stars",                "Data Cleaning",    "Missing values · Duplicates · Outliers · Feature Engineering"),
        ("bi-bar-chart-fill",       "Auto EDA",         "Statistics · Correlation · Trend Detection · Auto Charts"),
        ("bi-robot",                "ML Training",      "Classification · Regression · Cross-Validation · Feature Importance"),
        ("bi-cpu-fill",             "Deep Learning",    "MLP · LSTM Time-Series · Loss Curves · Predictions"),
        ("bi-journal-text",         "RAG Pipeline",     "PDF ingestion · ChromaDB · Semantic Search · Citations"),
        ("bi-lightbulb-fill",       "Business Insights","KPI Analysis · Root Cause · LLM Recommendations"),
        ("bi-file-earmark-pdf-fill","Report Export",    "Professional PDF · Executive Summary · Charts"),
        ("bi-chat-dots-fill",       "AI Chat",          "Natural language queries routed to correct agent"),
    ]
    for i, (icon, title, desc) in enumerate(capabilities):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:#7C3AED'>
              <div style='font-size:1.8rem;color:#7C3AED'><i class="bi {icon}"></i></div>
              <div style='font-weight:700;color:#0F172A;margin:6px 0;font-size:1.05rem'>{title}</div>
              <div style='font-size:13px;color:#64748B'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><i class='bi bi-rocket-takeoff-fill'></i> Quick Start</div>", unsafe_allow_html=True)
    st.info("👈 **Start by uploading a dataset** in the **Data Upload** section, then explore the other sections for analysis, ML training, and insights.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📁 Data Upload":
    st.markdown("<div class='section-header'><i class='bi bi-folder-fill'></i> Data Upload & Ingestion</div>", unsafe_allow_html=True)

    tab_file, tab_sqlite = st.tabs(["📂 File Upload", "🗄️ SQLite Database"])

    with tab_file:
        st.markdown("Upload CSV, Excel, or PDF files for analysis.")
        uploaded = st.file_uploader(
            "Drop your file here",
            type=["csv", "xlsx", "xls", "pdf"],
            accept_multiple_files=False,
            help="Max 200 MB. CSV and Excel get loaded as DataFrames; PDF tables are extracted automatically.",
        )

        if uploaded:
            with st.spinner("🔄 Ingesting file…"):
                try:
                    from agents.ingestion_agent import ingest_file
                    df, report = ingest_file(uploaded)
                    st.session_state["df"]               = df
                    st.session_state["df_raw"]           = df.copy()
                    st.session_state["ingestion_result"] = report
                    st.session_state["dataset_id"]       = report.get("dataset_id")
                    st.session_state["dataset_name"]     = uploaded.name
                    st.success(f"✅ **{uploaded.name}** ingested successfully!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        if st.session_state.get("ingestion_result"):
            rep = st.session_state["ingestion_result"]
            st.markdown("#### 📋 Validation Report")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows",       rep.get("rows", "?"))
            c2.metric("Columns",    rep.get("columns", "?"))
            c3.metric("Missing %",  f"{rep.get('missing_pct','?')}%")
            c4.metric("Duplicates", rep.get("duplicate_rows","?"))

            if rep.get("warnings"):
                for w in rep["warnings"]:
                    st.warning(w)

            df = _df()
            if df is not None:
                st.markdown("#### 👀 Data Preview")
                st.dataframe(df.head(50), use_container_width=True)

                st.markdown("#### 🏷️ Column Types")
                dtype_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type"  : df.dtypes.astype(str).values,
                    "Nulls" : df.isnull().sum().values,
                    "Unique": df.nunique().values,
                })
                st.dataframe(dtype_df, use_container_width=True)

    with tab_sqlite:
        st.markdown("Connect to an existing SQLite database file.")
        db_path = st.text_input("SQLite DB path", placeholder="./data/mydb.db")
        if db_path and Path(db_path).exists():
            try:
                from agents.ingestion_agent import list_sqlite_tables, ingest_sqlite_table
                tables = list_sqlite_tables(db_path)
                table  = st.selectbox("Select table", tables)
                if st.button("Load Table"):
                    with st.spinner("Loading…"):
                        df, report = ingest_sqlite_table(db_path, table)
                        st.session_state.update({
                            "df": df, "df_raw": df.copy(),
                            "ingestion_result": report,
                            "dataset_id": report.get("dataset_id"),
                            "dataset_name": f"{table} (SQLite)",
                        })
                    st.success(f"Loaded `{table}` — {len(df):,} rows")
                    st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(str(e))
        elif db_path:
            st.warning("File not found.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🧹 Data Cleaning":
    st.markdown("<div class='section-header'><i class='bi bi-stars'></i> Data Cleaning Pipeline</div>", unsafe_allow_html=True)
    df = _df()
    if df is None:
        st.info("Upload a dataset first.")
    else:
        _show_df_metrics(df)
        st.markdown("#### ⚙️ Cleaning Configuration")
        col1, col2 = st.columns(2)
        with col1:
            missing_strat = st.selectbox("Missing Value Strategy",
                ["auto (median/mode)", "mean", "median", "mode", "drop rows"])
            outlier_method = st.selectbox("Outlier Detection", ["IQR", "Z-Score", "None"])
        with col2:
            outlier_action = st.selectbox("Outlier Action", ["Cap", "Drop", "None"])
            encode_cats    = st.checkbox("One-hot encode categoricals (≤15 unique)", value=True)

        if st.button("🚀 Run Cleaning Pipeline", type="primary"):
            with st.spinner("Cleaning…"):
                try:
                    from agents.cleaning_agent import DataCleaner
                    strat_map = {"auto (median/mode)": "auto", "mean": "mean",
                                 "median": "median", "mode": "mode", "drop rows": "drop"}
                    cleaner   = DataCleaner(
                        missing_strategy   = strat_map[missing_strat],
                        outlier_method     = outlier_method.lower().replace("-",""),
                        outlier_action     = outlier_action.lower(),
                        encode_categoricals= encode_cats,
                    )
                    df_clean  = cleaner.run(df)
                    st.session_state["df"]              = df_clean
                    st.session_state["cleaning_result"] = {
                        "shape_before": df.shape,
                        "shape_after" : df_clean.shape,
                        "changes"     : cleaner.change_log,
                    }
                    st.success("✅ Cleaning complete!")
                except Exception as e:
                    st.error(str(e))

        if st.session_state.get("cleaning_result"):
            cr = st.session_state["cleaning_result"]
            st.markdown("#### 📋 Changes Applied")
            c1, c2 = st.columns(2)
            c1.metric("Rows Before → After",
                f"{cr['shape_before'][0]:,} → {cr['shape_after'][0]:,}")
            c2.metric("Cols Before → After",
                f"{cr['shape_before'][1]} → {cr['shape_after'][1]}")
            for ch in cr.get("changes", []):
                st.markdown(f"- ✅ {ch}")

            st.markdown("#### 🔍 Cleaned Data Preview")
            st.dataframe(st.session_state["df"].head(30), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 EDA Dashboard":
    st.markdown("<div class='section-header'><i class='bi bi-bar-chart-fill'></i> Exploratory Data Analysis</div>", unsafe_allow_html=True)
    df = _df()
    if df is None:
        st.info("Upload a dataset first.")
    else:
        if st.button("🔍 Run Auto-EDA", type="primary"):
            with st.spinner("Analysing…"):
                try:
                    from agents.eda_agent import compute_summary, compute_correlation, detect_trends, auto_charts, _llm_narrative
                    summary   = compute_summary(df)
                    corr      = compute_correlation(df)
                    trends    = detect_trends(df)
                    charts    = auto_charts(df)
                    narrative = _llm_narrative(summary, trends)
                    st.session_state["eda_result"] = {
                        "summary": summary, "correlation": corr,
                        "trends": trends, "_charts": charts, "narrative": narrative,
                    }
                except Exception as e:
                    st.error(str(e))

        eda = st.session_state.get("eda_result")
        if eda:
            tab_sum, tab_corr, tab_charts, tab_narr = st.tabs(
                ["📊 Summary", "🔗 Correlation", "📈 Charts", "💬 AI Narrative"]
            )

            with tab_sum:
                _show_df_metrics(df)
                st.markdown("**Statistical Summary**")
                st.dataframe(df.describe(include="all"), use_container_width=True)

                if eda["summary"].get("categorical"):
                    st.markdown("**Categorical Columns**")
                    cat_rows = [{"Column": k, "Unique": v["unique"], "Top Value": v["top"]}
                                for k, v in eda["summary"]["categorical"].items()]
                    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

            with tab_corr:
                corr_data = eda.get("correlation", {})
                if corr_data.get("top_pairs"):
                    st.markdown("**Top Correlated Pairs**")
                    st.dataframe(pd.DataFrame(corr_data["top_pairs"]).head(15), use_container_width=True)
                if corr_data.get("matrix"):
                    import plotly.express as px
                    corr_df = pd.DataFrame(corr_data["matrix"])
                    fig     = px.imshow(corr_df, text_auto=".2f", title="Correlation Matrix",
                                        color_continuous_scale="RdBu_r", aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)

            with tab_charts:
                charts = eda.get("_charts", [])
                if charts:
                    for fig in charts:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No charts generated.")

            with tab_narr:
                if eda.get("trends"):
                    st.markdown("**Trend Signals**")
                    for t in eda["trends"]:
                        st.markdown(f"- {t}")
                st.markdown("**AI Narrative**")
                st.markdown(eda.get("narrative", "Not generated yet."))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 ML Training":
    st.markdown("<div class='section-header'><i class='bi bi-robot'></i> Machine Learning Training Dashboard</div>", unsafe_allow_html=True)
    df = _df()
    if df is None:
        st.info("Upload a dataset first.")
    else:
        _show_df_metrics(df)
        st.markdown("#### ⚙️ Model Configuration")
        target_col = st.selectbox("Target Column", df.columns.tolist())
        task_auto  = st.checkbox("Auto-detect task (classification / regression)", value=True)
        task_manual= "auto"
        if not task_auto:
            task_manual = st.radio("Task", ["classification", "regression"], horizontal=True)

        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training — this may take a minute…"):
                try:
                    from agents.ml_agent import train_and_compare, ml_charts
                    result = train_and_compare(
                        df, target=target_col,
                        task=None if task_auto else task_manual,
                        dataset_id=st.session_state.get("dataset_id", 0),
                    )
                    result["_charts"] = ml_charts(result)
                    st.session_state["ml_result"]   = result
                    st.session_state["_target_col"] = target_col
                    st.success(f"✅ Best model: **{result['best_model']}**")
                except Exception as e:
                    st.error(str(e))

        ml = st.session_state.get("ml_result")
        if ml:
            tab_res, tab_fi, tab_cv, tab_charts = st.tabs(
                ["📊 Results", "🏆 Feature Importance", "📉 Cross-Validation", "📈 Charts"]
            )

            with tab_res:
                st.markdown(f"**Task:** `{ml.get('task')}` | **Best Model:** `{ml.get('best_model')}`")
                rows = []
                for name, data in ml.get("results", {}).items():
                    m = data.get("metrics", {})
                    rows.append({"Model": name, **m})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with tab_fi:
                fi = ml.get("feature_importance", {})
                if fi:
                    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"])
                    st.dataframe(fi_df, use_container_width=True)
                    import plotly.express as px
                    fig = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature",
                                 orientation="h", color="Importance",
                                 color_continuous_scale="Blues", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")

            with tab_cv:
                cv = ml.get("cv_scores", {})
                if cv:
                    rows = [{"Model": m, "Fold": i+1, "Score": s}
                            for m, scores in cv.items() for i, s in enumerate(scores)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No CV scores available.")

            with tab_charts:
                for fig in ml.get("_charts", []):
                    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DEEP LEARNING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🧠 Deep Learning":
    st.markdown("<div class='section-header'><i class='bi bi-cpu-fill'></i> Deep Learning Training</div>", unsafe_allow_html=True)
    df = _df()
    if df is None:
        st.info("Upload a dataset first.")
    else:
        _show_df_metrics(df)
        col1, col2 = st.columns(2)
        with col1:
            dl_mode  = st.selectbox("Model Type", ["MLP (Tabular)", "LSTM (Time-Series)"])
            target_c = st.selectbox("Target Column", df.columns.tolist())
        with col2:
            epochs = st.slider("Epochs", 10, 200, 50, step=10)
            seq_len = st.slider("Sequence Length (LSTM only)", 5, 50, 10) if "LSTM" in dl_mode else 10

        if st.button("🚀 Train Deep Learning Model", type="primary"):
            with st.spinner("Training neural network…"):
                try:
                    from agents.dl_agent import train_tabular_mlp, train_lstm_forecast, dl_charts
                    mode = "lstm" if "LSTM" in dl_mode else "mlp"
                    if mode == "lstm":
                        result = train_lstm_forecast(df, target=target_c, seq_len=seq_len, epochs=epochs)
                    else:
                        result = train_tabular_mlp(df, target=target_c, epochs=epochs)
                    result["_charts"] = dl_charts(result)
                    st.session_state["dl_result"] = result
                    st.success(f"✅ {result['model']} training complete!")
                except Exception as e:
                    st.error(str(e))

        dl = st.session_state.get("dl_result")
        if dl:
            tab_m, tab_hist, tab_pred = st.tabs(["📊 Metrics", "📉 Training History", "🔮 Predictions"])
            with tab_m:
                st.json(dl.get("metrics", {}))
            with tab_hist:
                for fig in dl.get("_charts", [])[:1]:
                    st.plotly_chart(fig, use_container_width=True)
            with tab_pred:
                for fig in dl.get("_charts", [])[1:]:
                    st.plotly_chart(fig, use_container_width=True)
                if dl.get("predictions"):
                    preview = pd.DataFrame({
                        "Actual"   : dl["actuals"][:50],
                        "Predicted": dl["predictions"][:50],
                    })
                    st.dataframe(preview, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RAG DOCUMENT CHAT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📚 RAG Document Chat":
    st.markdown("<div class='section-header'><i class='bi bi-journal-text'></i> RAG — Document Intelligence</div>", unsafe_allow_html=True)

    st.markdown("#### 📤 Upload Documents")
    rag_files = st.file_uploader(
        "Upload PDF or text documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="rag_uploader",
    )
    if rag_files and st.button("📥 Ingest Documents"):
        missing_keys = validate_env()
        if "GOOGLE_API_KEY" in missing_keys:
            st.error("GOOGLE_API_KEY required for embeddings.")
        else:
            with st.spinner("Embedding and storing…"):
                total = 0
                for f in rag_files:
                    try:
                        from agents.rag_agent import ingest_pdf_bytes, ingest_document
                        data = f.read()
                        if f.name.endswith(".pdf"):
                            n = ingest_pdf_bytes(f.name, data)
                        else:
                            n = ingest_document(f.name, data.decode("utf-8", errors="replace"))
                        total += n
                        st.success(f"✅ {f.name} → {n} chunks")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                st.session_state["rag_docs_count"] = st.session_state["rag_docs_count"] + total

    try:
        from agents.rag_agent import collection_stats
        stats = collection_stats()
        st.markdown(f"**ChromaDB:** `{stats.get('count', 0)}` chunks stored")
    except Exception:
        pass

    st.markdown("---")
    st.markdown("#### 💬 Ask Questions About Your Documents")

    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    rag_input = st.chat_input("Ask anything about your uploaded documents…")
    if rag_input:
        st.session_state["rag_messages"].append({"role": "user", "content": rag_input})
        with st.chat_message("user"):
            st.markdown(rag_input)

        missing_keys = validate_env()
        if missing_keys:
            answer = f"⚠️ Missing API keys: {', '.join(missing_keys)}"
        else:
            with st.spinner("🔍 Searching documents…"):
                try:
                    from agents.rag_agent import rag_answer
                    result = rag_answer(rag_input)
                    answer = result["answer"]
                    if result.get("sources"):
                        answer += f"\n\n**Sources:**\n{result['sources']}"
                except Exception as e:
                    answer = f"Error: {e}"

        st.session_state["rag_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💡 Business Insights":
    st.markdown("<div class='section-header'><i class='bi bi-lightbulb-fill'></i> Business Insights & Recommendations</div>", unsafe_allow_html=True)
    df = _df()
    if df is None:
        st.info("Upload a dataset first.")
    else:
        context = st.text_area(
            "Optional: Describe your business context",
            placeholder="e.g. This is e-commerce sales data for Q1 2024. We want to reduce churn.",
            height=80,
        )

        if st.button("💡 Generate Business Insights", type="primary"):
            missing_keys = validate_env()
            if "GROQ_API_KEY" in missing_keys:
                st.error("GROQ_API_KEY required.")
            else:
                with st.spinner("Generating insights…"):
                    try:
                        from agents.insight_agent import compute_kpis, root_cause_signals, generate_insights
                        kpis    = compute_kpis(df)
                        signals = root_cause_signals(df)
                        ml_sum  = None
                        if st.session_state.get("ml_result"):
                            ml  = st.session_state["ml_result"]
                            ml_sum = f"Best model: {ml.get('best_model')}. Task: {ml.get('task')}."
                        narr    = generate_insights(df, kpis, signals, ml_sum, context)
                        st.session_state["insight_result"] = {
                            "kpis": kpis, "signals": signals, "narrative": narr
                        }
                    except Exception as e:
                        st.error(str(e))

        ins = st.session_state.get("insight_result")
        if ins:
            tab_kpi, tab_sig, tab_narr = st.tabs(["📊 KPIs", "⚠️ Signals", "💬 AI Insights"])

            with tab_kpi:
                kpis = ins.get("kpis", [])
                if kpis:
                    cols = st.columns(min(len(kpis), 4))
                    for i, kpi in enumerate(kpis[:8]):
                        with cols[i % 4]:
                            st.metric(kpi["label"], kpi["value"], kpi.get("delta",""))

            with tab_sig:
                for sig in ins.get("signals", []):
                    st.warning(sig)

            with tab_narr:
                st.markdown(ins.get("narrative", "No narrative generated."))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📄 Report Generator":
    st.markdown("<div class='section-header'><i class='bi bi-file-earmark-pdf-fill'></i> Professional Report Generator</div>", unsafe_allow_html=True)

    df = _df()
    report_title = st.text_input("Report Title", value="InsightForge AI — Data Analysis Report")

    available = []
    if st.session_state.get("ingestion_result"):  available.append("Data Ingestion Summary")
    if st.session_state.get("cleaning_result"):   available.append("Data Cleaning Results")
    if st.session_state.get("eda_result"):        available.append("EDA Narrative")
    if st.session_state.get("ml_result"):         available.append("ML Model Results")
    if st.session_state.get("dl_result"):         available.append("Deep Learning Results")
    if st.session_state.get("insight_result"):    available.append("Business Insights")

    if available:
        st.markdown(f"**Sections available:** {', '.join(available)}")
    else:
        st.info("Run at least one analysis step before generating a report.")

    include_data = st.checkbox("Include data preview table in report", value=True)

    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Building PDF…"):
            try:
                from agents.report_agent import build_report
                from agents.supervisor import AgentState
                state: AgentState = {
                    "session_id"      : st.session_state["session_id"],
                    "dataset_id"      : st.session_state.get("dataset_id"),
                    "ingestion_result": st.session_state.get("ingestion_result"),
                    "cleaning_result" : st.session_state.get("cleaning_result"),
                    "eda_result"      : st.session_state.get("eda_result"),
                    "ml_result"       : st.session_state.get("ml_result"),
                    "dl_result"       : st.session_state.get("dl_result"),
                    "insight_result"  : st.session_state.get("insight_result"),
                }
                path = build_report(state, title=report_title, df=df if include_data else None)
                st.session_state["report_path"] = str(path)
                st.success(f"✅ Report generated: `{path.name}`")
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("report_path"):
        rp = Path(st.session_state["report_path"])
        if rp.exists():
            with open(rp, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=f.read(),
                    file_name=rp.name,
                    mime="application/pdf",
                    type="primary",
                )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI CHAT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💬 AI Chat":
    st.markdown("<div class='section-header'><i class='bi bi-chat-dots-fill'></i> AI Chat — Natural Language Analytics</div>", unsafe_allow_html=True)

    df = _df()
    if df is None:
        st.info("Upload a dataset for data-aware queries. Or ask general analytics questions.")

    st.markdown("**Suggested prompts:**")
    cols = st.columns(3)
    suggestions = [
        "Analyse my data and find patterns",
        "What are the key business insights?",
        "Train a classification model",
        "Clean my data and handle missing values",
        "Generate a business report",
        "Show me the feature importance",
    ]
    for i, sug in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}"):
                st.session_state["_pending_chat"] = sug

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("_pending_chat", None)

    chat_input = st.chat_input("Ask anything about your data…") or pending
    if chat_input:
        st.session_state["chat_messages"].append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

        missing_keys = validate_env()
        if missing_keys:
            answer = f"⚠️ Missing API keys: {', '.join(missing_keys)}. Please set them in Settings."
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    answer = _run_graph(chat_input)
                    eda = st.session_state.get("eda_result", {})
                    if eda.get("_charts"):
                        for fig in eda["_charts"][:3]:
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    answer = f"Error: {e}"

        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        try:
            from utils.db_utils import save_query
            save_query(
                session_id=st.session_state["session_id"],
                dataset_id=st.session_state.get("dataset_id"),
                question=chat_input, answer=answer, agent="supervisor",
            )
        except Exception:
            pass

    if st.session_state["chat_messages"]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.markdown("<div class='section-header'><i class='bi bi-gear-fill'></i> Settings & Configuration</div>", unsafe_allow_html=True)

    tab_api, tab_db, tab_about = st.tabs(["🔑 API Keys", "🗄️ Database", "ℹ️ About"])

    with tab_api:
        st.markdown("Configure API keys for the current session (or set them in `.env`).")
        groq_key   = st.text_input("Groq API Key",   type="password", value="")
        google_key = st.text_input("Google API Key", type="password", value="")
        if st.button("Save API Keys"):
            import os
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
                import config
                config.GROQ_API_KEY = groq_key
            if google_key:
                os.environ["GOOGLE_API_KEY"] = google_key
                import config
                config.GOOGLE_API_KEY = google_key
            st.success("✅ Keys updated for this session. Restart app to persist permanently.")

    with tab_db:
        st.markdown("**Dataset History**")
        try:
            from utils.db_utils import list_datasets, list_model_runs
            datasets = list_datasets()
            if datasets:
                st.dataframe(pd.DataFrame(datasets)[
                    ["id","name","file_type","rows","columns","created_at"]
                ], use_container_width=True)
            else:
                st.info("No datasets ingested yet.")

            st.markdown("**Model Run History**")
            runs = list_model_runs()
            if runs:
                rows = [{"ID": r["id"], "Model": r["model_type"], "Task": r["task"],
                         "Target": r["target_column"], "Created": r["created_at"]}
                        for r in runs]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No model runs yet.")
        except Exception as e:
            st.error(str(e))

    with tab_about:
        st.markdown("""
        ## 🔮 InsightForge AI
        **Version:** 1.0.0  
        **Stack:** Python · Streamlit · LangChain · LangGraph · Groq · ChromaDB · PyTorch · Scikit-Learn · Plotly

        ### Architecture
        - **Supervisor Agent** — LangGraph routing
        - **9 Specialized Agents** — Ingestion, Cleaning, EDA, ML, DL, RAG, Insight, Report
        - **Vector Store** — ChromaDB with Google embeddings
        - **Database** — SQLite for dataset & model history

        ### Links
        - [Groq Console](https://console.groq.com/)
        - [Google AI Studio](https://aistudio.google.com/)
        - [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
        """)