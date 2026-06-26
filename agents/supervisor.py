"""
agents/supervisor.py
─────────────────────
Defines the shared AgentState TypedDict and the supervisor routing logic.
The supervisor is just a Python function that LangGraph calls as a node.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Optional, Sequence
from typing_extensions import TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from loguru import logger


# ── Shared State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """State passed between every node in the LangGraph pipeline."""
    # Conversation
    messages       : Annotated[list[BaseMessage], operator.add]
    user_query     : str
    session_id     : str

    # Data
    dataframe_key  : Optional[str]   # key in st.session_state that holds the df
    dataset_id     : Optional[int]   # SQLite id
    dataset_name   : Optional[str]
    file_type      : Optional[str]

    # Agent outputs
    ingestion_result  : Optional[dict]
    cleaning_result   : Optional[dict]
    eda_result        : Optional[dict]
    ml_result         : Optional[dict]
    dl_result         : Optional[dict]
    rag_result        : Optional[dict]
    insight_result    : Optional[dict]
    report_result     : Optional[dict]

    # Routing
    next_agent     : Optional[str]
    final_answer   : Optional[str]
    error          : Optional[str]


# ── Intent keywords → agent mapping ──────────────────────────────────────────

_INTENT_MAP: list[tuple[list[str], str]] = [
    (["upload", "load", "import", "ingest", "read file", "open file"], "ingestion"),
    (["clean", "missing", "duplicate", "outlier", "null", "impute", "preprocess"], "cleaning"),
    (["eda", "explore", "distribution", "correlation", "visualize", "visualise",
      "chart", "plot", "statistic", "describe", "trend", "insight"],            "eda"),
    (["train", "model", "classify", "regression", "predict", "feature importance",
      "cross-validation", "accuracy", "precision", "recall", "auc"],            "ml"),
    (["neural", "deep learning", "lstm", "time series forecast",
      "text classif", "rnn", "transformer"],                                     "dl"),
    (["document", "pdf", "rag", "search", "retrieve", "vector", "embed",
      "source", "citation", "semantic"],                                          "rag"),
    (["insight", "recommend", "kpi", "root cause", "business", "strategy"],     "insight"),
    (["report", "export", "summary", "executive", "generate report",
      "download", "pdf report"],                                                  "report"),
]


def route_query(query: str) -> str:
    """Return the agent name best matching the user's intent."""
    q = query.lower()
    scores: dict[str, int] = {}
    for keywords, agent in _INTENT_MAP:
        score = sum(1 for kw in keywords if kw in q)
        if score:
            scores[agent] = scores.get(agent, 0) + score

    if not scores:
        return "eda"   # safe default

    return max(scores, key=lambda k: scores[k])


# ── Supervisor Node ───────────────────────────────────────────────────────────

def supervisor_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: inspect the latest user message and decide which agent runs next.
    """
    query = state.get("user_query", "")
    if not query and state.get("messages"):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

    target = route_query(query)
    logger.info(f"[Supervisor] routed '{query[:60]}...' → {target}")
    return {**state, "next_agent": target}


# ── END Node sentinel ─────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["ingestion","cleaning","eda","ml","dl","rag","insight","report","__end__"]:
    """Conditional edge: which node should run after the supervisor?"""
    next_a = state.get("next_agent", "__end__")
    valid  = {"ingestion","cleaning","eda","ml","dl","rag","insight","report"}
    return next_a if next_a in valid else "__end__"  # type: ignore
