"""
graph/pipeline.py
──────────────────
Builds and compiles the InsightForge AI LangGraph StateGraph.

Topology:
    START
      └─► supervisor
            └─► [ingestion | cleaning | eda | ml | dl | rag | insight | report]
                  └─► END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agents.supervisor   import AgentState, supervisor_agent, should_continue
from agents.ingestion_agent import ingestion_node
from agents.cleaning_agent  import cleaning_node
from agents.eda_agent       import eda_node
from agents.ml_agent        import ml_node
from agents.dl_agent        import dl_node
from agents.rag_agent       import rag_node
from agents.insight_agent   import insight_node
from agents.report_agent    import report_node
from loguru import logger


def build_graph():
    """Construct and compile the LangGraph pipeline. Returns a CompiledGraph."""
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("ingestion",  ingestion_node)
    builder.add_node("cleaning",   cleaning_node)
    builder.add_node("eda",        eda_node)
    builder.add_node("ml",         ml_node)
    builder.add_node("dl",         dl_node)
    builder.add_node("rag",        rag_node)
    builder.add_node("insight",    insight_node)
    builder.add_node("report",     report_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    # Conditional routing from supervisor
    builder.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "ingestion": "ingestion",
            "cleaning" : "cleaning",
            "eda"      : "eda",
            "ml"       : "ml",
            "dl"       : "dl",
            "rag"      : "rag",
            "insight"  : "insight",
            "report"   : "report",
            "__end__"  : END,
        },
    )

    # All agent nodes terminate at END
    for node in ["ingestion", "cleaning", "eda", "ml", "dl", "rag", "insight", "report"]:
        builder.add_edge(node, END)

    graph = builder.compile()
    logger.info("[Graph] LangGraph pipeline compiled successfully")
    return graph


# Singleton — compiled once at import time
_GRAPH = None

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_graph(state: AgentState) -> AgentState:
    """Invoke the compiled graph with the given state and return final state."""
    graph = get_graph()
    result = graph.invoke(state)
    return result
