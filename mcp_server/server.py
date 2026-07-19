"""
server.py
---------
MCP server entrypoint for InsightForge AI.

Exposes the existing LangGraph agent pipeline (agents/*.py, graph/pipeline.py)
as standard MCP tools, so any MCP-compatible client (Claude Desktop, Claude
Code, Cowork, or a custom client) can drive the platform directly -- the same
capabilities available through the Streamlit UI, reachable as tool calls.

Design notes (mirrors the choices made in the Streamlit app):
    - Heavy imports (LangGraph pipeline, PyTorch, sklearn, chromadb) happen
      lazily inside each tool handler, not at module level, so the server
      starts instantly even if a client only ever calls one tool.
    - Session state is backed by the same SQLite store the Streamlit app
      uses (utils/db_utils.py), so a dataset ingested via MCP is visible in
      the UI and vice versa. If that module isn't importable (e.g. running
      this file standalone for a demo), the server falls back to a simple
      in-memory session store so it still runs.
    - This module intentionally keeps the MCP <-> pipeline bridging logic
      inline rather than in a separate session_bridge.py, to keep the
      request/response mapping easy to follow end to end.

Run:
    python -m mcp_server.server --transport stdio
    python -m mcp_server.server --transport http --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any, Dict

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from mcp_server.tool_defs import TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insightforge.mcp_server")

server = Server("insightforge-ai")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
# Tries to use the app's real SQLite-backed session store so MCP and the
# Streamlit UI share state. Falls back to an in-memory dict if that module
# isn't available (e.g. this file is being run outside the full project).

try:
    from utils.db_utils import get_session, create_session, update_session  # type: ignore

    _USING_REAL_STORE = True
except ImportError:  # pragma: no cover - fallback path for standalone runs
    _USING_REAL_STORE = False
    _MEMORY_SESSIONS: Dict[str, Dict[str, Any]] = {}

    def create_session() -> str:
        session_id = str(uuid.uuid4())
        _MEMORY_SESSIONS[session_id] = {}
        return session_id

    def get_session(session_id: str) -> Dict[str, Any]:
        if session_id not in _MEMORY_SESSIONS:
            raise KeyError(f"Unknown session_id: {session_id}")
        return _MEMORY_SESSIONS[session_id]

    def update_session(session_id: str, **fields: Any) -> None:
        _MEMORY_SESSIONS.setdefault(session_id, {}).update(fields)


def _resolve_session(session_id: str | None) -> str:
    """Return an existing session_id, or create a new one if none was given."""
    if session_id:
        get_session(session_id)  # raises if it doesn't exist
        return session_id
    return create_session()


def _result(payload: Dict[str, Any]) -> list[TextContent]:
    """Wrap a dict result as the TextContent list MCP tool calls expect."""
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


# ---------------------------------------------------------------------------
# MCP protocol handlers
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
        for t in TOOLS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any] | None) -> list[TextContent]:
    arguments = arguments or {}
    logger.info("tool call: %s(%s)", name, arguments)

    handlers = {
        "ingest_dataset": _handle_ingest_dataset,
        "clean_dataset": _handle_clean_dataset,
        "run_eda": _handle_run_eda,
        "train_ml_models": _handle_train_ml_models,
        "forecast": _handle_forecast,
        "query_documents": _handle_query_documents,
        "generate_insights": _handle_generate_insights,
        "generate_report": _handle_generate_report,
        "run_full_pipeline": _handle_run_full_pipeline,
    }

    handler = handlers.get(name)
    if handler is None:
        return _result({"error": f"Unknown tool '{name}'"})

    try:
        return await handler(arguments)
    except Exception as exc:  # noqa: BLE001 - surface any agent error to the client
        logger.exception("tool '%s' failed", name)
        return _result({"error": str(exc), "tool": name})


# ---------------------------------------------------------------------------
# Tool handlers -- each lazily imports the matching agent and calls into the
# same AgentState-based functions the LangGraph nodes use.
# ---------------------------------------------------------------------------

async def _handle_ingest_dataset(args: Dict[str, Any]) -> list[TextContent]:
    from agents.ingestion_agent import ingest_file  # lazy import

    session_id = _resolve_session(args.get("session_id"))
    state = ingest_file(
        file_path=args["file_path"],
        file_type=args.get("file_type", "auto"),
    )
    update_session(session_id, dataset=state)
    return _result(
        {
            "session_id": session_id,
            "rows": state.get("rows"),
            "columns": state.get("columns"),
            "schema": state.get("schema"),
        }
    )


async def _handle_clean_dataset(args: Dict[str, Any]) -> list[TextContent]:
    from agents.cleaning_agent import clean_dataframe

    session_id = args["session_id"]
    state = get_session(session_id)
    cleaned_state, summary = clean_dataframe(
        state["dataset"],
        impute_strategy=args.get("impute_strategy", "median"),
        drop_duplicates=args.get("drop_duplicates", True),
        outlier_method=args.get("outlier_method", "iqr"),
    )
    update_session(session_id, dataset=cleaned_state)
    return _result({"session_id": session_id, "cleaning_summary": summary})


async def _handle_run_eda(args: Dict[str, Any]) -> list[TextContent]:
    from agents.eda_agent import run_eda

    session_id = args["session_id"]
    state = get_session(session_id)
    eda_result = run_eda(state["dataset"], columns=args.get("columns"))
    update_session(session_id, eda=eda_result)
    return _result({"session_id": session_id, "eda": eda_result})


async def _handle_train_ml_models(args: Dict[str, Any]) -> list[TextContent]:
    from agents.ml_agent import train_and_compare_models

    session_id = args["session_id"]
    state = get_session(session_id)
    ml_result = train_and_compare_models(
        state["dataset"],
        target_column=args["target_column"],
        task=args.get("task", "classification"),
    )
    update_session(session_id, ml=ml_result)
    return _result({"session_id": session_id, "ml_result": ml_result})


async def _handle_forecast(args: Dict[str, Any]) -> list[TextContent]:
    from agents.dl_agent import run_forecast

    session_id = args["session_id"]
    state = get_session(session_id)
    dl_result = run_forecast(
        state["dataset"],
        target_column=args["target_column"],
        horizon=args.get("horizon", 6),
        model_type=args.get("model_type", "auto"),
    )
    update_session(session_id, dl=dl_result)
    return _result({"session_id": session_id, "forecast": dl_result})


async def _handle_query_documents(args: Dict[str, Any]) -> list[TextContent]:
    from agents.rag_agent import answer_from_documents

    session_id = args["session_id"]
    state = get_session(session_id)
    answer = answer_from_documents(
        session_id=session_id,
        question=args["question"],
        top_k=args.get("top_k", 5),
    )
    return _result({"session_id": session_id, "answer": answer})


async def _handle_generate_insights(args: Dict[str, Any]) -> list[TextContent]:
    from agents.insight_agent import generate_insights

    session_id = args["session_id"]
    state = get_session(session_id)
    insights = generate_insights(state)
    update_session(session_id, insights=insights)
    return _result({"session_id": session_id, "insights": insights})


async def _handle_generate_report(args: Dict[str, Any]) -> list[TextContent]:
    from agents.report_agent import generate_pdf_report

    session_id = args["session_id"]
    state = get_session(session_id)
    report_path = generate_pdf_report(
        state, report_type=args.get("report_type", "full")
    )
    update_session(session_id, report_path=report_path)
    return _result({"session_id": session_id, "report_path": report_path})


async def _handle_run_full_pipeline(args: Dict[str, Any]) -> list[TextContent]:
    from graph.pipeline import get_compiled_graph

    session_id = _resolve_session(args.get("session_id"))
    graph = get_compiled_graph()  # compiled once, reused as a singleton

    initial_state: Dict[str, Any] = {
        "session_id": session_id,
        "request": args["request"],
    }
    if args.get("file_path"):
        initial_state["file_path"] = args["file_path"]

    final_state = graph.invoke(initial_state)
    update_session(session_id, **final_state)
    return _result({"session_id": session_id, "result": final_state})


# ---------------------------------------------------------------------------
# Entrypoint / transport selection
# ---------------------------------------------------------------------------

async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="insightforge-ai",
                server_version="1.0.0",
            ),
        )


async def _run_http(port: int) -> None:
    # Requires an ASGI-capable MCP transport (e.g. mcp.server.sse or a
    # streamable-http transport, depending on the installed SDK version).
    from mcp.server.sse import sse_server  # type: ignore

    logger.info("Starting InsightForge AI MCP server on http://0.0.0.0:%s", port)
    async with sse_server(server, port=port) as running_server:
        await running_server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightForge AI MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to serve the MCP protocol over.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind when using the http transport.",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        asyncio.run(_run_stdio())
    else:
        asyncio.run(_run_http(args.port))


if __name__ == "__main__":
    main()
