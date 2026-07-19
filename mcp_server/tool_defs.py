"""
tool_defs.py
------------
JSON-schema tool definitions for the InsightForge AI MCP server.

Each entry maps 1:1 to a LangGraph node (see graph/pipeline.py) except for
`run_full_pipeline`, which lets the supervisor_agent decide which nodes to
run based on a natural-language request -- the same behaviour as the
Streamlit chat box.

These definitions are consumed by server.py's `list_tools()` handler and
are also used to validate incoming arguments before they're forwarded to
the underlying agent functions.
"""

from typing import Any, Dict, List

# A minimal Tool representation: {name, description, inputSchema}.
# inputSchema follows standard JSON Schema (draft-07), as expected by the
# MCP spec for `tools/list` responses.

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "ingest_dataset",
        "description": (
            "Load a CSV, Excel, PDF, or SQLite file into a managed "
            "InsightForge session, normalising it into a single DataFrame "
            "shape for downstream agents. Returns the session_id, row/column "
            "counts, and inferred schema."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the source file.",
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Existing session to attach this dataset to. "
                        "Omit to create a new session."
                    ),
                },
                "file_type": {
                    "type": "string",
                    "enum": ["csv", "excel", "pdf", "sqlite", "auto"],
                    "default": "auto",
                    "description": "Force a file type instead of inferring from extension.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "clean_dataset",
        "description": (
            "Run missing-value imputation, duplicate removal, and outlier "
            "detection on a session's DataFrame. Returns a summary of the "
            "cleaning actions taken."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Target session."},
                "impute_strategy": {
                    "type": "string",
                    "enum": ["mean", "median", "mode", "drop"],
                    "default": "median",
                },
                "drop_duplicates": {"type": "boolean", "default": True},
                "outlier_method": {
                    "type": "string",
                    "enum": ["iqr", "zscore", "none"],
                    "default": "iqr",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "run_eda",
        "description": (
            "Compute descriptive statistics and correlations, generate "
            "plotly chart specs, and produce an LLM-written plain-English "
            "narrative for a session's dataset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of columns to analyse. Defaults to all.",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "train_ml_models",
        "description": (
            "Train and compare Logistic Regression, Random Forest, and "
            "Gradient Boosting models with 5-fold cross-validation against "
            "a target column. Returns metrics per model and the selected best model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "target_column": {"type": "string"},
                "task": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "default": "classification",
                },
            },
            "required": ["session_id", "target_column"],
        },
    },
    {
        "name": "forecast",
        "description": (
            "Run the PyTorch deep-learning forecaster (MLP for general "
            "prediction, LSTM for time-series shaped data) against a target "
            "column and return predicted values for the requested horizon."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "target_column": {"type": "string"},
                "horizon": {
                    "type": "integer",
                    "default": 6,
                    "description": "Number of future periods to forecast.",
                },
                "model_type": {
                    "type": "string",
                    "enum": ["auto", "mlp", "lstm"],
                    "default": "auto",
                },
            },
            "required": ["session_id", "target_column"],
        },
    },
    {
        "name": "query_documents",
        "description": (
            "Ask a retrieval-grounded question against PDFs ingested into "
            "this session. Retrieves the top-5 cosine-similar chunks from "
            "ChromaDB and answers only from that context, with inline "
            "source citations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "question": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["session_id", "question"],
        },
    },
    {
        "name": "generate_insights",
        "description": (
            "Synthesise everything computed so far in the session (EDA, ML, "
            "DL results) into KPIs, root-cause explanations, and LLM-written "
            "business recommendations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Assemble all prior results for a session into a downloadable "
            "PDF report via ReportLab Platypus. Returns the report file path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "report_type": {
                    "type": "string",
                    "enum": ["full", "summary"],
                    "default": "full",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "run_full_pipeline",
        "description": (
            "Natural-language entry point. Passes the request to the "
            "supervisor_agent, which decides which agents/nodes are needed "
            "and runs them in order -- identical behaviour to typing the "
            "request into the Streamlit chat box. Use this when you don't "
            "need fine-grained control over individual steps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Omit to create a new session.",
                },
                "request": {
                    "type": "string",
                    "description": (
                        "Natural-language instruction, e.g. "
                        "'clean this and forecast revenue for next 6 months'."
                    ),
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional dataset to ingest first, if this is a new session.",
                },
            },
            "required": ["request"],
        },
    },
]


def get_tool_names() -> List[str]:
    """Convenience helper: returns just the registered tool names."""
    return [tool["name"] for tool in TOOLS]


def get_tool_schema(name: str) -> Dict[str, Any]:
    """Look up a single tool's inputSchema by name, or raise KeyError."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["inputSchema"]
    raise KeyError(f"Unknown tool: {name}")
