# InsightForge AI - Multi-Agent Data Analytics Platform

**Built by [Divesh Kumar](https://github.com/diveshkumar2233)**

Production-ready end-to-end AI analytics platform powered by LangGraph, Groq, ChromaDB, PyTorch, Streamlit, and the **Model Context Protocol (MCP)**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-orange)
![MCP](https://img.shields.io/badge/MCP-tool%20server-purple)
## Screenshots
 
![InsightsForge AI Dashboard](insightsforge_dashboard.png)
*Main dashboard — pipeline run history, agent activation frequency, ingested file types, and RAG document Q&A*
 
![InsightsForge AI Prediction Module](insightsforge_prediction_report.png)
*Prediction module — actual vs. predicted, 6-month LSTM forecast, model comparison, and feature importance*
 
![InsightsForge AI Report Preview](insightsforge_report_preview.png)
*Generated PDF report preview — model performance, key findings, and business recommendations*
## 📑 Table of Contents
 
- What This Project Does
- Motivation
- How It Works — Architecture Deep-Dive
- MCP Integration (Model Context Protocol)
- Quick Start
- Project Structure
- LangGraph Architecture
- Database Schema
- Agent Reference
- Deployment
- Resume Description
- Interview Q&A
- Author
## What This Project Does
 
InsightForge AI turns a raw data file (CSV, Excel, PDF, or SQLite) into a complete analysis — cleaning, exploratory statistics, machine learning, deep learning forecasts, document Q&A, and a polished PDF report — all from a single natural language request, with no manual pipeline-wiring required.
 
In practice, a user uploads a dataset and types something like "show me trends and forecast next quarter". Instead of running that request through one generic model, the system reads the intent, decides which specialized capabilities are actually needed, and orchestrates them in the right order — the same way a small team of analysts would divide up the work.
 
With the MCP layer, that same capability is no longer locked inside the Streamlit UI — any MCP-compatible client (Claude Desktop, Claude Code, or a custom agent) can call InsightForge AI's agents directly as tools.
 
## Motivation
 
Most "AI analytics" demos are a single LLM call wrapped around a chart library — they can describe data but can't actually clean it, model it, or ground their answers in source documents. The goal behind InsightForge AI was to go further: build a system that behaves less like a chatbot and more like a junior data science team, where each "team member" (agent) has one job and does it well, and a supervisor decides who's needed for a given request.
 
This also served as a deliberate showcase of production-style AI engineering rather than a notebook prototype: stateful orchestration (LangGraph), retrieval-grounded answers instead of hallucinated ones (RAG + ChromaDB), classical and deep learning model pipelines (scikit-learn + PyTorch), a persistence layer (SQLite) so nothing is thrown away between sessions, and an MCP server so the whole pipeline is reachable as standard, discoverable tools rather than being trapped behind one UI.
 
## How It Works — Architecture Deep-Dive
 
### 1. The Core Idea: A Supervisor, Not a Script
 
A traditional pipeline runs every step every time, whether they're needed or not. InsightForge AI instead uses a LangGraph state graph, where a `supervisor_agent` looks at the incoming query, scores it against keywords associated with each downstream capability ("forecast," "predict," "correlate," "summarize this PDF," etc.), and routes execution only through the agents relevant to that request. A request like "clean this and show me a correlation matrix" never touches the ML or DL nodes; a request like "forecast revenue for next 6 months" skips straight past EDA-only steps.
 
This routing logic is what makes the system feel less like a fixed script and more like a dispatcher delegating to the right specialist.
 
### 2. Shared State Flows Through the Graph
 
Every agent reads from and writes to a single shared `AgentState` object as it passes through the graph. This is the backbone of the whole system: instead of each agent being an isolated function call, the DataFrame, cleaning decisions, computed stats, and intermediate results all travel together through the pipeline, so later agents (like the report generator) can reference everything earlier agents discovered, without re-deriving it.
 
```
START
  -> supervisor_agent   (keyword-scored intent routing)
  -> ingestion_node     CSV/Excel/PDF -> DataFrame
  -> cleaning_node      impute/dedup/encode
  -> eda_node            stats + plotly charts + LLM narrative
  -> ml_node              3-model sklearn comparison + CV
  -> dl_node              PyTorch MLP or LSTM
  -> rag_node            ChromaDB retrieve + Groq answer
  -> insight_node        KPI + root-cause + LLM report
  -> report_node         PDF via ReportLab Platypus
  -> END
```
 
### 3. What Each Stage Actually Does
 
- **Ingestion** normalizes wildly different file types (CSV, Excel via openpyxl, PDF via pdfplumber, or an existing SQLite table) into one consistent DataFrame shape, so every downstream agent can assume clean, uniform input.
- **Cleaning** handles the unglamorous-but-critical work: missing value imputation, duplicate removal, and outlier detection using pandas/scipy, so models aren't trained on garbage.
- **EDA** computes descriptive statistics and correlations, generates plotly charts, and — notably — asks the LLM to turn the raw numbers into a plain-English narrative, so the output reads like an analyst's summary rather than a stats dump.
- **ML** trains and compares three scikit-learn models (Logistic Regression, Random Forest, Gradient Boosting) with 5-fold cross-validation, so the "best" model is chosen on evidence rather than a single default.
- **DL** steps in for problems that benefit from deep learning specifically — a PyTorch MLP for general prediction, or an LSTM when the data has a time-series shape, for forecasting tasks the classical models handle poorly.
- **RAG** lets users ask questions about uploaded documents rather than just the tabular data — see the retrieval details below.
- **Insight** synthesizes everything computed so far into KPIs, root-cause explanations, and LLM-written business recommendations.
- **Report** assembles all of the above into a downloadable PDF using ReportLab's Platypus layout engine.
### 4. How RAG Avoids Hallucinating
 
A key design decision was making document Q&A trustworthy, not just plausible-sounding:
 
- pdfplumber extracts raw text from uploaded PDFs.
- Text is chunked into 800-character windows with 150-character overlap (overlap prevents losing context at chunk boundaries).
- Each chunk is embedded using Google's `embedding-001` model and stored in ChromaDB.
- At query time, the top-5 most cosine-similar chunks are retrieved and passed to the LLM.
- The system prompt explicitly restricts the LLM to answering only from the retrieved context, and every chunk carries its source filename and index — so answers come with inline citations rather than being invented.
### 5. Why LangGraph Instead of a Simple Chain
 
A linear chain (A → B → C → D) can't express "skip B and C if the query doesn't need them" without scattering if/else logic throughout the code. LangGraph makes that branching structural — the routing decision lives in one place (the supervisor), and the graph topology itself documents which paths exist. This made the system far easier to extend: adding a 9th agent means adding a node and an edge, not rewriting conditional logic spread across multiple files.
 
### 6. Performance and Persistence Choices
 
- The LangGraph pipeline is compiled once as a singleton rather than rebuilt per request.
- ChromaDB initializes lazily — only when a RAG query actually happens — so simple CSV-only sessions don't pay that startup cost.
- Heavy imports (PyTorch, sklearn, etc.) live inside the agent functions, not at module level, so Streamlit's app startup stays fast even though the full ML/DL stack is available.
- SQLite persists datasets, query history, model run metrics, and generated reports across sessions — so users can return to past analyses without re-uploading or re-running anything.
## MCP Integration (Model Context Protocol)
 
InsightForge AI ships an optional **MCP server** (`mcp_server/server.py`) that exposes the same LangGraph pipeline and agents as standard MCP tools. This means any MCP-compatible client — Claude Desktop, Claude Code, Cowork, or a custom-built MCP client — can drive the platform directly, without going through the Streamlit UI.
 
### Why add MCP on top of an already-working Streamlit app?
 
- **Reusable tools, not a walled-off UI.** The agents already do real work (cleaning, modeling, forecasting, RAG). MCP turns that work into tools any LLM client can call, instead of logic trapped behind one app's buttons.
- **Composability.** A user working inside Claude Desktop can ask "pull last week's sales CSV from my Downloads and forecast next month" and have InsightForge's `ingestion`, `cleaning`, and `dl` agents invoked as part of a larger multi-tool conversation, alongside file-system or calendar tools.
- **No UI required for automation.** Scheduled jobs, CI pipelines, or other agents can call the MCP tools headlessly.
### Exposed MCP Tools
 
| Tool | Maps to | Description |
|---|---|---|
| `ingest_dataset` | `ingestion_node` | Load a CSV/Excel/PDF/SQLite file into a managed session |
| `clean_dataset` | `cleaning_node` | Impute, dedupe, and flag outliers for a session's DataFrame |
| `run_eda` | `eda_node` | Return stats, correlations, and an LLM narrative summary |
| `train_ml_models` | `ml_node` | Run the 3-model sklearn comparison with 5-fold CV |
| `forecast` | `dl_node` | Run the PyTorch MLP/LSTM forecaster for a target column |
| `query_documents` | `rag_node` | Ask a cited, retrieval-grounded question against uploaded PDFs |
| `generate_insights` | `insight_node` | Produce KPI + root-cause + business recommendations |
| `generate_report` | `report_node` | Build and return a downloadable PDF report |
| `run_full_pipeline` | full graph | Natural-language entry point — lets the supervisor decide which agents to run, same as the Streamlit chat box |
 
### How It's Wired
 
```
mcp_server/
├── server.py          # MCP server entrypoint (stdio or HTTP transport)
├── tool_defs.py        # JSON-schema tool definitions
└── session_bridge.py    # Maps MCP tool calls -> existing AgentState / LangGraph graph
```
 
The MCP server does **not** duplicate any agent logic — it's a thin adapter layer that:
 
1. Receives a tool call (e.g. `forecast`) with a `session_id` and parameters.
2. Loads or creates the corresponding `AgentState` from the same SQLite-backed session store the Streamlit app uses.
3. Invokes the relevant compiled LangGraph node(s) directly.
4. Serializes the result (DataFrame summaries, metrics, chart data, PDF path) back as an MCP tool result.
This means the Streamlit UI and any MCP client are just two different front doors to the same underlying graph and session state — a change to an agent is instantly available through both.
 
### Running the MCP Server
 
```bash
# stdio transport (for Claude Desktop / Claude Code local config)
python -m mcp_server.server --transport stdio
 
# HTTP transport (for remote MCP clients)
python -m mcp_server.server --transport http --port 8765
```
 
Example Claude Desktop config entry (`claude_desktop_config.json`):
 
```json
{
  "mcpServers": {
    "insightforge-ai": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "--transport", "stdio"],
      "cwd": "/path/to/insightforge_ai"
    }
  }
}
```
 
Once connected, a client can call `run_full_pipeline` with a natural-language request the same way a Streamlit user would type it into the chat box.
 
## Quick Start
 
### 1. Clone and Install
 
```bash
git clone https://github.com/yourname/insightforge_ai.git
cd insightforge_ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
 
### 2. Configure API Keys
 
```bash
cp .env.example .env
# Edit .env -- add your GROQ_API_KEY and GOOGLE_API_KEY
```
 
| Key | Source |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com/ |
| `GOOGLE_API_KEY` | https://aistudio.google.com/ |
 
### 3. Run (Streamlit UI)
 
```bash
streamlit run app.py
```
 
### 3b. Run (MCP Server)
 
```bash
python -m mcp_server.server --transport stdio
```
 
### 4. Tests
 
```bash
pytest tests/ -v
```
 
## Project Structure
 
```
insightforge_ai/
├── app.py                     # Streamlit multi-page UI (entry point)
├── config.py                  # Centralised env-based configuration
├── requirements.txt
├── .env.example
├── agents/
│   ├── supervisor.py          # AgentState + LangGraph routing logic
│   ├── ingestion_agent.py     # CSV / Excel / PDF / SQLite ingestion
│   ├── cleaning_agent.py      # Missing values, duplicates, outliers
│   ├── eda_agent.py           # Stats, correlation, charts, LLM narrative
│   ├── ml_agent.py            # sklearn model comparison + CV
│   ├── dl_agent.py            # PyTorch MLP + LSTM forecaster
│   ├── rag_agent.py           # ChromaDB RAG pipeline
│   ├── insight_agent.py       # KPI + root-cause + LLM insights
│   └── report_agent.py        # ReportLab PDF generation
├── graph/
│   └── pipeline.py            # LangGraph StateGraph build + compile
├── mcp_server/
│   ├── server.py               # MCP server entrypoint (stdio/http)
│   └── tool_defs.py             # MCP tool schema definitions
├── utils/
│   ├── data_utils.py          # Loaders, validators, profilers
│   ├── db_utils.py            # SQLite CRUD helpers
│   └── pdf_utils.py           # ReportLab PDF helpers
├── data/
│   ├── uploads/                # Raw uploaded files
│   ├── processed/               # Cleaned datasets
│   └── app.db                  # SQLite application database
├── vector_store/
│   └── chroma_db/               # ChromaDB persistence directory
├── reports/                    # Generated PDF reports
└── tests/
    └── test_agents.py          # pytest unit tests (25 tests)
```
 
## LangGraph Architecture
 
```
START
  -> supervisor_agent   (keyword-scored intent routing)
  -> ingestion_node     CSV/Excel/PDF -> DataFrame
  -> cleaning_node      impute/dedup/encode
  -> eda_node            stats + plotly charts + LLM narrative
  -> ml_node              3-model sklearn comparison + CV
  -> dl_node              PyTorch MLP or LSTM
  -> rag_node            ChromaDB retrieve + Groq answer
  -> insight_node        KPI + root-cause + LLM report
  -> report_node         PDF via ReportLab Platypus
  -> END
```
 
The MCP server calls into this same compiled graph — either invoking individual nodes for single-tool calls, or the full graph for `run_full_pipeline`.
 
## Database Schema (SQLite)
 
```sql
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, file_type TEXT, rows INTEGER, columns INTEGER,
    file_path TEXT, hash TEXT, created_at TEXT
);
 
CREATE TABLE query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, dataset_id INTEGER,
    question TEXT, answer TEXT, agent TEXT, created_at TEXT
);
 
CREATE TABLE model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER, model_type TEXT, task TEXT,
    target_column TEXT, metrics TEXT, params TEXT, created_at TEXT
);
 
CREATE TABLE reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER, report_type TEXT,
    file_path TEXT, created_at TEXT
);
```
 
## Agent Reference
 
| Agent | Job | Stack |
|---|---|---|
| Supervisor | Route queries | LangGraph |
| Ingestion | Load + validate files | pdfplumber, openpyxl |
| Cleaning | Fix data quality | pandas, scipy |
| EDA | Auto charts + narrative | plotly, groq |
| ML | Model comparison + CV | scikit-learn |
| DL | MLP + LSTM | PyTorch |
| RAG | Document Q&A | chromadb, google-genai |
| Insight | Business recommendations | groq |
| Report | PDF export | reportlab |
| MCP Server | Expose agents as external tools | mcp (Model Context Protocol SDK) |
 
## Deployment
 
### Streamlit Cloud
 
1. Push to GitHub
2. Open share.streamlit.io
3. Add `GROQ_API_KEY` and `GOOGLE_API_KEY` as Secrets
4. Set main file to `app.py` and deploy
### Docker
 
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
 
The MCP server can be run as a second process/container alongside the Streamlit app, sharing the same `data/app.db` and `vector_store/` volumes.
 
## Resume Description
 
**InsightForge AI - Multi-Agent Data Analytics Platform** | Python, LangGraph, Groq, ChromaDB, PyTorch, MCP
 
Architected and built a production-grade AI analytics platform using a LangGraph Supervisor Agent that dynamically routes natural language queries to 8 specialised agents. Implemented full ML pipelines (Logistic Regression, Random Forest, Gradient Boosting with 5-fold CV), PyTorch deep learning (MLP + LSTM time-series), and a RAG pipeline with ChromaDB and Google embeddings. Exposed the full agent pipeline as a Model Context Protocol (MCP) server, enabling external MCP clients like Claude Desktop to invoke the platform's analytics tools directly. Delivered a professional Streamlit UI with real-time dashboards, automatic EDA chart generation, and ReportLab PDF export. Used Groq llama3-70b for LLM-powered narratives, business insights, and document Q&A with source citations.
 
## Interview Q&A
 
**Q: Why LangGraph over a simple chain?**
LangGraph supports stateful graphs with conditional routing — different queries need fundamentally different agent pipelines. A chain requires scattered branching logic; LangGraph makes topology explicit.
 
**Q: How does RAG handle large PDFs?**
pdfplumber extracts text, chunked into 800-char windows with 150-char overlap. Each chunk is embedded with Google embedding-001 and stored in ChromaDB. Top-5 cosine-similar chunks are retrieved at query time.
 
**Q: How do you prevent hallucination?**
System prompt instructs the LLM to answer only from the provided context. Each chunk is tagged with source filename and chunk index for inline citations.
 
**Q: How do you keep the app fast?**
LangGraph compiled once as singleton; ChromaDB lazily initialised; heavy imports inside agent functions (not module level); SQLite needs no server process.
 
**Q: Why add an MCP server instead of just a REST API?**
A REST API needs a custom client and custom auth/wiring per consumer. MCP is a standardized protocol that any compliant client (Claude Desktop, Claude Code, other agents) already knows how to speak — tool discovery, schemas, and invocation are handled by the protocol, so adding a new consumer means zero new integration code on the server side.
 
**Q: How would you scale to production?**
Replace SQLite with PostgreSQL, ChromaDB with Pinecone/Weaviate, add async LangGraph execution, containerise with Docker, deploy on Kubernetes, add Redis for session state and caching, and run the MCP server over HTTP with proper auth for remote clients.
 
## Author
 
**Divesh Kumar**
 
Designed and built InsightForge AI end-to-end — agent architecture, LangGraph orchestration, ML/DL pipelines, RAG system, MCP server, and the Streamlit UI.
 
- 💼 GitHub: https://github.com/diveshkumar2233
- 📧 Email: diveshkumar4464@gmail.com
If you found this project useful or interesting, consider ⭐ starring the repo!
 

