"""
utils/db_utils.py
──────────────────
SQLite helpers — schema bootstrap + CRUD for datasets, queries, model runs.
"""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SQLITE_DB_PATH
from loguru import logger

# ── Schema ───────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,
    rows        INTEGER,
    columns     INTEGER,
    file_path   TEXT,
    hash        TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS query_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    dataset_id  INTEGER REFERENCES datasets(id),
    question    TEXT    NOT NULL,
    answer      TEXT,
    agent       TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER REFERENCES datasets(id),
    model_type      TEXT,
    task            TEXT,
    target_column   TEXT,
    metrics         TEXT,   -- JSON
    params          TEXT,   -- JSON
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER REFERENCES datasets(id),
    report_type TEXT,
    file_path   TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def bootstrap_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(_DDL)
    logger.info("SQLite schema bootstrapped")


# ── Dataset CRUD ─────────────────────────────────────────────────────────────

def insert_dataset(name: str, file_type: str, rows: int, columns: int,
                   file_path: str = "", hash_: str = "") -> int:
    sql = """INSERT INTO datasets (name, file_type, rows, columns, file_path, hash)
             VALUES (?, ?, ?, ?, ?, ?)"""
    with get_connection() as conn:
        cur = conn.execute(sql, (name, file_type, rows, columns, file_path, hash_))
        return cur.lastrowid  # type: ignore


def list_datasets() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


# ── Query History ─────────────────────────────────────────────────────────────

def save_query(session_id: str, dataset_id: int | None,
               question: str, answer: str, agent: str = "") -> None:
    sql = """INSERT INTO query_history (session_id, dataset_id, question, answer, agent)
             VALUES (?, ?, ?, ?, ?)"""
    with get_connection() as conn:
        conn.execute(sql, (session_id, dataset_id, question, answer, agent))


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    sql = "SELECT * FROM query_history WHERE session_id=? ORDER BY id DESC LIMIT ?"
    with get_connection() as conn:
        rows = conn.execute(sql, (session_id, limit)).fetchall()
    return [dict(r) for r in rows]


# ── Model Runs ────────────────────────────────────────────────────────────────

def save_model_run(dataset_id: int, model_type: str, task: str,
                   target_column: str, metrics: dict, params: dict) -> int:
    sql = """INSERT INTO model_runs (dataset_id, model_type, task, target_column, metrics, params)
             VALUES (?, ?, ?, ?, ?, ?)"""
    with get_connection() as conn:
        cur = conn.execute(sql, (dataset_id, model_type, task, target_column,
                                 json.dumps(metrics), json.dumps(params)))
        return cur.lastrowid  # type: ignore


def list_model_runs(dataset_id: int | None = None) -> list[dict]:
    sql = "SELECT * FROM model_runs ORDER BY id DESC"
    params: tuple = ()
    if dataset_id:
        sql = "SELECT * FROM model_runs WHERE dataset_id=? ORDER BY id DESC"
        params = (dataset_id,)
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"] or "{}")
        d["params"]  = json.loads(d["params"]  or "{}")
        result.append(d)
    return result
