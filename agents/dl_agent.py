"""
agents/dl_agent.py
───────────────────
Deep Learning agent using PyTorch:
  • Tabular classification / regression (MLP)
  • Time-series forecasting (LSTM)
  • Text classification (Embedding + LSTM)
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from agents.supervisor import AgentState

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── MLP for tabular data ──────────────────────────────────────────────────────

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── LSTM for time-series ──────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2, out: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def _train_loop(
    model: nn.Module, X_tr, y_tr, X_val, y_val,
    epochs: int = 30, lr: float = 1e-3, task: str = "regression",
) -> dict:
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    history   = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(X_val)
            val_loss = criterion(val_out, y_val).item()

        scheduler.step(val_loss)
        history["train_loss"].append(round(loss.item(), 6))
        history["val_loss"].append(round(val_loss, 6))

        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch+1}/{epochs}  train={loss.item():.4f}  val={val_loss:.4f}")

    return history


# ── Public interfaces ─────────────────────────────────────────────────────────

def train_tabular_mlp(
    df: pd.DataFrame, target: str, task: Optional[str] = None, epochs: int = 50,
) -> dict:
    df   = df.copy()
    le   = None
    if df[target].dtype == object:
        le       = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))

    # Prepare features
    dt_cols = df.select_dtypes(include=["datetime64"]).columns
    df.drop(columns=dt_cols, inplace=True)
    for col in df.select_dtypes(include="object").columns:
        if col == target:
            continue
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=[target]).values.astype(np.float32)
    y = df[target].values

    task = task or ("classification" if le or df[target].nunique() <= 20 else "regression")

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_tr_t  = _to_tensor(X_tr)
    X_val_t = _to_tensor(X_val)

    if task == "classification":
        out_dim  = int(np.max(y) + 1)
        y_tr_t   = _to_tensor(y_tr, torch.long)
        y_val_t  = _to_tensor(y_val, torch.long)
    else:
        out_dim  = 1
        y_tr_t   = _to_tensor(y_tr.reshape(-1, 1))
        y_val_t  = _to_tensor(y_val.reshape(-1, 1))

    model   = TabularMLP(X.shape[1], [128, 64, 32], out_dim).to(DEVICE)
    history = _train_loop(model, X_tr_t, y_tr_t, X_val_t, y_val_t, epochs=epochs, task=task)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t)
        if task == "classification":
            pred_labels = preds.argmax(dim=1).cpu().numpy()
            from sklearn.metrics import accuracy_score, f1_score
            metrics = {
                "accuracy"  : round(float(accuracy_score(y_val, pred_labels)), 4),
                "f1_weighted": round(float(f1_score(y_val, pred_labels, average="weighted", zero_division=0)), 4),
            }
        else:
            pred_vals = preds.cpu().numpy().flatten()
            from sklearn.metrics import mean_squared_error, r2_score
            metrics = {
                "rmse": round(float(np.sqrt(mean_squared_error(y_val, pred_vals))), 4),
                "r2"  : round(float(r2_score(y_val, pred_vals)), 4),
            }

    return {"model": "MLP", "task": task, "history": history, "metrics": metrics}


def train_lstm_forecast(df: pd.DataFrame, target: str, seq_len: int = 10, epochs: int = 50) -> dict:
    """Univariate LSTM time-series forecaster."""
    series   = df[target].dropna().values.astype(np.float32)
    scaler   = MinMaxScaler()
    series_s = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    def make_sequences(data, n):
        xs, ys = [], []
        for i in range(len(data) - n):
            xs.append(data[i:i+n])
            ys.append(data[i+n])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = make_sequences(series_s, seq_len)
    split        = int(len(X_seq) * 0.8)
    X_tr, X_val  = X_seq[:split], X_seq[split:]
    y_tr, y_val  = y_seq[:split], y_seq[split:]

    X_tr_t  = _to_tensor(X_tr).unsqueeze(-1)
    X_val_t = _to_tensor(X_val).unsqueeze(-1)
    y_tr_t  = _to_tensor(y_tr.reshape(-1, 1))
    y_val_t = _to_tensor(y_val.reshape(-1, 1))

    model   = LSTMForecaster(input_size=1).to(DEVICE)
    history = _train_loop(model, X_tr_t, y_tr_t, X_val_t, y_val_t, epochs=epochs, task="regression")

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy().flatten()
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_inv     = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    from sklearn.metrics import mean_squared_error, r2_score
    metrics = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_inv, preds_inv))), 4),
        "r2"  : round(float(r2_score(y_inv, preds_inv)), 4),
    }
    return {
        "model": "LSTM", "task": "time_series_forecast",
        "history": history, "metrics": metrics,
        "predictions": preds_inv.tolist(),
        "actuals"    : y_inv.tolist(),
    }


def dl_charts(result: dict) -> list[go.Figure]:
    figs = []
    history = result.get("history", {})
    if history.get("train_loss"):
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig    = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Train Loss"))
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],   mode="lines", name="Val Loss"))
        fig.update_layout(title=f"{result['model']} Training History",
                          xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white")
        figs.append(fig)

    if result.get("predictions") and result.get("actuals"):
        n   = min(len(result["actuals"]), 200)
        idx = list(range(n))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=result["actuals"][:n],      name="Actual",    mode="lines"))
        fig.add_trace(go.Scatter(x=idx, y=result["predictions"][:n],  name="Predicted", mode="lines", line=dict(dash="dash")))
        fig.update_layout(title="Forecast vs Actual", template="plotly_white")
        figs.append(fig)

    return figs


# ── LangGraph Node ────────────────────────────────────────────────────────────

def dl_node(state: AgentState) -> AgentState:
    df         = state.get("_df")
    target_col = state.get("_target_col")
    dl_mode    = state.get("_dl_mode", "mlp")   # mlp | lstm

    if df is None:
        return {**state, "error": "No dataframe for DL."}
    if not target_col:
        from utils.data_utils import infer_target_column
        target_col = infer_target_column(df)

    try:
        if dl_mode == "lstm":
            result = train_lstm_forecast(df, target=target_col)
        else:
            result = train_tabular_mlp(df, target=target_col)

        charts  = dl_charts(result)
        metrics = result.get("metrics", {})
        answer  = (
            f"🧠 **Deep Learning ({result['model']}) Complete**\n\n"
            "**Metrics:**\n" +
            "\n".join(f"- {k}: `{v}`" for k, v in metrics.items())
        )
        return {**state, "dl_result": {**result, "_charts": charts}, "final_answer": answer}
    except Exception as e:
        logger.exception(f"[DL] node error: {e}")
        return {**state, "error": str(e)}
