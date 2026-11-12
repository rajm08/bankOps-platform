import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_service import ModelService
from rag_service import RAGService

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app"))
DATA_DIR = PROJECT_ROOT / "DataSets"
TASKS_DIR = PROJECT_ROOT / "Tasks"

# Load local environment variables when running outside Docker.
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="BankOps Platform API",
    description="Inference, log insights, KPI monitoring, and incident analysis",
    version="1.0.0",
)

models = ModelService(TASKS_DIR)
rag = RAGService(
    persist_directory=str(PROJECT_ROOT / "banking_knowledge_db"),
    collection_name=os.getenv("CHROMA_COLLECTION", "banking_incident_knowledge"),
)

class IncidentRequest(BaseModel):
    description: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "bankops-backend",
        "models": models.status(),
        "rag": rag.status(),
    }

@app.get("/api/kpis")
def get_kpis():
    """Return the latest infrastructure KPI row from the existing CSV."""
    path = DATA_DIR / "infrastructure_metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise HTTPException(status_code=422, detail="timestamp column is missing")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise HTTPException(status_code=422, detail="No valid timestamp rows found")

    row = df.iloc[-1]
    result: dict[str, Any] = {"timestamp": row["timestamp"].isoformat()}
    for col in df.columns:
        if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
            value = row[col]
            result[col] = None if pd.isna(value) else float(value)
    return result

@app.get("/api/models")
def model_status():
    return models.status()

@app.get("/api/forecast")
def forecast():
    """
    Forecast adapter intentionally remains unimplemented until the saved
    timeseries_forecasting.pkl artifact's structure, feature order, scaler,
    and sequence length are verified. This avoids silently producing invalid
    forecasts from an unknown model interface.
    """
    info = models.describe("timeseries_forecasting.pkl")
    return {
        "status": "not_configured",
        "message": "Connect the saved forecasting model, scaler, feature order, and sequence construction.",
        "artifact": info,
    }

@app.get("/api/logs/insights")
def log_insights(limit: int = 100):
    path = DATA_DIR / "application_logs.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {path}")

    df = pd.read_csv(path).tail(max(1, min(limit, 1000)))
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "records": df.astype(object).where(pd.notna(df), None).to_dict(orient="records"),
        "classification_status": "pending_model_adapter",
        "message": "The log clustering artifact must be inspected to identify its vectorizer, model, and any label mapping before predictions are enabled.",
        "artifact": models.describe("banking_log_clustering_model.pkl"),
    }

@app.post("/api/incidents/analyze")
def analyze_incident(request: IncidentRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Incident description cannot be empty")
    try:
        return rag.analyze(request.description)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
