import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from generate_mock_data import generate
from data_processing import clean
from feature_engineering import engineer_features
from classify import train, predict_new
from risk_scoring import compute_risk
from real_pipeline import run_real_pipeline, DEFAULT_CSV

app = FastAPI(title="Industrial Fire Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "scored_df": None,
    "report": None,
    "confusion": None,
    "trained_at": None,
    "n_records": 0,
    "source": None,
}


def _run_mock_pipeline(n_records: int = 1200, seed: int = 42):
    raw_df = generate(n=n_records, seed=seed)
    cleaned_df = clean(raw_df)
    feature_df = engineer_features(cleaned_df)
    pipe, (X_test, y_test, y_pred) = train(feature_df)

    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    classes = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, y_pred, labels=classes)

    incoming_df = feature_df.loc[X_test.index].drop(columns=["true_class"])
    classified_df = predict_new(pipe, incoming_df)
    classified_df["true_class"] = feature_df.loc[X_test.index, "true_class"]

    scored_df = compute_risk(classified_df)
    scored_df["acq_date"] = pd.to_datetime(scored_df["acq_date"]).dt.strftime("%Y-%m-%d")
    return scored_df, report, {"labels": classes, "matrix": matrix.tolist()}


def _run_pipeline(source: str = "real", n_records: int = 1200, seed: int = 42,
                   sample_size: int = 8000, refresh_osm: bool = False):
    if source == "real" and Path(DEFAULT_CSV).exists():
        scored_df, report, confusion = run_real_pipeline(
            sample_size=sample_size, seed=seed, refresh_osm=refresh_osm
        )
    else:
        source = "mock"
        scored_df, report, confusion = _run_mock_pipeline(n_records=n_records, seed=seed)

    STATE["scored_df"] = scored_df
    STATE["report"] = report
    STATE["confusion"] = confusion
    STATE["trained_at"] = time.time()
    STATE["n_records"] = len(scored_df)
    STATE["source"] = source


@app.on_event("startup")
def _startup():
    _run_pipeline(source="real", sample_size=8000)


@app.post("/api/run")
def run_pipeline(source: str = "real", n_records: int = 1200, seed: int = 42,
                  sample_size: int = 8000, refresh_osm: bool = False):
    _run_pipeline(source=source, n_records=n_records, seed=seed,
                  sample_size=sample_size, refresh_osm=refresh_osm)
    return {
        "status": "ok",
        "source": STATE["source"],
        "n_records": STATE["n_records"],
        "trained_at": STATE["trained_at"],
    }


@app.get("/api/results")
def get_results(
    risk_level: Optional[str] = None,
    predicted_class: Optional[str] = None,
    region: Optional[str] = None,
    min_score: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 5000,
):
    if STATE["scored_df"] is None:
        raise HTTPException(503, "Pipeline has not run yet")
    df = STATE["scored_df"]
    if risk_level:
        df = df[df["risk_level"] == risk_level]
    if predicted_class:
        df = df[df["predicted_class"] == predicted_class]
    if region:
        df = df[df["region"] == region]
    if min_score is not None:
        df = df[df["risk_score"] >= min_score]
    if search:
        df = df[df["hotspot_id"].str.contains(search, case=False, na=False)]
    df = df.sort_values("risk_score", ascending=False).head(limit)
    return df.to_dict(orient="records")


@app.get("/api/summary")
def get_summary():
    if STATE["scored_df"] is None:
        raise HTTPException(503, "Pipeline has not run yet")
    df = STATE["scored_df"]
    return {
        "n_records": len(df),
        "source": STATE["source"],
        "trained_at": STATE["trained_at"],
        "risk_level_counts": df["risk_level"].value_counts().to_dict(),
        "predicted_class_counts": df["predicted_class"].value_counts().to_dict(),
        "region_counts": df["region"].value_counts().to_dict(),
        "regions": sorted(df["region"].unique().tolist()),
        "predicted_classes": sorted(df["predicted_class"].unique().tolist()),
        "avg_risk_score": round(float(df["risk_score"].mean()), 2),
        "high_risk_count": int((df["risk_level"] == "High").sum()),
        "accuracy": STATE["report"]["accuracy"] if STATE["report"] else None,
        "report": STATE["report"],
        "confusion": STATE["confusion"],
    }


@app.get("/api/hotspot/{hotspot_id}")
def get_hotspot(hotspot_id: str):
    if STATE["scored_df"] is None:
        raise HTTPException(503, "Pipeline has not run yet")
    df = STATE["scored_df"]
    row = df[df["hotspot_id"] == hotspot_id]
    if row.empty:
        raise HTTPException(404, "Hotspot not found")
    return row.to_dict(orient="records")[0]


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
