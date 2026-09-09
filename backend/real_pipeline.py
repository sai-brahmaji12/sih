"""
real_pipeline.py
==================
Runs the same shape of pipeline as main.py (clean -> engineer features
-> train/classify -> risk score), but for real VIIRS data instead of the
mock generator, using OSM for industrial-proximity features and
labeling.py for the training labels real data doesn't come with.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_processing import clean
from feature_engineering import compute_persistence, LAND_COVER_CATEGORIES
from real_data_loader import load_viirs_csv
from labeling import heuristic_label
from classify import build_pipeline, predict_new
from risk_scoring import compute_risk
import osm_module

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CSV = DATA_DIR / "viirs-snpp_2023_India.csv"

NUMERIC_FEATURES = [
    "frp_mw", "brightness_k", "confidence", "frp_norm",
    "dist_to_industry_km", "is_night", "persistence_count", "day_of_year",
]
CATEGORICAL_FEATURES = ["land_cover"]


def _land_cover_proxy(dist_km: pd.Series, category: pd.Series) -> pd.Series:
    """
    Real FIRMS data has no land-cover raster attached. As a lightweight
    stand-in, anything close to a known OSM industrial/mining/power site
    is labeled by that site's category; everything else falls back to
    "other". Swap in a real land-cover product (e.g. ESA WorldCover)
    here for production use.
    """
    near = dist_km <= 3
    out = pd.Series("other", index=dist_km.index)
    out[near] = category[near].map({
        "mining": "mining",
        "power": "industrial",
        "industrial": "industrial",
    }).fillna("other")
    return out


def engineer_features_real(df: pd.DataFrame, osm_sites) -> pd.DataFrame:
    df = df.copy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    df["dist_to_industry_km"] = osm_module.nearest_industry_distance_km(lat, lon, osm_sites)
    nearest_category = pd.Series(
        osm_module.nearest_industry_category(lat, lon, osm_sites), index=df.index
    )
    df["nearest_industry_category"] = nearest_category
    df["land_cover"] = _land_cover_proxy(df["dist_to_industry_km"], nearest_category)
    df["land_cover"] = df["land_cover"].where(df["land_cover"].isin(LAND_COVER_CATEGORIES + ["other"]), "other")

    df["is_night"] = (df["acq_daynight"] == "N").astype(int)
    df["persistence_count"] = compute_persistence(df)
    df["day_of_year"] = pd.to_datetime(df["acq_date"]).dt.dayofyear

    denom = df["frp_mw"].max() - df["frp_mw"].min() + 1e-9
    df["frp_norm"] = (df["frp_mw"] - df["frp_mw"].min()) / denom

    df["true_class"] = heuristic_label(df)

    meta_cols = ["hotspot_id", "latitude", "longitude", "acq_date", "region"]
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[meta_cols + feature_cols + ["true_class"]].copy()


def run_real_pipeline(
    csv_path: str = str(DEFAULT_CSV),
    sample_size: Optional[int] = 8000,
    seed: int = 42,
    refresh_osm: bool = False,
) -> Tuple[pd.DataFrame, dict, dict]:
    raw_df = load_viirs_csv(csv_path, sample_size=sample_size, seed=seed)
    cleaned_df = clean(raw_df)

    osm_sites = osm_module.fetch_osm_industrial_sites(force_refresh=refresh_osm)
    print(f"[real_pipeline] Using {len(osm_sites)} OSM industrial/mining/power sites")

    feature_df = engineer_features_real(cleaned_df, osm_sites)
    print(f"[real_pipeline] Built feature set with {len(feature_df)} rows")

    X = feature_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = feature_df["true_class"]
    # Stratified split needs >=2 examples per class; rare classes (e.g. very
    # few "Industrial Fire" hits in a small/offline-fallback sample) can dip
    # below that, so fall back to a plain random split in that case.
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=stratify
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    classes = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, y_pred, labels=classes)

    incoming_df = feature_df.loc[X_test.index].drop(columns=["true_class"])
    classified_df = predict_new(pipe, incoming_df)
    classified_df["true_class"] = feature_df.loc[X_test.index, "true_class"]

    scored_df = compute_risk(classified_df)
    scored_df["acq_date"] = pd.to_datetime(scored_df["acq_date"]).dt.strftime("%Y-%m-%d")

    confusion = {"labels": classes, "matrix": matrix.tolist()}
    return scored_df, report, confusion
