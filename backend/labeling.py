"""
labeling.py
============
Real FIRMS hotspots don't come with a ground-truth "what kind of fire is
this" label the way the mock data did — nobody has hand-labeled 580,000
satellite pixels. So we derive a starting label with simple, explainable
rules built from what we *do* know (distance to OSM industrial/mining
infrastructure, fire intensity, how often the same spot re-fires, day vs
night), matching the category set from the SIH problem statement:

    Industrial Fire | Persistent Industrial Source | Mining Activity |
    Agricultural Burning | Natural Fire | Other

The classifier is then trained on these rule-based labels, which lets it
generalise beyond the exact rule thresholds and gives you feature
importances / SHAP explainability on top of a defensible starting point.
Swap this out for real ground truth (e.g. verified incident reports)
whenever you have it — nothing downstream needs to change.
"""

import numpy as np
import pandas as pd

CLASSES = [
    "Industrial Fire",
    "Persistent Industrial Source",
    "Mining Activity",
    "Agricultural Burning",
    "Natural Fire",
    "Other",
]


def heuristic_label(df: pd.DataFrame) -> pd.Series:
    """
    df must have: dist_to_industry_km, nearest_industry_category,
    frp_mw, persistence_count, is_night
    """
    near_industry = df["dist_to_industry_km"] <= 3
    near_mining = df["nearest_industry_category"] == "mining"
    is_persistent = df["persistence_count"] >= 5
    high_intensity = df["frp_mw"] >= 40
    low_intensity_daytime = (df["frp_mw"] < 15) & (df["is_night"] == 0)

    conditions = [
        near_industry & near_mining,
        near_industry & is_persistent,
        near_industry,
        is_persistent & ~near_industry,
        low_intensity_daytime & ~near_industry,
        high_intensity & ~near_industry,
    ]
    choices = [
        "Mining Activity",
        "Persistent Industrial Source",
        "Industrial Fire",
        "Persistent Industrial Source",
        "Agricultural Burning",
        "Natural Fire",
    ]
    return pd.Series(
        np.select(conditions, choices, default="Other"),
        index=df.index,
        name="true_class",
    )
