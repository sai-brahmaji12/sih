import pandas as pd
import numpy as np


def compute_risk(
    df: pd.DataFrame,
) -> pd.DataFrame:

    output = df.copy()

    intensity = np.clip(
        output["frp_mw"] / 50.0,
        0,
        1,
    )

    persistence = np.clip(
        output["persistence_count"]
        / 10.0,
        0,
        1,
    )

    proximity = np.clip(
        1
        - output["dist_to_industry_km"]
        / 100.0,
        0,
        1,
    )

    confidence = np.clip(
        output[
            "predicted_confidence"
        ],
        0,
        1,
    )

    output["risk_score"] = (
        0.35 * intensity
        + 0.25 * persistence
        + 0.20 * proximity
        + 0.20 * confidence
    ) * 100

    output["risk_score"] = (
        output["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    output["risk_level"] = np.select(
        [
            output["risk_score"] >= 70,
            output["risk_score"] >= 40,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    return output
