import numpy as np
import pandas as pd

LAND_COVER_CATEGORIES = [
    "industrial",
    "agricultural",
    "forest",
    "mining",
    "urban",
    "barren",
    "grassland",
    "other",
]


def compute_persistence(
    df: pd.DataFrame,
    radius_km=1.0,
) -> pd.Series:

    lat_bin = (
        df["latitude"]
        / (radius_km / 111.0)
    ).round()

    lon_bin = (
        df["longitude"]
        / (
            radius_km
            / (
                111.0
                * np.cos(
                    np.radians(
                        df["latitude"]
                    )
                ).clip(
                    lower=0.2
                )
            )
        )
    ).round()

    group_key = (
        lat_bin.astype(str)
        + "_"
        + lon_bin.astype(str)
    )

    return group_key.map(
        group_key.value_counts()
    )


def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    from generate_mock_data import (
        INDUSTRIAL_SITES,
        haversine_km,
    )

    def distance_to_nearest(
        lat,
        lon,
    ):
        distances = [
            haversine_km(
                lat,
                lon,
                site_lat,
                site_lon,
            )
            for site_lat, site_lon
            in INDUSTRIAL_SITES
        ]

        return min(distances)

    df["dist_to_industry_km"] = df.apply(
        lambda row: distance_to_nearest(
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )

    df["land_cover"] = (
        df["land_cover"]
        .where(
            df["land_cover"].isin(
                LAND_COVER_CATEGORIES
            ),
            "other",
        )
    )

    df["is_night"] = (
        df["acq_daynight"] == "N"
    ).astype(int)

    df["persistence_count"] = (
        compute_persistence(df)
    )

    df["day_of_year"] = (
        pd.to_datetime(
            df["acq_date"]
        ).dt.dayofyear
    )

    minimum = df["frp_mw"].min()
    maximum = df["frp_mw"].max()

    denominator = (
        maximum
        - minimum
        + 1e-9
    )

    df["frp_norm"] = (
        df["frp_mw"] - minimum
    ) / denominator

    feature_cols = [
        "frp_mw",
        "brightness_k",
        "confidence",
        "frp_norm",
        "dist_to_industry_km",
        "is_night",
        "persistence_count",
        "day_of_year",
        "land_cover",
    ]

    meta_cols = [
        "hotspot_id",
        "latitude",
        "longitude",
        "acq_date",
        "region",
    ]

    label_col = (
        ["true_class"]
        if "true_class" in df.columns
        else []
    )

    return df[
        meta_cols
        + feature_cols
        + label_col
    ].copy()
