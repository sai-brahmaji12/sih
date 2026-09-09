from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

CONFIDENCE_MAP = {
    "l": 25,
    "n": 60,
    "h": 90,
    "low": 25,
    "nominal": 60,
    "high": 90,
}

STATE_CENTROIDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Telangana": (18.1124, 79.0193),
    "Tamil Nadu": (11.1271, 78.6569),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Maharashtra": (19.7515, 75.7139),
    "Odisha": (20.9517, 85.0985),
    "West Bengal": (22.9868, 87.8550),
    "Chhattisgarh": (21.2787, 81.8661),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Gujarat": (22.2587, 71.1924),
    "Rajasthan": (27.0238, 74.2179),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Bihar": (25.0961, 85.3131),
    "Jharkhand": (23.6102, 85.2799),
    "Punjab": (31.1471, 75.3412),
    "Haryana": (29.0588, 76.0856),
    "Assam": (26.2006, 92.9376),
    "Uttarakhand": (30.0668, 79.0193),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jammu and Kashmir": (33.7782, 76.5762),
}


def _nearest_region(
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    names = list(STATE_CENTROIDS.keys())

    centroids = np.array(
        [STATE_CENTROIDS[name] for name in names],
        dtype=float,
    )

    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]

    centroid_lat = np.radians(centroids[:, 0])[None, :]
    centroid_lon = np.radians(centroids[:, 1])[None, :]

    dlat = centroid_lat - lat_r
    dlon = centroid_lon - lon_r

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r)
        * np.cos(centroid_lat)
        * np.sin(dlon / 2) ** 2
    )

    dist = 2 * 6371.0 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    nearest = dist.argmin(axis=1)

    return np.array(names)[nearest]


def keep_india_land(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    world_url = (
        "https://raw.githubusercontent.com/nvkelso/"
        "natural-earth-vector/master/geojson/"
        "ne_110m_admin_0_countries.geojson"
    )

    try:
        world = gpd.read_file(world_url)

        india = world[
            world["ADMIN"].astype(str).str.strip().str.lower()
            == "india"
        ]

        if india.empty:
            return df

        geometry = [
            Point(float(lon), float(lat))
            for lat, lon in zip(
                df["latitude"],
                df["longitude"],
            )
        ]

        points = gpd.GeoDataFrame(
            df.copy(),
            geometry=geometry,
            crs="EPSG:4326",
        )

        try:
            india_geometry = india.geometry.union_all()
        except AttributeError:
            india_geometry = india.geometry.unary_union

        mask = points.geometry.intersects(india_geometry)

        result = points.loc[
            mask
        ].drop(
            columns=["geometry"]
        )

        return pd.DataFrame(result)

    except Exception as exc:
        print(
            f"[real_data_loader] Land filter failed: {exc}"
        )
        return df


def load_viirs_csv(
    path: str,
    sample_size: Optional[int] = 8000,
    seed: int = 42,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"VIIRS CSV not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "latitude",
        "longitude",
        "acq_date",
        "daynight",
        "confidence",
        "frp",
        "bright_ti4",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"VIIRS CSV is missing expected columns: {sorted(missing)}"
        )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    df["bright_ti4"] = pd.to_numeric(
        df["bright_ti4"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "frp",
            "bright_ti4",
        ]
    ).copy()

    df = df[
        df["latitude"].between(6.0, 37.5)
        & df["longitude"].between(68.0, 98.0)
    ].copy()

    df = keep_india_land(df)

    df = df[
        df["frp"].between(0, 200)
        & df["bright_ti4"].between(250, 450)
    ].copy()

    if (
        sample_size is not None
        and len(df) > sample_size
    ):
        df = df.sample(
            n=sample_size,
            random_state=seed,
        )

    df = df.reset_index(drop=True)

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    df["confidence"] = (
        df["confidence"]
        .astype(str)
        .str.lower()
        .map(CONFIDENCE_MAP)
        .fillna(50)
        .astype(int)
    )

    out = pd.DataFrame(
        {
            "hotspot_id": [
                f"HS_{i:06d}"
                for i in range(1, len(df) + 1)
            ],
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
            "acq_date": df["acq_date"],
            "acq_daynight": df["daynight"].astype(str),
            "brightness_k": df["bright_ti4"].astype(float),
            "frp_mw": df["frp"].astype(float),
            "confidence": df["confidence"].astype(int),
        }
    )

    out["region"] = _nearest_region(
        out["latitude"].to_numpy(),
        out["longitude"].to_numpy(),
    )

    return out.reset_index(drop=True)
