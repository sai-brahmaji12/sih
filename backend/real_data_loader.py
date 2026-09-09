"""
real_data_loader.py
====================
Loads a real VIIRS/FIRMS hotspot CSV (like the one exported from NASA
FIRMS) and reshapes it into the same column layout your mock generator
produced, so it can flow through data_processing.clean() and
feature_engineering unchanged.

VIIRS confidence comes as l/n/h (low/nominal/high), not 0-100 like MODIS,
so it's mapped to a numeric scale. Region is approximated by nearest
Indian state capital, since the raw file has no admin-region column.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# VIIRS confidence is categorical (low / nominal / high) -> numeric proxy
CONFIDENCE_MAP = {"l": 25, "n": 60, "h": 90}

# Approximate Indian state centroids, used only to give each hotspot a
# human-readable region label (nearest centroid "wins"). This is a rough
# stand-in for real admin-boundary geocoding.
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


def _nearest_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    names = list(STATE_CENTROIDS.keys())
    clat = np.radians(np.array([v[0] for v in STATE_CENTROIDS.values()]))
    clon = np.radians(np.array([v[1] for v in STATE_CENTROIDS.values()]))
    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]
    dlat = clat[None, :] - lat_r
    dlon = clon[None, :] - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(clat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    nearest_idx = dist.argmin(axis=1)
    return np.array(names)[nearest_idx]


def keep_india_land(df):
    """Keep only VIIRS points that fall on Indian land."""

    world_url = (
        "https://raw.githubusercontent.com/nvkelso/"
        "natural-earth-vector/master/geojson/"
        "ne_110m_admin_0_countries.geojson"
    )

    world = gpd.read_file(world_url)

    india = world[world["ADMIN"] == "India"]

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df["longitude"], df["latitude"])
    ]

    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=geometry,
        crs="EPSG:4326"
    )

    # Keep points that are actually inside India's boundary
    points = points[points.geometry.within(india.geometry.union_all())]

    return pd.DataFrame(points.drop(columns="geometry"))

def load_viirs_csv(
    path: str,
    sample_size: Optional[int] = 8000,
    seed: int = 42,
) -> pd.DataFrame:
  
   df = pd.read_csv(path)

# Basic India coordinate range
df = df[
    df["latitude"].between(6.0, 37.0) &
    df["longitude"].between(68.0, 98.0)
].copy()

# Remove points that are actually in the sea
df = keep_india_land(df)
    required = {
        "latitude", "longitude", "acq_date", "daynight",
        "confidence", "frp", "bright_ti4",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"VIIRS CSV is missing expected columns: {missing}")

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    out = pd.DataFrame({
        "hotspot_id": [f"HS_{i:06d}" for i in range(1, len(df) + 1)],
        "latitude": df["latitude"].astype(float),
        "longitude": df["longitude"].astype(float),
        "acq_date": pd.to_datetime(df["acq_date"]),
        "acq_daynight": df["daynight"],
        "brightness_k": df["bright_ti4"].astype(float),
        "frp_mw": df["frp"].astype(float),
        "confidence": df["confidence"].map(CONFIDENCE_MAP).fillna(50).astype(int),
    })
    out["region"] = _nearest_region(out["latitude"].to_numpy(), out["longitude"].to_numpy())
    return out
