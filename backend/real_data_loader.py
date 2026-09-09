from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ---------------------------------------------------------
# Indian state approximate centroids
# Used for assigning a region to each hotspot
# ---------------------------------------------------------

STATE_CENTROIDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
}


# ---------------------------------------------------------
# Confidence conversion
# ---------------------------------------------------------

CONFIDENCE_MAP = {
    "l": 35,
    "n": 65,
    "h": 90,
    "low": 35,
    "nominal": 65,
    "high": 90,
}


# ---------------------------------------------------------
# Find nearest state/region
# ---------------------------------------------------------

def _nearest_region(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> list:

    states = list(STATE_CENTROIDS.keys())

    centroids = np.array(
        [STATE_CENTROIDS[state] for state in states],
        dtype=float,
    )

    result = []

    for lat, lon in zip(latitudes, longitudes):

        # Simple squared-distance calculation.
        # Good enough for approximate region assignment.
        distances = (
            (centroids[:, 0] - lat) ** 2
            + (centroids[:, 1] - lon) ** 2
        )

        nearest_index = int(np.argmin(distances))

        result.append(states[nearest_index])

    return result


# ---------------------------------------------------------
# Keep only points falling on Indian land
# ---------------------------------------------------------

def keep_india_land(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove VIIRS hotspots that fall in the sea.

    Uses Natural Earth country boundaries and keeps
    only points located inside India's land polygon.
    """

    if df.empty:
        return df

    world_url = (
        "https://raw.githubusercontent.com/nvkelso/"
        "natural-earth-vector/master/geojson/"
        "ne_110m_admin_0_countries.geojson"
    )

    try:

        world = gpd.read_file(world_url)

        # Select India
        india = world[
            world["ADMIN"].astype(str).str.strip().str.lower()
            == "india"
        ]

        if india.empty:
            print("Warning: India boundary not found.")
            return df

        # Create Point geometries.
        # IMPORTANT:
        # Shapely uses longitude first, latitude second.
        points = [
            Point(float(lon), float(lat))
            for lat, lon in zip(
                df["latitude"],
                df["longitude"]
            )
        ]

        points_gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=points,
            crs="EPSG:4326",
        )

        # Combine India's geometry
        try:
            india_geometry = india.geometry.union_all()
        except AttributeError:
            # Compatibility with older GeoPandas
            india_geometry = india.geometry.unary_union

        # Keep only points inside Indian land
        mask = points_gdf.geometry.within(india_geometry)

        filtered = points_gdf.loc[
            mask
        ].drop(
            columns=["geometry"]
        )

        print(
            f"Land filter: {len(df)} → "
            f"{len(filtered)} hotspots"
        )

        return pd.DataFrame(filtered)

    except Exception as exc:

        print(
            "Warning: India land filtering failed:"
            f" {exc}"
        )

        # Do not crash the whole application if the
        # external boundary file cannot be downloaded.
        return df


# ---------------------------------------------------------
# Load VIIRS CSV
# ---------------------------------------------------------

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

    # -----------------------------------------------------
    # Read CSV
    # -----------------------------------------------------

    df = pd.read_csv(path)

    print(
        f"Loaded {len(df)} rows from {path.name}"
    )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

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
            "VIIRS CSV is missing expected columns: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Convert coordinates to numeric
    # -----------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    # Remove invalid coordinates
    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).copy()

    # -----------------------------------------------------
    # Basic India bounding box
    #
    # This removes obviously impossible points before
    # performing the more expensive land check.
    # -----------------------------------------------------

    df = df[
        df["latitude"].between(6.0, 37.0)
        & df["longitude"].between(68.0, 98.0)
    ].copy()

    print(
        f"After India bounding box: {len(df)} hotspots"
    )

    # -----------------------------------------------------
    # Remove sea/offshore points
    # -----------------------------------------------------

    df = keep_india_land(df)

    # -----------------------------------------------------
    # Convert important numerical columns
    # -----------------------------------------------------

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    df["bright_ti4"] = pd.to_numeric(
        df["bright_ti4"],
        errors="coerce",
    )

    # Remove rows with invalid fire measurements
    df = df.dropna(
        subset=[
            "frp",
            "bright_ti4",
        ]
    ).copy()

    # -----------------------------------------------------
    # Sample data if requested
    # -----------------------------------------------------

    if (
        sample_size is not None
        and len(df) > sample_size
    ):

        df = df.sample(
            n=sample_size,
            random_state=seed,
        ).reset_index(drop=True)

    else:

        df = df.reset_index(drop=True)

    # -----------------------------------------------------
    # Convert acquisition date
    # -----------------------------------------------------

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Create output dataframe
    # -----------------------------------------------------

    out = pd.DataFrame(
        {
            "hotspot_id": [
                f"HS_{i:06d}"
                for i in range(
                    1,
                    len(df) + 1,
                )
            ],

            "latitude": df[
                "latitude"
            ].astype(float),

            "longitude": df[
                "longitude"
            ].astype(float),

            "acq_date": df[
                "acq_date"
            ],

            "acq_daynight": df[
                "daynight"
            ],

            "brightness_k": df[
                "bright_ti4"
            ].astype(float),

            "frp_mw": df[
                "frp"
            ].astype(float),

            "confidence": df[
                "confidence"
            ]
            .astype(str)
            .str.lower()
            .map(CONFIDENCE_MAP)
            .fillna(50)
            .astype(int),
        }
    )

    # -----------------------------------------------------
    # Assign approximate region/state
    # -----------------------------------------------------

    out["region"] = _nearest_region(
        out["latitude"].to_numpy(),
        out["longitude"].to_numpy(),
    )

    # -----------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------

    out = out.reset_index(drop=True)

    print(
        f"Final VIIRS hotspots: {len(out)}"
    )

    return out
