import numpy as np
import pandas as pd

INDUSTRIAL_SITES = [
    (16.5062, 80.6480),
    (17.3850, 78.4867),
    (13.0827, 80.2707),
    (12.9716, 77.5946),
    (11.0168, 76.9558),
    (17.6868, 83.2185)
]

LAND_COVER_CATEGORIES = [
    "industrial", "agricultural", "forest", "mining",
    "urban", "barren", "grassland", "other"
]

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))

def generate(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    regions = [
        "Andhra Pradesh", "Telangana", "Tamil Nadu",
        "Karnataka", "Odisha", "Maharashtra"
    ]
    land_cover = rng.choice(LAND_COVER_CATEGORIES, n)
    region = rng.choice(regions, n)
    latitude = rng.uniform(8.0, 20.0, n)
    longitude = rng.uniform(74.0, 85.0, n)
    acq_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n), unit="D"
    )
    acq_daynight = rng.choice(["D", "N"], n, p=[0.7, 0.3])
    brightness_k = np.clip(rng.normal(325, 18, n), 280, 400)
    frp_mw = np.clip(rng.gamma(2.2, 7.0, n), 0.1, 200)
    confidence = rng.integers(0, 101, n)

    true_class = np.select(
        [
            land_cover.eq if False else np.array(land_cover) == "forest",
            np.array(land_cover) == "agricultural",
            np.array(land_cover) == "industrial",
            frp_mw > 35
        ],
        [
            "Forest Fire",
            "Crop Residue Fire",
            "Industrial Fire",
            "High Intensity Fire"
        ],
        default="Other Fire"
    )

    df = pd.DataFrame({
        "hotspot_id": [f"HS_{i:05d}" for i in range(1, n + 1)],
        "latitude": latitude.round(6),
        "longitude": longitude.round(6),
        "acq_date": acq_date,
        "acq_daynight": acq_daynight,
        "brightness_k": brightness_k.round(2),
        "frp_mw": frp_mw.round(2),
        "confidence": confidence,
        "land_cover": land_cover,
        "region": region,
        "true_class": true_class
    })
    return df
