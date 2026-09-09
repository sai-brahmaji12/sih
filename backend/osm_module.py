import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import requests


OVERPASS_URL = (
    "https://overpass-api.de/api/interpreter"
)

CACHE_PATH = (
    Path(__file__).parent
    / "data"
    / "osm_industrial_cache.json"
)

INDIA_BBOX = (
    6.0,
    68.0,
    37.5,
    98.0,
)

FALLBACK_SITES = [
    {
        "lat": 16.5062,
        "lon": 80.6480,
        "category": "industrial",
    },
    {
        "lat": 17.3850,
        "lon": 78.4867,
        "category": "industrial",
    },
    {
        "lat": 13.0827,
        "lon": 80.2707,
        "category": "industrial",
    },
    {
        "lat": 12.9716,
        "lon": 77.5946,
        "category": "industrial",
    },
    {
        "lat": 11.0168,
        "lon": 76.9558,
        "category": "industrial",
    },
    {
        "lat": 17.6868,
        "lon": 83.2185,
        "category": "industrial",
    },
]


OVERPASS_QUERY_TEMPLATE = """
[out:json][timeout:{timeout}];
(
  node["landuse"="industrial"]({bbox});
  way["landuse"="industrial"]({bbox});
  node["man_made"="works"]({bbox});
  node["power"="plant"]({bbox});
  way["power"="plant"]({bbox});
  node["landuse"="quarry"]({bbox});
  way["landuse"="quarry"]({bbox});
  node["man_made"="mineshaft"]({bbox});
);
out center;
"""


def _bbox_str(bbox):
    south, west, north, east = bbox
    return f"{south},{west},{north},{east}"


def _category_for_tags(tags):
    if tags.get("power") == "plant":
        return "power"

    if (
        tags.get("landuse") == "quarry"
        or tags.get("man_made") == "mineshaft"
    ):
        return "mining"

    return "industrial"


def fetch_osm_industrial_sites(
    bbox=INDIA_BBOX,
    cache_path=CACHE_PATH,
    max_age_days=30,
    timeout=120,
    force_refresh=False,
) -> List[Dict]:

    cache_path = Path(cache_path)

    if (
        not force_refresh
        and cache_path.exists()
    ):
        age_days = (
            time.time()
            - cache_path.stat().st_mtime
        ) / 86400

        if age_days <= max_age_days:
            try:
                cached = json.loads(
                    cache_path.read_text()
                )

                if cached:
                    return cached

            except Exception:
                pass

    query = OVERPASS_QUERY_TEMPLATE.format(
        bbox=_bbox_str(bbox),
        timeout=timeout,
    )

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=timeout,
        )

        response.raise_for_status()

        elements = response.json().get(
            "elements",
            [],
        )

        sites = []

        for element in elements:

            if element.get("type") == "node":
                lat = element.get("lat")
                lon = element.get("lon")
            else:
                center = (
                    element.get("center")
                    or {}
                )

                lat = center.get("lat")
                lon = center.get("lon")

            if lat is None or lon is None:
                continue

            if not (
                6.0 <= float(lat) <= 37.5
                and 68.0 <= float(lon) <= 98.0
            ):
                continue

            sites.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "category": _category_for_tags(
                        element.get("tags", {})
                    ),
                }
            )

        if sites:
            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cache_path.write_text(
                json.dumps(sites)
            )

            return sites

    except Exception as exc:
        print(
            f"[osm_module] OSM unavailable: {exc}"
        )

    return FALLBACK_SITES


def _distance_matrix(
    lat,
    lon,
    sites,
):
    lat = np.asarray(
        lat,
        dtype=float,
    )

    lon = np.asarray(
        lon,
        dtype=float,
    )

    if lat.ndim == 0:
        lat = lat.reshape(1)

    if lon.ndim == 0:
        lon = lon.reshape(1)

    site_lat = np.radians(
        np.array(
            [s["lat"] for s in sites],
            dtype=float,
        )
    )

    site_lon = np.radians(
        np.array(
            [s["lon"] for s in sites],
            dtype=float,
        )
    )

    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]

    dlat = (
        site_lat[None, :]
        - lat_r
    )

    dlon = (
        site_lon[None, :]
        - lon_r
    )

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r)
        * np.cos(site_lat[None, :])
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * 6371.0
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0, 1)
            )
        )
    )


def nearest_industry_distance_km(
    lat,
    lon,
    sites,
):
    if not sites:
        return np.full(
            len(np.asarray(lat)),
            9999.0,
        )

    distances = _distance_matrix(
        lat,
        lon,
        sites,
    )

    return distances.min(axis=1)


def nearest_industry_category(
    lat,
    lon,
    sites,
):
    if not sites:
        return np.array(
            ["industrial"]
            * len(np.asarray(lat))
        )

    distances = _distance_matrix(
        lat,
        lon,
        sites,
    )

    nearest_idx = distances.argmin(
        axis=1
    )

    categories = np.array(
        [
            s["category"]
            for s in sites
        ]
    )

    return categories[
        nearest_idx
    ]
