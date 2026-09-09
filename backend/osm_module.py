"""
osm_module.py
==============
Pulls real industrial infrastructure locations (factories, power plants,
mines/quarries) from OpenStreetMap via the Overpass API, so the fire
classifier can measure "how close is this hotspot to a known industrial
site" using real-world data instead of a hardcoded list.

Design goals:
- Works offline after the first successful fetch (results are cached to disk).
- Never crashes the app if there's no internet or Overpass is slow/down —
  falls back to a small built-in list of major Indian industrial hubs.
- Cheap to query repeatedly: distance calculations are fully vectorised
  with numpy so they stay fast even for hundreds of thousands of hotspots.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_PATH = Path(__file__).parent / "data" / "osm_industrial_cache.json"

# Rough bounding box covering mainland India: (south, west, north, east)
INDIA_BBOX = (8.0, 68.0, 35.5, 97.5)

# Used only if OSM can't be reached (no internet, Overpass down/timed out).
# Same fallback the mock pipeline already used, so the app still runs.
FALLBACK_SITES = [
    {"lat": 16.5062, "lon": 80.6480, "category": "industrial"},
    {"lat": 17.3850, "lon": 78.4867, "category": "industrial"},
    {"lat": 13.0827, "lon": 80.2707, "category": "industrial"},
    {"lat": 12.9716, "lon": 77.5946, "category": "industrial"},
    {"lat": 11.0168, "lon": 76.9558, "category": "industrial"},
    {"lat": 17.6868, "lon": 83.2185, "category": "industrial"},
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


def _bbox_str(bbox) -> str:
    south, west, north, east = bbox
    return f"{south},{west},{north},{east}"


def _category_for_tags(tags: dict) -> str:
    if tags.get("power") == "plant":
        return "power"
    if tags.get("landuse") == "quarry" or tags.get("man_made") == "mineshaft":
        return "mining"
    return "industrial"


def fetch_osm_industrial_sites(
    bbox=INDIA_BBOX,
    cache_path: Path = CACHE_PATH,
    max_age_days: float = 30,
    timeout: int = 180,
    force_refresh: bool = False,
) -> List[Dict]:
    """
    Returns a list of {"lat", "lon", "category"} dicts for industrial,
    power-plant, and mining/quarry sites in the given bounding box.

    Tries, in order:
      1. A fresh-enough disk cache (fast, no network needed).
      2. A live Overpass API query (needs internet; can take 10-60s for
         a bbox the size of India).
      3. The small built-in fallback list, if both of the above fail.
    """
    cache_path = Path(cache_path)

    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            try:
                cached = json.loads(cache_path.read_text())
                if cached:
                    return cached
            except (json.JSONDecodeError, OSError):
                pass  # fall through and re-fetch

    query = OVERPASS_QUERY_TEMPLATE.format(bbox=_bbox_str(bbox), timeout=timeout)
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        sites = []
        for el in elements:
            if el.get("type") == "node":
                lat, lon = el.get("lat"), el.get("lon")
            else:  # way -> use computed center
                center = el.get("center") or {}
                lat, lon = center.get("lat"), center.get("lon")
            if lat is None or lon is None:
                continue
            sites.append({
                "lat": lat,
                "lon": lon,
                "category": _category_for_tags(el.get("tags", {})),
            })
        if sites:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(sites))
            return sites
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        print(f"[osm_module] Overpass fetch failed ({exc}); using fallback site list")

    return FALLBACK_SITES


def nearest_industry_distance_km(
    lat: np.ndarray, lon: np.ndarray, sites: List[Dict]
) -> np.ndarray:
    """
    Vectorised distance (km) from each (lat, lon) to the nearest site in
    `sites`. lat/lon can be scalars or numpy arrays of the same length.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    site_lat = np.radians(np.array([s["lat"] for s in sites]))
    site_lon = np.radians(np.array([s["lon"] for s in sites]))

    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]

    dlat = site_lat[None, :] - lat_r
    dlon = site_lon[None, :] - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(site_lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return dist.min(axis=1)


def nearest_industry_category(
    lat: np.ndarray, lon: np.ndarray, sites: List[Dict]
) -> np.ndarray:
    """Category ('industrial' / 'power' / 'mining') of the nearest site."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    site_lat = np.radians(np.array([s["lat"] for s in sites]))
    site_lon = np.radians(np.array([s["lon"] for s in sites]))
    categories = np.array([s["category"] for s in sites])

    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]
    dlat = site_lat[None, :] - lat_r
    dlon = site_lon[None, :] - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(site_lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    nearest_idx = dist.argmin(axis=1)
    return categories[nearest_idx]
