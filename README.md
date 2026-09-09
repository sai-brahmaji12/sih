# Thermal Watch — Industrial Fire Detection App

A self-contained web app built on top of your pipeline. It now runs on
your **real VIIRS FIRMS data** (`backend/data/viirs-snpp_2023_India.csv`,
~580K hotspots for India, 2023) by default, fused with **real OpenStreetMap
industrial infrastructure** (factories, power plants, mines/quarries) —
your original mock-data pipeline (`generate_mock_data.py` →
`data_processing.py` → `feature_engineering.py` → `classify.py` →
`risk_scoring.py`) is untouched and still available as a fallback/demo mode.

## New: real data + OSM

- **`backend/real_data_loader.py`** — reads the VIIRS CSV, converts it into
  the same column shape your mock pipeline used (renames `frp`→`frp_mw`,
  `bright_ti4`→`brightness_k`, maps VIIRS's l/n/h confidence to a numeric
  scale, approximates a state `region` from the nearest state centroid).
- **`backend/osm_module.py`** — queries the OpenStreetMap Overpass API for
  real industrial land use, power plants, and mines/quarries across India,
  caches the result to disk (`backend/data/osm_industrial_cache.json`) so
  it doesn't have to re-fetch every run, and falls back to a small built-in
  site list if there's no internet or Overpass is unreachable (this sandbox
  couldn't reach it, so it ran on the fallback list when I tested it —
  it'll use real OSM data automatically the first time you run it with
  internet access).
- **`backend/labeling.py`** — real satellite hotspots don't come with a
  "what kind of fire is this" label the way mock data did, so this applies
  simple, explainable rules (distance to OSM industrial sites, how often a
  spot re-fires, intensity, day/night) to assign one of: **Industrial
  Fire, Persistent Industrial Source, Mining Activity, Agricultural
  Burning, Natural Fire, Other** — the categories from your SIH problem
  statement. The classifier is trained on these rule-based labels, so its
  reported "accuracy" measures how well it reproduces the rule, not
  ground truth against real incident reports — read `predicted_class`
  as the rule-based classification made explainable and generalizable,
  not verified ground truth.
- **`backend/real_pipeline.py`** — wires the above together with your
  existing `data_processing.clean()`, `feature_engineering.compute_persistence`,
  and `classify.build_pipeline()`, reusing your original code wherever the
  column shapes line up.

In the app header there's a record-count dropdown (4K–50K) and a
"Retrain on real data" button — it re-samples that many rows from the
580K-row file, re-fetches/reuses OSM sites, and retrains live. Full-file
(580K row) runs are supported from Python directly (see below) but are
too many markers for a smooth in-browser map, so the dashboard samples.

## What you get

- **Live map** of hotspots (Leaflet), colored and sized by risk level/score
- **KPI strip**: total hotspots, high/medium/low risk counts, avg. risk score
- **Filters**: risk level, predicted class, region, min. risk score, ID search
- **Predicted-class breakdown** chart
- **Ranked table** of hotspots, click a row to fly to it on the map
- **Retrain button** — reruns the whole pipeline on a fresh mock batch
- Model accuracy and record count shown in the header

## Run it

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser. The pipeline runs
once automatically on startup (same as running `main.py`), and the
dashboard loads its output immediately.

## How it's wired

- `backend/app.py` — FastAPI service. On startup it runs your pipeline
  (mock data → clean → engineer features → train + classify → risk score)
  and caches the result in memory. Endpoints:
  - `GET /api/summary` — KPIs, class/region counts, model accuracy
  - `GET /api/results` — filtered/sorted hotspot records (risk_level,
    predicted_class, region, min_score, search query params)
  - `GET /api/hotspot/{id}` — single record
  - `POST /api/run` — retrain on a fresh mock batch (optional `seed`,
    `n_records`)
- `frontend/index.html` — single-file dashboard (vanilla JS + Leaflet),
  served by FastAPI at `/`. No build step required.

## Running the full 580K-row file (outside the dashboard)

```python
from real_pipeline import run_real_pipeline
scored_df, report, confusion = run_real_pipeline(sample_size=None)  # None = every row
scored_df.to_csv("full_year_scored.csv", index=False)
```

This can take a few minutes for the full file — the OSM fetch and
feature engineering are vectorized, but training a Random Forest on
500K+ rows and writing the output all take real time.

## Using your own CSV / different year

`backend/real_pipeline.DEFAULT_CSV` points at
`backend/data/viirs-snpp_2023_India.csv`. Pass any other FIRMS CSV with
the same VIIRS column layout (`latitude, longitude, bright_ti4, acq_date,
confidence, frp, daynight, ...`) via `run_real_pipeline(csv_path=...)`,
or drop the new file in `backend/data/` and point `DEFAULT_CSV` at it.

## Notes

- Both pipelines train a fresh classifier on every startup/"Retrain" —
  fine for a demo, but for the real SIH build you'll want to persist the
  trained model and only retrain on demand.
- No database is used — results live in memory for the process lifetime,
  in keeping with the "avoid overengineering" direction from earlier.
- The OSM industrial-site cache (`backend/data/osm_industrial_cache.json`)
  refreshes automatically after 30 days, or immediately if you pass
  `refresh_osm=true` to `/api/run`.
- `land_cover` for real data is a lightweight proxy (near an OSM
  industrial/mining/power site → that category, else "other") since no
  land-cover raster is attached to the FIRMS file. Swap in a real product
  like ESA WorldCover later for more accurate land-cover categories.
