# Contact Volume Forecast Dashboard

An internal WFM dashboard for hourly contact-center volume forecasting, built on the same
data ingestion, feature engineering, and recursive forecasting logic as the original EDA
notebook (`notebooks/contact_volume_forecasting (naive + recursive + holidays).ipynb`).

Server-rendered Flask + Jinja2 + htmx + Plotly.js — no React/Vue/SPA framework, no build
step required.

## Project structure

```
ml-mvp/
├── app.py                  # entrypoint: python app.py
├── config.py                # env-driven settings
├── forecasting/              # shared core module — imported by app AND training notebook
│   ├── ingestion.py            # Google Sheets pull + datetime construction + clean_series
│   ├── features.py              # calendar/holiday/lag/rolling feature engineering
│   ├── metrics.py                # MAE/RMSE/sMAPE + confidence labeling
│   ├── models.py                  # all 8 models, holdout training, full-history retraining
│   ├── recursive.py                # recursive forecast loop, seasonal-naive, blending, Monday alignment
│   ├── pipeline.py                  # orchestration: run_holdout_validation(), run_forecast()
│   ├── export.py                     # Google Sheets / CSV export (hourly/daily/weekly/metrics blocks)
│   └── registry.py                    # SQLite run registry (atomic active-run switching)
├── app/
│   ├── services.py            # glue: cache lookups, KPI/staleness computation, Plotly figure building
│   ├── cache.py                 # diskcache wrapper keyed by (model, horizon, data-as-of)
│   ├── routes/
│   │   ├── pages.py                # full-page routes
│   │   └── fragments.py             # htmx partial routes
│   └── templates/                # Jinja2: base shell, dashboard, fragments/
├── notebooks/
│   ├── contact_volume_forecasting (...).ipynb   # original EDA/validation notebook, untouched
│   └── training.ipynb                            # companion training notebook
├── static/                  # app.css, app.js (Tailwind/htmx/Plotly loaded via CDN in base.html)
└── data/
    ├── registry.db             # SQLite run registry (created on first run)
    └── cache/                    # diskcache forecast cache (created on first run)
```

## 1. Service account setup

1. In Google Cloud Console, create a service account and enable the Sheets API and Drive API.
2. Generate a JSON key for the service account and save it somewhere on disk (default
   expected path: `service_account.json` in the project root — configurable, see below).
3. Share both the source spreadsheet (hourly contacts) and the destination spreadsheet
   (forecast exports) with the service account's email address (Editor access).

## 2. Configuration

Copy `.env.example` to `.env` and fill in:

```
SERVICE_ACCOUNT_FILE=service_account.json
SOURCE_URL=https://docs.google.com/spreadsheets/d/.../edit
SOURCE_SHEET_NAME=Historical Contacts (Hourly)
DEST_URL=https://docs.google.com/spreadsheets/d/.../edit
```

`DATE_COL`, `HOUR_COL`, `TARGET_COL` default to `DAY`, `HOUR_OF_DAY`, `TOTAL_VOLUME`
(matching the source sheet's long format) and rarely need changing.

## 3. Install

```
pip install -r requirements.txt
```

No build step is required — Tailwind, htmx, and Plotly.js load from CDN in
`app/templates/base.html`.

## 4. Run the training notebook (first time, and whenever you retune models)

```
jupyter notebook notebooks/training.ipynb
```

This notebook:
- Loads live data via `forecasting.ingestion.load_series()`.
- Trains/validates every model via `forecasting.pipeline.run_holdout_validation()` (same
  30-day holdout, same MAE/RMSE/sMAPE functions the app would use).
- Logs each run's params and metrics to `data/registry.db`, along with a series-stats
  snapshot used later for staleness comparison.
- Marks each newly logged run as the **active** run per model — the dashboard always
  reads whichever run is marked active, so you can keep experimenting in the notebook
  without affecting the live app until you choose to promote a run.

The app will not train any models on page load — it only reads metadata/metrics from
this registry and computes/caches forecasts on demand.

## 5. Launch the app

```
python app.py
```

Visit `http://localhost:5000`. Defaults to whichever active model has the best RMSE and
a 4-week horizon.

## Notes

- **Forecast horizon control** maps weeks → hours internally (`FORECAST_HORIZON` in the
  original notebook's terms) and always starts at the Monday 00:00 following the last
  fully completed week in the data, so weekly aggregates are always clean, complete weeks.
- **Caching**: forecast outputs are cached on disk keyed by `(model, horizon, data-as-of)`.
  Revisiting a previously viewed combination is instant; only a genuinely new combination
  for the current data snapshot computes live, in the request path.
- **Models**: SARIMA, XGBoost, LightGBM, Random Forest, Seasonal-Naive, Holt-Winters (ETS),
  Prophet, and Ridge Regression, plus a "Compare All" overlay mode. All recursive (tree/
  linear) and native-horizon (SARIMA/ETS/Prophet) models converge on a single shared
  seasonal-naive blending step in `forecasting.pipeline.run_forecast()`.
