"""
Read forecast results that the training notebook published to Google Sheets.

The forecast spreadsheet has:
  - One worksheet per run named  Forecast_YYYYMMDD_HHMMSS
  - A _meta worksheet with columns: sheet_name | run_at | data_as_of | horizon_hours

Each forecast worksheet has the horizontal three-block layout written by
export.build_forecast_df():
  cols 0 … BLOCK_SIZE-1   : hourly  (Datetime col + one col per model)
  col  BLOCK_SIZE          : separator (blank header)
  cols BLOCK_SIZE+1 … 2*BLOCK_SIZE : daily  (Date col + one col per model)
  col  2*BLOCK_SIZE+1      : separator
  cols 2*BLOCK_SIZE+2 … 3*BLOCK_SIZE+1 : weekly (Week_Start col + one col per model)

Column positions are read from the actual header row rather than hardcoded
offsets so they stay correct even if MODEL_NAMES grows.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from gspread_dataframe import get_as_dataframe

import config
from forecasting.ingestion import get_client

MODEL_NAMES = ["XGBoost", "LightGBM", "RandomForest", "SeasonalNaive", "HoltWinters", "Prophet", "Ridge"]
MODEL_LABELS = {
    "XGBoost":      "XGBoost",
    "LightGBM":     "LightGBM",
    "RandomForest": "Random Forest",
    "SeasonalNaive":"Seasonal-Naive Baseline",
    "HoltWinters":  "Holt-Winters (ETS)",
    "Prophet":      "Prophet",
    "Ridge":        "Ridge Regression",
}

_META_SHEET = "_meta"

# Pre-compute expected column header names for fast lookup
_HOURLY_TS_COL  = "Datetime"
_DAILY_TS_COL   = "Date"
_WEEKLY_TS_COL  = "Week_Start"


# ---------------------------------------------------------------------------
# Meta / discovery
# ---------------------------------------------------------------------------

def get_latest_meta(dest_url: str = None) -> dict | None:
    """
    Return the most recent row from _meta as a dict:
      {sheet_name, run_at (datetime), data_as_of (date), horizon_hours (int)}
    Returns None if _meta doesn't exist or has no data rows.
    """
    dest_url = dest_url or config.DEST_URL
    client = get_client()
    try:
        spreadsheet = client.open_by_url(dest_url)
        ws = spreadsheet.worksheet(_META_SHEET)
    except Exception:
        return None

    df = get_as_dataframe(ws, evaluate_formulas=False).dropna(how="all")
    if df.empty:
        return None

    last = df.iloc[-1]

    # Read per-model metrics if present (written by publish_forecast_run)
    metrics_by_model = {}
    for name in MODEL_NAMES:
        mae_col   = f"{name}_mae"
        rmse_col  = f"{name}_rmse"
        smape_col = f"{name}_smape"
        if rmse_col in last.index and pd.notna(last[rmse_col]):
            metrics_by_model[name] = {
                "mae":   float(last[mae_col])   if mae_col   in last.index and pd.notna(last[mae_col])   else None,
                "rmse":  float(last[rmse_col])  if rmse_col  in last.index and pd.notna(last[rmse_col])  else None,
                "smape": float(last[smape_col]) if smape_col in last.index and pd.notna(last[smape_col]) else None,
            }

    return {
        "sheet_name":      str(last["sheet_name"]),
        "run_at":          datetime.strptime(str(last["run_at"]), "%Y-%m-%d %H:%M:%S"),
        "data_as_of":      pd.Timestamp(str(last["data_as_of"])),
        "horizon_hours":   int(float(last["horizon_hours"])),
        "metrics_by_model": metrics_by_model,
    }


# ---------------------------------------------------------------------------
# Sheet reader
# ---------------------------------------------------------------------------

def read_forecast_sheet(
    sheet_name: str,
    dest_url: str = None,
) -> dict:
    """
    Read a forecast worksheet and return a dict with three DataFrames:
      {
        "hourly":  DataFrame(index=DatetimeIndex, columns=[model_name, ...]),
        "daily":   DataFrame(index=DatetimeIndex, columns=[model_name, ...]),
        "weekly":  DataFrame(index=DatetimeIndex, columns=[model_name, ...]),
      }

    Column names in each DataFrame are the internal model_name keys
    (e.g. "Ridge", "XGBoost"), not the display labels.
    """
    dest_url = dest_url or config.DEST_URL
    client = get_client()
    spreadsheet = client.open_by_url(dest_url)
    ws = spreadsheet.worksheet(sheet_name)

    # Read the whole sheet as a DataFrame; gspread pads short rows with NaN
    raw = get_as_dataframe(ws, evaluate_formulas=False, header=0)

    headers = list(raw.columns)

    def _extract_block(ts_col_name: str) -> pd.DataFrame:
        """
        Find the timestamp column by name, take the next len(MODEL_NAMES)
        columns as model values, return a clean DataFrame indexed by datetime.
        """
        try:
            ts_idx = headers.index(ts_col_name)
        except ValueError:
            raise ValueError(
                f"Timestamp column '{ts_col_name}' not found in sheet '{sheet_name}'. "
                f"Headers: {headers[:30]}"
            )

        model_cols = headers[ts_idx + 1: ts_idx + 1 + len(MODEL_NAMES)]
        block = raw.iloc[:, ts_idx: ts_idx + 1 + len(MODEL_NAMES)].copy()
        block.columns = [ts_col_name] + model_cols

        # Drop rows where the timestamp is blank/NaN (padding from shorter blocks)
        block = block.dropna(subset=[ts_col_name])
        block = block[block[ts_col_name].astype(str).str.strip() != ""]
        block.index = pd.to_datetime(block[ts_col_name])
        block = block.drop(columns=[ts_col_name])

        # Coerce model columns to float
        for col in block.columns:
            block[col] = pd.to_numeric(block[col], errors="coerce")

        # Rename display-label columns back to internal model_name keys
        label_to_name = {f"{MODEL_LABELS.get(n, n)}_{suffix}": n
                         for n in MODEL_NAMES
                         for suffix in ("Hourly", "Daily", "Weekly")}
        block = block.rename(columns=label_to_name)

        # Keep only columns that are known MODEL_NAMES
        block = block[[c for c in block.columns if c in MODEL_NAMES]]
        return block

    return {
        "hourly": _extract_block(_HOURLY_TS_COL),
        "daily":  _extract_block(_DAILY_TS_COL),
        "weekly": _extract_block(_WEEKLY_TS_COL),
    }


def get_model_series(
    blocks: dict,
    model_name: str,
    granularity: str,
) -> pd.Series:
    """
    Extract a single model's forecast series from the block dict returned by
    read_forecast_sheet().  granularity is 'hourly' | 'daily' | 'weekly'.
    """
    gran = granularity if granularity in ("hourly", "daily", "weekly") else "daily"
    df = blocks[gran]
    if model_name not in df.columns:
        # Fallback: first available model
        model_name = df.columns[0]
    return df[model_name].dropna()
