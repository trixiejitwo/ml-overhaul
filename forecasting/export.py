"""
Export utilities:
- publish_forecast_run(): called from the notebook after training; writes one
  timestamped worksheet (horizontal layout: hourly | daily | weekly blocks) to
  the forecast spreadsheet and upserts a row in the _meta registry sheet.
- build_export_df() / export_to_sheets() / export_to_csv_bytes(): legacy
  per-model export kept for the CSV download route.
"""
import io
from datetime import datetime

import pandas as pd
from gspread_dataframe import get_as_dataframe, set_with_dataframe

import config
from forecasting.ingestion import get_client

MODEL_NAMES = [
    "XGBoost", "LightGBM", "RandomForest", "SeasonalNaive", "HoltWinters", "Prophet", "Ridge",
    "XGBoost+Sales", "LightGBM+Sales", "RandomForest+Sales", "Ridge+Sales", "Prophet+Sales",
]
MODEL_LABELS = {
    "XGBoost":            "XGBoost",
    "LightGBM":           "LightGBM",
    "RandomForest":       "Random Forest",
    "SeasonalNaive":      "Seasonal-Naive Baseline",
    "HoltWinters":        "Holt-Winters (ETS)",
    "Prophet":            "Prophet",
    "Ridge":              "Ridge Regression",
    "XGBoost+Sales":      "XGBoost + Sales",
    "LightGBM+Sales":     "LightGBM + Sales",
    "RandomForest+Sales": "Random Forest + Sales",
    "Ridge+Sales":        "Ridge + Sales",
    "Prophet+Sales":      "Prophet + Sales",
}

# Fixed column positions in the horizontal layout so the reader never has to
# guess.  Each block is: 1 timestamp col + N model cols.  Separator cols
# between blocks are single blank-header columns.
#
# With 7 models (MODEL_NAMES order):
#   cols 0-7   : hourly block  (Datetime + 7 model cols)
#   col  8     : separator
#   cols 9-16  : daily block   (Date + 7 model cols)
#   col  17    : separator
#   cols 18-25 : weekly block  (Week_Start + 7 model cols)
#
# If MODEL_NAMES ever changes length these offsets shift — recalculate via
# BLOCK_SIZE below rather than hardcoding magic numbers.
BLOCK_SIZE = 1 + len(MODEL_NAMES)   # timestamp col + one col per model
HOURLY_START  = 0
DAILY_START   = BLOCK_SIZE + 1      # +1 for separator
WEEKLY_START  = BLOCK_SIZE * 2 + 2  # +2 for two separators

META_SHEET = "_meta"


def _model_col_names(suffix: str) -> list[str]:
    """Return column headers for one block, e.g. ['Datetime', 'XGBoost_Hourly', ...]."""
    ts_col = {"Hourly": "Datetime", "Daily": "Date", "Weekly": "Week_Start"}[suffix]
    return [ts_col] + [f"{MODEL_LABELS.get(n, n)}_{suffix}" for n in MODEL_NAMES]


def build_forecast_df(forecasts: dict) -> pd.DataFrame:
    """
    Build the horizontal three-block export DataFrame from a dict of
    model_name -> hourly blended pd.Series.  Only models present in
    `forecasts` are written; missing models produce no column (so a run
    without +Sales models still produces a valid sheet).
    """
    active_names = [n for n in MODEL_NAMES if n in forecasts]

    # --- Block A: Hourly ---
    first = forecasts[active_names[0]]
    block_a = pd.DataFrame({"Datetime": first.index.strftime("%Y-%m-%d %H:%M")})
    for name in active_names:
        block_a[f"{MODEL_LABELS.get(name, name)}_Hourly"] = forecasts[name].values

    # --- Block B: Daily ---
    daily = {n: forecasts[n].resample("D").sum() for n in active_names}
    block_b = pd.DataFrame({"Date": daily[active_names[0]].index.strftime("%Y-%m-%d")})
    for name in active_names:
        block_b[f"{MODEL_LABELS.get(name, name)}_Daily"] = daily[name].values

    # --- Block C: Weekly (Mon-anchored) ---
    weekly = {n: daily[n].resample("W-MON", label="left", closed="left").sum()
              for n in active_names}
    block_c = pd.DataFrame({"Week_Start": weekly[active_names[0]].index.strftime("%Y-%m-%d")})
    for name in active_names:
        block_c[f"{MODEL_LABELS.get(name, name)}_Weekly"] = weekly[name].values

    max_rows = max(len(block_a), len(block_b), len(block_c))

    def pad(df, n):
        if len(df) < n:
            empty = pd.DataFrame("", index=range(n - len(df)), columns=df.columns)
            return pd.concat([df, empty], ignore_index=True)
        return df

    sep = pd.DataFrame({"": [""] * max_rows})
    return pd.concat(
        [pad(block_a, max_rows),
         sep,
         pad(block_b, max_rows),
         sep.rename(columns={"": " "}),
         pad(block_c, max_rows)],
        axis=1,
    )


def publish_forecast_run(
    forecasts: dict,
    data_as_of: pd.Timestamp,
    dest_url: str = None,
    metrics_by_model: dict = None,
) -> str:
    """
    Write one timestamped worksheet to the forecast spreadsheet and upsert a
    row in the _meta sheet.  Called from the training notebook after all models
    have been forecast.

    forecasts: dict of model_name -> hourly blended pd.Series (all MODEL_NAMES).
    data_as_of: last observed timestamp from the ingestion series.
    Returns the new sheet name (e.g. 'Forecast_20260701_143200').
    """
    dest_url = dest_url or config.DEST_URL
    run_at = datetime.now()
    sheet_name = run_at.strftime("Forecast_%Y%m%d_%H%M%S")

    export_df = build_forecast_df(forecasts)
    client = get_client()
    spreadsheet = client.open_by_url(dest_url)

    # Write the forecast sheet
    ws = spreadsheet.add_worksheet(title=sheet_name,
                                   rows=len(export_df) + 5, cols=len(export_df.columns) + 2)
    set_with_dataframe(ws, export_df, include_index=False, include_column_header=True)

    # Bold + freeze header row
    header_range = f"A1:{_col_letter(len(export_df.columns))}1"
    ws.format(header_range, {"textFormat": {"bold": True}})
    spreadsheet.batch_update({"requests": [{
        "updateSheetProperties": {
            "properties": {"sheetId": ws.id, "gridProperties": {"frozenRowCount": 1}},
            "fields": "gridProperties.frozenRowCount",
        }
    }]})

    # Upsert _meta sheet — one row per run, never deleted
    _upsert_meta(spreadsheet, sheet_name, run_at, data_as_of,
                 horizon_hours=len(forecasts[MODEL_NAMES[0]]),
                 metrics_by_model=metrics_by_model)

    print(f"Published: {sheet_name}  ({len(export_df)} rows, data_as_of={data_as_of.date()})")
    return sheet_name


def _col_letter(n: int) -> str:
    """Convert 1-based column number to a Sheets column letter (A, B, … Z, AA…)."""
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def _upsert_meta(spreadsheet, sheet_name: str, run_at: datetime,
                 data_as_of: pd.Timestamp, horizon_hours: int,
                 metrics_by_model: dict = None):
    """
    Append a row to _meta, creating the sheet if it doesn't exist yet.
    If metric columns are missing from an existing header row, they are added
    before appending so the new row always lands in the right columns.
    """
    metric_cols = []
    for name in MODEL_NAMES:
        metric_cols += [f"{name}_mae", f"{name}_rmse", f"{name}_smape"]

    try:
        meta_ws = spreadsheet.worksheet(META_SHEET)
        # Check whether the metric columns already exist in the header
        existing_header = meta_ws.row_values(1)
        missing = [c for c in metric_cols if c not in existing_header]
        if missing:
            # Extend the header row with the missing metric columns
            new_header = existing_header + missing
            meta_ws.update([new_header], "A1")
    except Exception:
        meta_ws = spreadsheet.add_worksheet(title=META_SHEET, rows=500, cols=50)
        header = ["sheet_name", "run_at", "data_as_of", "horizon_hours"] + metric_cols
        meta_ws.append_row(header, value_input_option="RAW")

    # Re-read the header to know which column each value belongs to
    header = meta_ws.row_values(1)
    col_index = {h: i for i, h in enumerate(header)}

    base_row = [sheet_name,
                run_at.strftime("%Y-%m-%d %H:%M:%S"),
                str(data_as_of.date()),
                horizon_hours]
    row = [""] * len(header)
    for i, v in enumerate(base_row):
        row[i] = v

    if metrics_by_model:
        for name in MODEL_NAMES:
            m = metrics_by_model.get(name, {})
            for suffix, key in (("_mae", "mae"), ("_rmse", "rmse"), ("_smape", "smape")):
                col = f"{name}{suffix}"
                if col in col_index:
                    row[col_index[col]] = m.get(key, "")

    meta_ws.append_row(row, value_input_option="RAW")


# ---------------------------------------------------------------------------
# Legacy helpers kept for the CSV download route
# ---------------------------------------------------------------------------

def build_export_df(forecasts: dict, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Original layout used by export_to_csv_bytes."""
    names = list(forecasts.keys())
    labels = {name: MODEL_LABELS.get(name, name) for name in names}

    first = forecasts[names[0]]
    block_a = pd.DataFrame({"Datetime": first.index.strftime("%Y-%m-%d %H:%M:%S")})
    for name in names:
        block_a[f"{labels[name]}_Hourly"] = forecasts[name].values

    daily = {name: forecasts[name].resample("D").sum() for name in names}
    block_b = pd.DataFrame({"Date": daily[names[0]].index.strftime("%Y-%m-%d")})
    for name in names:
        block_b[f"{labels[name]}_Daily"] = daily[name].values

    weekly = {name: daily[name].resample("W-MON", label="left", closed="left").sum()
              for name in names}
    block_c = pd.DataFrame({"Week_Start": weekly[names[0]].index.strftime("%Y-%m-%d")})
    for name in names:
        block_c[f"{labels[name]}_Weekly"] = weekly[name].values

    block_d = metrics_df[["Model", "MAE", "RMSE", "sMAPE"]].copy()
    max_rows = max(len(block_a), len(block_b), len(block_c), len(block_d))

    def pad_df(df, n):
        if len(df) < n:
            empty = pd.DataFrame("", index=range(n - len(df)), columns=df.columns)
            return pd.concat([df, empty], ignore_index=True)
        return df

    sep = pd.DataFrame({"": [""] * max_rows})
    return pd.concat(
        [pad_df(block_a, max_rows), sep,
         pad_df(block_b, max_rows), sep.rename(columns={"": " "}),
         pad_df(block_c, max_rows), sep.rename(columns={"": "  "}),
         pad_df(block_d, max_rows)],
        axis=1,
    )


def export_to_sheets(forecasts: dict, metrics_df: pd.DataFrame, dest_url: str = None) -> str:
    dest_url = dest_url or config.DEST_URL
    sheet_name = datetime.now().strftime("Forecast_%Y%m%d_%H%M%S")
    export_df = build_export_df(forecasts, metrics_df)
    client = get_client()
    spreadsheet = client.open_by_url(dest_url)
    worksheet = spreadsheet.add_worksheet(title=sheet_name,
                                          rows=len(export_df) + 10, cols=50)
    set_with_dataframe(worksheet, export_df, include_index=False, include_column_header=True)
    header_range = f"A1:{chr(64 + len(export_df.columns))}1"
    worksheet.format(header_range, {"textFormat": {"bold": True}})
    spreadsheet.batch_update({"requests": [{
        "updateSheetProperties": {
            "properties": {"sheetId": worksheet.id, "gridProperties": {"frozenRowCount": 1}},
            "fields": "gridProperties.frozenRowCount",
        }
    }]})
    return sheet_name


def export_to_csv_bytes(forecasts: dict, metrics_df: pd.DataFrame) -> bytes:
    export_df = build_export_df(forecasts, metrics_df)
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
