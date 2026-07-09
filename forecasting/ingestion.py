"""
Data ingestion: pull the long-format hourly contacts sheet via a Google
service account and collapse it to a single uniform-frequency hourly series.

Ported unchanged in behavior from the original notebook's Section 4
(`ingest_hourly_sheet`, `clean_series`), with column names and credentials
parameterized instead of hardcoded.
"""
import time

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
from gspread.exceptions import APIError

import config


def _sheets_call_with_retry(fn, retries=4, base_delay=15):
    """Call fn(); retry up to `retries` times on 429 with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except APIError as e:
            if e.response.status_code == 429 and attempt < retries:
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
            else:
                raise


def get_client() -> gspread.Client:
    """Return an authenticated gspread client."""
    creds = Credentials.from_service_account_file(
        config.SERVICE_ACCOUNT_FILE, scopes=config.SCOPES
    )
    return gspread.authorize(creds)


def get_worksheet(spreadsheet_url: str, sheet_index: int = 0):
    """Open a spreadsheet by URL and return the worksheet at the given index."""
    client = get_client()
    spreadsheet = client.open_by_url(spreadsheet_url)
    return spreadsheet.get_worksheet(sheet_index)


def ingest_hourly_sheet(spreadsheet_url: str, sheet_name: str) -> pd.DataFrame:
    """
    Pull the hourly contacts sheet.
    The sheet has a title row (row 0) followed by a header row (row 1),
    so we skip the first row when reading via gspread-dataframe.
    """
    def _fetch():
        client = get_client()
        spreadsheet = client.open_by_url(spreadsheet_url)
        ws = spreadsheet.worksheet(sheet_name)
        df = get_as_dataframe(ws, evaluate_formulas=True, header=1)
        df.dropna(how="all", inplace=True)
        return df

    return _sheets_call_with_retry(_fetch)


def clean_series(
    df: pd.DataFrame,
    date_col: str = None,
    hour_col: str = None,
    target_col: str = None,
) -> pd.Series:
    """
    Combine the date and hour-of-day columns into a single datetime index,
    aggregate the target across all category rows for each (day, hour) pair,
    and resample to a uniform hourly frequency.

    The source sheet is long-format: multiple rows share the same (date, hour)
    pair (one row per category), so volumes must be summed before resampling.
    This datetime construction is performed first, before any feature
    engineering, since every downstream step depends on a clean hourly index.
    """
    date_col = date_col or config.DATE_COL
    hour_col = hour_col or config.HOUR_COL
    target_col = target_col or config.TARGET_COL

    df = df[[date_col, hour_col, target_col]].copy()

    # Parse and validate date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    null_dates = df[date_col].isna().sum()
    df.dropna(subset=[date_col], inplace=True)

    # Coerce hour and volume to numeric
    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    # Construct the datetime index: DAY + timedelta(hours=HOUR_OF_DAY)
    df["datetime"] = df[date_col] + pd.to_timedelta(df[hour_col], unit="h")

    # Aggregate across categories: sum volumes for each (day, hour) slot
    grouped = df.groupby("datetime")[target_col].sum().sort_index()

    # Resample to uniform hourly grid (fills any missing hours with 0)
    series_hourly = grouped.resample("h").sum()

    return series_hourly


def ingestion_summary(series: pd.Series, null_dates: int = 0) -> dict:
    """Plain-language ingestion stats for display/logging (no DataFrame reprs)."""
    return {
        "rows": int(series.shape[0]),
        "start": series.index.min(),
        "end": series.index.max(),
        "null_dates_removed": int(null_dates),
        "zero_filled_hours": int((series == 0).sum()),
    }


def load_series() -> pd.Series:
    """Convenience entrypoint: ingest + clean using settings from config."""
    raw_df = ingest_hourly_sheet(config.SOURCE_URL, config.SOURCE_SHEET_NAME)
    return clean_series(raw_df, config.DATE_COL, config.HOUR_COL, config.TARGET_COL)
