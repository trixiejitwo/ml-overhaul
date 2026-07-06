"""
Feature engineering: calendar features, US holiday flag, lag features,
rolling aggregations. Ported unchanged in behavior from the original
notebook's Section 6 (`build_features`) and Section 11.1
(`build_features_for_forecast`, `FORECAST_FEATURE_COLS`).
"""
import holidays
import numpy as np
import pandas as pd

# Features retained for the recursive forward forecast: short lags/rolling
# windows (1-48h) are dropped because they are the first to drift once they
# are computed from the model's own prior predictions rather than ground
# truth. Only daily/weekly seasonal structure is kept.
FORECAST_FEATURE_COLS = [
    "hour", "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "is_weekend", "is_holiday",
    "lag_24", "lag_48", "lag_72", "lag_168",
    "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168",
    "expanding_mean",
]

# Same as above plus the sales regressor column (1-week lagged weekly rings).
FORECAST_FEATURE_COLS_SALES = FORECAST_FEATURE_COLS + ["sales_lag1w"]

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [6, 12, 24, 48, 168]


def us_holiday_dates(start_year: int, end_year: int) -> set:
    """Set of US federal holiday dates (date objects) spanning start_year..end_year inclusive."""
    cal = holidays.US(years=range(start_year, end_year + 1))
    return set(cal.keys())


def _add_calendar_and_holiday(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    holiday_dates = us_holiday_dates(df.index.year.min(), df.index.year.max())
    df["is_holiday"] = df.index.normalize().isin(pd.to_datetime(sorted(holiday_dates))).astype(int)
    return df


def build_features(s: pd.Series) -> pd.DataFrame:
    """Full feature matrix used for holdout-validation training. Drops NaN rows
    produced by lag/rolling warmup."""
    df = s.to_frame(name="target")
    df = _add_calendar_and_holiday(df)

    for lag in LAG_HOURS:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = df["target"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["target"].shift(1).rolling(w).std()

    df["expanding_mean"] = df["target"].shift(1).expanding().mean()

    df.dropna(inplace=True)
    return df


def build_features_for_forecast(s: pd.Series) -> pd.DataFrame:
    """Same feature set as build_features, but without dropping NaN rows —
    the last row (the one being forecast) only has lag/rolling inputs."""
    df = s.to_frame(name="target")
    df = _add_calendar_and_holiday(df)

    for lag in LAG_HOURS:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = df["target"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["target"].shift(1).rolling(w).std()

    df["expanding_mean"] = df["target"].shift(1).expanding().mean()

    return df


def add_sales_regressor(feature_df: pd.DataFrame, sales: pd.Series, lag_weeks: int = 1) -> pd.DataFrame:
    """
    Join a 1-week-lagged weekly sales regressor onto an hourly feature matrix.

    sales: weekly pd.Series indexed by week-start Monday timestamps (from the
           Historical Sales sheet). Both actuals and forecast values are expected
           to already be combined in a single series before calling this.
    lag_weeks: how many weeks to lag the sales signal (default 1 — sales from
               week t-1 predicts contacts in week t).

    Each hourly row gets the sales value from the week that is lag_weeks prior
    to the row's own week. Weeks with no sales value are forward-filled then
    back-filled so there are no NaN gaps at boundaries.
    """
    # Snap each hourly timestamp to its Monday week-start (floor to nearest Mon)
    days_since_mon = feature_df.index.dayofweek  # Mon=0 … Sun=6
    hour_week = (feature_df.index - pd.to_timedelta(days_since_mon, unit="D")).normalize()

    # Build a lagged sales lookup: week W gets sales[W - lag_weeks]
    sales_shifted = sales.copy()
    sales_shifted.index = sales_shifted.index + pd.Timedelta(weeks=lag_weeks)
    # Reindex to every week present in the feature matrix, fill gaps
    all_weeks = pd.DatetimeIndex(sorted(set(hour_week)))
    sales_aligned = sales_shifted.reindex(all_weeks).ffill().bfill()

    # Map back to hourly rows
    feature_df = feature_df.copy()
    feature_df["sales_lag1w"] = pd.Series(hour_week, index=feature_df.index).map(sales_aligned).values
    # Remaining NaNs (start of history before first sales week) → series median
    med = feature_df["sales_lag1w"].median()
    feature_df["sales_lag1w"] = feature_df["sales_lag1w"].fillna(med if pd.notna(med) else 0.0)
    return feature_df
