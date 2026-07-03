"""
Recursive forecasting mechanics: Monday-aligned horizon start, hour-by-hour
recursive prediction with feature recomputation, seasonal-naive baseline,
and linear-decay blending. Ported unchanged in behavior from the original
notebook's Section 11 (`next_monday_after_last_complete_week`,
`seasonal_naive_baseline`, `blend_with_naive`, `recursive_forecast`).
"""
import numpy as np
import pandas as pd

from forecasting.features import build_features_for_forecast


def next_monday_after_last_complete_week(s: pd.Series) -> pd.Timestamp:
    """
    Return the Monday 00:00 of the current (possibly incomplete) week that
    contains the last observation. The forecast starts at the hour immediately
    after the last historical point, and the weekly aggregation view folds the
    forecasted partial-week hours back into that Monday's bucket alongside the
    available actuals for that week.

    This means:
    - No gap between last historical point and first forecast point.
    - If history ends mid-week (e.g. Wednesday), the forecast starts Thursday
      00:00 and the weekly bar for that Monday includes both the Mon-Wed
      actuals and the Thu-Sun forecasted values.
    """
    last_ts = s.index.max()
    current_week_monday = last_ts.normalize() - pd.Timedelta(days=last_ts.dayofweek)
    return current_week_monday


def forecast_start_from_series(s: pd.Series) -> pd.Timestamp:
    """Return the first hour to forecast: the hour immediately after the last
    observed timestamp in `s`."""
    return s.index.max() + pd.Timedelta(hours=1)


def seasonal_naive_baseline(history: pd.Series, future_index: pd.DatetimeIndex, weeks: int = 8) -> pd.Series:
    """
    For each timestamp in `future_index`, take the median of the actual
    values observed at the same hour-of-week (hour x day-of-week) over the
    trailing `weeks` weeks of history. Used both as a standalone selectable
    model and to bound recursive forecast drift over long horizons via
    blending.
    """
    lookback_start = history.index.max() - pd.Timedelta(weeks=weeks)
    recent = history.loc[history.index > lookback_start]
    by_hour_of_week = recent.groupby([recent.index.dayofweek, recent.index.hour]).median()

    keys = list(zip(future_index.dayofweek, future_index.hour))
    values = [by_hour_of_week.get(k, recent.median()) for k in keys]
    return pd.Series(values, index=future_index)


def blend_with_naive(ml_forecast: pd.Series, naive: pd.Series) -> pd.Series:
    """
    Blend a model's raw forecast with the seasonal-naive baseline. Weight on
    the model forecast decays linearly from 1.0 at the first forecast hour to
    0.2 at the final horizon hour, so the blend never fully abandons the
    model but bounds how far compounding/long-horizon error can drift.

    This is the single shared blending entrypoint used by every model in
    pipeline.run_forecast(), regardless of whether the raw forecast came from
    the recursive lag-feature loop (tree models, linear/ridge) or a native
    multi-step predict (SARIMA, ETS, Prophet) -- so blend behavior cannot
    drift between model code paths.
    """
    n = len(ml_forecast)
    ml_weight = np.linspace(1.0, 0.2, n)
    naive_aligned = naive.reindex(ml_forecast.index).values
    blended = ml_weight * ml_forecast.values + (1 - ml_weight) * naive_aligned
    return pd.Series(np.clip(blended, 0, None), index=ml_forecast.index)


def recursive_forecast(model, history: pd.Series, start: pd.Timestamp, horizon: int, feature_cols: list) -> pd.Series:
    """
    Iteratively forecast `horizon` hours starting at `start` (which may be
    later than history.index.max() + 1h — any gap is bridged recursively so
    lag/rolling features stay continuous, but only the [start, start+horizon)
    window is returned).

    `model` must expose `.predict(X)` on a DataFrame restricted to
    `feature_cols`, returning an array-like of length 1 per call here.
    """
    working = history.copy()
    last_needed = start + pd.Timedelta(hours=horizon - 1)
    full_index = pd.date_range(start=history.index[-1] + pd.Timedelta(hours=1), end=last_needed, freq="h")
    preds = {}

    for ts in full_index:
        working.loc[ts] = np.nan  # placeholder so build_features can compute calendar feats for ts
        feat_row = build_features_for_forecast(working).loc[[ts]]
        y_hat = model.predict(feat_row[feature_cols])[0]
        y_hat = max(y_hat, 0.0)  # contacts cannot be negative
        working.loc[ts] = y_hat
        if ts >= start:
            preds[ts] = y_hat

    out_index = pd.date_range(start=start, periods=horizon, freq="h")
    return pd.Series([preds[ts] for ts in out_index], index=out_index)
