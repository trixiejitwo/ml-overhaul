"""
Orchestration layer: the functions both the Flask app and the training
notebook call. Wraps the lower-level modules (ingestion, features, models,
recursive) into two entrypoints:

- run_holdout_validation(): trains every requested model on a temporal
  train/holdout split and returns metrics, exactly mirroring the notebook's
  Sections 7-10 (TEST_SIZE-hour holdout, same metric functions).
- run_forecast(): retrains a model on full history, produces its raw
  forward forecast (recursive loop for tree/linear models, native multi-step
  predict for SARIMA/ETS/Prophet, the seasonal-naive series itself for the
  naive baseline), and blends it with the seasonal-naive baseline via the
  single shared blend_with_naive() call -- this is the one place blending
  happens, regardless of which code path produced the raw forecast.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from forecasting.features import FORECAST_FEATURE_COLS, build_features
from forecasting.metrics import compute_metrics
from forecasting.models import MODEL_KIND, MODEL_LABELS, MODEL_NAMES, train_full, train_holdout
from forecasting.recursive import (
    blend_with_naive,
    forecast_start_from_series,
    next_monday_after_last_complete_week,
    recursive_forecast,
    seasonal_naive_baseline,
)
from forecasting.utils import trailing_window


@dataclass
class ForecastResult:
    model_name: str
    forecast_start: pd.Timestamp
    week_anchor: pd.Timestamp   # Monday of the (possibly partial) week containing forecast_start
    horizon_hours: int
    raw: pd.Series          # the model's own forecast, pre-blend
    naive_baseline: pd.Series
    blended: pd.Series      # what the app displays as "the forecast"
    history_tail: pd.Series  # trailing actuals for chart continuity


@dataclass
class HoldoutResult:
    metrics_df: pd.DataFrame
    predictions: dict = field(default_factory=dict)  # model_name -> pd.Series
    series_test: pd.Series = None


def run_holdout_validation(
    series: pd.Series,
    model_names: list = None,
    test_size: int = None,
    params_by_model: dict = None,
) -> HoldoutResult:
    """
    Train/evaluate `model_names` (default: all registered models) on a fixed
    temporal holdout of the last `test_size` hours, exactly mirroring the
    notebook's Section 7-10 split and metrics.
    """
    model_names = model_names or MODEL_NAMES
    test_size = test_size or config.TEST_SIZE
    params_by_model = params_by_model or {}

    feature_df = build_features(series)
    X = feature_df.drop(columns="target")
    y = feature_df["target"]

    split_idx = len(X) - test_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    series_train = series.loc[y_train.index]
    series_test = series.loc[y_test.index]

    rows = []
    predictions = {}
    for name in model_names:
        params = params_by_model.get(name)
        pred, metrics = train_holdout(name, X_train, y_train, X_test, y_test, series_train, series_test, params)
        rows.append(metrics)
        predictions[name] = pred

    metrics_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return HoldoutResult(metrics_df=metrics_df, predictions=predictions, series_test=series_test)


def run_forecast(
    series: pd.Series,
    model_name: str,
    horizon_hours: int,
    params: dict = None,
    naive_weeks: int = 8,
) -> ForecastResult:
    """
    Retrain `model_name` on the full history and produce its blended forward
    forecast. The forecast starts at the hour immediately after the last
    observed point — no gap. The week_anchor (Monday of the current partial
    week) is stored on the result so the weekly chart can fold partial-week
    forecast hours into the same bucket as the actuals for that week.
    """
    forecast_start = forecast_start_from_series(series)
    week_anchor = next_monday_after_last_complete_week(series)
    naive_index = pd.date_range(start=forecast_start, periods=horizon_hours, freq="h")
    naive_baseline = seasonal_naive_baseline(series, naive_index, weeks=naive_weeks)

    kind = MODEL_KIND[model_name]

    if kind == "naive":
        raw = naive_baseline.copy()

    elif kind == "recursive":
        feature_df = build_features(series)
        X_full = feature_df[FORECAST_FEATURE_COLS]
        y_full = feature_df["target"]
        model = train_full(model_name, X_full, y_full, series, params)
        raw = recursive_forecast(model, series, forecast_start, horizon_hours, FORECAST_FEATURE_COLS)

    elif kind == "native":
        model = train_full(model_name, None, None, series, params)
        raw = model.forecast_horizon(forecast_start, horizon_hours)

    else:
        raise ValueError(f"Unknown model kind for {model_name}: {kind}")

    # Single shared blending step for every model kind.
    blended = blend_with_naive(raw, naive_baseline)

    history_tail = trailing_window(series, 30)

    return ForecastResult(
        model_name=model_name,
        forecast_start=forecast_start,
        week_anchor=week_anchor,
        horizon_hours=horizon_hours,
        raw=raw,
        naive_baseline=naive_baseline,
        blended=blended,
        history_tail=history_tail,
    )


def series_stats_snapshot(series: pd.Series, trend_window_days: int = 30) -> dict:
    """
    Basic series statistics for drift/staleness comparison: mean, std, and a
    simple recent-trend slope (linear fit of the trailing window's daily
    totals against day index).
    """
    tail = trailing_window(series, trend_window_days).resample("d").sum()
    if len(tail) >= 2:
        x = np.arange(len(tail))
        slope = float(np.polyfit(x, tail.values, 1)[0])
    else:
        slope = 0.0

    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "recent_trend": slope,
    }
