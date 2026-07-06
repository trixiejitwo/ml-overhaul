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
from forecasting.features import (
    FORECAST_FEATURE_COLS,
    FORECAST_FEATURE_COLS_SALES,
    add_sales_regressor,
    build_features,
)
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
    sales: pd.Series = None,
) -> HoldoutResult:
    """
    Train/evaluate `model_names` (default: all registered models) on a fixed
    temporal holdout of the last `test_size` hours.

    sales: combined actual+forecast weekly sales series (pd.Series indexed by
           Monday week-start). Required when model_names contains any +Sales
           models. Base models ignore it even if provided.
    """
    model_names = model_names or MODEL_NAMES
    test_size = test_size or config.TEST_SIZE
    params_by_model = params_by_model or {}

    feature_df = build_features(series)

    # Build sales-augmented feature matrix once if any +Sales model is requested
    has_sales_models = any(MODEL_KIND.get(n, "").endswith("_sales") for n in model_names)
    if has_sales_models and sales is not None:
        feature_df_sales = add_sales_regressor(feature_df, sales)
    else:
        feature_df_sales = None

    X_base = feature_df.drop(columns="target")
    y = feature_df["target"]

    split_idx = len(X_base) - test_size
    X_train_base, X_test_base = X_base.iloc[:split_idx], X_base.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    series_train = series.loc[y_train.index]
    series_test  = series.loc[y_test.index]

    if feature_df_sales is not None:
        X_sales = feature_df_sales.drop(columns="target")
        X_train_sales, X_test_sales = X_sales.iloc[:split_idx], X_sales.iloc[split_idx:]
    else:
        X_train_sales = X_test_sales = None

    rows = []
    predictions = {}
    for name in model_names:
        kind = MODEL_KIND.get(name, "recursive")
        params = params_by_model.get(name)

        if kind.endswith("_sales"):
            if X_train_sales is None:
                print(f"  [{name}] skipped — no sales data provided")
                continue
            cols = FORECAST_FEATURE_COLS_SALES
            pred, metrics = train_holdout(
                name,
                X_train_sales[cols], y_train,
                X_test_sales[cols],  y_test,
                series_train, series_test, params,
            )
        else:
            if kind == "recursive":
                cols = FORECAST_FEATURE_COLS
                pred, metrics = train_holdout(
                    name,
                    X_train_base[cols], y_train,
                    X_test_base[cols],  y_test,
                    series_train, series_test, params,
                )
            else:
                pred, metrics = train_holdout(
                    name,
                    X_train_base, y_train,
                    X_test_base,  y_test,
                    series_train, series_test, params,
                )

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
    sales: pd.Series = None,
) -> ForecastResult:
    """
    Retrain `model_name` on the full history and produce its blended forward
    forecast.

    sales: combined actual+forecast weekly sales series. Required for +Sales
           models. The forecast portion (already in the sheet) provides the
           future regressor values for the horizon.
    """
    forecast_start = forecast_start_from_series(series)
    week_anchor    = next_monday_after_last_complete_week(series)
    naive_index    = pd.date_range(start=forecast_start, periods=horizon_hours, freq="h")
    naive_baseline = seasonal_naive_baseline(series, naive_index, weeks=naive_weeks)

    kind = MODEL_KIND[model_name]

    if kind == "naive":
        raw = naive_baseline.copy()

    elif kind == "recursive":
        feature_df = build_features(series)
        X_full = feature_df[FORECAST_FEATURE_COLS]
        y_full = feature_df["target"]
        model  = train_full(model_name, X_full, y_full, series, params)
        raw    = recursive_forecast(model, series, forecast_start, horizon_hours, FORECAST_FEATURE_COLS)

    elif kind == "recursive_sales":
        if sales is None:
            raise ValueError(f"{model_name} requires sales data")
        feature_df       = build_features(series)
        feature_df_sales = add_sales_regressor(feature_df, sales)
        X_full = feature_df_sales[FORECAST_FEATURE_COLS_SALES]
        y_full = feature_df_sales["target"]
        # Use only the rows that survived build_features NaN-drop for series alignment
        series_aligned = series.loc[feature_df_sales.index]
        model  = train_full(model_name, X_full, y_full, series_aligned, params)
        future_sales_hourly = _make_future_sales(sales, forecast_start, horizon_hours)
        raw = recursive_forecast(
            model, series, forecast_start, horizon_hours,
            FORECAST_FEATURE_COLS_SALES,
            extra_features=future_sales_hourly,
        )

    elif kind == "native":
        model = train_full(model_name, None, None, series, params)
        raw   = model.forecast_horizon(forecast_start, horizon_hours)

    elif kind == "native_sales":
        if sales is None:
            raise ValueError(f"{model_name} requires sales data")
        feature_df       = build_features(series)
        feature_df_sales = add_sales_regressor(feature_df, sales)
        X_full = feature_df_sales[FORECAST_FEATURE_COLS_SALES]
        y_full = feature_df_sales["target"]
        # Align series to the rows that survived NaN-drop
        series_aligned = series.loc[feature_df_sales.index]
        future_sales_hourly = _make_future_sales(sales, forecast_start, horizon_hours)
        model = train_full(model_name, X_full, y_full, series_aligned, params,
                           future_sales=future_sales_hourly)
        raw = model.forecast_horizon(forecast_start, horizon_hours)

    else:
        raise ValueError(f"Unknown model kind for {model_name}: {kind}")

    blended      = blend_with_naive(raw, naive_baseline)
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


def _make_future_sales(sales: pd.Series, forecast_start: pd.Timestamp, horizon_hours: int) -> pd.Series:
    """
    Build an hourly sales_lag1w series for the forecast horizon. For each
    hour in [forecast_start, forecast_start + horizon), use the sales value
    from the prior week (1-week lag). The sales series already contains both
    actuals and forecast values so we just shift by 1 week and broadcast to
    hourly resolution.
    """
    horizon_index = pd.date_range(start=forecast_start, periods=horizon_hours, freq="h")
    # Snap each hour to its Monday week-start
    days_since_mon = horizon_index.dayofweek
    hour_week = (horizon_index - pd.to_timedelta(days_since_mon, unit="D")).normalize()
    # 1-week lag: week W looks up sales[W - 1w]
    lagged_sales = sales.copy()
    lagged_sales.index = lagged_sales.index + pd.Timedelta(weeks=1)
    all_weeks = pd.DatetimeIndex(sorted(set(hour_week)))
    sales_aligned = lagged_sales.reindex(all_weeks).ffill().bfill()
    hourly = pd.Series(hour_week, index=horizon_index).map(sales_aligned)
    hourly = hourly.ffill().bfill()
    # If still NaN (horizon extends past all known sales), fill with last known value
    if hourly.isna().any():
        last_val = sales_aligned.dropna().iloc[-1] if len(sales_aligned.dropna()) else 0.0
        hourly = hourly.fillna(last_val)
    result = pd.Series(hourly.values, index=horizon_index)
    result.name = "sales_lag1w"
    return result


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
