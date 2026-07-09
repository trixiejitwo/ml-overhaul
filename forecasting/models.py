"""
Model definitions and a registry that drives both holdout validation and
forward forecasting. Each entry describes how to train a model for holdout
evaluation, how to retrain it on full history, and how it produces a raw
multi-step-ahead forecast.

Three forecast "kinds" exist, reflecting how each model is fundamentally
shaped -- this is a structural property of the model, not a stylistic
choice, so it is *not* something pipeline.run_forecast() can paper over:

- "recursive": trained on the engineered feature matrix; forward forecast
  comes from forecasting.recursive.recursive_forecast() (hour-by-hour, each
  prediction fed back in as input to the next step).
- "native": forecasts its own multi-step horizon directly from the raw
  series (ETS, Prophet) -- no engineered feature matrix involved.
- "naive": the seasonal-naive baseline itself, selectable as a standalone
  model rather than only an internal blending term.

Regardless of kind, every model's raw forecast passes through the same
forecasting.recursive.blend_with_naive() call in pipeline.run_forecast() --
the blending step is not duplicated per model.

Note: SARIMA is excluded. At m=24 on a full ~12k-hour history, auto_arima
can select seasonal orders whose Kalman filter state covariance array
(state_dim^2 * n_obs) is too large to allocate. Re-add only after
constraining the data window or reducing seasonal period.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecasting.features import us_holiday_dates
from forecasting.metrics import compute_metrics
from forecasting.recursive import seasonal_naive_baseline

DEFAULT_PARAMS = {
    "XGBoost": dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ),
    "LightGBM": dict(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ),
    "RandomForest": dict(
        n_estimators=300, max_depth=15, min_samples_leaf=4, random_state=42,
    ),
    "SeasonalNaive": dict(weeks=8),
    "HoltWinters": dict(
        trend="add", seasonal="add", seasonal_periods=168, damped_trend=True,
    ),
    "Prophet": dict(
        yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True,
    ),
    "Ridge": dict(alpha=1.0, random_state=42),
    # +Sales variants inherit base model params; no separate defaults needed
    "XGBoost+Sales":     dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "LightGBM+Sales":    dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "RandomForest+Sales":dict(n_estimators=300, max_depth=15, min_samples_leaf=4, random_state=42),
    "Ridge+Sales":       dict(alpha=1.0, random_state=42),
    "Prophet+Sales":     dict(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True),
}

MODEL_KIND = {
    "XGBoost":            "recursive",
    "LightGBM":           "recursive",
    "RandomForest":       "recursive",
    "SeasonalNaive":      "naive",
    "HoltWinters":        "native",
    "Prophet":            "native",
    "Ridge":              "recursive",
    "XGBoost+Sales":      "recursive_sales",
    "LightGBM+Sales":     "recursive_sales",
    "RandomForest+Sales": "recursive_sales",
    "Ridge+Sales":        "recursive_sales",
    "Prophet+Sales":      "native_sales",
}

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

# Base 7 models — always trained; +Sales variants trained separately when
# sales data is provided to the pipeline.
BASE_MODEL_NAMES  = ["XGBoost", "LightGBM", "RandomForest", "SeasonalNaive", "HoltWinters", "Prophet", "Ridge"]
SALES_MODEL_NAMES = ["XGBoost+Sales", "LightGBM+Sales", "RandomForest+Sales", "Ridge+Sales", "Prophet+Sales"]
MODEL_NAMES       = BASE_MODEL_NAMES + SALES_MODEL_NAMES


def _prophet_holidays_df(years) -> pd.DataFrame:
    dates = sorted(us_holiday_dates(min(years), max(years)))
    return pd.DataFrame({
        "holiday": "us_federal_holiday",
        "ds": pd.to_datetime(dates),
        "lower_window": 0,
        "upper_window": 0,
    })


# ---------------------------------------------------------------------------
# Holdout training: each model is trained on the train split and evaluated
# on the 30-day (TEST_SIZE) holdout. Tree/linear models use the engineered
# feature matrix (X_train/X_test); series-native models use the raw
# train/test series.
# ---------------------------------------------------------------------------

def _base_name(model_name: str) -> str:
    """Strip +Sales suffix to get the underlying estimator type."""
    return model_name.replace("+Sales", "")


def train_holdout(model_name: str, X_train, y_train, X_test, y_test,
                   series_train: pd.Series, series_test: pd.Series, params: dict = None) -> tuple:
    """Returns (predictions: pd.Series indexed like y_test, metrics: dict).

    For +Sales models, X_train/X_test must already contain the sales_lag1w
    column (added by pipeline via add_sales_regressor before calling here).
    """
    params = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}
    base = _base_name(model_name)

    if base == "XGBoost":
        fit_params = {k: v for k, v in params.items() if k not in ("early_stopping_rounds",)}
        model = xgb.XGBRegressor(
            **fit_params, early_stopping_rounds=params.get("early_stopping_rounds", 50), verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = np.clip(model.predict(X_test), 0, None)
        pred = pd.Series(y_pred, index=y_test.index)

    elif base == "LightGBM":
        model = lgb.LGBMRegressor(**params, verbosity=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        y_pred = np.clip(model.predict(X_test), 0, None)
        pred = pd.Series(y_pred, index=y_test.index)

    elif base == "RandomForest":
        model = RandomForestRegressor(**params, n_jobs=1, verbose=0)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)
        pred = pd.Series(y_pred, index=y_test.index)

    elif base == "Ridge":
        model = Ridge(**params)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)
        pred = pd.Series(y_pred, index=y_test.index)

    elif model_name == "SeasonalNaive":
        pred = seasonal_naive_baseline(series_train, series_test.index, weeks=params.get("weeks", 8))

    elif model_name == "HoltWinters":
        model = ExponentialSmoothing(
            series_train,
            trend=params.get("trend"), seasonal=params.get("seasonal"),
            seasonal_periods=params.get("seasonal_periods", 168),
            damped_trend=params.get("damped_trend", True),
        ).fit()
        y_pred = np.clip(model.forecast(len(series_test)), 0, None)
        pred = pd.Series(y_pred.values, index=series_test.index)

    elif base == "Prophet":
        years = range(series_train.index.year.min(), series_train.index.year.max() + 1)
        prophet_df = pd.DataFrame({"ds": series_train.index, "y": series_train.values})
        p_params = {k: v for k, v in params.items()}
        model = Prophet(holidays=_prophet_holidays_df(years), **p_params)
        if model_name == "Prophet+Sales":
            # sales_lag1w is a column in X_train/X_test — extract as regressor series
            model.add_regressor("sales_lag1w")
            prophet_df["sales_lag1w"] = X_train["sales_lag1w"].values
        model.fit(prophet_df)
        future = pd.DataFrame({"ds": series_test.index})
        if model_name == "Prophet+Sales":
            future["sales_lag1w"] = X_test["sales_lag1w"].values
        forecast = model.predict(future)
        y_pred = np.clip(forecast["yhat"].values, 0, None)
        pred = pd.Series(y_pred, index=series_test.index)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    metrics = compute_metrics(series_test.values, pred.values, MODEL_LABELS.get(model_name, model_name))
    return pred, metrics


# ---------------------------------------------------------------------------
# Full-history retraining + forward forecast generation. Recursive-kind
# models return a fitted estimator; native/naive-kind models return a small
# wrapper exposing a uniform `.forecast_horizon(start, horizon)` so
# pipeline.run_forecast() can treat all kinds uniformly after this point.
# ---------------------------------------------------------------------------

class NativeHorizonModel:
    """Wraps a series-native model (SARIMA/ETS/Prophet) so pipeline code can
    request an arbitrary forward window without caring which library is
    underneath."""

    def __init__(self, kind: str, fitted_model, last_history_ts: pd.Timestamp):
        self.kind = kind
        self.fitted_model = fitted_model
        self.last_history_ts = last_history_ts

    def forecast_horizon(self, start: pd.Timestamp, horizon: int) -> pd.Series:
        # Bridge any gap between end-of-history and the (possibly later,
        # Monday-aligned) requested start, then slice to [start, start+horizon).
        last_needed = start + pd.Timedelta(hours=horizon - 1)
        n_periods = int((last_needed - self.last_history_ts) / pd.Timedelta(hours=1))
        full_index = pd.date_range(
            start=self.last_history_ts + pd.Timedelta(hours=1), periods=n_periods, freq="h"
        )

        if self.kind == "HoltWinters":
            y_pred = np.clip(self.fitted_model.forecast(n_periods), 0, None)
            full = pd.Series(y_pred.values, index=full_index)
        elif self.kind == "Prophet":
            future = pd.DataFrame({"ds": full_index})
            forecast = self.fitted_model.predict(future)
            full = pd.Series(np.clip(forecast["yhat"].values, 0, None), index=full_index)
        else:
            raise ValueError(f"Unsupported native kind: {self.kind!r}")

        return full.loc[start: start + pd.Timedelta(hours=horizon - 1)]


class NativeSalesHorizonModel:
    """Like NativeHorizonModel but carries future sales values for Prophet+Sales."""

    def __init__(self, fitted_model, last_history_ts: pd.Timestamp, future_sales: pd.Series):
        self.fitted_model = fitted_model
        self.last_history_ts = last_history_ts
        self.future_sales = future_sales  # hourly sales_lag1w for the forecast window

    def forecast_horizon(self, start: pd.Timestamp, horizon: int) -> pd.Series:
        last_needed = start + pd.Timedelta(hours=horizon - 1)
        n_periods = int((last_needed - self.last_history_ts) / pd.Timedelta(hours=1))
        full_index = pd.date_range(
            start=self.last_history_ts + pd.Timedelta(hours=1), periods=n_periods, freq="h"
        )
        future = pd.DataFrame({"ds": full_index})
        # Align future sales to the forecast index
        sales_aligned = self.future_sales.reindex(full_index).ffill().bfill()
        future["sales_lag1w"] = sales_aligned.values
        forecast = self.fitted_model.predict(future)
        full = pd.Series(np.clip(forecast["yhat"].values, 0, None), index=full_index)
        return full.loc[start: start + pd.Timedelta(hours=horizon - 1)]


def train_full(model_name: str, X_full: pd.DataFrame, y_full: pd.Series,
                series_full: pd.Series, params: dict = None, future_sales: pd.Series = None):
    """
    Refit `model_name` on the full history. Returns either a fitted
    estimator with `.predict(X)` (recursive-kind models, used directly by
    forecasting.recursive.recursive_forecast) or a NativeHorizonModel
    (native-kind models). SeasonalNaive has no fitted model -- callers
    should special-case it.

    For +Sales models, X_full must already contain sales_lag1w. For
    Prophet+Sales, future_sales (hourly series covering the forecast window)
    must also be supplied.
    """
    params = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}
    kind = MODEL_KIND[model_name]
    base = _base_name(model_name)

    if kind == "naive":
        return None

    if model_name == "HoltWinters":
        model = ExponentialSmoothing(
            series_full,
            trend=params.get("trend"), seasonal=params.get("seasonal"),
            seasonal_periods=params.get("seasonal_periods", 168),
            damped_trend=params.get("damped_trend", True),
        ).fit()
        return NativeHorizonModel("HoltWinters", model, series_full.index.max())

    if base == "Prophet":
        years = range(series_full.index.year.min(), series_full.index.year.max() + 1)
        prophet_df = pd.DataFrame({"ds": series_full.index, "y": series_full.values})
        model = Prophet(holidays=_prophet_holidays_df(years), **params)
        if model_name == "Prophet+Sales":
            model.add_regressor("sales_lag1w")
            prophet_df["sales_lag1w"] = X_full["sales_lag1w"].values
        model.fit(prophet_df)
        if model_name == "Prophet+Sales":
            return NativeSalesHorizonModel(model, series_full.index.max(), future_sales)
        return NativeHorizonModel("Prophet", model, series_full.index.max())

    if base == "XGBoost":
        model = xgb.XGBRegressor(**params, verbosity=0)
        model.fit(X_full, y_full)
        return model

    if base == "LightGBM":
        model = lgb.LGBMRegressor(**params, verbosity=-1)
        model.fit(X_full, y_full)
        return model

    if base == "RandomForest":
        model = RandomForestRegressor(**params, n_jobs=1, verbose=0)
        model.fit(X_full, y_full)
        return model

    if base == "Ridge":
        model = Ridge(**params)
        model.fit(X_full, y_full)
        return model

    raise ValueError(f"Unknown model: {model_name}")
