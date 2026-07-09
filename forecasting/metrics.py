"""
Evaluation metrics. Ported unchanged from the original notebook's Section 8.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    return 100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def mape_weekly(series_true: "pd.Series", series_pred: "pd.Series") -> float:
    """MAPE on weekly-resampled totals. Both inputs must be pd.Series with a DatetimeIndex."""
    import pandas as pd
    true_w = series_true.resample("W-MON", label="left", closed="left").sum()
    pred_w = series_pred.resample("W-MON", label="left", closed="left").sum()
    idx = true_w.index.intersection(pred_w.index)
    true_w, pred_w = true_w.loc[idx].values, pred_w.loc[idx].values
    mask = true_w != 0
    return 100 * float(np.mean(np.abs((true_w[mask] - pred_w[mask]) / true_w[mask])))


def compute_metrics(series_true: "pd.Series", series_pred: "pd.Series", model_name: str) -> dict:
    """
    series_true / series_pred: pd.Series with a DatetimeIndex (hourly).
    Both .values and the index are used — MAPE_weekly resamples to weekly totals.
    """
    y_true, y_pred = series_true.values, series_pred.values
    return {
        "Model":       model_name,
        "MAE":         round(mae(y_true, y_pred), 4),
        "RMSE":        round(rmse(y_true, y_pred), 4),
        "MAPE":        round(mape(y_true, y_pred), 4),
        "MAPE_weekly": round(mape_weekly(series_true, series_pred), 4),
        "sMAPE":       round(smape(y_true, y_pred), 4),
    }


def smape_to_confidence_label(smape_value: float) -> str:
    """
    Map a holdout sMAPE to a plain-language confidence label for non-technical
    audiences (KPI cards), instead of surfacing the raw metric. Thresholds are
    a deliberately coarse, defensible banding for contact-volume forecasting,
    not a statistically derived cutoff.
    """
    if smape_value < 10:
        return "High"
    if smape_value < 20:
        return "Good"
    if smape_value < 35:
        return "Moderate"
    return "Low"
