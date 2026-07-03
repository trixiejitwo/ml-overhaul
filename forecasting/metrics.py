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


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    return {
        "Model": model_name,
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
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
