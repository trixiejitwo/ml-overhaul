"""
Glue layer between Flask routes and the data sources.

Two independent in-process caches with 15-minute TTL background refresh:
  - Ingestion cache: the live hourly contacts series from the source sheet.
  - Forecast cache:  the latest forecast run read from the forecast sheet.

Neither cache blocks a request on miss beyond the initial cold start — after
the first successful load each is refreshed by a daemon thread so requests
always see stale-but-fast data rather than a slow inline Sheets API call.
"""
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import config
from forecasting.forecast_reader import (
    get_latest_meta,
    get_model_series,
    read_forecast_sheet,
)
from forecasting.ingestion import load_series

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


def _smape_to_confidence_label(smape: float | None) -> str:
    if smape is None:
        return "Unrated"
    if smape <= 10:
        return "High"
    if smape <= 20:
        return "Medium"
    return "Low"

MODEL_COLORS = {
    "XGBoost":      "#f59e0b",
    "LightGBM":     "#10b981",
    "RandomForest": "#8b5cf6",
    "SeasonalNaive":"#94a3b8",
    "HoltWinters":  "#ec4899",
    "Prophet":      "#06b6d4",
    "Ridge":        "#84cc16",
}

# ---------------------------------------------------------------------------
# Ingestion cache — live hourly contacts series
# ---------------------------------------------------------------------------

_ing_lock  = threading.Lock()
_ing_state = {
    "series":      None,
    "data_as_of":  None,
    "loaded_at":   None,   # time.monotonic() of last successful load
}
_ING_TTL = config.INGESTION_POLL_MINUTES * 60


def _load_ingestion():
    series = load_series()
    with _ing_lock:
        _ing_state["series"]     = series
        _ing_state["data_as_of"] = series.index.max()
        _ing_state["loaded_at"]  = time.monotonic()


def _ensure_ingestion_loaded():
    with _ing_lock:
        loaded = _ing_state["loaded_at"]
    if loaded is None:
        _load_ingestion()
    elif time.monotonic() - loaded > _ING_TTL:
        t = threading.Thread(target=_load_ingestion, daemon=True)
        t.start()


def get_series() -> pd.Series:
    _ensure_ingestion_loaded()
    with _ing_lock:
        return _ing_state["series"]


def get_ingestion_as_of() -> pd.Timestamp:
    _ensure_ingestion_loaded()
    with _ing_lock:
        return _ing_state["data_as_of"]


def refresh_ingestion() -> dict:
    """Force-reload ingestion from Sheets. Returns status dict."""
    _load_ingestion()
    return get_data_status()


# ---------------------------------------------------------------------------
# Forecast cache — latest run from the forecast spreadsheet
# ---------------------------------------------------------------------------

_fc_lock  = threading.Lock()
_fc_state = {
    "meta":       None,   # dict from get_latest_meta()
    "blocks":     None,   # dict {"hourly": df, "daily": df, "weekly": df}
    "loaded_at":  None,   # time.monotonic() of last successful load
}
_FC_TTL = config.FORECAST_POLL_MINUTES * 60


def _load_forecast():
    meta = get_latest_meta()
    if meta is None:
        return  # no forecast published yet — leave existing state
    blocks = read_forecast_sheet(meta["sheet_name"])
    with _fc_lock:
        _fc_state["meta"]      = meta
        _fc_state["blocks"]    = blocks
        _fc_state["loaded_at"] = time.monotonic()


def _ensure_forecast_loaded():
    with _fc_lock:
        loaded = _fc_state["loaded_at"]
    if loaded is None:
        _load_forecast()
    elif time.monotonic() - loaded > _FC_TTL:
        t = threading.Thread(target=_load_forecast, daemon=True)
        t.start()


def get_forecast_meta() -> dict | None:
    _ensure_forecast_loaded()
    with _fc_lock:
        return _fc_state["meta"]


def get_forecast_blocks() -> dict | None:
    _ensure_forecast_loaded()
    with _fc_lock:
        return _fc_state["blocks"]


def get_forecast_series(model_name: str, granularity: str) -> pd.Series | None:
    blocks = get_forecast_blocks()
    if blocks is None:
        return None
    return get_model_series(blocks, model_name, granularity)


def refresh_forecast() -> dict:
    """Force-reload from the forecast spreadsheet. Returns status dict."""
    _load_forecast()
    return get_data_status()


def refresh_data() -> dict:
    """Refresh both ingestion and forecast. Called by the Refresh Data button."""
    _load_ingestion()
    _load_forecast()
    return get_data_status()


# ---------------------------------------------------------------------------
# Best model helper
# ---------------------------------------------------------------------------

def best_active_model() -> str:
    """Lowest-RMSE model from _meta metrics, fallback to first MODEL_NAME."""
    meta = get_forecast_meta()
    if meta is None:
        return MODEL_NAMES[0]
    metrics = meta.get("metrics_by_model", {})
    ranked = [(n, m["rmse"]) for n, m in metrics.items() if m.get("rmse") is not None]
    if not ranked:
        return MODEL_NAMES[0]
    return min(ranked, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Status pill
# ---------------------------------------------------------------------------

def get_data_status() -> dict:
    """Freshness summary for the top-bar status pill."""
    _ensure_ingestion_loaded()
    _ensure_forecast_loaded()

    with _ing_lock:
        data_as_of = _ing_state["data_as_of"]
    with _fc_lock:
        meta = _fc_state["meta"]

    hours_since_data = 0
    data_stale = False
    if data_as_of is not None:
        hours_since_data = (pd.Timestamp.now() - data_as_of.tz_localize(None)) / pd.Timedelta(hours=1)
        data_stale = hours_since_data > 48

    forecast_stale = meta is None

    return {
        "data_as_of":     data_as_of,
        "data_stale":     data_stale,
        "forecast_stale": forecast_stale,
        "is_stale":       data_stale or forecast_stale,
        "run_at":         meta["run_at"] if meta else None,
    }


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

def kpi_strip_data(model_name: str, granularity: str) -> dict:
    meta   = get_forecast_meta()
    blocks = get_forecast_blocks()
    series = get_series()

    # Hourly forecast always used for totals regardless of chart granularity
    fc_hourly = get_model_series(blocks, model_name, "hourly") if blocks else None

    # Next-week volume from first 168 hourly forecast hours
    next_week_vol = None
    forecast_start = None
    forecast_end   = None
    pct_change     = None
    if fc_hourly is not None and len(fc_hourly):
        forecast_start  = fc_hourly.index[0]
        forecast_end    = fc_hourly.index[-1]
        first_week      = fc_hourly.iloc[:min(168, len(fc_hourly))]
        next_week_vol   = float(first_week.sum())

        # % change vs last comparable period in actuals
        if series is not None:
            last_comparable = series.iloc[-168:].sum()
            if last_comparable and last_comparable != 0:
                pct_change = (next_week_vol - last_comparable) / last_comparable * 100

    # Accuracy from _meta metrics (written by notebook at publish time)
    meta_model = (meta.get("metrics_by_model", {}) or {}).get(model_name) if meta else None
    smape = meta_model.get("smape") if meta_model else None
    confidence_label = _smape_to_confidence_label(smape)
    accuracy_pct = round(100 - smape, 1) if smape is not None else None

    return {
        "model_name":              model_name,
        "model_label":             MODEL_LABELS.get(model_name, model_name),
        "next_week_vol":           round(next_week_vol) if next_week_vol else None,
        "pct_change_vs_last_week": pct_change,
        "confidence_label":        confidence_label,
        "accuracy_pct":            accuracy_pct,
        "forecast_start":          forecast_start,
        "forecast_end":            forecast_end,
        "run_at":                  meta["run_at"] if meta else None,
        "data_as_of":              meta["data_as_of"] if meta else None,
    }


# ---------------------------------------------------------------------------
# Holiday helpers
# ---------------------------------------------------------------------------

def _holiday_ticks(index: pd.DatetimeIndex) -> list[tuple]:
    if len(index) == 0:
        return []
    try:
        import holidays as _holidays
        years = range(index.min().year, index.max().year + 1)
        cal   = _holidays.US(years=list(years))
        lo, hi = index.min().date(), index.max().date()
        return [(d, name) for d, name in cal.items() if lo <= d <= hi]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Hero chart — actuals (area) + forecast (line) + live-validation overlay
# ---------------------------------------------------------------------------

def _resample_for_granularity(s: pd.Series, granularity: str) -> pd.Series:
    if granularity == "daily":
        return s.resample("D").sum()
    elif granularity == "weekly":
        return s.resample("W-MON", label="left", closed="left").sum()
    return s  # hourly


def _drop_incomplete_trailing_bucket(s: pd.Series, granularity: str) -> pd.Series:
    """
    Drop every trailing bucket that covers a period not yet fully elapsed.
    Applied to the actuals side only (never to forecast).
    """
    if s.empty:
        return s
    now = pd.Timestamp.now()
    if granularity == "daily":
        # Drop any bucket whose day >= today (today is partial)
        return s[s.index.normalize() < now.normalize()]
    elif granularity == "weekly":
        # W-MON bucket; drop if the 7-day window hasn't closed yet
        return s[s.index + pd.Timedelta(days=7) <= now]
    else:  # hourly
        return s[s.index < now.floor("h")]


def build_hero_chart(model_name: str, granularity: str = "daily") -> str:
    """
    Hero time-series chart.
    Seam fix: concat actuals-tail + forecast at hourly resolution BEFORE
    resampling, then split at the first forecast bucket so both sides of the
    seam land in the same resampled bin and the join is continuous.
    """
    def trailing_window(s, days):
        cutoff = s.index.max() - pd.Timedelta(days=days)
        return s.loc[s.index > cutoff]

    series = get_series()
    blocks = get_forecast_blocks()
    meta   = get_forecast_meta()

    trailing_days = {"hourly": 30, "daily": 60, "weekly": 90}.get(granularity, 60)
    hover_fmt = "%{x|%a %b %d, %H:%M}" if granularity == "hourly" else "%{x|%b %d, %Y}"

    fig = go.Figure()

    # --- Build actuals and forecast with seam-continuous resampling ---
    fc_raw = None
    hist_resampled = None
    fc_resampled   = None

    if series is not None:
        history_raw = trailing_window(series, trailing_days)

        if blocks is not None:
            fc_raw = get_model_series(blocks, model_name, "hourly")  # always hourly for seam

        if fc_raw is not None and len(fc_raw):
            # Concat at hourly resolution so the boundary hour lands in the
            # same resampled bucket on both sides, then resample once.
            combined = pd.concat([history_raw, fc_raw])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined_r = _resample_for_granularity(combined, granularity)

            split_ts = _resample_for_granularity(fc_raw, granularity).index[0]
            hist_resampled = combined_r[combined_r.index < split_ts]
            fc_resampled   = combined_r[combined_r.index >= split_ts]
        else:
            hist_resampled = _resample_for_granularity(history_raw, granularity)

    # Drop incomplete trailing bucket from actuals
    if hist_resampled is not None:
        hist_resampled = _drop_incomplete_trailing_bucket(hist_resampled, granularity)

    # --- Actuals trace ---
    if hist_resampled is not None and len(hist_resampled):
        fig.add_trace(go.Scatter(
            x=hist_resampled.index, y=hist_resampled.values,
            mode="lines", name="Actuals",
            line=dict(color="#a5b4fc", width=2),
            fill="tozeroy", fillcolor="rgba(165,180,252,0.15)",
            hovertemplate=hover_fmt + "<br>%{y:,.0f} contacts<extra>Actuals</extra>",
        ))

    # --- Forecast trace ---
    if fc_resampled is not None and len(fc_resampled):
        fc_start_hourly = fc_raw.index[0] if fc_raw is not None else None
        in_overlap = (series is not None and fc_start_hourly is not None
                      and series.index.max() > fc_start_hourly)

        fc_label = MODEL_LABELS.get(model_name, model_name)
        fc_color = MODEL_COLORS.get(model_name, "#6366f1")

        # Prepend the last actual point so the line visually connects
        if hist_resampled is not None and len(hist_resampled):
            fc_x = [hist_resampled.index[-1]] + list(fc_resampled.index)
            fc_y = [float(hist_resampled.values[-1])] + list(fc_resampled.values)
        else:
            fc_x = list(fc_resampled.index)
            fc_y = list(fc_resampled.values)

        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            mode="lines", name=fc_label,
            line=dict(color=fc_color, width=2,
                      dash="dot" if in_overlap else "solid"),
            opacity=0.55 if in_overlap else 1.0,
            hovertemplate=hover_fmt + "<br>%{y:,.0f} contacts<extra>" + fc_label + "</extra>",
        ))

        # Forecast-start vline
        if meta and fc_resampled is not None:
            split_ts = fc_resampled.index[0]
            fig.add_vline(
                x=split_ts.timestamp() * 1000,
                line_width=1.5, line_dash="dash", line_color="#475569",
            )
            fig.add_annotation(
                x=split_ts, y=1.0, yref="paper", showarrow=False,
                text="Forecast start", font=dict(size=11, color="#64748b"),
                yanchor="bottom",
            )

    # --- Holiday ticks (hover-only) ---
    all_x = []
    if hist_resampled is not None:
        all_x += list(hist_resampled.index)
    if fc_raw is not None:
        all_x += list(fc_raw.index)
    if all_x:
        holidays_in_range = _holiday_ticks(pd.DatetimeIndex(all_x))
        for hdate, _ in holidays_in_range:
            fig.add_vline(x=pd.Timestamp(hdate).timestamp() * 1000,
                          line_width=1, line_color="#fbbf24", opacity=0.35)
        if holidays_in_range:
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(d) for d, _ in holidays_in_range],
                y=[0] * len(holidays_in_range),
                mode="markers",
                marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(width=0)),
                hovertemplate="%{customdata[0]}<br><span style='color:#fbbf24'>%{customdata[1]}</span><extra></extra>",
                customdata=[[d.strftime("%b %d, %Y"), n] for d, n in holidays_in_range],
                name="Holiday", showlegend=False,
            ))

    # Pin x-axis: left = start of actuals window, right = end of forecast
    x_start = None
    x_end   = None
    if series is not None:
        x_start = (series.index.max() - pd.Timedelta(days=trailing_days)).normalize()
    if fc_raw is not None and len(fc_raw):
        x_end = fc_raw.index[-1]
    elif series is not None:
        x_end = series.index.max()

    fig.update_layout(
        template="none",
        margin=dict(l=40, r=20, t=20, b=40),
        height=420,
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            showgrid=False, title=None, color="#64748b",
            tickfont=dict(color="#64748b"), linecolor="#1e293b",
            range=[x_start, x_end] if x_start and x_end else None,
        ),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", title="Contact Volume",
                   color="#64748b", tickfont=dict(color="#64748b")),
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#94a3b8"),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155",
                        font=dict(color="#e2e8f0")),
    )
    return pio.to_json(fig)


# ---------------------------------------------------------------------------
# Weekly bar chart — forecast bars with actuals overlay line
# ---------------------------------------------------------------------------

def build_weekly_chart(model_name: str) -> str:
    """
    Weekly volume chart:
      - Forecast bars (the published weekly block from the sheet).
      - Actuals as a line overlaid on top, extending into the forecast period
        as time passes and new data arrives (live validation).
    """
    def trailing_window(s, days):
        cutoff = s.index.max() - pd.Timedelta(days=days)
        return s.loc[s.index > cutoff]

    series = get_series()
    blocks = get_forecast_blocks()
    meta   = get_forecast_meta()

    now = pd.Timestamp.now()
    current_week_start = (now - pd.Timedelta(days=now.weekday())).normalize()

    fig = go.Figure()

    # --- Build weekly actuals first (needed for stacked bar calculation) ---
    weekly_actuals = None
    if series is not None:
        fc_start = blocks["weekly"].index[0] if (blocks is not None and len(blocks["weekly"])) else None
        lookback_days = max(90, (now - fc_start).days + 30) if fc_start is not None else 90
        history_raw = trailing_window(series, lookback_days)
        all_weekly = history_raw.resample("W-MON", label="left", closed="left").sum()
        # Only complete weeks (week bucket + 7 days <= now)
        weekly_actuals = all_weekly[all_weekly.index + pd.Timedelta(days=7) <= now]

    # --- Forecast bars ---
    # For a week that has already started (current_week_start), we render two
    # stacked segments: base = hours already logged, top = remaining forecast.
    # For all future complete weeks: single bar = full forecast value.
    if blocks is not None:
        fc_weekly = get_model_series(blocks, model_name, "weekly")
        if fc_weekly is not None and len(fc_weekly):
            fc_weekly = fc_weekly.copy()
            fc_label = MODEL_LABELS.get(model_name, model_name)
            fc_color = MODEL_COLORS.get(model_name, "#6366f1")

            first_week = fc_weekly.index[0]
            is_current_week = (first_week == current_week_start)

            if is_current_week and series is not None:
                # Hours already accumulated in the current partial week
                hours_so_far = float(series[series.index >= current_week_start].sum())
                remaining    = max(0.0, float(fc_weekly.iloc[0]) - hours_so_far)

                # Base segment — what's already happened this week
                fig.add_trace(go.Bar(
                    x=[first_week], y=[hours_so_far],
                    name="This week (actual so far)",
                    marker_color="#a5b4fc",
                    opacity=0.75,
                    hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} contacts so far<extra>Actual so far</extra>",
                ))
                # Top segment — remaining forecast
                fig.add_trace(go.Bar(
                    x=[first_week], y=[remaining],
                    name=fc_label + " (remaining)",
                    marker_color=fc_color,
                    opacity=0.75,
                    hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} contacts remaining<extra>Forecast remaining</extra>",
                ))
                # Future complete weeks
                future = fc_weekly.iloc[1:]
            else:
                future = fc_weekly

            if len(future):
                fig.add_trace(go.Bar(
                    x=future.index, y=future.values,
                    name=fc_label,
                    marker_color=fc_color,
                    opacity=0.75,
                    hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} contacts<extra>Forecast</extra>",
                ))

    # --- Actuals line — only complete weeks, no current partial week ---
    if weekly_actuals is not None and len(weekly_actuals):
        fig.add_trace(go.Scatter(
            x=weekly_actuals.index, y=weekly_actuals.values,
            mode="lines+markers", name="Actuals",
            line=dict(color="#a5b4fc", width=2),
            marker=dict(size=5, color="#a5b4fc"),
            hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} contacts<extra>Actuals</extra>",
        ))

    fig.update_layout(
        template="none",
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        barmode="stack",
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, title=None, color="#64748b",
                   tickfont=dict(color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", title="Weekly Total",
                   color="#64748b", tickfont=dict(color="#64748b")),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#94a3b8"),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155",
                        font=dict(color="#e2e8f0")),
    )
    return pio.to_json(fig)


# ---------------------------------------------------------------------------
# Model comparison (from registry — unchanged)
# ---------------------------------------------------------------------------

def _rmse_to_gradient_color(rank: int, total: int) -> str:
    if total <= 1:
        return "#22c55e"
    t = rank / (total - 1)
    r = int(round(0x22 + t * (0xef - 0x22)))
    g = int(round(0xc5 + t * (0x44 - 0xc5)))
    b = int(round(0x5e + t * (0x44 - 0x5e)))
    return f"#{r:02x}{g:02x}{b:02x}"


def model_comparison_data() -> list:
    meta = get_forecast_meta()
    metrics = (meta.get("metrics_by_model") or {}) if meta else {}

    ranked = sorted(
        [(n, m) for n, m in metrics.items() if m.get("rmse") is not None],
        key=lambda x: x[1]["rmse"],
    )
    rank_map = {n: i for i, (n, _) in enumerate(ranked)}
    total = len(ranked)

    rows = []
    for name in MODEL_NAMES:
        m = metrics.get(name)
        rows.append({
            "model_name":  name,
            "model_label": MODEL_LABELS.get(name, name),
            "mae":   m["mae"]   if m else None,
            "rmse":  m["rmse"]  if m else None,
            "smape": m["smape"] if m else None,
            "color": _rmse_to_gradient_color(rank_map[name], total)
                     if name in rank_map else "#334155",
        })
    return rows
