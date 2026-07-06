"""
Analytics data layer — reads historical sheets from the same Google Sheets
source the forecast ingestion uses. Cached in-process with the same TTL
pattern as services.py (background refresh after INGESTION_POLL_MINUTES).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from gspread_dataframe import get_as_dataframe

import config
from forecasting.ingestion import get_client

# ---------------------------------------------------------------------------
# Palette (dark surface — matched to existing app theme)
# ---------------------------------------------------------------------------
_CAT = [
    "#3987e5",  # blue
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
    "#d55181",  # magenta
]
_SURFACE = "#0f172a"
_GRID    = "#1e293b"
_MUTED   = "#64748b"
_INK     = "#e2e8f0"
_INK2    = "#94a3b8"

_LAYOUT_BASE = dict(
    template="none",
    paper_bgcolor=_SURFACE,
    plot_bgcolor=_SURFACE,
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=_INK2),
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color=_INK)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11, color=_INK2), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=48, r=16, t=28, b=40),
)

_L1_SHOW = [
    "Managing an order",
    "Interacting with hardware and accessories",
    "Using the Oura App and software",
    "Managing my relationship with Oura",
    "Preparing for purchase",
    "Activating a product",
    "[Merged ticket, no code]",
]
_L1_SHORT = {
    "Managing an order":                         "Orders",
    "Interacting with hardware and accessories":  "Hardware",
    "Using the Oura App and software":            "App / Software",
    "Managing my relationship with Oura":         "Relationship",
    "Preparing for purchase":                     "Pre-purchase",
    "Activating a product":                       "Activation",
    "[Merged ticket, no code]":                   "Other",
}


def _rgba(hex6: str, alpha: float) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Sheet readers
# ---------------------------------------------------------------------------

def _read_daily_l1(spreadsheet) -> pd.DataFrame:
    """Daily contacts grouped by L1 category."""
    ws  = spreadsheet.worksheet("Historical Contacts (Daily)")
    df  = get_as_dataframe(ws, evaluate_formulas=True, header=1).dropna(how="all")
    df  = df.iloc[:, :5].copy()
    df.columns = ["DAY", "L1", "L2", "L3", "VOLUME"]
    df  = df.dropna(subset=["DAY"])
    df["DAY"]    = pd.to_datetime(df["DAY"], errors="coerce")
    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce").fillna(0)
    df  = df.dropna(subset=["DAY"])

    daily_l1 = (df[df["L1"].isin(_L1_SHOW)]
                .groupby(["DAY", "L1"])["VOLUME"].sum()
                .unstack(fill_value=0)
                .sort_index())
    for cat in _L1_SHOW:
        if cat not in daily_l1.columns:
            daily_l1[cat] = 0
    return daily_l1[_L1_SHOW]


def _read_hourly(spreadsheet) -> pd.DataFrame:
    """Hourly contacts with L1 category."""
    ws  = spreadsheet.worksheet("Historical Contacts (Hourly)")
    df  = get_as_dataframe(ws, evaluate_formulas=True, header=1).dropna(how="all")
    df  = df.iloc[:, :4].copy()
    df.columns = ["DAY", "HOUR_OF_DAY", "L1", "VOLUME"]
    df  = df.dropna(subset=["DAY"])
    df["DAY"]        = pd.to_datetime(df["DAY"], errors="coerce")
    df["HOUR_OF_DAY"] = pd.to_numeric(df["HOUR_OF_DAY"], errors="coerce").fillna(0).astype(int)
    df["VOLUME"]     = pd.to_numeric(df["VOLUME"], errors="coerce").fillna(0)
    return df.dropna(subset=["DAY"])


def _read_customers(spreadsheet) -> pd.Series:
    ws  = spreadsheet.worksheet("Active Customers (with Active S")
    df  = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
    df  = df.iloc[:, :2].copy()
    df.columns = ["Date", "PayingUsers"]
    df["Date"]       = pd.to_datetime(df["Date"], errors="coerce")
    df["PayingUsers"] = pd.to_numeric(df["PayingUsers"], errors="coerce")
    df  = df.dropna()
    return df.set_index("Date").sort_index()["PayingUsers"]


def _read_sales(spreadsheet) -> dict:
    ws    = spreadsheet.worksheet("Historical Sales")
    raw   = get_as_dataframe(ws, evaluate_formulas=True, header=None).dropna(how="all")
    # Row 0 = dates, Row 2 = Actual/Forecast flag, Row 24 = M7 total sell-through
    dates = pd.to_datetime(raw.iloc[0, 4:], errors="coerce")
    flags = raw.iloc[2, 4:]
    vals  = pd.to_numeric(raw.iloc[24, 4:], errors="coerce")
    valid = dates.notna() & vals.notna()
    df    = pd.DataFrame({
        "date": dates[valid].values,
        "flag": flags[valid].values,
        "val":  vals[valid].values,
    }).sort_values("date")
    actual   = pd.Series(df[df["flag"] == "Actual"]["val"].values,
                         index=df[df["flag"] == "Actual"]["date"].values)
    forecast = pd.Series(df[df["flag"] == "Forecast"]["val"].values,
                         index=df[df["flag"] == "Forecast"]["date"].values)
    return {"actual": actual, "forecast": forecast}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_lock      = threading.Lock()
_cache: dict | None = None
_loaded_at: float | None = None
_TTL = config.INGESTION_POLL_MINUTES * 60


def _load() -> dict:
    client      = get_client()
    spreadsheet = client.open_by_url(config.SOURCE_URL)

    daily_l1  = _read_daily_l1(spreadsheet)
    hourly    = _read_hourly(spreadsheet)
    customers = _read_customers(spreadsheet)
    sales     = _read_sales(spreadsheet)

    daily_total = daily_l1.sum(axis=1)

    heatmap = (hourly.groupby([hourly["DAY"].dt.dayofweek, "HOUR_OF_DAY"])["VOLUME"]
               .mean().unstack(fill_value=0))

    return dict(
        daily_total=daily_total,
        daily_l1=daily_l1,
        customers=customers,
        sales=sales,
        heatmap=heatmap,
        hourly=hourly,
    )


def _get() -> dict:
    global _cache, _loaded_at
    with _lock:
        loaded = _loaded_at
    if loaded is None:
        with _lock:
            if _cache is None:
                _cache     = _load()
                _loaded_at = time.monotonic()
    elif time.monotonic() - loaded > _TTL:
        t = threading.Thread(target=_refresh, daemon=True)
        t.start()
    with _lock:
        return _cache


def _refresh():
    global _cache, _loaded_at
    try:
        data = _load()
        with _lock:
            _cache     = data
            _loaded_at = time.monotonic()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

def kpi_data() -> dict:
    d = _get()
    daily_total = d["daily_total"]
    customers   = d["customers"]
    sales       = d["sales"]

    latest_year = daily_total.index.max().year
    ytd = int(daily_total[daily_total.index.year == latest_year].sum())

    now        = daily_total.index.max()
    week_end   = (now - pd.Timedelta(days=now.weekday() + 1)).normalize()
    week_start = week_end - pd.Timedelta(days=6)
    prev_end   = week_start - pd.Timedelta(days=1)
    prev_start = prev_end  - pd.Timedelta(days=6)
    last_wk    = float(daily_total[week_start:week_end].sum())
    prior_wk   = float(daily_total[prev_start:prev_end].sum())
    wow_pct    = (last_wk - prior_wk) / prior_wk * 100 if prior_wk else None

    latest_users = int(customers.iloc[-1])
    users_prev   = int(customers.iloc[-5]) if len(customers) >= 5 else None
    users_pct    = ((latest_users - users_prev) / users_prev * 100) if users_prev else None

    contact_rate = round(last_wk / latest_users * 1000, 2) if latest_users else None
    latest_sales = int(sales["actual"].iloc[-1]) if len(sales["actual"]) else None

    return dict(
        ytd_contacts=ytd,
        weekly_contacts=round(last_wk),
        wow_pct=round(wow_pct, 1) if wow_pct is not None else None,
        paying_users=latest_users,
        users_pct=round(users_pct, 1) if users_pct is not None else None,
        contact_rate=contact_rate,
        weekly_sales=latest_sales,
        data_through=daily_total.index.max().strftime("%b %d, %Y"),
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _axis_x(**kw):
    return dict(showgrid=False, color=_MUTED, tickfont=dict(color=_MUTED),
                linecolor=_GRID, **kw)


def _axis_y(**kw):
    return dict(showgrid=True, gridcolor=_GRID, color=_MUTED,
                tickfont=dict(color=_MUTED), zeroline=False, **kw)


def build_stacked_area(trailing_days: int = 90) -> str:
    d  = _get()
    df = d["daily_l1"]
    cutoff = df.index.max() - pd.Timedelta(days=trailing_days)
    df = df[df.index > cutoff]

    fig = go.Figure()
    for i, cat in enumerate(_L1_SHOW):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[cat],
            name=_L1_SHORT[cat],
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5, color=_CAT[i]),
            fillcolor=_rgba(_CAT[i], 0.75),
            hovertemplate="%{x|%b %d}<br>%{y:,.0f} contacts<extra>" + _L1_SHORT[cat] + "</extra>",
        ))
    fig.update_layout(**_LAYOUT_BASE, height=340,
                      xaxis=_axis_x(title=None),
                      yaxis=_axis_y(title="Daily contacts"),
                      hovermode="x unified")
    return pio.to_json(fig)


def build_paying_users() -> str:
    s = _get()["customers"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color=_CAT[0], width=2),
        fill="tozeroy", fillcolor=_rgba(_CAT[0], 0.13),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} paying users<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_LAYOUT_BASE, height=260,
                      xaxis=_axis_x(title=None),
                      yaxis=_axis_y(title="Paying users"),
                      hovermode="x unified")
    return pio.to_json(fig)


def build_weekly_sales() -> str:
    sales    = _get()["sales"]
    actual   = sales["actual"]
    forecast = sales["forecast"]

    fig = go.Figure()

    # Faded forecast band — vrect equivalent via a filled Scatter behind the bars
    if len(forecast):
        fc_start = forecast.index[0]
        fc_end   = forecast.index[-1] + pd.Timedelta(days=7)
        # Use a shape for the background band
        fig.add_vrect(
            x0=fc_start, x1=fc_end,
            fillcolor=_rgba(_CAT[1], 0.08),
            line_width=0,
            layer="below",
        )
        # Dashed divider line at forecast start
        fig.add_vline(
            x=fc_start.timestamp() * 1000,
            line_width=1, line_dash="dash", line_color=_MUTED,
        )
        fig.add_annotation(
            x=fc_start, y=1.0, yref="paper", showarrow=False,
            text="Forecast", font=dict(size=10, color=_MUTED),
            xanchor="left", yanchor="bottom", xshift=4,
        )

    # Actual bars
    if len(actual):
        fig.add_trace(go.Bar(
            x=actual.index, y=actual.values,
            name="Actual",
            marker_color=_CAT[1], opacity=0.85,
            hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} rings<extra>Actual</extra>",
        ))

    # Forecast bars — faded
    if len(forecast):
        fig.add_trace(go.Bar(
            x=forecast.index, y=forecast.values,
            name="Forecast",
            marker_color=_CAT[1], opacity=0.35,
            hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} rings<extra>Forecast</extra>",
        ))

    fig.update_layout(**_LAYOUT_BASE, height=260, bargap=0.25,
                      xaxis=_axis_x(title=None),
                      yaxis=_axis_y(title="Rings sold"),
                      hovermode="x unified")
    return pio.to_json(fig)


def build_category_mix(trailing_days: int = 30) -> str:
    d      = _get()
    df     = d["daily_l1"]
    cutoff = df.index.max() - pd.Timedelta(days=trailing_days)
    recent = df[df.index > cutoff].sum()
    total  = recent.sum()
    pct    = (recent / total * 100).sort_values()
    labels = [_L1_SHORT[c] for c in pct.index]
    colors = [_CAT[_L1_SHOW.index(c)] for c in pct.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pct.values, y=labels, orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"{v:.1f}%" for v in pct.values],
        textposition="outside",
        textfont=dict(color=_INK2, size=11),
        hovertemplate="%{y}<br>%{x:.1f}% of contacts<extra></extra>",
        showlegend=False,
    ))
    layout = {**_LAYOUT_BASE, "margin": dict(l=120, r=70, t=28, b=56)}
    fig.update_layout(**layout, height=300,
                      xaxis=_axis_x(title="% of total contacts", ticksuffix="%",
                                    title_standoff=12,
                                    titlefont=dict(color=_MUTED, size=11)),
                      yaxis=dict(showgrid=False, color=_MUTED, tickfont=dict(color=_INK2),
                                 automargin=True),
                      hovermode="y unified")
    return pio.to_json(fig)


def build_heatmap() -> str:
    hm   = _get()["heatmap"].reindex(range(7)).fillna(0)
    days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [f"{h:02d}:00" for h in range(24)]

    fig = go.Figure(go.Heatmap(
        z=hm.values.tolist(), x=hours, y=days,
        colorscale=[[0, "#dbeafe"], [0.15, "#93c5fd"], [0.4, "#3b82f6"],
                    [0.7, "#1d4ed8"], [1.0, "#1e3a8a"]],
        hoverongaps=False,
        hovertemplate="%{y} %{x}<br>avg %{z:,.0f} contacts<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=12, len=0.9,
                      tickfont=dict(color=_MUTED, size=10), outlinewidth=0),
    ))
    layout = {**_LAYOUT_BASE, "margin": dict(l=48, r=60, t=28, b=40)}
    fig.update_layout(**layout, height=260,
                      xaxis=dict(showgrid=False, tickangle=-45,
                                 tickfont=dict(color=_MUTED, size=10),
                                 color=_MUTED, side="bottom"),
                      yaxis=dict(showgrid=False, tickfont=dict(color=_INK2),
                                 color=_MUTED, autorange="reversed"))
    return pio.to_json(fig)


def build_contact_rate() -> str:
    d           = _get()
    daily_total = d["daily_total"]
    customers   = d["customers"]

    weekly   = daily_total.resample("W-MON", label="left", closed="left").sum()
    cust_wkly = customers.resample("W-MON", label="left", closed="left").last().ffill()
    common   = weekly.index.intersection(cust_wkly.index)
    rate     = (weekly[common] / cust_wkly[common] * 1000).dropna()
    rate     = rate[rate.index >= "2025-01-01"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rate.index, y=rate.values, mode="lines+markers",
        line=dict(color=_CAT[4], width=2),
        marker=dict(size=5, color=_CAT[4]),
        fill="tozeroy", fillcolor=_rgba(_CAT[4], 0.13),
        hovertemplate="Week of %{x|%b %d}<br>%{y:.2f} contacts / 1k users<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_LAYOUT_BASE, height=260,
                      xaxis=_axis_x(title=None),
                      yaxis=_axis_y(title="Contacts / 1k users"),
                      hovermode="x unified")
    return pio.to_json(fig)
