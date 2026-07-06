"""
Analytics data layer — reads the local Excel file and builds Plotly figures
for the Insights dashboard. All heavy parsing is done once and cached in
module-level state (re-read on app restart only; the file doesn't change live).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

EXCEL_PATH = Path(__file__).parent.parent / "Oura Complete Data (2).xlsx"

# ---------------------------------------------------------------------------
# Palette (dark surface #0f172a — matched to the existing app theme)
# ---------------------------------------------------------------------------
_CAT = [
    "#3987e5",  # blue        — Managing an order
    "#199e70",  # aqua        — Interacting with hardware
    "#c98500",  # yellow      — Using the Oura App
    "#008300",  # green       — Managing my relationship
    "#9085e9",  # violet      — Preparing for purchase
    "#e66767",  # red         — Activating a product
    "#d55181",  # magenta     — Other / merged / archived
]


def _rgba(hex6: str, alpha: float) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
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

# L1 categories to show (drop archived/merged in charts)
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
    "Managing an order":                          "Orders",
    "Interacting with hardware and accessories":  "Hardware",
    "Using the Oura App and software":            "App / Software",
    "Managing my relationship with Oura":         "Relationship",
    "Preparing for purchase":                     "Pre-purchase",
    "Activating a product":                       "Activation",
    "[Merged ticket, no code]":                   "Other",
}

# ---------------------------------------------------------------------------
# Raw data (loaded once)
# ---------------------------------------------------------------------------
_lock   = threading.Lock()
_cache: dict | None = None


def _load() -> dict:
    xl = pd.ExcelFile(EXCEL_PATH)

    # --- Hourly contacts ---
    hourly = xl.parse("Historical Contacts (Hourly)", header=1)
    hourly.columns = ["DAY", "HOUR_OF_DAY", "L1", "VOLUME"]
    hourly = hourly.dropna(subset=["DAY"]).copy()
    hourly["DAY"]    = pd.to_datetime(hourly["DAY"])
    hourly["VOLUME"] = pd.to_numeric(hourly["VOLUME"], errors="coerce").fillna(0)

    # --- Daily contacts ---
    daily = xl.parse("Historical Contacts (Daily)", header=1)
    daily.columns = ["DAY", "L1", "L2", "L3", "VOLUME"]
    daily = daily.dropna(subset=["DAY"]).copy()
    daily["DAY"]    = pd.to_datetime(daily["DAY"])
    daily["VOLUME"] = pd.to_numeric(daily["VOLUME"], errors="coerce").fillna(0)

    # Aggregate daily total per day
    daily_total = daily.groupby("DAY")["VOLUME"].sum().sort_index()

    # Daily by L1 (keep only named categories)
    daily_l1 = (daily[daily["L1"].isin(_L1_SHOW)]
                .groupby(["DAY", "L1"])["VOLUME"].sum()
                .unstack(fill_value=0)
                .sort_index())
    # Ensure all _L1_SHOW columns present
    for cat in _L1_SHOW:
        if cat not in daily_l1.columns:
            daily_l1[cat] = 0
    daily_l1 = daily_l1[_L1_SHOW]

    # --- Active customers (weekly) ---
    customers = xl.parse("Active Customers (with Active S")
    customers.columns = ["Date", "PayingUsers"]
    customers["Date"]       = pd.to_datetime(customers["Date"])
    customers["PayingUsers"] = pd.to_numeric(customers["PayingUsers"], errors="coerce")
    customers = customers.dropna().set_index("Date").sort_index()

    # --- Historical sales (weekly ring sell-through) ---
    raw_sales = xl.parse("Historical Sales", header=None)
    dates_row = raw_sales.iloc[1, 4:]
    total_row = raw_sales.iloc[7, 4:]
    flag_row  = raw_sales.iloc[3, 4:]
    valid     = pd.to_datetime(dates_row, errors="coerce").notna()
    sales_dates  = pd.to_datetime(dates_row[valid])
    sales_vals   = pd.to_numeric(total_row[valid], errors="coerce")
    sales_flags  = flag_row[valid]
    sales_all    = pd.Series(sales_vals.values, index=sales_dates.values)
    sales_actual = sales_all[sales_flags.values == "Actual"].dropna()
    # Drop the last partial week if suspiciously small (< 20% of median)
    if len(sales_actual) > 4:
        med = sales_actual.median()
        if sales_actual.iloc[-1] < 0.2 * med:
            sales_actual = sales_actual.iloc[:-1]

    # --- Heatmap: avg contacts by (day-of-week, hour) ---
    heatmap = (hourly.groupby([hourly["DAY"].dt.dayofweek, "HOUR_OF_DAY"])["VOLUME"]
               .mean().unstack(fill_value=0))

    return dict(
        daily_total=daily_total,
        daily_l1=daily_l1,
        customers=customers,
        sales_actual=sales_actual,
        heatmap=heatmap,
        hourly=hourly,
        daily=daily,
    )


def _get() -> dict:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load()
        return _cache


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

def kpi_data() -> dict:
    d = _get()
    daily_total = d["daily_total"]
    customers   = d["customers"]
    sales       = d["sales_actual"]

    # YTD contacts (calendar year of latest data point)
    latest_year = daily_total.index.max().year
    ytd = int(daily_total[daily_total.index.year == latest_year].sum())

    # WoW contact change (last full week vs prior week)
    now = daily_total.index.max()
    week_end   = (now - pd.Timedelta(days=now.weekday() + 1)).normalize()
    week_start = week_end - pd.Timedelta(days=6)
    prev_end   = week_start - pd.Timedelta(days=1)
    prev_start = prev_end  - pd.Timedelta(days=6)
    last_wk  = float(daily_total[week_start:week_end].sum())
    prior_wk = float(daily_total[prev_start:prev_end].sum())
    wow_pct  = (last_wk - prior_wk) / prior_wk * 100 if prior_wk else None

    # Latest paying users
    latest_users = int(customers["PayingUsers"].iloc[-1])
    users_prev   = int(customers["PayingUsers"].iloc[-5]) if len(customers) >= 5 else None
    users_pct    = ((latest_users - users_prev) / users_prev * 100) if users_prev else None

    # Contact rate per 1k users (last full week)
    contact_rate = round(last_wk / latest_users * 1000, 2) if latest_users else None

    # Latest weekly sales
    latest_sales = int(sales.iloc[-1]) if len(sales) else None

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
    """Daily stacked area — contact volume by L1 category."""
    d = _get()
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

    fig.update_layout(
        **_LAYOUT_BASE,
        height=340,
        xaxis=_axis_x(title=None),
        yaxis=_axis_y(title="Daily contacts"),
        hovermode="x unified",
    )
    return pio.to_json(fig)


def build_paying_users() -> str:
    """Paying users over time — line chart."""
    d = _get()
    s = d["customers"]["PayingUsers"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values,
        mode="lines",
        line=dict(color=_CAT[0], width=2),
        fill="tozeroy", fillcolor=_rgba(_CAT[0], 0.13),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} paying users<extra></extra>",
        name="Paying users",
        showlegend=False,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        xaxis=_axis_x(title=None),
        yaxis=_axis_y(title="Paying users"),
        hovermode="x unified",
    )
    return pio.to_json(fig)


def build_weekly_sales() -> str:
    """Weekly ring sell-through — bar chart."""
    d = _get()
    s = d["sales_actual"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=s.index, y=s.values,
        marker_color=_CAT[1],
        opacity=0.85,
        hovertemplate="Week of %{x|%b %d}<br>%{y:,.0f} rings<extra>Sell-through</extra>",
        name="Weekly sales",
        showlegend=False,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        bargap=0.25,
        xaxis=_axis_x(title=None),
        yaxis=_axis_y(title="Rings sold"),
        hovermode="x unified",
    )
    return pio.to_json(fig)


def build_category_mix(trailing_days: int = 30) -> str:
    """Horizontal bar — % share by L1 for the last N days."""
    d = _get()
    df = d["daily_l1"]
    cutoff = df.index.max() - pd.Timedelta(days=trailing_days)
    recent = df[df.index > cutoff].sum()
    total  = recent.sum()
    pct    = (recent / total * 100).sort_values()

    labels = [_L1_SHORT[c] for c in pct.index]
    colors = [_CAT[_L1_SHOW.index(c)] for c in pct.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pct.values,
        y=labels,
        orientation="h",
        marker_color=colors,
        opacity=0.85,
        text=[f"{v:.1f}%" for v in pct.values],
        textposition="outside",
        textfont=dict(color=_INK2, size=11),
        hovertemplate="%{y}<br>%{x:.1f}% of contacts<extra></extra>",
        showlegend=False,
    ))
    layout = {**_LAYOUT_BASE, "margin": dict(l=110, r=60, t=28, b=24)}
    fig.update_layout(
        **layout,
        height=280,
        xaxis=_axis_x(title="% of total contacts", ticksuffix="%"),
        yaxis=dict(showgrid=False, color=_MUTED, tickfont=dict(color=_INK2)),
        hovermode="y unified",
    )
    return pio.to_json(fig)


def build_heatmap() -> str:
    """Average contact volume by hour of day × day of week."""
    d = _get()
    hm = d["heatmap"].reindex(range(7)).fillna(0)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [f"{h:02d}:00" for h in range(24)]

    z = hm.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=hours,
        y=days,
        colorscale=[[0, _SURFACE], [0.001, "#1e3a5f"], [0.4, "#2a78d6"],
                    [0.7, "#3987e5"], [1.0, "#86b6ef"]],
        hoverongaps=False,
        hovertemplate="%{y} %{x}<br>avg %{z:,.0f} contacts<extra></extra>",
        showscale=True,
        colorbar=dict(
            thickness=12, len=0.9,
            tickfont=dict(color=_MUTED, size=10),
            outlinewidth=0,
        ),
    ))
    layout = {**_LAYOUT_BASE, "margin": dict(l=48, r=60, t=28, b=40)}
    fig.update_layout(
        **layout,
        height=260,
        xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(color=_MUTED, size=10),
                   color=_MUTED, side="bottom"),
        yaxis=dict(showgrid=False, tickfont=dict(color=_INK2), color=_MUTED,
                   autorange="reversed"),
    )
    return pio.to_json(fig)


def build_contact_rate() -> str:
    """Weekly contact rate (contacts per 1k paying users)."""
    d = _get()
    daily_total = d["daily_total"]
    customers   = d["customers"]["PayingUsers"]

    weekly = daily_total.resample("W-MON", label="left", closed="left").sum()
    # Align to closest customer snapshot (weekly)
    cust_weekly = customers.resample("W-MON", label="left", closed="left").last().ffill()

    common = weekly.index.intersection(cust_weekly.index)
    rate   = (weekly[common] / cust_weekly[common] * 1000).dropna()
    rate   = rate[rate.index >= "2025-01-01"]  # only since contact data starts

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rate.index, y=rate.values,
        mode="lines+markers",
        line=dict(color=_CAT[4], width=2),
        marker=dict(size=5, color=_CAT[4]),
        fill="tozeroy", fillcolor=_rgba(_CAT[4], 0.13),
        hovertemplate="Week of %{x|%b %d}<br>%{y:.2f} contacts / 1k users<extra></extra>",
        name="Contact rate",
        showlegend=False,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        xaxis=_axis_x(title=None),
        yaxis=_axis_y(title="Contacts / 1k users"),
        hovermode="x unified",
    )
    return pio.to_json(fig)
