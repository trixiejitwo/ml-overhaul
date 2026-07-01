"""
htmx fragment routes.  The app no longer runs forecasts inline; it reads the
latest published forecast from the forecast spreadsheet via services.py.
Controls are model (which column to read) and granularity (which block).
"""
import io

from flask import Blueprint, render_template, request, send_file

from app import services
from forecasting.models import MODEL_LABELS, MODEL_NAMES

fragments_bp = Blueprint("fragments", __name__)


def _parse_controls():
    model_name = request.values.get("model", "")
    if model_name not in MODEL_NAMES:
        model_name = services.best_active_model()

    granularity = request.values.get("granularity", "daily")
    if granularity not in ("hourly", "daily", "weekly"):
        granularity = "daily"

    return model_name, granularity


@fragments_bp.route("/status")
def status():
    st = services.get_data_status()
    return render_template("fragments/status_pill.html", status=st, oob_refresh=False)


@fragments_bp.route("/dashboard-data")
def dashboard_data():
    model_name, granularity = _parse_controls()

    kpis       = services.kpi_strip_data(model_name, granularity)
    hero_chart = services.build_hero_chart(model_name, granularity)
    weekly_chart = services.build_weekly_chart(model_name)
    comparison = services.model_comparison_data()

    return render_template(
        "fragments/dashboard_data.html",
        kpis=kpis,
        hero_chart=hero_chart,
        weekly_chart=weekly_chart,
        comparison=comparison,
        model_name=model_name,
        granularity=granularity,
        model_options=sorted_model_options(),
    )


@fragments_bp.route("/refresh-data", methods=["POST"])
def refresh_data():
    services.refresh_data()
    model_name, granularity = _parse_controls()
    st           = services.get_data_status()
    kpis         = services.kpi_strip_data(model_name, granularity)
    hero_chart   = services.build_hero_chart(model_name, granularity)
    weekly_chart = services.build_weekly_chart(model_name)
    comparison   = services.model_comparison_data()
    return render_template(
        "fragments/refresh_data.html",
        status=st,
        kpis=kpis,
        hero_chart=hero_chart,
        weekly_chart=weekly_chart,
        comparison=comparison,
        model_name=model_name,
        granularity=granularity,
        model_options=sorted_model_options(),
    )


def sorted_model_options():
    """Return MODEL_NAMES ordered best→worst by RMSE; unranked models at end."""
    active = services.model_comparison_data()  # already includes all models
    ranked   = [r for r in active if r["rmse"] is not None]
    unranked = [r for r in active if r["rmse"] is None]
    ranked.sort(key=lambda r: r["rmse"])
    ordered = ranked + unranked
    return [(r["model_name"], r["model_label"]) for r in ordered]


@fragments_bp.route("/export/csv")
def export_csv():
    """Download the current model's forecast series as CSV."""
    model_name, granularity = _parse_controls()
    fc = services.get_forecast_series(model_name, granularity)
    if fc is None:
        return "No forecast available", 404
    buf = io.StringIO()
    fc.to_csv(buf, header=["contacts"])
    filename = f"forecast_{model_name}_{granularity}.csv"
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )
