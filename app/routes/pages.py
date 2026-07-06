"""Full-page route: dashboard shell rendered instantly, real data loaded via htmx."""
from flask import Blueprint, render_template, request

from app import services, analytics
from app.services import MODEL_LABELS, MODEL_NAMES

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/insights")
def insights():
    kpis        = analytics.kpi_data()
    stacked     = analytics.build_stacked_area()
    users_chart = analytics.build_paying_users()
    sales_chart = analytics.build_weekly_sales()
    mix_chart   = analytics.build_category_mix()
    heatmap     = analytics.build_heatmap()
    rate_chart  = analytics.build_contact_rate()
    return render_template(
        "insights.html",
        kpis=kpis,
        stacked=stacked,
        users_chart=users_chart,
        sales_chart=sales_chart,
        mix_chart=mix_chart,
        heatmap=heatmap,
        rate_chart=rate_chart,
    )


@pages_bp.route("/")
def dashboard():
    model_name = request.args.get("model", "")
    if model_name not in MODEL_NAMES:
        model_name = ""  # resolved after first data load

    granularity = request.args.get("granularity", "daily")
    if granularity not in ("hourly", "daily", "weekly"):
        granularity = "daily"

    return render_template(
        "dashboard.html",
        model_name=model_name,
        granularity=granularity,
        model_options=[(name, MODEL_LABELS[name]) for name in MODEL_NAMES],
    )
