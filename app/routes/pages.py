"""Full-page routes."""
from flask import Blueprint, redirect, render_template, request, url_for

from app import services
from app.services import MODEL_LABELS, MODEL_NAMES

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return redirect(url_for("pages.insights"))


@pages_bp.route("/insights")
def insights():
    return render_template("insights.html")


@pages_bp.route("/forecast")
def dashboard():
    model_name = request.args.get("model", "")
    if model_name not in MODEL_NAMES:
        model_name = ""

    granularity = request.args.get("granularity", "daily")
    if granularity not in ("hourly", "daily", "weekly"):
        granularity = "daily"

    return render_template(
        "dashboard.html",
        model_name=model_name,
        granularity=granularity,
        model_options=[(name, MODEL_LABELS[name]) for name in MODEL_NAMES],
    )
