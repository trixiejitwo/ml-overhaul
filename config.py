"""
Centralized, env-driven configuration. Mirrors the original notebook's
Section 2 configuration cell, but sourced from environment variables /
a .env file instead of hardcoded constants, so the same settings can be
shared by the Flask app and the training notebook without duplication.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# --- Credentials ---
# Resolved against the project root, not the process's working directory, so
# it works the same whether launched as `python app.py` from the project
# root or from a Jupyter kernel rooted in notebooks/.
SERVICE_ACCOUNT_FILE = str(BASE_DIR / os.environ.get("SERVICE_ACCOUNT_FILE", "service_account.json"))

# --- Data source ---
SOURCE_URL = os.environ.get("SOURCE_URL", "")
SOURCE_SHEET_NAME = os.environ.get("SOURCE_SHEET_NAME", "Historical Contacts (Hourly)")
DATE_COL = os.environ.get("DATE_COL", "DAY")
HOUR_COL = os.environ.get("HOUR_COL", "HOUR_OF_DAY")
TARGET_COL = os.environ.get("TARGET_COL", "TOTAL_VOLUME")

# --- Export destination ---
DEST_URL = os.environ.get("DEST_URL", "")

# --- Modeling ---
TEST_SIZE = int(os.environ.get("TEST_SIZE_HOURS", 24 * 30))          # 30-day holdout, hours
SEASONAL_PERIOD = int(os.environ.get("SEASONAL_PERIOD_HOURS", 24 * 7))  # weekly seasonality, hours
DEFAULT_HORIZON_WEEKS = int(os.environ.get("DEFAULT_HORIZON_WEEKS", 12))
FORECAST_HORIZON_HOURS = int(os.environ.get("FORECAST_HORIZON_HOURS", 24 * 7 * 13))  # ~3 months

# --- Polling intervals ---
FORECAST_POLL_MINUTES = int(os.environ.get("FORECAST_POLL_MINUTES", 15))
INGESTION_POLL_MINUTES = int(os.environ.get("INGESTION_POLL_MINUTES", 15))

# --- Google Sheets scopes ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# --- Storage ---
REGISTRY_DB_PATH = str(BASE_DIR / os.environ.get("REGISTRY_DB_PATH", "data/registry.db"))
CACHE_DIR = str(BASE_DIR / os.environ.get("CACHE_DIR", "data/cache"))

# --- Flask ---
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-key-not-for-production")
FLASK_DEBUG = _bool(os.environ.get("FLASK_DEBUG"), default=False)
PORT = int(os.environ.get("PORT", 5000))
