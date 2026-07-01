"""Entrypoint: `python app.py` runs the dashboard with no build step."""
import base64
import json
import os

# On Railway (and any cloud host), the service account JSON is injected as a
# base64-encoded environment variable rather than a file on disk.
# Decode it back to a file at startup so the rest of the code is unchanged.
_creds_b64 = os.environ.get("GOOGLE_CREDENTIALS_B64")
if _creds_b64:
    _creds_path = os.path.join(os.path.dirname(__file__), "service_account.json")
    with open(_creds_path, "w") as _f:
        _f.write(base64.b64decode(_creds_b64).decode("utf-8"))

import config
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.FLASK_DEBUG)
