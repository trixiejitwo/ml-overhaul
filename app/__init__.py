"""Flask application factory."""
from flask import Flask

import config


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="../static", static_url_path="/static")
    app.secret_key = config.FLASK_SECRET_KEY

    from forecasting import registry
    registry.init_db()

    from app.routes.pages import pages_bp
    from app.routes.fragments import fragments_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(fragments_bp, url_prefix="/fragments")

    @app.errorhandler(Exception)
    def handle_error(err):
        app.logger.exception("Unhandled error")
        from flask import render_template, request
        message = "Something went wrong loading this. Please try refreshing the page."
        if request.headers.get("HX-Request"):
            return render_template("fragments/error.html", message=message), 200
        return render_template("error.html", message=message), 200

    return app
