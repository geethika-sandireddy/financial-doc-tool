"""Local dev entrypoint. Production deployments should run via gunicorn:
gunicorn "financial_doc_tool.api.app:create_app()" --bind 0.0.0.0:${PORT:-7860}
"""

from financial_doc_tool.api.app import create_app
from financial_doc_tool.config import settings

app = create_app()

if __name__ == "__main__":
    app.run(debug=settings.flask_debug)
