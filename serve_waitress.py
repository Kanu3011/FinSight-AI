import os

from waitress import serve

from wsgi import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    serve(app, host="0.0.0.0", port=port)

