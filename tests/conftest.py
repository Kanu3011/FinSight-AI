from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(r"C:\Users\3011k\OneDrive\Desktop\website - Clean")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_finsight.db"
    monkeypatch.setenv("FINSIGHT_DB_PATH", str(test_db_path))
    monkeypatch.setenv("DATABASE_URL", "")

    import importlib
    import app as app_module

    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(app_module, "DATABASE_URL", None)

    flask_app = app_module.create_app()
    flask_app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
    )
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


def register(client, email: str, name: str = "Test User", password: str = "Password123!"):
    return client.post(
        "/register",
        data={"name": name, "email": email, "password": password},
        follow_redirects=True,
    )


def login(client, email: str, password: str = "Password123!"):
    return client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=True,
    )


@pytest.fixture()
def auth_client(client):
    register(client, "tester@example.com")
    return client


@pytest.fixture()
def fraud_csv():
    csv_text = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,"
        "V22,V23,V24,V25,V26,V27,V28,Amount\n"
        "0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,"
        "0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,"
        "-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,"
        "-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,"
        "-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,"
        "-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62\n"
    )
    return io.BytesIO(csv_text.encode("utf-8"))


@pytest.fixture()
def portfolio_csv():
    csv_text = (
        "Date,AssetA,AssetB,AssetC\n"
        "2024-01-01,100,120,90\n"
        "2024-01-02,101,121,91\n"
        "2024-01-03,103,119,93\n"
        "2024-01-04,104,122,94\n"
        "2024-01-05,106,124,96\n"
    )
    return io.BytesIO(csv_text.encode("utf-8"))
