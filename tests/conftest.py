from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from services.credit_risk_service import FEATURE_COLUMNS, get_form_options

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
def credit_xlsx():
    options = get_form_options()
    row = {}
    for column in FEATURE_COLUMNS:
        if column in options:
            row[column] = options[column][0]
        else:
            row[column] = 1
    frame = pd.DataFrame([row])
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture()
def fraud_xlsx():
    frame = pd.DataFrame(
        [
            {
                "Time": 0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Amount": 149.62,
            }
        ]
    )
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


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


@pytest.fixture()
def portfolio_xlsx():
    frame = pd.DataFrame(
        [
            {"Date": "2024-01-01", "AssetA": 100, "AssetB": 120, "AssetC": 90},
            {"Date": "2024-01-02", "AssetA": 101, "AssetB": 121, "AssetC": 91},
            {"Date": "2024-01-03", "AssetA": 103, "AssetB": 119, "AssetC": 93},
            {"Date": "2024-01-04", "AssetA": 104, "AssetB": 122, "AssetC": 94},
            {"Date": "2024-01-05", "AssetA": 106, "AssetB": 124, "AssetC": 96},
        ]
    )
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer
