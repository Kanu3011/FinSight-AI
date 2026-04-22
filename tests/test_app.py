from __future__ import annotations

import re
import app as app_module

from conftest import login, register
from services.credit_risk_service import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, get_form_options


def test_protected_route_redirects_to_signin(client):
    response = client.get("/dashboard", follow_redirects=False)

    assert response.status_code == 302
    assert "/signin" in response.headers["Location"]


def test_register_and_login_flow(client):
    response = register(client, "auth@example.com", name="Auth User")

    assert response.status_code == 200
    assert b"Dashboard" in response.data

    client.post("/logout", follow_redirects=True)
    response = login(client, "auth@example.com")

    assert response.status_code == 200
    assert b"Dashboard" in response.data


def test_credit_risk_individual_flow(auth_client):
    options = get_form_options()
    payload = {field: options[field][0] for field in CATEGORICAL_COLUMNS}
    payload.update({field: "1" for field in NUMERIC_COLUMNS})

    response = auth_client.post("/credit-risk/analyze/individual", data=payload, follow_redirects=True)

    assert response.status_code == 200
    assert b"Latest Credit-Risk Result" in response.data
    assert b"Risk Score" in response.data


def test_credit_risk_excel_batch_flow(auth_client, credit_xlsx):
    response = auth_client.post(
        "/credit-risk/analyze/batch",
        data={"dataset": (credit_xlsx, "credit_sample.xlsx")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Latest Credit-Risk Result" in response.data
    assert b"Rows Processed" in response.data


def test_fraud_batch_flow(auth_client, fraud_csv):
    response = auth_client.post(
        "/fraud/analyze/batch",
        data={"dataset": (fraud_csv, "fraud_sample.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Latest Fraud Result" in response.data
    assert b"Flagged" in response.data
    assert b"Open Full Detail" in response.data


def test_fraud_excel_flow(auth_client, fraud_xlsx):
    response = auth_client.post(
        "/fraud/analyze/batch",
        data={"dataset": (fraud_xlsx, "fraud_sample.xlsx")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Latest Fraud Result" in response.data
    assert b"Download Report" in response.data


def test_portfolio_batch_flow(auth_client, portfolio_csv):
    response = auth_client.post(
        "/portfolio/analyze/batch",
        data={
            "risk_free_rate": "0.02",
            "dataset": (portfolio_csv, "portfolio_sample.csv"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Latest Portfolio Result" in response.data
    assert b"Best Sharpe" in response.data


def test_portfolio_excel_flow(auth_client, portfolio_xlsx):
    response = auth_client.post(
        "/portfolio/analyze/batch",
        data={
            "risk_free_rate": "0.02",
            "dataset": (portfolio_xlsx, "portfolio_sample.xlsx"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Latest Portfolio Result" in response.data
    assert b"Download Report" in response.data


def test_history_is_scoped_to_current_user(client):
    register(client, "user1@example.com", name="User One")
    options = get_form_options()
    payload = {field: options[field][0] for field in CATEGORICAL_COLUMNS}
    payload.update({field: "1" for field in NUMERIC_COLUMNS})
    client.post("/credit-risk/analyze/individual", data=payload, follow_redirects=True)
    client.post("/logout", follow_redirects=True)

    register(client, "user2@example.com", name="User Two")
    response = client.get("/history", follow_redirects=True)

    assert response.status_code == 200
    assert b"individual assessment" not in response.data


def test_analysis_report_download(auth_client):
    options = get_form_options()
    payload = {field: options[field][0] for field in CATEGORICAL_COLUMNS}
    payload.update({field: "1" for field in NUMERIC_COLUMNS})

    response = auth_client.post("/credit-risk/analyze/individual", data=payload, follow_redirects=True)
    match = re.search(rb'href="(/analysis/\d+/report\.json)"', response.data)
    assert match is not None
    report_response = auth_client.get(match.group(1).decode("utf-8"), follow_redirects=False)

    assert report_response.status_code == 200
    assert report_response.mimetype == "application/json"
    assert b'"analysis_type": "credit_risk"' in report_response.data


def test_forgot_password_flow(client):
    register(client, "reset@example.com", password="Password123!")

    response = client.post(
        "/forgot-password",
        data={"email": "reset@example.com"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Reset Link" in response.data

    match = re.search(rb"/reset-password/([A-Za-z0-9_\-]+)", response.data)
    assert match is not None
    token = match.group(1).decode("utf-8")

    reset_response = client.post(
        f"/reset-password/{token}",
        data={"password": "NewPassword123!", "confirm_password": "NewPassword123!"},
        follow_redirects=True,
    )

    assert reset_response.status_code == 200
    assert b"Your password has been reset" in reset_response.data

    login_response = login(client, "reset@example.com", password="NewPassword123!")
    assert login_response.status_code == 200
    assert b"Dashboard" in login_response.data


def test_extract_result_payload_accepts_decoded_json():
    payload = app_module._extract_result_payload(
        {
            "analysis_type": "fraud",
            "result_json": None,
            "fraud_result_json": {"flagged_count": 2, "top_rows": []},
            "portfolio_result_json": None,
        }
    )

    assert payload["flagged_count"] == 2
