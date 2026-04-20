from __future__ import annotations

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
    assert b"Analysis Detail" in response.data
    assert b"Risk Score" in response.data


def test_fraud_batch_flow(auth_client, fraud_csv):
    response = auth_client.post(
        "/fraud/analyze/batch",
        data={"dataset": (fraud_csv, "fraud_sample.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Flagged Count" in response.data
    assert b"Flagged Transactions" in response.data


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
    assert b"Recommended Allocation" in response.data
    assert b"Best Sharpe" in response.data


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

    response = auth_client.post("/credit-risk/analyze/individual", data=payload, follow_redirects=False)
    report_response = auth_client.get(response.headers["Location"] + "/report.json", follow_redirects=False)

    assert report_response.status_code == 200
    assert report_response.mimetype == "application/json"
    assert b'"analysis_type": "credit_risk"' in report_response.data
