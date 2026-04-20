from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "german_credit_data.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifacts" / "credit_risk_pipeline.pkl"

FEATURE_COLUMNS = [
    "status_account",
    "month_duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "status_savings",
    "years_employment",
    "payment_to_income_ratio",
    "status_and_sex",
    "secondary_obligor",
    "residence_since",
    "collateral",
    "age",
    "other_installment_plans",
    "housing",
    "n_credits",
    "job",
    "n_guarantors",
    "telephone",
    "is_foreign_worker",
]

NUMERIC_COLUMNS = [
    "month_duration",
    "credit_amount",
    "payment_to_income_ratio",
    "residence_since",
    "age",
    "n_credits",
    "n_guarantors",
]

CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]

FIELD_LABELS = {
    "status_account": "Account Status",
    "month_duration": "Loan Duration (Months)",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "credit_amount": "Credit Amount",
    "status_savings": "Savings Status",
    "years_employment": "Employment Length",
    "payment_to_income_ratio": "Payment To Income Ratio",
    "status_and_sex": "Applicant Profile",
    "secondary_obligor": "Secondary Obligor",
    "residence_since": "Years At Current Address",
    "collateral": "Collateral",
    "age": "Age",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing",
    "n_credits": "Number Of Existing Credits",
    "job": "Job",
    "n_guarantors": "Number Of Dependants / Guarantors",
    "telephone": "Telephone",
    "is_foreign_worker": "Foreign Worker",
}

INDIVIDUAL_FORM_FIELDS = [
    "status_account",
    "month_duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "status_savings",
    "years_employment",
    "payment_to_income_ratio",
    "status_and_sex",
    "age",
    "housing",
]

INDIVIDUAL_DEFAULTS = {
    "secondary_obligor": "none",
    "residence_since": 2,
    "collateral": "none",
    "other_installment_plans": "none",
    "n_credits": 1,
    "job": "skilled employee/ official",
    "n_guarantors": 1,
    "telephone": "none",
    "is_foreign_worker": "yes",
}


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _train_bundle() -> dict[str, Any]:
    df = _load_training_frame()
    X = df[FEATURE_COLUMNS].copy()
    y = (df["target"].str.lower() == "bad").astype(int)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_options": {
            column: sorted(df[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_COLUMNS
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as artifact_file:
        pickle.dump(bundle, artifact_file)

    return bundle


def load_bundle() -> dict[str, Any]:
    if ARTIFACT_PATH.exists():
        with ARTIFACT_PATH.open("rb") as artifact_file:
            return pickle.load(artifact_file)
    return _train_bundle()


def get_form_options() -> dict[str, list[str]]:
    bundle = load_bundle()
    options = {key: list(value) for key, value in bundle["categorical_options"].items()}
    profile_options = options.get("status_and_sex", [])
    if "female : single" not in profile_options:
        profile_options.append("female : single")
        options["status_and_sex"] = sorted(profile_options)
    return options


def build_individual_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    payload = {field: (form_data.get(field) or "") for field in INDIVIDUAL_FORM_FIELDS}
    payload.update(INDIVIDUAL_DEFAULTS)
    return payload


def get_individual_field_config() -> list[dict[str, Any]]:
    options = get_form_options()
    field_config: list[dict[str, Any]] = []

    for field in INDIVIDUAL_FORM_FIELDS:
        field_entry: dict[str, Any] = {
            "name": field,
            "label": FIELD_LABELS.get(field, field.replace("_", " ").title()),
        }
        if field in NUMERIC_COLUMNS:
            field_entry["type"] = "number"
            field_entry["step"] = "any"
        else:
            field_entry["type"] = "select"
            field_entry["options"] = options[field]
        field_config.append(field_entry)

    return field_config


def _prepare_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    frame = frame[FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_bundle()
    frame = _prepare_frame([payload])
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(frame)[0]
    probability_bad = float(probabilities[1])
    probability_good = float(probabilities[0])
    risk_score = round(probability_bad * 100, 2)

    if risk_score >= 70:
        band = "High risk"
        recommendation = "Review manually before approval."
    elif risk_score >= 40:
        band = "Medium risk"
        recommendation = "Request supporting documents and affordability checks."
    else:
        band = "Low risk"
        recommendation = "Applicant appears suitable for standard review."

    return {
        "predicted_label": "bad" if probability_bad >= 0.5 else "good",
        "risk_band": band,
        "risk_score": risk_score,
        "probability_bad": round(probability_bad, 4),
        "probability_good": round(probability_good, 4),
        "recommendation": recommendation,
        "input_snapshot": {column: _json_ready(frame.iloc[0][column]) for column in FEATURE_COLUMNS},
    }


def predict_batch(file_bytes: bytes) -> dict[str, Any]:
    bundle = load_bundle()
    frame = pd.read_csv(io.BytesIO(file_bytes))
    prepared = _prepare_frame(frame.to_dict(orient="records"))
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(prepared)

    results = prepared.copy()
    results["probability_good"] = probabilities[:, 0]
    results["probability_bad"] = probabilities[:, 1]
    results["predicted_label"] = results["probability_bad"].apply(lambda value: "bad" if value >= 0.5 else "good")
    results["risk_score"] = (results["probability_bad"] * 100).round(2)

    flagged = results[results["predicted_label"] == "bad"].sort_values("risk_score", ascending=False)
    preview = flagged.head(10) if not flagged.empty else results.sort_values("risk_score", ascending=False).head(10)

    return {
        "rows_processed": int(len(results)),
        "high_risk_count": int((results["risk_score"] >= 70).sum()),
        "medium_risk_count": int(((results["risk_score"] >= 40) & (results["risk_score"] < 70)).sum()),
        "low_risk_count": int((results["risk_score"] < 40).sum()),
        "average_risk_score": round(float(results["risk_score"].mean()), 2),
        "predicted_bad_count": int((results["predicted_label"] == "bad").sum()),
        "predicted_good_count": int((results["predicted_label"] == "good").sum()),
        "top_rows": json.loads(preview.to_json(orient="records")),
    }
