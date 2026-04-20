from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "creditcard.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifacts" / "fraud_pipeline.pkl"

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    for column in FEATURE_COLUMNS + ["Class"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=FEATURE_COLUMNS + ["Class"])


def _train_bundle() -> dict[str, Any]:
    df = _load_training_frame()
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0].sample(n=min(20000, int((df["Class"] == 0).sum())), random_state=42)
    training_frame = pd.concat([fraud, normal], ignore_index=True).sample(frac=1, random_state=42)

    X = training_frame[FEATURE_COLUMNS]
    y = training_frame["Class"].astype(int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X, y)

    bundle = {"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS}
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as artifact_file:
        pickle.dump(bundle, artifact_file)
    return bundle


def load_bundle() -> dict[str, Any]:
    if ARTIFACT_PATH.exists():
        with ARTIFACT_PATH.open("rb") as artifact_file:
            return pickle.load(artifact_file)
    return _train_bundle()


def _prepare_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "CSV format not recognised for fraud analysis. "
            "Please use the fraud transaction template columns."
        )

    frame = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("The uploaded fraud file contains blank or non-numeric transaction feature values.")
    return frame


def predict_batch(file_bytes: bytes) -> dict[str, Any]:
    bundle = load_bundle()
    frame = pd.read_csv(io.BytesIO(file_bytes))
    prepared = _prepare_frame(frame.to_dict(orient="records"))
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(prepared)[:, 1]

    results = prepared.copy()
    results["fraud_probability"] = probabilities
    results["predicted_label"] = results["fraud_probability"].apply(lambda value: "fraud" if value >= 0.5 else "legitimate")
    results["risk_score"] = (results["fraud_probability"] * 100).round(2)

    flagged = results[results["predicted_label"] == "fraud"].sort_values("risk_score", ascending=False)
    preview = flagged.head(20) if not flagged.empty else results.sort_values("risk_score", ascending=False).head(20)

    return {
        "rows_processed": int(len(results)),
        "flagged_count": int((results["predicted_label"] == "fraud").sum()),
        "clear_count": int((results["predicted_label"] == "legitimate").sum()),
        "average_risk_score": round(float(results["risk_score"].mean()), 2),
        "max_risk_score": round(float(results["risk_score"].max()), 2),
        "recommendation": "Review all flagged transactions before approval or settlement.",
        "top_rows": json.loads(preview.to_json(orient="records")),
    }
