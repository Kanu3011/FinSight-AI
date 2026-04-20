from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence

import psycopg
from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from psycopg.errors import OperationalError, UniqueViolation
from psycopg.rows import dict_row
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from services.credit_risk_service import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FIELD_LABELS,
    NUMERIC_COLUMNS,
    build_individual_payload,
    get_form_options,
    get_individual_field_config,
    predict_batch,
    predict_one,
)
from services.fraud_service import FEATURE_COLUMNS as FRAUD_FEATURE_COLUMNS
from services.fraud_service import predict_batch as predict_fraud_batch
from services.portfolio_service import optimize_portfolio


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "finsight.db"
DATABASE_URL = os.environ.get("DATABASE_URL")


def _is_postgres() -> bool:
    return bool(DATABASE_URL) and DATABASE_URL.startswith(("postgres://", "postgresql://"))


def _connect_sqlite() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _sqlite_exec(query: str, params: Sequence[Any] = ()) -> None:
    connection = _connect_sqlite()
    try:
        connection.execute(query, params)
        connection.commit()
    finally:
        connection.close()


def _sqlite_fetchone(query: str, params: Sequence[Any] = ()):
    connection = _connect_sqlite()
    try:
        return connection.execute(query, params).fetchone()
    finally:
        connection.close()


def _sqlite_fetchall(query: str, params: Sequence[Any] = ()):
    connection = _connect_sqlite()
    try:
        return connection.execute(query, params).fetchall()
    finally:
        connection.close()


def _sqlite_insert_returning_id(query: str, params: Sequence[Any] = ()) -> int:
    connection = _connect_sqlite()
    try:
        cursor = connection.execute(query, params)
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def _pg_exec(query: str, params: Sequence[Any] = ()) -> None:
    assert DATABASE_URL
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)


def _pg_fetchone(query: str, params: Sequence[Any] = ()):
    assert DATABASE_URL
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()


def _pg_fetchall(query: str, params: Sequence[Any] = ()):
    assert DATABASE_URL
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()


def _pg_insert_returning_id(query: str, params: Sequence[Any] = ()) -> int:
    assert DATABASE_URL
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            if not row:
                raise RuntimeError("Expected RETURNING id")
            return int(row[0])


def db_exec(query: str, params: Sequence[Any] = ()) -> None:
    if _is_postgres():
        _pg_exec(query, params)
    else:
        _sqlite_exec(query, params)


def db_fetchone(query: str, params: Sequence[Any] = ()):
    if _is_postgres():
        return _pg_fetchone(query, params)
    return _sqlite_fetchone(query, params)


def db_fetchall(query: str, params: Sequence[Any] = ()):
    if _is_postgres():
        return _pg_fetchall(query, params)
    return _sqlite_fetchall(query, params)


def db_insert_returning_id(query: str, params: Sequence[Any] = ()) -> int:
    if _is_postgres():
        return _pg_insert_returning_id(query, params)
    return _sqlite_insert_returning_id(query, params)


def init_db() -> None:
    if _is_postgres():
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                analysis_type TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                original_filename TEXT,
                summary TEXT,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS credit_risk_results (
                id BIGSERIAL PRIMARY KEY,
                analysis_run_id BIGINT NOT NULL UNIQUE REFERENCES analysis_runs(id) ON DELETE CASCADE,
                predicted_label TEXT NOT NULL,
                risk_band TEXT NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                probability_good DOUBLE PRECISION NOT NULL,
                probability_bad DOUBLE PRECISION NOT NULL,
                recommendation TEXT NOT NULL,
                result_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS fraud_results (
                id BIGSERIAL PRIMARY KEY,
                analysis_run_id BIGINT NOT NULL UNIQUE REFERENCES analysis_runs(id) ON DELETE CASCADE,
                flagged_count INTEGER NOT NULL,
                clear_count INTEGER NOT NULL,
                average_risk_score DOUBLE PRECISION NOT NULL,
                max_risk_score DOUBLE PRECISION NOT NULL,
                recommendation TEXT NOT NULL,
                result_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS portfolio_results (
                id BIGSERIAL PRIMARY KEY,
                analysis_run_id BIGINT NOT NULL UNIQUE REFERENCES analysis_runs(id) ON DELETE CASCADE,
                simulations INTEGER NOT NULL,
                best_return DOUBLE PRECISION NOT NULL,
                best_volatility DOUBLE PRECISION NOT NULL,
                best_sharpe_ratio DOUBLE PRECISION NOT NULL,
                recommendation TEXT NOT NULL,
                result_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    else:
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                original_filename TEXT,
                summary TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS credit_risk_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL UNIQUE,
                predicted_label TEXT NOT NULL,
                risk_band TEXT NOT NULL,
                risk_score REAL NOT NULL,
                probability_good REAL NOT NULL,
                probability_bad REAL NOT NULL,
                recommendation TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS fraud_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL UNIQUE,
                flagged_count INTEGER NOT NULL,
                clear_count INTEGER NOT NULL,
                average_risk_score REAL NOT NULL,
                max_risk_score REAL NOT NULL,
                recommendation TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
            )
            """
        )
        db_exec(
            """
            CREATE TABLE IF NOT EXISTS portfolio_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL UNIQUE,
                simulations INTEGER NOT NULL,
                best_return REAL NOT NULL,
                best_volatility REAL NOT NULL,
                best_sharpe_ratio REAL NOT NULL,
                recommendation TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
            )
            """
        )


@dataclass
class User(UserMixin):
    id: int
    email: str
    full_name: str

    @staticmethod
    def get_by_id(user_id: str) -> Optional["User"]:
        if _is_postgres():
            row = db_fetchone(
                "SELECT id, email, full_name FROM users WHERE id = %s",
                (int(user_id),),
            )
        else:
            row = db_fetchone(
                "SELECT id, email, full_name FROM users WHERE id = ?",
                (int(user_id),),
            )

        if not row:
            return None
        return User(id=int(row["id"]), email=row["email"], full_name=row["full_name"])

    @staticmethod
    def get_by_email(email: str) -> Optional["User"]:
        normalized = email.strip().lower()
        if _is_postgres():
            row = db_fetchone(
                "SELECT id, email, full_name FROM users WHERE lower(email) = %s",
                (normalized,),
            )
        else:
            row = db_fetchone(
                "SELECT id, email, full_name FROM users WHERE lower(email) = ?",
                (normalized,),
            )

        if not row:
            return None
        return User(id=int(row["id"]), email=row["email"], full_name=row["full_name"])


def _create_credit_risk_run(
    user_id: int,
    input_mode: str,
    summary: str,
    result: dict[str, Any],
    original_filename: Optional[str] = None,
) -> int:
    payload = json.dumps(result)
    if _is_postgres():
        run_id = db_insert_returning_id(
            """
            INSERT INTO analysis_runs (user_id, analysis_type, input_mode, status, original_filename, summary, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (user_id, "credit_risk", input_mode, "completed", original_filename, summary),
        )
        db_exec(
            """
            INSERT INTO credit_risk_results (
                analysis_run_id, predicted_label, risk_band, risk_score,
                probability_good, probability_bad, recommendation, result_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                run_id,
                result["predicted_label"],
                result["risk_band"],
                result["risk_score"],
                result["probability_good"],
                result["probability_bad"],
                result["recommendation"],
                payload,
            ),
        )
        return run_id

    run_id = db_insert_returning_id(
        """
        INSERT INTO analysis_runs (user_id, analysis_type, input_mode, status, original_filename, summary, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (user_id, "credit_risk", input_mode, "completed", original_filename, summary),
    )
    db_exec(
        """
        INSERT INTO credit_risk_results (
            analysis_run_id, predicted_label, risk_band, risk_score,
            probability_good, probability_bad, recommendation, result_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            result["predicted_label"],
            result["risk_band"],
            result["risk_score"],
            result["probability_good"],
            result["probability_bad"],
            result["recommendation"],
            payload,
        ),
    )
    return run_id


def _create_fraud_run(
    user_id: int,
    summary: str,
    result: dict[str, Any],
    original_filename: Optional[str] = None,
) -> int:
    payload = json.dumps(result)
    if _is_postgres():
        run_id = db_insert_returning_id(
            """
            INSERT INTO analysis_runs (user_id, analysis_type, input_mode, status, original_filename, summary, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (user_id, "fraud", "batch", "completed", original_filename, summary),
        )
        db_exec(
            """
            INSERT INTO fraud_results (
                analysis_run_id, flagged_count, clear_count, average_risk_score,
                max_risk_score, recommendation, result_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                run_id,
                result["flagged_count"],
                result["clear_count"],
                result["average_risk_score"],
                result["max_risk_score"],
                result["recommendation"],
                payload,
            ),
        )
        return run_id

    run_id = db_insert_returning_id(
        """
        INSERT INTO analysis_runs (user_id, analysis_type, input_mode, status, original_filename, summary, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (user_id, "fraud", "batch", "completed", original_filename, summary),
    )
    db_exec(
        """
        INSERT INTO fraud_results (
            analysis_run_id, flagged_count, clear_count, average_risk_score,
            max_risk_score, recommendation, result_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            result["flagged_count"],
            result["clear_count"],
            result["average_risk_score"],
            result["max_risk_score"],
            result["recommendation"],
            payload,
        ),
    )
    return run_id


def _create_portfolio_run(
    user_id: int,
    summary: str,
    result: dict[str, Any],
    original_filename: Optional[str] = None,
) -> int:
    payload = json.dumps(result)
    if _is_postgres():
        run_id = db_insert_returning_id(
            """
            INSERT INTO analysis_runs (user_id, analysis_type, input_mode, status, original_filename, summary, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (user_id, "portfolio", "batch", "completed", original_filename, summary),
        )
        db_exec(
            """
            INSERT INTO portfolio_results (
                analysis_run_id, simulations, best_return, best_volatility,
                best_sharpe_ratio, recommendation, result_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                run_id,
                result["simulations"],
                result["best_return"],
                result["best_volatility"],
                result["best_sharpe_ratio"],
                result["recommendation"],
                payload,
            ),
        )
        return run_id

    run_id = db_insert_returning_id(
        """
        INSERT INTO analysis_runs (user_id, analysis_type, input_mode, status, original_filename, summary, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (user_id, "portfolio", "batch", "completed", original_filename, summary),
    )
    db_exec(
        """
        INSERT INTO portfolio_results (
            analysis_run_id, simulations, best_return, best_volatility,
            best_sharpe_ratio, recommendation, result_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            result["simulations"],
            result["best_return"],
            result["best_volatility"],
            result["best_sharpe_ratio"],
            result["recommendation"],
            payload,
        ),
    )
    return run_id


def _fetch_analysis_row(run_id: int, user_id: int):
    placeholder = "%s" if _is_postgres() else "?"
    return db_fetchone(
        f"""
        SELECT ar.id, ar.user_id, ar.analysis_type, ar.input_mode, ar.status, ar.original_filename,
               ar.summary, ar.created_at, cr.predicted_label, cr.risk_band, cr.risk_score,
               cr.probability_good, cr.probability_bad, cr.recommendation, cr.result_json,
               fr.flagged_count, fr.clear_count, fr.average_risk_score, fr.max_risk_score,
               fr.recommendation AS fraud_recommendation, fr.result_json AS fraud_result_json,
               pr.simulations, pr.best_return, pr.best_volatility, pr.best_sharpe_ratio,
               pr.recommendation AS portfolio_recommendation, pr.result_json AS portfolio_result_json
        FROM analysis_runs ar
        LEFT JOIN credit_risk_results cr ON cr.analysis_run_id = ar.id
        LEFT JOIN fraud_results fr ON fr.analysis_run_id = ar.id
        LEFT JOIN portfolio_results pr ON pr.analysis_run_id = ar.id
        WHERE ar.id = {placeholder} AND ar.user_id = {placeholder}
        """,
        (run_id, user_id),
    )


def _extract_result_payload(row: Any) -> dict[str, Any]:
    if row["analysis_type"] == "credit_risk":
        raw_payload = row["result_json"]
    elif row["analysis_type"] == "fraud":
        raw_payload = row["fraud_result_json"]
    else:
        raw_payload = row["portfolio_result_json"]
    return json.loads(raw_payload) if raw_payload else {}


def _build_report_payload(row: Any, result_payload: dict[str, Any]) -> dict[str, Any]:
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "id": row["id"],
            "analysis_type": row["analysis_type"],
            "input_mode": row["input_mode"],
            "status": row["status"],
            "summary": row["summary"],
            "created_at": row["created_at"],
            "original_filename": row["original_filename"],
        },
        "result": result_payload,
    }

    if row["analysis_type"] == "credit_risk":
        report["headline_metrics"] = {
            "predicted_label": row["predicted_label"],
            "risk_band": row["risk_band"],
            "risk_score": row["risk_score"],
            "probability_good": row["probability_good"],
            "probability_bad": row["probability_bad"],
            "recommendation": row["recommendation"],
        }
    elif row["analysis_type"] == "fraud":
        report["headline_metrics"] = {
            "flagged_count": row["flagged_count"],
            "clear_count": row["clear_count"],
            "average_risk_score": row["average_risk_score"],
            "max_risk_score": row["max_risk_score"],
            "recommendation": row["fraud_recommendation"],
        }
    else:
        report["headline_metrics"] = {
            "simulations": row["simulations"],
            "best_return": row["best_return"],
            "best_volatility": row["best_volatility"],
            "best_sharpe_ratio": row["best_sharpe_ratio"],
            "recommendation": row["portfolio_recommendation"],
        }

    return report


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
        REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )

    init_db()

    login_manager = LoginManager()
    login_manager.login_view = "signin"
    login_manager.login_message_category = "error"
    login_manager.init_app(app)

    CSRFProtect(app)

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    )

    @login_manager.user_loader
    def load_user(user_id: str) -> Optional[User]:
        return User.get_by_id(user_id)

    @app.get("/healthz")
    def healthz():
        try:
            if _is_postgres():
                db_fetchone("SELECT 1", ())
            else:
                db_fetchone("SELECT 1", ())
        except OperationalError:
            return {"ok": False, "db": "down"}, 503
        except Exception:
            return {"ok": False}, 503
        return {"ok": True}, 200

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/signin")
    def signin():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return render_template("signin.html")

    @app.post("/login")
    @limiter.limit("10/minute;100/day")
    def login():
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        remember = request.form.get("remember") == "on"

        if not email or not password:
            flash("Please enter your email and password.", "error")
            return redirect(url_for("signin"))

        user = User.get_by_email(email)
        if not user:
            flash("Invalid email or password.", "error")
            return redirect(url_for("signin"))

        if _is_postgres():
            row = db_fetchone(
                "SELECT id, email, full_name, password_hash FROM users WHERE lower(email) = %s",
                (email,),
            )
        else:
            row = db_fetchone(
                "SELECT id, email, full_name, password_hash FROM users WHERE lower(email) = ?",
                (email,),
            )

        if not row or not check_password_hash(row["password_hash"], password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("signin"))

        login_user(user, remember=remember)
        return redirect(url_for("dashboard"))

    @app.get("/register")
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return render_template("register.html")

    @app.post("/register")
    @limiter.limit("5/minute;20/day")
    def register_post():
        full_name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not full_name or not email or not password:
            flash("Please fill out all fields.", "error")
            return redirect(url_for("register"))

        if "@" not in email or "." not in email.split("@")[-1]:
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        try:
            if _is_postgres():
                user_id = db_insert_returning_id(
                    """
                    INSERT INTO users (full_name, email, password_hash)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (full_name, email, password_hash),
                )
            else:
                user_id = db_insert_returning_id(
                    """
                    INSERT INTO users (full_name, email, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (full_name, email, password_hash),
                )
        except sqlite3.IntegrityError:
            flash("That email is already registered. Please sign in.", "error")
            return redirect(url_for("signin"))
        except UniqueViolation:
            flash("That email is already registered. Please sign in.", "error")
            return redirect(url_for("signin"))
        except Exception:
            flash("We could not create your account right now. Please try again.", "error")
            return redirect(url_for("register"))

        login_user(User(id=user_id, email=email, full_name=full_name))
        flash("Account created successfully.", "success")
        return redirect(url_for("dashboard"))

    @app.post("/logout")
    @login_required
    def logout():
        logout_user()
        flash("You have been signed out.", "success")
        return redirect(url_for("home"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        user_id = int(current_user.id)
        placeholder = "%s" if _is_postgres() else "?"

        total_runs = db_fetchone(
            f"SELECT COUNT(*) AS count FROM analysis_runs WHERE user_id = {placeholder}",
            (user_id,),
        )["count"]
        completed_runs = db_fetchone(
            f"SELECT COUNT(*) AS count FROM analysis_runs WHERE user_id = {placeholder} AND status = 'completed'",
            (user_id,),
        )["count"]
        avg_risk_score = db_fetchone(
            f"""
            SELECT AVG(cr.risk_score) AS avg_score
            FROM credit_risk_results cr
            JOIN analysis_runs ar ON ar.id = cr.analysis_run_id
            WHERE ar.user_id = {placeholder}
            """,
            (user_id,),
        )["avg_score"]
        high_risk_runs = db_fetchone(
            f"""
            SELECT COUNT(*) AS count
            FROM credit_risk_results cr
            JOIN analysis_runs ar ON ar.id = cr.analysis_run_id
            WHERE ar.user_id = {placeholder} AND cr.risk_score >= 70
            """,
            (user_id,),
        )["count"]
        fraud_alerts = db_fetchone(
            f"""
            SELECT COALESCE(SUM(fr.flagged_count), 0) AS count
            FROM fraud_results fr
            JOIN analysis_runs ar ON ar.id = fr.analysis_run_id
            WHERE ar.user_id = {placeholder}
            """,
            (user_id,),
        )["count"]
        avg_sharpe = db_fetchone(
            f"""
            SELECT AVG(pr.best_sharpe_ratio) AS avg_sharpe
            FROM portfolio_results pr
            JOIN analysis_runs ar ON ar.id = pr.analysis_run_id
            WHERE ar.user_id = {placeholder}
            """,
            (user_id,),
        )["avg_sharpe"]
        module_breakdown = db_fetchall(
            f"""
            SELECT analysis_type, COUNT(*) AS count
            FROM analysis_runs
            WHERE user_id = {placeholder}
            GROUP BY analysis_type
            ORDER BY count DESC
            """,
            (user_id,),
        )
        recent_runs = db_fetchall(
            f"""
            SELECT ar.id, ar.analysis_type, ar.input_mode, ar.summary, ar.created_at,
                   cr.risk_band, cr.risk_score, fr.flagged_count, pr.best_sharpe_ratio
            FROM analysis_runs ar
            LEFT JOIN credit_risk_results cr ON cr.analysis_run_id = ar.id
            LEFT JOIN fraud_results fr ON fr.analysis_run_id = ar.id
            LEFT JOIN portfolio_results pr ON pr.analysis_run_id = ar.id
            WHERE ar.user_id = {placeholder}
            ORDER BY ar.created_at DESC
            LIMIT 5
            """,
            (user_id,),
        )

        return render_template(
            "dashboard.html",
            total_runs=total_runs,
            completed_runs=completed_runs,
            avg_risk_score=round(float(avg_risk_score), 2) if avg_risk_score is not None else None,
            high_risk_runs=high_risk_runs,
            fraud_alerts=fraud_alerts,
            avg_sharpe=round(float(avg_sharpe), 4) if avg_sharpe is not None else None,
            module_breakdown=module_breakdown,
            recent_runs=recent_runs,
        )

    @app.get("/credit-risk")
    @login_required
    def credit_risk():
        return render_template(
            "credit_risk.html",
            categorical_options=get_form_options(),
            numeric_fields=NUMERIC_COLUMNS,
            feature_columns=FEATURE_COLUMNS,
            categorical_fields=CATEGORICAL_COLUMNS,
            individual_fields=get_individual_field_config(),
            field_labels=FIELD_LABELS,
            result=None,
            run_id=None,
            result_mode=None,
        )

    @app.post("/credit-risk/analyze/individual")
    @login_required
    def credit_risk_individual():
        payload = {
            key: value.strip() if isinstance(value, str) else value
            for key, value in build_individual_payload(request.form.to_dict()).items()
        }
        required_fields = get_individual_field_config()
        missing_labels = [
            field["label"]
            for field in required_fields
            if not str(payload.get(field["name"], "")).strip()
        ]
        if missing_labels:
            flash("Please complete every field before running an analysis.", "error")
            return redirect(url_for("credit_risk"))

        try:
            result = predict_one(payload)
            summary = f"{result['risk_band']} individual assessment"
            run_id = _create_credit_risk_run(int(current_user.id), "individual", summary, result)
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("credit_risk"))
        except Exception:
            flash("The credit-risk analysis could not be completed right now.", "error")
            return redirect(url_for("credit_risk"))

        flash("Credit-risk analysis completed successfully.", "success")
        return render_template(
            "credit_risk.html",
            categorical_options=get_form_options(),
            numeric_fields=NUMERIC_COLUMNS,
            feature_columns=FEATURE_COLUMNS,
            categorical_fields=CATEGORICAL_COLUMNS,
            individual_fields=get_individual_field_config(),
            field_labels=FIELD_LABELS,
            result=result,
            run_id=run_id,
            result_mode="individual",
        )

    @app.post("/credit-risk/analyze/batch")
    @login_required
    def credit_risk_batch():
        uploaded_file = request.files.get("dataset")
        if not uploaded_file or not uploaded_file.filename:
            flash("Please choose a CSV file to analyze.", "error")
            return redirect(url_for("credit_risk"))

        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith(".csv"):
            flash("Only CSV files are supported for batch analysis.", "error")
            return redirect(url_for("credit_risk"))

        try:
            result = predict_batch(uploaded_file.read())
            summary = (
                f"Batch file analyzed: {result['rows_processed']} rows, "
                f"{result['predicted_bad_count']} predicted bad"
            )
            result.update(
                {
                    "predicted_label": "bad" if result["predicted_bad_count"] else "good",
                    "risk_band": "High risk"
                    if result["average_risk_score"] >= 70
                    else "Medium risk"
                    if result["average_risk_score"] >= 40
                    else "Low risk",
                    "risk_score": result["average_risk_score"],
                    "probability_good": round(result["predicted_good_count"] / max(result["rows_processed"], 1), 4),
                    "probability_bad": round(result["predicted_bad_count"] / max(result["rows_processed"], 1), 4),
                    "recommendation": "Review the flagged records shown below before approving any applications.",
                }
            )
            run_id = _create_credit_risk_run(
                int(current_user.id),
                "batch",
                summary,
                result,
                original_filename=filename,
            )
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("credit_risk"))
        except Exception:
            flash("We could not process that CSV file. Please verify the expected columns and try again.", "error")
            return redirect(url_for("credit_risk"))

        flash("Batch credit-risk analysis completed successfully.", "success")
        return render_template(
            "credit_risk.html",
            categorical_options=get_form_options(),
            numeric_fields=NUMERIC_COLUMNS,
            feature_columns=FEATURE_COLUMNS,
            categorical_fields=CATEGORICAL_COLUMNS,
            individual_fields=get_individual_field_config(),
            field_labels=FIELD_LABELS,
            result=result,
            run_id=run_id,
            result_mode="batch",
        )

    @app.get("/fraud")
    @login_required
    def fraud():
        return render_template(
            "fraud.html",
            fraud_feature_columns=FRAUD_FEATURE_COLUMNS,
            result=None,
            run_id=None,
        )

    @app.post("/fraud/analyze/batch")
    @login_required
    def fraud_batch():
        uploaded_file = request.files.get("dataset")
        if not uploaded_file or not uploaded_file.filename:
            flash("Please choose a CSV file to analyze.", "error")
            return redirect(url_for("fraud"))

        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith(".csv"):
            flash("Only CSV files are supported for fraud analysis.", "error")
            return redirect(url_for("fraud"))

        try:
            result = predict_fraud_batch(uploaded_file.read())
            summary = (
                f"Fraud scan completed: {result['rows_processed']} rows, "
                f"{result['flagged_count']} flagged"
            )
            run_id = _create_fraud_run(
                int(current_user.id),
                summary,
                result,
                original_filename=filename,
            )
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("fraud"))
        except Exception:
            flash("We could not process that fraud dataset. Check the expected schema and try again.", "error")
            return redirect(url_for("fraud"))

        flash("Fraud analysis completed successfully.", "success")
        return render_template(
            "fraud.html",
            fraud_feature_columns=FRAUD_FEATURE_COLUMNS,
            result=result,
            run_id=run_id,
        )

    @app.get("/portfolio")
    @login_required
    def portfolio():
        return render_template("portfolio.html", result=None, run_id=None)

    @app.post("/portfolio/analyze/batch")
    @login_required
    def portfolio_batch():
        uploaded_file = request.files.get("dataset")
        if not uploaded_file or not uploaded_file.filename:
            flash("Please choose a CSV file to optimize.", "error")
            return redirect(url_for("portfolio"))

        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith(".csv"):
            flash("Only CSV files are supported for portfolio optimization.", "error")
            return redirect(url_for("portfolio"))

        try:
            risk_free_rate = float(request.form.get("risk_free_rate", "0.02") or "0.02")
            result = optimize_portfolio(uploaded_file.read(), risk_free_rate=risk_free_rate)
            summary = (
                f"Portfolio optimized: {len(result['assets'])} assets, "
                f"Sharpe {result['best_sharpe_ratio']}"
            )
            run_id = _create_portfolio_run(
                int(current_user.id),
                summary,
                result,
                original_filename=filename,
            )
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("portfolio"))
        except Exception:
            flash("We could not optimize that portfolio file. Check the CSV format and try again.", "error")
            return redirect(url_for("portfolio"))

        flash("Portfolio optimization completed successfully.", "success")
        return render_template("portfolio.html", result=result, run_id=run_id)

    @app.get("/history")
    @login_required
    def history():
        placeholder = "%s" if _is_postgres() else "?"
        rows = db_fetchall(
            f"""
            SELECT ar.id, ar.analysis_type, ar.input_mode, ar.status, ar.original_filename, ar.summary,
                   ar.created_at, cr.risk_band, cr.risk_score, fr.flagged_count, fr.max_risk_score,
                   pr.best_sharpe_ratio, pr.best_return
            FROM analysis_runs ar
            LEFT JOIN credit_risk_results cr ON cr.analysis_run_id = ar.id
            LEFT JOIN fraud_results fr ON fr.analysis_run_id = ar.id
            LEFT JOIN portfolio_results pr ON pr.analysis_run_id = ar.id
            WHERE ar.user_id = {placeholder}
            ORDER BY ar.created_at DESC
            LIMIT 25
            """,
            (int(current_user.id),),
        )
        return render_template("history.html", runs=rows)

    @app.get("/analysis/<int:run_id>")
    @login_required
    def analysis_detail(run_id: int):
        row = _fetch_analysis_row(run_id, int(current_user.id))
        if not row:
            abort(404)

        try:
            result_payload = _extract_result_payload(row)
            return render_template("analysis_detail.html", run=row, result=result_payload)
        except Exception:
            flash("The analysis was saved, but the detail page could not be displayed right now. Please try again from history.", "error")
            return redirect(url_for("history"))

    @app.get("/analysis/<int:run_id>/report.json")
    @login_required
    def analysis_report_json(run_id: int):
        row = _fetch_analysis_row(run_id, int(current_user.id))
        if not row:
            abort(404)

        result_payload = _extract_result_payload(row)
        report_payload = _build_report_payload(row, result_payload)
        report_bytes = json.dumps(report_payload, indent=2).encode("utf-8")

        return send_file(
            BytesIO(report_bytes),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"analysis_{run_id}_report.json",
        )

    return app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
