# FinSight AI

FinSight AI is a Flask-based financial analytics platform built as a final-year project. It combines secure user authentication with three real, database-backed analytics workflows:

- Credit-risk assessment for individual applicants and CSV batches
- Fraud detection on transaction datasets
- Portfolio optimization using historical asset price series

The project is designed to feel like a real-world analytics product rather than a static demo. Users can register, sign in, run analyses, review stored history, inspect detailed result pages, and download machine-readable JSON reports.

## Core Features

- Secure registration and login with password hashing
- CSRF protection and rate-limited authentication endpoints
- SQLite for local development and PostgreSQL support for deployment
- Real saved analysis history scoped to each user
- Exportable JSON reports for every completed run
- Automated tests for auth, route protection, and all three analysis modules

## Analytics Modules

### 1. Credit Risk

- Dataset: `german_credit_data.csv`
- Service: `services/credit_risk_service.py`
- Inputs:
  - Individual applicant form
  - Batch CSV upload with the German credit feature schema
- Output:
  - Predicted class
  - Risk band
  - Risk score
  - Probability estimates
  - Saved result detail page

### 2. Fraud Detection

- Dataset: `creditcard.csv`
- Service: `services/fraud_service.py`
- Input:
  - Batch CSV upload with fraud transaction feature columns
- Output:
  - Flagged transaction count
  - Average and maximum fraud risk
  - Preview of flagged rows
  - Saved result detail page

### 3. Portfolio Optimization

- Service: `services/portfolio_service.py`
- Input:
  - CSV where the first column is a date/index and remaining columns are asset price series
- Output:
  - Maximum-Sharpe portfolio allocation
  - Best return, volatility, and Sharpe ratio
  - Top allocation weights
  - Saved result detail page

## Project Structure

```text
website/
├─ app.py
├─ wsgi.py
├─ serve_waitress.py
├─ requirements.txt
├─ render.yaml
├─ finsight.db
├─ services/
│  ├─ credit_risk_service.py
│  ├─ fraud_service.py
│  └─ portfolio_service.py
├─ templates/
│  ├─ base.html
│  ├─ dashboard.html
│  ├─ credit_risk.html
│  ├─ fraud.html
│  ├─ portfolio.html
│  ├─ history.html
│  └─ analysis_detail.html
├─ static/
│  └─ styles.css
└─ tests/
   ├─ conftest.py
   └─ test_app.py
```

## Database Design

The application currently creates and uses these tables:

- `users`
- `analysis_runs`
- `credit_risk_results`
- `fraud_results`
- `portfolio_results`

`analysis_runs` is the shared backbone for the application. Each completed module writes a row to `analysis_runs`, then stores module-specific output in its own result table. This design makes the dashboard and history page easy to extend.

## Local Setup

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest tests -q
```

Current test coverage includes:

- registration and login
- protection of authenticated routes
- credit-risk analysis flow
- fraud analysis flow
- portfolio optimization flow
- user-scoped history behavior

## Environment Variables

Use `.env.example` as a reference.

- `FLASK_ENV`
- `FLASK_SECRET_KEY`
- `RATELIMIT_STORAGE_URI`
- `DATABASE_URL`

## Deployment

The project includes `render.yaml` for deployment on Render.

### Production notes

- Set a strong `FLASK_SECRET_KEY`
- Prefer PostgreSQL over SQLite in production
- Run behind Gunicorn or Waitress
- Serve over HTTPS

### Linux/macOS

```bash
python -m pip install -r requirements.txt
export FLASK_ENV=production
export FLASK_SECRET_KEY="your-long-random-secret"
gunicorn -w 2 -b 0.0.0.0:8000 wsgi:app
```

### Windows

```bash
python -m pip install -r requirements.txt
set FLASK_ENV=production
set FLASK_SECRET_KEY=your-long-random-secret
python serve_waitress.py
```

## Final-Year Project Positioning

This project demonstrates:

- full-stack web development with Flask
- secure authentication and protected workflows
- applied machine learning integration into a web product
- persistence of analytical outputs
- test-driven verification of important user flows
- deployment readiness for a real hosted demo

## Recommended Next Improvements

- Add PDF export in addition to JSON reports
- Add dashboard charts rendered from stored database results
- Add migrations with Flask-Migrate or Alembic
- Add password reset and email verification
- Expand the test suite with negative cases and export-route tests
