from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd


def _clean_frame(file_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(file_bytes))
    if frame.shape[1] < 2:
        raise ValueError("Portfolio upload needs at least two columns: a date/index column and one asset column.")

    numeric_frame = frame.copy()
    first_column = numeric_frame.columns[0]
    numeric_frame = numeric_frame.drop(columns=[first_column], errors="ignore")
    numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if numeric_frame.shape[1] < 2:
        raise ValueError("Please upload a CSV with at least two numeric asset series.")

    if (numeric_frame <= 0).any().any():
        raise ValueError("Portfolio optimization expects positive price values so returns can be calculated safely.")

    return numeric_frame


def optimize_portfolio(file_bytes: bytes, risk_free_rate: float = 0.02, simulations: int = 4000) -> dict[str, Any]:
    price_frame = _clean_frame(file_bytes)
    returns = price_frame.pct_change().dropna()

    if len(returns) < 2:
        raise ValueError("The uploaded file needs enough rows to calculate historical returns.")

    mean_returns = returns.mean() * 252
    covariance = returns.cov() * 252
    asset_names = mean_returns.index.tolist()
    n_assets = len(asset_names)

    np.random.seed(42)
    weights = np.random.random((simulations, n_assets))
    weights = weights / weights.sum(axis=1, keepdims=True)

    portfolio_returns = weights @ mean_returns.to_numpy()
    portfolio_volatility = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance.to_numpy(), weights))
    sharpe_ratios = np.divide(
        portfolio_returns - risk_free_rate,
        portfolio_volatility,
        out=np.zeros_like(portfolio_returns),
        where=portfolio_volatility != 0,
    )

    best_index = int(np.argmax(sharpe_ratios))
    low_vol_index = int(np.argmin(portfolio_volatility))

    best_weights = weights[best_index]
    low_vol_weights = weights[low_vol_index]

    def _allocation_map(weight_vector: np.ndarray) -> dict[str, float]:
        return {asset_names[i]: round(float(weight_vector[i]) * 100, 2) for i in range(n_assets)}

    best_allocation = _allocation_map(best_weights)
    low_vol_allocation = _allocation_map(low_vol_weights)

    top_assets = sorted(best_allocation.items(), key=lambda item: item[1], reverse=True)[:5]

    frontier_preview = [
        {
            "expected_return": round(float(portfolio_returns[i]) * 100, 2),
            "volatility": round(float(portfolio_volatility[i]) * 100, 2),
            "sharpe_ratio": round(float(sharpe_ratios[i]), 4),
        }
        for i in np.linspace(0, simulations - 1, num=min(40, simulations), dtype=int)
    ]

    return {
        "assets": asset_names,
        "rows_processed": int(len(price_frame)),
        "simulations": simulations,
        "best_return": round(float(portfolio_returns[best_index]) * 100, 2),
        "best_volatility": round(float(portfolio_volatility[best_index]) * 100, 2),
        "best_sharpe_ratio": round(float(sharpe_ratios[best_index]), 4),
        "min_volatility_return": round(float(portfolio_returns[low_vol_index]) * 100, 2),
        "min_volatility": round(float(portfolio_volatility[low_vol_index]) * 100, 2),
        "risk_free_rate": round(risk_free_rate * 100, 2),
        "recommended_allocation": best_allocation,
        "minimum_volatility_allocation": low_vol_allocation,
        "top_assets": [{"asset": asset, "weight": weight} for asset, weight in top_assets],
        "frontier_preview": frontier_preview,
        "recommendation": "Use the maximum-Sharpe allocation as the base case, then compare it against the minimum-volatility mix for a more defensive stance.",
    }
