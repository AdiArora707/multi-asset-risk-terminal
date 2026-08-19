"""Return linking, component risk, leverage, beta, and concentration diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .metrics import calculate_metrics


def geometric_linked_contributions(
    portfolio_returns: pd.Series,
    daily_contributions: pd.DataFrame,
) -> pd.Series:
    """Carino-link daily arithmetic contributions to total compounded return.

    The linked asset contributions reconcile exactly to the portfolio's total
    return while retaining the day-by-day economic contribution information.
    """

    aligned = daily_contributions.reindex(portfolio_returns.index).fillna(0.0)
    returns = portfolio_returns.astype(float)
    total_return = float((1.0 + returns).prod() - 1.0)
    period_k = pd.Series(
        np.where(
            np.isclose(returns, 0.0),
            1.0,
            np.log1p(returns) / returns,
        ),
        index=returns.index,
    )
    total_k = np.log1p(total_return) / total_return if not np.isclose(total_return, 0.0) else 1.0
    linked = aligned.mul(period_k, axis=0).sum(axis=0).div(total_k)
    residual = total_return - float(linked.sum())
    if abs(residual) > 1e-10:
        linked.iloc[-1] += residual
    linked.name = "Linked Return Contribution"
    return linked.sort_values(ascending=False)


def component_risk_contributions(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Euler-decompose portfolio volatility into asset risk contributions."""

    weight_vector = pd.Series(weights, dtype=float).reindex(asset_returns.columns).fillna(0.0)
    covariance = asset_returns.cov() * periods_per_year
    portfolio_variance = float(weight_vector @ covariance @ weight_vector)
    if portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive for risk attribution.")
    portfolio_volatility = np.sqrt(portfolio_variance)
    marginal = covariance @ weight_vector / portfolio_volatility
    component = weight_vector * marginal
    percent = component / portfolio_volatility
    return pd.DataFrame(
        {
            "Weight": weight_vector,
            "Marginal Risk": marginal,
            "Component Risk": component,
            "Percent of Total Risk": percent,
        }
    ).sort_values("Percent of Total Risk", ascending=False)


def exposure_diagnostics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_daily: pd.Series,
    weights: Mapping[str, float],
    periods_per_year: int = 252,
) -> pd.Series:
    """Summarize whether results reflect leverage, beta, alpha, or rare days."""

    stats_row = calculate_metrics(
        portfolio_returns,
        risk_free_daily=risk_free_daily,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
    )
    gross = float(sum(abs(value) for value in weights.values()))
    net = float(sum(weights.values()))
    beta_return = float(stats_row["Beta"]) * float(
        (benchmark_returns - risk_free_daily.reindex(benchmark_returns.index).fillna(0.0)).mean()
        * periods_per_year
    )
    return pd.Series(
        {
            "Gross Exposure": gross,
            "Net Risky Exposure": net,
            "Financing Weight at Rebalance": 1.0 - net,
            "Leveraged": gross > 1.0 + 1e-10,
            "Portfolio Beta": stats_row["Beta"],
            "Jensen Alpha": stats_row["Jensen Alpha"],
            "CAPM R-squared": stats_row["R-squared"],
            "Approx. Annual Beta Return": beta_return,
            "Sharpe Ratio": stats_row["Sharpe Ratio"],
            "CAGR": stats_row["CAGR"],
            "CAGR Without Best 10 Days": stats_row["CAGR Without Best 10 Days"],
            "Best 10 Days / Positive Returns": stats_row["Best 10 Days / Positive Returns"],
        },
        name="Exposure Diagnostic",
    )
