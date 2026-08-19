"""Return transformations and a financing-aware portfolio simulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate discrete returns without pandas' implicit forward fill."""

    returns = prices.sort_index().pct_change(fill_method=None)
    return returns.replace([np.inf, -np.inf], np.nan).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate continuously compounded returns from strictly positive prices."""

    if (prices <= 0).any().any():
        raise ValueError("Log returns require strictly positive prices.")
    returns = np.log(prices).diff()
    return returns.replace([np.inf, -np.inf], np.nan).iloc[1:]


@dataclass(frozen=True)
class PortfolioPath:
    """Daily output of the self-financing portfolio simulation."""

    returns: pd.Series
    contributions: pd.DataFrame
    weights: pd.DataFrame
    cash_weight: pd.Series


def construct_portfolio(
    asset_returns: pd.DataFrame,
    target_weights: Mapping[str, float],
    risk_free_daily: pd.Series | float = 0.0,
    rebalance_frequency: str | None = "M",
) -> PortfolioPath:
    """Simulate a portfolio with drifting weights and explicit financing.

    At each rebalance the risky assets are reset to ``target_weights``. Residual
    capital is invested at the risk-free rate; a negative residual is borrowing.
    Between rebalances every asset weight drifts with realized performance.
    """

    weights = pd.Series(target_weights, dtype=float)
    missing = weights.index.difference(asset_returns.columns)
    if not missing.empty:
        raise ValueError(f"Returns are missing configured assets: {missing.tolist()}")
    returns = asset_returns.loc[:, weights.index].dropna(how="any").sort_index()
    if returns.empty:
        raise ValueError("No complete return rows are available for portfolio construction.")

    if isinstance(risk_free_daily, pd.Series):
        rf = risk_free_daily.reindex(returns.index).ffill().fillna(0.0).astype(float)
    else:
        rf = pd.Series(float(risk_free_daily), index=returns.index)

    period_labels = (
        returns.index.to_period(rebalance_frequency)
        if rebalance_frequency
        else pd.Index([0] * len(returns))
    )
    risky_weights = weights.copy()
    cash_weight = 1.0 - float(weights.sum())
    previous_period: object | None = None
    portfolio_values: list[float] = []
    cash_weights: list[float] = []
    contribution_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []

    for position, (timestamp, row) in enumerate(returns.iterrows()):
        period = period_labels[position]
        if previous_period is not None and period != previous_period:
            risky_weights = weights.copy()
            cash_weight = 1.0 - float(weights.sum())

        weight_rows.append(risky_weights.copy())
        cash_weights.append(cash_weight)
        asset_contribution = risky_weights * row
        financing_contribution = cash_weight * rf.loc[timestamp]
        portfolio_return = float(asset_contribution.sum() + financing_contribution)
        if not np.isfinite(portfolio_return) or portfolio_return <= -1.0:
            raise ValueError(
                f"Portfolio lost 100% or produced a non-finite return on {timestamp.date()}; "
                "check leverage and input returns."
            )
        contribution_rows.append(
            pd.concat([asset_contribution, pd.Series({"Financing": financing_contribution})])
        )
        portfolio_values.append(portfolio_return)

        denominator = 1.0 + portfolio_return
        risky_weights = risky_weights.mul(1.0 + row).div(denominator)
        cash_weight = cash_weight * (1.0 + rf.loc[timestamp]) / denominator
        previous_period = period

    portfolio = pd.Series(portfolio_values, index=returns.index, name="Portfolio")
    contributions = pd.DataFrame(contribution_rows, index=returns.index)
    weight_history = pd.DataFrame(weight_rows, index=returns.index)
    cash_history = pd.Series(cash_weights, index=returns.index, name="Financing")
    if not np.allclose(contributions.sum(axis=1), portfolio, atol=1e-12):
        raise RuntimeError("Portfolio contribution reconciliation failed.")
    return PortfolioPath(portfolio, contributions, weight_history, cash_history)
