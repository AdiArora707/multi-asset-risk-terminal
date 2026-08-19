import numpy as np
import pandas as pd
import pytest

from multi_asset_terminal.returns import construct_portfolio, log_returns, simple_returns


def test_return_transformations_do_not_implicitly_fill():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"A": [100.0, 101.0, np.nan, 103.0]}, index=index)
    discrete = simple_returns(prices)
    continuous = log_returns(prices.fillna(102.0))
    assert discrete["A"].isna().sum() == 2
    assert continuous.iloc[0, 0] == pytest.approx(np.log(101 / 100))


def test_daily_contributions_reconcile_and_weights_drift():
    index = pd.date_range("2024-01-02", periods=5, freq="B")
    returns = pd.DataFrame({"A": [0.01] * 5, "B": [0.0] * 5}, index=index)
    path = construct_portfolio(returns, {"A": 0.5, "B": 0.5}, rebalance_frequency=None)
    pd.testing.assert_series_equal(path.contributions.sum(axis=1), path.returns, check_names=False)
    assert path.weights.iloc[-1]["A"] > 0.5
    assert path.cash_weight.abs().max() == pytest.approx(0.0)


def test_leveraged_portfolio_pays_financing_rate():
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    returns = pd.DataFrame({"A": [0.01] * 3}, index=index)
    path = construct_portfolio(
        returns,
        {"A": 1.5},
        risk_free_daily=pd.Series(0.001, index=index),
        rebalance_frequency="D",
    )
    assert path.returns.iloc[0] == pytest.approx(1.5 * 0.01 - 0.5 * 0.001)
    assert path.contributions.iloc[0]["Financing"] == pytest.approx(-0.0005)
