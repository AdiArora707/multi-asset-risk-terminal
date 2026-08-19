import numpy as np
import pandas as pd
import pytest

from multi_asset_terminal.attribution import (
    component_risk_contributions,
    geometric_linked_contributions,
)


def test_linked_contributions_sum_to_compounded_return():
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    portfolio = pd.Series([0.01, -0.02, 0.015, 0.005], index=index)
    contributions = pd.DataFrame({"A": portfolio * 0.6, "B": portfolio * 0.4}, index=index)
    linked = geometric_linked_contributions(portfolio, contributions)
    assert linked.sum() == pytest.approx((1 + portfolio).prod() - 1, abs=1e-12)


def test_euler_component_risk_sums_to_portfolio_volatility():
    rng = np.random.default_rng(12)
    index = pd.date_range("2022-01-01", periods=500, freq="B")
    returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, len(index)), "B": rng.normal(0, 0.006, len(index))},
        index=index,
    )
    weights = {"A": 0.65, "B": 0.35}
    result = component_risk_contributions(returns, weights)
    vector = pd.Series(weights)
    expected_volatility = np.sqrt(vector @ (returns.cov() * 252) @ vector)
    assert result["Component Risk"].sum() == pytest.approx(expected_volatility)
    assert result["Percent of Total Risk"].sum() == pytest.approx(1.0)
