import numpy as np
import pandas as pd
import pytest

from multi_asset_terminal.metrics import (
    calculate_metrics,
    drawdown_series,
    performance_table,
    rolling_statistics,
)


def test_drawdown_and_compounding_metrics_have_known_values():
    index = pd.date_range("2020-01-01", periods=4, freq="YS")
    returns = pd.Series([0.10, -0.20, 0.25, 0.10], index=index, name="Test")
    drawdowns = drawdown_series(returns)
    assert drawdowns.iloc[1] == pytest.approx(-0.20)
    metrics = calculate_metrics(returns, periods_per_year=1, confidence=0.75, best_n=1)
    assert metrics["Cumulative Return"] == pytest.approx(np.prod(1 + returns) - 1)
    assert metrics["Max Drawdown"] == pytest.approx(-0.20)


def test_performance_and_rolling_tables_are_column_consistent():
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    benchmark = pd.Series(rng.normal(0.0003, 0.01, len(index)), index=index, name="Benchmark")
    asset = 0.0001 + 0.8 * benchmark + pd.Series(rng.normal(0, 0.004, len(index)), index=index)
    returns = pd.DataFrame({"Portfolio": asset, "Benchmark": benchmark})
    table = performance_table(returns, 0.0, benchmark)
    rolling = rolling_statistics(returns, 0.0, benchmark, window=63)
    assert table.loc["Portfolio", "Beta"] == pytest.approx(0.8, abs=0.08)
    assert set(rolling) == {"volatility", "sharpe", "beta"}
    assert rolling["sharpe"]["Portfolio"].notna().sum() == len(index) - 62
