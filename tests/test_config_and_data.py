import numpy as np
import pandas as pd
import pytest

from multi_asset_terminal.config import TerminalConfig
from multi_asset_terminal.data import align_macro_features, align_prices


def test_config_accepts_explicit_leverage_but_rejects_label_mismatch():
    config = TerminalConfig(assets={"Equity": "SPY"}, weights={"Equity": 1.5})
    assert config.gross_exposure == 1.5
    assert config.net_exposure == 1.5
    with pytest.raises(ValueError, match="identical labels"):
        TerminalConfig(assets={"Equity": "SPY"}, weights={"Bond": 1.0})


def test_align_prices_does_not_forward_fill_missing_observations():
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, np.nan, 102.0], "B": [50.0, 51.0, 52.0]}, index=index)
    aligned = align_prices(prices, ["A", "B"], mode="intersection")
    assert list(aligned.index) == [index[0], index[2]]
    assert len(aligned) == 2


def test_macro_alignment_converts_annual_yield_and_trailing_cpi():
    daily_index = pd.date_range("2023-01-02", "2024-12-31", freq="B")
    yield_series = pd.Series([5.0, 4.0], index=[daily_index[0], pd.Timestamp("2024-01-02")])
    cpi_index = pd.date_range("2022-01-01", periods=36, freq="MS")
    cpi = pd.Series(100 * 1.002 ** np.arange(36), index=cpi_index)
    macro = align_macro_features(daily_index, yield_series, cpi)
    expected = (1.05 ** (1 / 252)) - 1
    assert macro.iloc[0]["daily_risk_free_rate"] == pytest.approx(expected)
    assert macro.iloc[-1]["inflation_yoy"] > 0
