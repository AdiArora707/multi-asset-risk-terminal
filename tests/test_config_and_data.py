import numpy as np
import pandas as pd
import pytest
import requests

from multi_asset_terminal.config import TerminalConfig
from multi_asset_terminal.data import (
    DataDownloadError,
    align_macro_features,
    align_prices,
    download_fred_series,
)


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


def test_fred_rejects_invalid_api_key_without_exposing_it(tmp_path):
    with pytest.raises(DataDownloadError, match="32 lowercase alphanumeric"):
        download_fred_series(
            "DGS3MO",
            "2024-01-01",
            api_key="not-a-valid-key",
            cache_dir=tmp_path,
        )


def test_fred_api_failure_uses_public_csv_without_leaking_key(monkeypatch, tmp_path):
    secret = "a" * 32

    class CsvResponse:
        text = "DATE,DGS3MO\n2024-01-02,5.40\n2024-01-03,5.38\n"

        @staticmethod
        def raise_for_status():
            return None

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.RequestException(f"request failed with api_key={secret}")
            return CsvResponse()

        @staticmethod
        def close():
            return None

    monkeypatch.setattr("multi_asset_terminal.data._requests_session", FakeSession)
    series = download_fred_series(
        "DGS3MO",
        "2024-01-01",
        api_key=secret,
        cache_dir=tmp_path,
        refresh=True,
    )
    assert series.iloc[0] == pytest.approx(5.40)
    assert secret not in series.to_string()
