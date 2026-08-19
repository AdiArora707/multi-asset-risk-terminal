import numpy as np
import pandas as pd
import pytest

from multi_asset_terminal import pipeline
from multi_asset_terminal.config import TerminalConfig
from multi_asset_terminal.reporting import export_analysis_artifacts


def test_synthetic_end_to_end_pipeline_and_export(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    index = pd.date_range("2019-01-02", periods=1_100, freq="B")
    tickers = ["AAA", "BBB", "CCC", "SPY"]
    innovations = rng.normal(
        loc=[0.0004, 0.0002, 0.0001, 0.00035],
        scale=[0.011, 0.007, 0.004, 0.010],
        size=(len(index), len(tickers)),
    )
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(innovations, axis=0)), index=index, columns=tickers
    )
    cpi_index = pd.date_range("2017-01-01", index.max(), freq="MS")

    def fake_prices(config, refresh=False):
        return prices

    def fake_fred(series_id, *args, **kwargs):
        if series_id == "DGS3MO":
            return pd.Series(4.0, index=index, name=series_id)
        return pd.Series(100 * 1.002 ** np.arange(len(cpi_index)), index=cpi_index, name=series_id)

    monkeypatch.setattr(pipeline, "download_adjusted_prices", fake_prices)
    monkeypatch.setattr(pipeline, "download_fred_series", fake_fred)
    config = TerminalConfig(
        assets={"Equity": "AAA", "Bond": "BBB", "Gold": "CCC"},
        weights={"Equity": 0.5, "Bond": 0.3, "Gold": 0.2},
        benchmark="SPY",
        start="2019-01-01",
        rolling_window=126,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "output"),
    )

    results = pipeline.run_analysis(config, build_figures=True)
    assert len(results.figures) == 10
    assert results.metrics.loc["Portfolio", "Observations"] == len(index) - 1
    assert (
        abs(
            results.return_contributions.sum()
            - results.metrics.loc["Portfolio", "Cumulative Return"]
        )
        < 1e-10
    )
    assert results.risk_contributions["Percent of Total Risk"].sum() == pytest.approx(1.0)

    paths = export_analysis_artifacts(results, include_quantstats=False)
    assert paths["tear_sheet"].exists()
    assert "Multi-Asset Performance and Risk Terminal" in paths["tear_sheet"].read_text(
        encoding="utf-8"
    )
    assert paths["performance_metrics"].exists()
