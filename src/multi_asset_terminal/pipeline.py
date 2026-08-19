"""End-to-end orchestration for market data, analytics, and visual outputs."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .attribution import (
    component_risk_contributions,
    exposure_diagnostics,
    geometric_linked_contributions,
)
from .config import TerminalConfig
from .data import (
    DataDownloadError,
    align_macro_features,
    align_prices,
    download_adjusted_prices,
    download_fred_series,
)
from .metrics import (
    calendar_year_metrics,
    performance_table,
    regression_table,
    rolling_statistics,
)
from .returns import PortfolioPath, construct_portfolio, log_returns, simple_returns
from .visualizations import build_figure_set

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResults:
    """All tabular and visual artifacts from one terminal run."""

    config: TerminalConfig
    prices: pd.DataFrame
    simple_returns: pd.DataFrame
    log_returns: pd.DataFrame
    macro: pd.DataFrame
    portfolio_path: PortfolioPath
    evaluation_returns: pd.DataFrame
    metrics: pd.DataFrame
    rolling: dict[str, pd.DataFrame]
    calendar_years: pd.DataFrame
    regressions: pd.DataFrame
    return_contributions: pd.Series
    risk_contributions: pd.DataFrame
    diagnostics: pd.Series
    figures: dict[str, go.Figure] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _fallback_macro_series(
    config: TerminalConfig, price_index: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    annual_yield = pd.Series(
        config.fallback_annual_risk_free_rate * 100.0,
        index=price_index,
        name=config.risk_free_series,
    )
    # An all-NaN CPI series explicitly marks inflation unavailable rather than fabricated.
    monthly_index = pd.date_range(price_index.min(), price_index.max(), freq="MS")
    cpi = pd.Series(np.nan, index=monthly_index, name=config.inflation_series)
    return annual_yield, cpi


def run_analysis(
    config: TerminalConfig | None = None,
    *,
    refresh: bool = False,
    fred_api_key: str | None = None,
    build_figures: bool = True,
) -> AnalysisResults:
    """Execute the complete reproducible research pipeline.

    Parameters
    ----------
    config:
        Validated asset universe and analytics assumptions.
    refresh:
        Ignore on-disk data caches and call the remote sources again.
    fred_api_key:
        Optional FRED API key. If omitted, ``FRED_API_KEY`` is read from the
        environment; if still absent, the official FRED CSV endpoint is used.
    build_figures:
        Set false for batch calculations or unit tests that do not need Plotly.
    """

    config = config or TerminalConfig()
    run_warnings: list[str] = []
    prices_raw = download_adjusted_prices(config, refresh=refresh)
    prices = align_prices(prices_raw, config.all_tickers, mode=config.calendar_mode)
    if config.calendar_mode == "union":
        complete = prices.dropna(how="any")
        if len(complete) < config.periods_per_year:
            raise ValueError(
                "Union-calendar data do not contain one full year of complete cross-asset rows."
            )
        prices = complete

    prices = prices.rename(columns=config.ticker_to_label)
    label_order = [*config.assets.keys(), "Benchmark"]
    prices = prices.loc[:, label_order]

    api_key = fred_api_key or os.getenv("FRED_API_KEY")
    try:
        annual_yield = download_fred_series(
            config.risk_free_series,
            config.start,
            config.end,
            api_key=api_key,
            cache_dir=config.cache_dir,
            refresh=refresh,
        )
        cpi = download_fred_series(
            config.inflation_series,
            config.start,
            config.end,
            api_key=api_key,
            cache_dir=config.cache_dir,
            refresh=refresh,
        )
    except DataDownloadError as exc:
        message = (
            f"FRED macro data were unavailable ({exc}). The configured fallback risk-free "
            "rate is being used and inflation output should be treated as unavailable."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        run_warnings.append(message)
        annual_yield, cpi = _fallback_macro_series(config, prices.index)

    raw_simple = simple_returns(prices)
    raw_log = log_returns(prices)
    macro = align_macro_features(
        raw_simple.index,
        annual_yield,
        cpi,
        periods_per_year=config.periods_per_year,
        fallback_annual_rate=config.fallback_annual_risk_free_rate,
        inflation_release_lag_days=config.inflation_release_lag_days,
    )
    daily_rf = macro["daily_risk_free_rate"]

    asset_returns = raw_simple.loc[:, list(config.assets)]
    benchmark = raw_simple["Benchmark"].rename("Benchmark")
    portfolio_path = construct_portfolio(
        asset_returns,
        config.weights,
        risk_free_daily=daily_rf,
        rebalance_frequency=config.rebalance_frequency,
    )

    common_index = portfolio_path.returns.index.intersection(benchmark.dropna().index)
    evaluation = pd.concat(
        [
            portfolio_path.returns.reindex(common_index),
            benchmark.reindex(common_index),
            asset_returns.reindex(common_index),
        ],
        axis=1,
    ).dropna(how="any")
    daily_rf = (
        daily_rf.reindex(evaluation.index)
        .ffill()
        .fillna(np.expm1(np.log1p(config.fallback_annual_risk_free_rate) / config.periods_per_year))
    )
    macro = macro.reindex(evaluation.index)

    metrics = performance_table(
        evaluation,
        risk_free_daily=daily_rf,
        benchmark_returns=evaluation["Benchmark"],
        periods_per_year=config.periods_per_year,
        confidence=config.var_confidence,
    )
    rolling = rolling_statistics(
        evaluation,
        daily_rf,
        evaluation["Benchmark"],
        window=config.rolling_window,
        periods_per_year=config.periods_per_year,
    )
    calendar_years = calendar_year_metrics(
        evaluation,
        daily_rf,
        evaluation["Benchmark"],
        periods_per_year=config.periods_per_year,
    )
    regressions = regression_table(
        evaluation.drop(columns="Benchmark"),
        evaluation["Benchmark"],
        daily_rf,
        periods_per_year=config.periods_per_year,
    )
    path_returns = portfolio_path.returns.reindex(evaluation.index)
    path_contributions = portfolio_path.contributions.reindex(evaluation.index)
    return_contributions = geometric_linked_contributions(path_returns, path_contributions)
    risk_contributions = component_risk_contributions(
        asset_returns.reindex(evaluation.index),
        config.weights,
        periods_per_year=config.periods_per_year,
    )
    diagnostics = exposure_diagnostics(
        evaluation["Portfolio"],
        evaluation["Benchmark"],
        daily_rf,
        config.weights,
        periods_per_year=config.periods_per_year,
    )
    figures = (
        build_figure_set(
            evaluation,
            rolling,
            metrics,
            return_contributions,
            risk_contributions,
        )
        if build_figures
        else {}
    )
    return AnalysisResults(
        config=config,
        prices=prices.reindex(evaluation.index.union([prices.index[0]])).sort_index(),
        simple_returns=raw_simple.reindex(evaluation.index),
        log_returns=raw_log.reindex(evaluation.index),
        macro=macro,
        portfolio_path=portfolio_path,
        evaluation_returns=evaluation,
        metrics=metrics,
        rolling=rolling,
        calendar_years=calendar_years,
        regressions=regressions,
        return_contributions=return_contributions,
        risk_contributions=risk_contributions,
        diagnostics=diagnostics,
        figures=figures,
        warnings=tuple(run_warnings),
    )
