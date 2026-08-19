"""Reusable performance, drawdown, tail-risk, and regression analytics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _clean_series(values: pd.Series) -> pd.Series:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()


def growth_index(returns: pd.Series | pd.DataFrame, initial_value: float = 1.0):
    """Compound simple returns into a wealth index."""

    return initial_value * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series | pd.DataFrame):
    """Return the percentage decline from the prior wealth peak."""

    wealth = growth_index(returns)
    return wealth.div(wealth.cummax()).sub(1.0)


def cagr(returns: pd.Series) -> float:
    """Geometrically annualized return using actual elapsed calendar time."""

    values = _clean_series(returns)
    if len(values) < 2:
        return np.nan
    total_growth = float((1.0 + values).prod())
    elapsed_years = (values.index[-1] - values.index[0]).days / 365.25
    if elapsed_years <= 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (1.0 / elapsed_years) - 1.0


def max_drawdown_duration(drawdowns: pd.Series) -> int:
    """Longest number of observations spent below a previous high-water mark."""

    underwater = drawdowns.fillna(0.0) < 0.0
    groups = (~underwater).cumsum()
    durations = underwater.groupby(groups).cumsum()
    return int(durations.max()) if not durations.empty else 0


def capm_regression(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_daily: pd.Series | float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Estimate Jensen alpha and beta with heteroskedasticity/autocorrelation robust errors."""

    joined = pd.concat(
        [asset_returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(joined) < 30 or joined["benchmark"].var() <= 0:
        return pd.Series(
            {
                "Beta": np.nan,
                "Jensen Alpha": np.nan,
                "R-squared": np.nan,
                "Alpha t-stat": np.nan,
                "Beta t-stat": np.nan,
            }
        )
    if isinstance(risk_free_daily, pd.Series):
        rf = risk_free_daily.reindex(joined.index).ffill().fillna(0.0)
    else:
        rf = pd.Series(float(risk_free_daily), index=joined.index)
    y = (joined["asset"] - rf).rename("asset")
    market_excess = (joined["benchmark"] - rf).rename("benchmark")
    x = sm.add_constant(market_excess, has_constant="add")
    fit = sm.OLS(y, x, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    alpha_daily = float(fit.params["const"])
    return pd.Series(
        {
            "Beta": float(fit.params["benchmark"]),
            "Jensen Alpha": (1.0 + alpha_daily) ** periods_per_year - 1.0,
            "R-squared": float(fit.rsquared),
            "Alpha t-stat": float(fit.tvalues["const"]),
            "Beta t-stat": float(fit.tvalues["benchmark"]),
        }
    )


def calculate_metrics(
    returns: pd.Series,
    risk_free_daily: pd.Series | float = 0.0,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    confidence: float = 0.95,
    best_n: int = 10,
) -> pd.Series:
    """Calculate standardized absolute, relative, and tail-risk statistics."""

    values = _clean_series(returns)
    if len(values) < 2:
        raise ValueError("At least two valid return observations are required.")
    if isinstance(risk_free_daily, pd.Series):
        rf = risk_free_daily.reindex(values.index).ffill().fillna(0.0).astype(float)
    else:
        rf = pd.Series(float(risk_free_daily), index=values.index)
    excess = values - rf

    mean_excess = float(excess.mean())
    excess_vol = float(excess.std(ddof=1))
    volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))
    annual_excess = mean_excess * periods_per_year
    sharpe = mean_excess / excess_vol * np.sqrt(periods_per_year) if excess_vol > 0 else np.nan
    sortino = annual_excess / downside_deviation if downside_deviation > 0 else np.nan

    drawdowns = drawdown_series(values)
    maximum_drawdown = float(drawdowns.min())
    compound_return = cagr(values)
    calmar = compound_return / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan

    left_quantile = float(values.quantile(1.0 - confidence))
    tail = values[values <= left_quantile]
    var = -left_quantile
    cvar = -float(tail.mean()) if not tail.empty else np.nan
    upper_quantile = float(values.quantile(confidence))
    tail_ratio = upper_quantile / abs(left_quantile) if left_quantile != 0 else np.nan
    positive_sum = float(values[values > 0].sum())
    negative_sum = abs(float(values[values < 0].sum()))
    profit_factor = positive_sum / negative_sum if negative_sum > 0 else np.nan

    without_best = values.drop(values.nlargest(min(best_n, len(values))).index)
    cagr_without_best = cagr(without_best) if len(without_best) >= 2 else np.nan
    best_day_share = (
        float(values.nlargest(min(best_n, len(values))).sum()) / positive_sum
        if positive_sum > 0
        else np.nan
    )

    relative = {
        "Beta": np.nan,
        "Jensen Alpha": np.nan,
        "R-squared": np.nan,
        "Tracking Error": np.nan,
        "Information Ratio": np.nan,
        "Up Capture": np.nan,
        "Down Capture": np.nan,
    }
    if benchmark_returns is not None:
        joined = pd.concat(
            [values.rename("asset"), benchmark_returns.rename("benchmark")], axis=1
        ).dropna()
        if len(joined) >= 2:
            regression = capm_regression(joined["asset"], joined["benchmark"], rf, periods_per_year)
            relative.update(regression[["Beta", "Jensen Alpha", "R-squared"]].to_dict())
            active = joined["asset"] - joined["benchmark"]
            tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
            relative["Tracking Error"] = tracking_error
            relative["Information Ratio"] = (
                float(active.mean() * periods_per_year / tracking_error)
                if tracking_error > 0
                else np.nan
            )
            up = joined["benchmark"] > 0
            down = joined["benchmark"] < 0
            relative["Up Capture"] = (
                float(joined.loc[up, "asset"].mean() / joined.loc[up, "benchmark"].mean())
                if up.any() and joined.loc[up, "benchmark"].mean() != 0
                else np.nan
            )
            relative["Down Capture"] = (
                float(joined.loc[down, "asset"].mean() / joined.loc[down, "benchmark"].mean())
                if down.any() and joined.loc[down, "benchmark"].mean() != 0
                else np.nan
            )

    output = {
        "Observations": len(values),
        "Cumulative Return": float((1.0 + values).prod() - 1.0),
        "CAGR": compound_return,
        "Annualized Volatility": volatility,
        "Annualized Downside Deviation": downside_deviation,
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Calmar Ratio": float(calmar),
        "Max Drawdown": maximum_drawdown,
        "Max Drawdown Duration": max_drawdown_duration(drawdowns),
        "Skewness": float(stats.skew(values, bias=False)),
        "Excess Kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        f"Historical VaR {confidence:.0%}": var,
        f"Historical CVaR {confidence:.0%}": cvar,
        "Tail Ratio": tail_ratio,
        "Profit Factor": profit_factor,
        "Hit Rate": float((values > 0).mean()),
        "Best Day": float(values.max()),
        "Worst Day": float(values.min()),
        f"CAGR Without Best {best_n} Days": cagr_without_best,
        f"Best {best_n} Days / Positive Returns": best_day_share,
        **relative,
    }
    return pd.Series(output, name=returns.name)


def performance_table(
    returns: pd.DataFrame,
    risk_free_daily: pd.Series | float = 0.0,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate the same metrics for every return stream."""

    rows = {
        column: calculate_metrics(
            returns[column],
            risk_free_daily=risk_free_daily,
            benchmark_returns=benchmark_returns,
            periods_per_year=periods_per_year,
            confidence=confidence,
        )
        for column in returns.columns
    }
    return pd.DataFrame(rows).T


def rolling_statistics(
    returns: pd.DataFrame,
    risk_free_daily: pd.Series | float,
    benchmark_returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> dict[str, pd.DataFrame]:
    """Vectorized rolling volatility, Sharpe ratio, and beta."""

    if isinstance(risk_free_daily, pd.Series):
        rf = risk_free_daily.reindex(returns.index).ffill().fillna(0.0)
    else:
        rf = pd.Series(float(risk_free_daily), index=returns.index)
    excess = returns.sub(rf, axis=0)
    rolling_vol = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )
    rolling_sharpe = (
        excess.rolling(window, min_periods=window)
        .mean()
        .div(excess.rolling(window, min_periods=window).std(ddof=1))
        .mul(np.sqrt(periods_per_year))
    )
    benchmark_variance = benchmark_returns.rolling(window, min_periods=window).var(ddof=1)
    rolling_beta = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for column in returns:
        rolling_beta[column] = (
            returns[column]
            .rolling(window, min_periods=window)
            .cov(benchmark_returns)
            .div(benchmark_variance)
        )
    return {"volatility": rolling_vol, "sharpe": rolling_sharpe, "beta": rolling_beta}


def calendar_year_metrics(
    returns: pd.DataFrame,
    risk_free_daily: pd.Series | float,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Create a tidy calendar-year table for subperiod stability analysis."""

    records: list[dict[str, object]] = []
    for year, frame in returns.groupby(returns.index.year):
        for column in frame.columns:
            rf_slice = (
                risk_free_daily.reindex(frame.index)
                if isinstance(risk_free_daily, pd.Series)
                else risk_free_daily
            )
            stats_row = calculate_metrics(
                frame[column],
                risk_free_daily=rf_slice,
                benchmark_returns=benchmark_returns.reindex(frame.index),
                periods_per_year=periods_per_year,
            )
            records.append(
                {
                    "Year": int(year),
                    "Series": column,
                    "Return": stats_row["Cumulative Return"],
                    "Volatility": stats_row["Annualized Volatility"],
                    "Sharpe": stats_row["Sharpe Ratio"],
                    "Max Drawdown": stats_row["Max Drawdown"],
                    "Beta": stats_row["Beta"],
                }
            )
    return pd.DataFrame.from_records(records).set_index(["Year", "Series"]).sort_index()


def regression_table(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_free_daily: pd.Series | float,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """CAPM diagnostics for all assets and the portfolio."""

    rows: Mapping[str, pd.Series] = {
        column: capm_regression(
            returns[column], benchmark_returns, risk_free_daily, periods_per_year
        )
        for column in returns
    }
    return pd.DataFrame(rows).T
