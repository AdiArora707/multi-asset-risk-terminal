"""Generate the committed Colab notebook from concise, reviewable cell sources."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# Multi-Asset Performance and Risk Terminal

**Equities · Bonds · Commodities · Real Estate · Cash**

This Colab is the research interface to a reusable, tested Python analytics package. It downloads
adjusted ETF prices and FRED macro series, constructs a financing-aware multi-asset portfolio,
separates beta from alpha, measures tail risk and return concentration, and exports an interactive
HTML tear sheet.

> Educational research only—not investment advice. Review the provider terms before using market
> data outside a personal/research setting."""
    ),
    markdown(
        """## 0. Colab setup

When running in Google Colab, the setup cell clones the published GitHub repository. A local Jupyter
session automatically uses the checked-out project. Dependency installation is explicit and repeatable."""
    ),
    code(
        """#@title Install the project
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/AdiArora707/multi-asset-risk-terminal.git"  #@param {type:"string"}

if IN_COLAB:
    project_dir = Path("/content/multi-asset-risk-terminal")
    if not project_dir.exists():
        if "YOUR_USERNAME" in REPO_URL or not REPO_URL.startswith("https://github.com/"):
            raise ValueError("REPO_URL must point to a valid GitHub repository.")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(project_dir)], check=True)
    os.chdir(project_dir)
else:
    project_dir = Path.cwd()
    if not (project_dir / "pyproject.toml").exists():
        raise RuntimeError("Start Jupyter from the repository root.")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".[notebook]"],
    check=True,
)
print(f"Project ready: {project_dir}")"""
    ),
    markdown(
        """## 1. Imports and display configuration

The notebook imports the public project API. All finance formulas live in `src/`, where they can be
unit-tested and reused by the command-line job."""
    ),
    code(
        """import os
from IPython.display import HTML, display
import pandas as pd
import plotly.io as pio

from multi_asset_terminal import TerminalConfig, export_analysis_artifacts, run_analysis
from multi_asset_terminal.visualizations import monthly_return_matrix

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
pio.templates.default = "plotly_white"""
    ),
    markdown(
        """## 2. Investment policy and run configuration

Weights may sum above 100%. Any negative residual is treated as borrowing and charged the daily
FRED risk-free rate. The default portfolio is unlevered and rebalanced monthly."""
    ),
    code(
        """#@title Configure the analysis
START_DATE = "2015-01-01"  #@param {type:"date"}
END_DATE = ""  #@param {type:"string"}
BENCHMARK = "SPY"  #@param {type:"string"}
ROLLING_WINDOW = 252  #@param {type:"integer"}
REFRESH_DATA = False  #@param {type:"boolean"}

assets = {
    "US Equity": "VTI",
    "International Equity": "VXUS",
    "US Bonds": "BND",
    "Gold": "GLD",
    "Real Estate": "VNQ",
    "Cash ETF": "BIL",
}
weights = {
    "US Equity": 0.30,
    "International Equity": 0.15,
    "US Bonds": 0.25,
    "Gold": 0.10,
    "Real Estate": 0.10,
    "Cash ETF": 0.10,
}

config = TerminalConfig(
    assets=assets,
    weights=weights,
    benchmark=BENCHMARK,
    start=START_DATE,
    end=END_DATE or None,
    rebalance_frequency="M",
    rolling_window=ROLLING_WINDOW,
)
print(f"Gross exposure: {config.gross_exposure:.1%}")
print(f"Net risky exposure: {config.net_exposure:.1%}")
display(pd.DataFrame({"Ticker": config.assets, "Target Weight": config.weights}))"""
    ),
    markdown(
        """## 3. Data collection pipeline

The run uses adjusted Yahoo Finance closes (`auto_adjust=True`) plus FRED `DGS3MO` and `CPIAUCSL`.
Set `FRED_API_KEY` in the environment to use the authenticated observations API. Otherwise, the
official FRED CSV endpoint is used. Downloads are retried, validated, and cached by parameters."""
    ),
    code(
        """# Optional in Colab: avoid displaying or committing the key.
# os.environ["FRED_API_KEY"] = "paste_your_key_here"

results = run_analysis(
    config,
    refresh=REFRESH_DATA,
    fred_api_key=os.getenv("FRED_API_KEY"),
)

print(f"Price observations: {len(results.prices):,}")
print(f"Analysis observations: {len(results.evaluation_returns):,}")
print(f"Sample: {results.evaluation_returns.index.min().date()} to "
      f"{results.evaluation_returns.index.max().date()}")
if results.warnings:
    for warning in results.warnings:
        print(f"WARNING: {warning}")
display(results.prices.tail())
display(results.macro.tail())"""
    ),
    markdown(
        """## 4. Data cleaning and feature engineering

The common calendar uses complete observations and never forward-fills prices. This avoids creating
false zero returns. Simple returns drive compounding; log returns are retained for additive and
distributional research. Annual Treasury yields are converted into equivalent daily compounded rates,
and CPI levels become trailing 12-month inflation."""
    ),
    code(
        """quality = pd.DataFrame({
    "Price rows": results.prices.count(),
    "Missing prices": results.prices.isna().sum(),
    "Simple return rows": results.simple_returns.count(),
    "Mean daily return": results.simple_returns.mean(),
    "Daily volatility": results.simple_returns.std(),
})
display(quality)
display(results.simple_returns.tail())
display(results.log_returns.tail())"""
    ),
    markdown(
        """## 5. Portfolio construction and reconciliation

Weights reset at the first observation of each month and drift between rebalances. `Financing` is
positive cash or negative borrowing. Daily asset contributions must sum exactly to each portfolio
return—this is an important production invariant."""
    ),
    code(
        """path = results.portfolio_path
reconciliation_error = (
    path.contributions.sum(axis=1) - path.returns
).abs().max()

print(f"Maximum contribution reconciliation error: {reconciliation_error:.3e}")
display(path.weights.tail())
display(path.cash_weight.tail().to_frame())
display(path.contributions.tail())"""
    ),
    markdown(
        """## 6. Standardized performance and risk statistics

All return streams use the same CAGR, volatility, downside, drawdown, benchmark-relative, higher-
moment, and empirical tail-loss functions. Percentage metrics are formatted only for presentation;
the underlying tables remain numeric and export cleanly."""
    ),
    code(
        """percent_metrics = [
    "Cumulative Return", "CAGR", "Annualized Volatility",
    "Annualized Downside Deviation", "Max Drawdown", "Jensen Alpha",
    "Tracking Error", "Hit Rate", "Best Day", "Worst Day",
    "CAGR Without Best 10 Days", "Best 10 Days / Positive Returns",
]
display(
    results.metrics.style
    .format({column: "{:.2%}" for column in percent_metrics})
    .format({
        "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}",
        "Calmar Ratio": "{:.2f}", "Beta": "{:.2f}", "R-squared": "{:.2f}",
    })
    .background_gradient(subset=["Sharpe Ratio", "Sortino Ratio"], cmap="RdYlGn")
)"""
    ),
    markdown(
        """## 7. Beta, leverage, tail, and concentration diagnostics

This panel answers the core investment question: did performance come from leverage, benchmark beta,
statistically estimated residual alpha, or a few exceptional trading days? Alpha is diagnostic—not a
causal claim—and its robust t-stat appears in the CAPM table."""
    ),
    code(
        """display(results.diagnostics.to_frame("Value"))
display(results.regressions.style.format("{:.3f}"))

portfolio_metrics = results.metrics.loc["Portfolio"]
print(f"Portfolio CAGR: {portfolio_metrics['CAGR']:.2%}")
print(f"Without the best 10 days: {portfolio_metrics['CAGR Without Best 10 Days']:.2%}")
print(f"Historical CVaR: {portfolio_metrics[f'Historical CVaR {config.var_confidence:.0%}']:.2%}")"""
    ),
    markdown(
        """## 8. Return and risk attribution

Carino-linked contributions sum to compounded total return. Euler component risks sum to annualized
portfolio volatility. Return leadership and risk-budget consumption can therefore be compared without
mixing incompatible units."""
    ),
    code(
        """total_return = (1 + results.evaluation_returns["Portfolio"]).prod() - 1
linked_sum = results.return_contributions.sum()
risk_sum = results.risk_contributions["Component Risk"].sum()
portfolio_vol = results.metrics.loc["Portfolio", "Annualized Volatility"]

print(f"Linked contributions / total return: {linked_sum:.4%} / {total_return:.4%}")
print(f"Component risk / portfolio volatility: {risk_sum:.4%} / {portfolio_vol:.4%}")
display(results.return_contributions.to_frame())
display(results.risk_contributions.style.format("{:.2%}"))
results.figures["attribution"].show()"""
    ),
    markdown(
        """## 9. Rolling and subperiod analysis

Rolling 252-trading-day windows reveal whether full-sample statistics are stable. Calendar-year tables
surface regime dependence and unusually strong or weak subperiods."""
    ),
    code(
        """display(results.calendar_years.xs("Portfolio", level="Series").style.format("{:.2%}"))
results.figures["rolling_sharpe"].show()
results.figures["rolling_volatility"].show()
results.figures["rolling_beta"].show()"""
    ),
    markdown(
        """## 10. Interactive performance dashboard

The same Plotly figure objects are used in Colab and the exported report, preventing notebook/report
metric drift."""
    ),
    code(
        """for chart_name in [
    "scorecard", "growth", "drawdown", "risk_return", "correlation", "monthly_returns"
]:
    results.figures[chart_name].show()"""
    ),
    markdown(
        """## 11. Monthly return table

Each monthly cell and YTD total is compounded from daily simple returns."""
    ),
    code(
        """monthly = monthly_return_matrix(results.evaluation_returns["Portfolio"])
display(monthly.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=None))"""
    ),
    markdown(
        """## 12. Automated tear sheet and research artifacts

This exports the custom interactive HTML report, all clean CSV tables, the exact run configuration,
and—when its plotting dependencies are compatible—an additional QuantStats report."""
    ),
    code(
        """artifacts = export_analysis_artifacts(results, include_quantstats=True)
for name, path in artifacts.items():
    print(f"{name:24s} -> {path.resolve()}")

tear_sheet = artifacts["tear_sheet"]
display(HTML(f'<a href="{tear_sheet}" target="_blank"><b>Open interactive tear sheet</b></a>'))

if IN_COLAB:
    from google.colab import files
    files.download(str(tear_sheet))"""
    ),
    markdown(
        """## 13. Interpretation checklist and next steps

Before presenting results, ask:

1. Is the Sharpe ratio persistent across rolling windows and calendar years?
2. Does CAPM beta and \\(R^2\\) explain most of the return?
3. Is Jensen alpha economically meaningful, and is its robust t-stat credible?
4. Does gross exposure exceed 100%, and is financing treated consistently?
5. Which assets dominate the volatility budget versus linked total return?
6. How much CAGR disappears when the ten best days are removed?
7. Are drawdown, CVaR, and downside capture acceptable for the mandate?
8. Are ETF proxies, survivorship, fees, liquidity, taxes, and data licensing suitable for the use case?

Professional upgrades include ALFRED vintages, multi-factor attribution, transaction costs, stress
testing, bootstrap confidence intervals, optimization constraints, institutional data adapters, and a
scheduled Streamlit/Dash deployment."""
    ),
]


def main() -> None:
    for position, cell in enumerate(CELLS):
        cell["id"] = f"terminal-cell-{position:02d}"
    notebook = {
        "cells": CELLS,
        "metadata": {
            "colab": {"name": "Multi_Asset_Performance_Risk_Terminal.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path("notebooks/Multi_Asset_Performance_Risk_Terminal.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {destination} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
