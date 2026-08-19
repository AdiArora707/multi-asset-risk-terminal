"""CSV, Plotly HTML, and optional QuantStats report generation."""

from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio

from .pipeline import AnalysisResults
from .visualizations import monthly_return_matrix

LOGGER = logging.getLogger(__name__)


def _format_value(name: str, value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if not isinstance(value, (int, float, np.integer, np.floating)) or pd.isna(value):
        return "—" if pd.isna(value) else html.escape(str(value))
    percent_tokens = (
        "Return",
        "CAGR",
        "Volatility",
        "Deviation",
        "Drawdown",
        "VaR",
        "CVaR",
        "Hit Rate",
        "Alpha",
        "Capture",
        "Exposure",
        "Weight",
        "Positive Returns",
    )
    if any(token in name for token in percent_tokens):
        return f"{float(value):.2%}"
    if name in {"Observations", "Max Drawdown Duration"}:
        return f"{int(value):,}"
    return f"{float(value):.3f}"


def _html_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    display = frame.head(max_rows) if max_rows else frame
    formatted = display.copy()
    for column in formatted.columns:
        formatted[column] = [_format_value(str(column), value) for value in formatted[column]]
    return formatted.to_html(classes="dataframe", border=0, escape=False)


def _diagnostics_html(series: pd.Series) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(name))}</th><td>{_format_value(str(name), value)}</td></tr>"
        for name, value in series.items()
    )
    return f'<table class="dataframe diagnostics"><tbody>{rows}</tbody></table>'


def generate_html_tearsheet(results: AnalysisResults, path: str | Path) -> Path:
    """Write a single portable dashboard-style HTML tear sheet."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chart_blocks: list[str] = []
    for position, (name, figure) in enumerate(results.figures.items()):
        chart = pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs="cdn" if position == 0 else False,
            config={"displaylogo": False, "responsive": True},
        )
        chart_blocks.append(f'<section class="chart" id="{html.escape(name)}">{chart}</section>')

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in results.warnings)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Asset Performance and Risk Tear Sheet</title>
  <style>
    :root{{--ink:#111827;--muted:#6B7280;--line:#E5E7EB;--blue:#2563EB;--paper:#F3F4F6}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
      font-family:Inter,Arial,sans-serif}} main{{max-width:1440px;margin:auto;padding:32px}}
    header,.panel,.chart{{background:white;border:1px solid var(--line);border-radius:14px;
      box-shadow:0 1px 2px rgba(0,0,0,.04)}} header{{padding:30px;margin-bottom:20px;
      border-top:5px solid var(--blue)}} h1{{margin:0 0 8px;font-size:30px}} h2{{font-size:19px}}
    .subtitle,.meta,.disclaimer{{color:var(--muted)}} .meta{{font-size:13px;margin-top:12px}}
    .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-bottom:20px}}
    .panel{{padding:20px;overflow:auto}} .chart{{padding:10px;margin-bottom:20px;overflow:hidden}}
    table.dataframe{{border-collapse:collapse;width:100%;font-size:13px}} table.dataframe th,
    table.dataframe td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;
      white-space:nowrap}} table.dataframe th:first-child,table.dataframe td:first-child{{text-align:left}}
    table.dataframe thead th{{background:#111827;color:white;position:sticky;top:0}}
    .diagnostics th{{width:65%}} .warning{{color:#92400E;background:#FFFBEB;padding:12px;border-radius:8px}}
    .disclaimer{{font-size:12px;line-height:1.5;margin:24px 8px}}
    @media(max-width:900px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
  </style>
</head>
<body><main>
  <header>
    <h1>Multi-Asset Performance and Risk Terminal</h1>
    <div class="subtitle">Cross-asset performance, factor exposure, tail risk, and contribution analysis</div>
    <div class="meta">Analysis: {results.evaluation_returns.index.min().date()} to
      {results.evaluation_returns.index.max().date()} · Generated {generated} · Benchmark:
      {html.escape(results.config.benchmark)}</div>
  </header>
  <div class="grid">
    <section class="panel"><h2>Exposure and Return Diagnostics</h2>{_diagnostics_html(results.diagnostics)}</section>
    <section class="panel"><h2>Portfolio Configuration</h2>{_html_table(pd.DataFrame({"Ticker": results.config.assets, "Weight": results.config.weights}))}</section>
  </div>
  {f'<div class="warning"><strong>Data warning</strong><ul>{warning_html}</ul></div>' if warning_html else ""}
  {"".join(chart_blocks)}
  <div class="grid">
    <section class="panel"><h2>CAPM Regression Diagnostics</h2>{_html_table(results.regressions)}</section>
    <section class="panel"><h2>Calendar-Year Portfolio Results</h2>{_html_table(results.calendar_years.xs("Portfolio", level="Series"))}</section>
  </div>
  <p class="disclaimer"><strong>Research use only.</strong> ETF prices are adjusted historical data and
  may be revised. Historical VaR is not a worst-case loss, beta is benchmark-dependent, and all results
  are backward-looking. This report is educational and is not investment advice.</p>
</main></body></html>"""
    destination.write_text(document, encoding="utf-8")
    return destination


def generate_quantstats_report(results: AnalysisResults, path: str | Path) -> Path | None:
    """Generate a secondary QuantStats report without making it a hard dependency at runtime."""

    destination = Path(path)
    try:
        import quantstats as qs

        annual_rf = float(results.macro["annual_risk_free_rate"].mean())
        qs.reports.html(
            results.evaluation_returns["Portfolio"],
            benchmark=results.evaluation_returns["Benchmark"],
            rf=annual_rf,
            title="Multi-Asset Portfolio — QuantStats Tear Sheet",
            output=str(destination),
            periods_per_year=results.config.periods_per_year,
        )
        return destination
    except Exception as exc:  # noqa: BLE001 - optional third-party report must not abort exports
        LOGGER.warning("QuantStats report was skipped: %s", exc)
        return None


def export_analysis_artifacts(
    results: AnalysisResults,
    output_dir: str | Path | None = None,
    *,
    include_quantstats: bool = True,
) -> dict[str, Path]:
    """Export clean research tables, configuration, and automated tear sheets."""

    directory = Path(output_dir or results.config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {
        "adjusted_prices": results.prices,
        "simple_returns": results.simple_returns,
        "log_returns": results.log_returns,
        "macro_features": results.macro,
        "performance_metrics": results.metrics,
        "calendar_year_metrics": results.calendar_years,
        "capm_regressions": results.regressions,
        "risk_contributions": results.risk_contributions,
        "monthly_portfolio_returns": monthly_return_matrix(results.evaluation_returns["Portfolio"]),
    }
    for name, frame in tables.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path)
        paths[name] = path

    return_path = directory / "linked_return_contributions.csv"
    results.return_contributions.to_csv(return_path, header=True)
    paths["return_contributions"] = return_path
    diagnostic_path = directory / "exposure_diagnostics.csv"
    results.diagnostics.to_csv(diagnostic_path, header=True)
    paths["diagnostics"] = diagnostic_path

    config_path = directory / "run_config.json"
    config_path.write_text(json.dumps(asdict(results.config), indent=2), encoding="utf-8")
    paths["config"] = config_path
    tearsheet_path = generate_html_tearsheet(results, directory / "multi_asset_tear_sheet.html")
    paths["tear_sheet"] = tearsheet_path
    if include_quantstats:
        quantstats_path = generate_quantstats_report(results, directory / "quantstats_report.html")
        if quantstats_path:
            paths["quantstats"] = quantstats_path
    return paths
