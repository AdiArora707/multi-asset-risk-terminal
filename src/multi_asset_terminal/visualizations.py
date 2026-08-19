"""Professional Plotly charts used by the notebook and HTML tear sheet."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .metrics import drawdown_series, growth_index

COLORS = [
    "#2563EB",
    "#DC2626",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#0891B2",
    "#4B5563",
    "#DB2777",
]


def _style_figure(fig: go.Figure, title: str, yaxis_title: str = "") -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        colorway=COLORS,
        font={"family": "Inter, Arial, sans-serif", "color": "#1F2937"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 65, "r": 30, "t": 90, "b": 55},
        xaxis={"showgrid": False, "rangeslider": {"visible": False}},
        yaxis={"title": yaxis_title, "gridcolor": "#E5E7EB", "zerolinecolor": "#9CA3AF"},
    )
    return fig


def growth_of_dollar_figure(returns: pd.DataFrame, title: str = "Growth of $1") -> go.Figure:
    wealth = growth_index(returns)
    fig = go.Figure()
    for i, column in enumerate(wealth.columns):
        width = 3.2 if column in {"Portfolio", "Benchmark"} else 1.5
        fig.add_trace(
            go.Scatter(
                x=wealth.index,
                y=wealth[column],
                name=column,
                mode="lines",
                line={"width": width, "color": COLORS[i % len(COLORS)]},
                hovertemplate="%{y:$,.2f}<extra>%{fullData.name}</extra>",
            )
        )
    return _style_figure(fig, title, "Portfolio value")


def drawdown_figure(returns: pd.DataFrame, title: str = "Drawdown from Prior Peak") -> go.Figure:
    drawdowns = drawdown_series(returns)
    fig = go.Figure()
    for i, column in enumerate(drawdowns.columns):
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns[column],
                name=column,
                mode="lines",
                fill="tozeroy" if column == "Portfolio" else None,
                line={"width": 2.5 if column == "Portfolio" else 1.4, "color": COLORS[i]},
                hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
    fig = _style_figure(fig, title, "Drawdown")
    fig.update_yaxes(tickformat=".0%")
    return fig


def rolling_line_figure(
    values: pd.DataFrame,
    title: str,
    yaxis_title: str,
    percent: bool = False,
    zero_line: bool = True,
) -> go.Figure:
    fig = go.Figure()
    for i, column in enumerate(values.columns):
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values[column],
                name=column,
                mode="lines",
                line={"width": 2.5 if column == "Portfolio" else 1.3, "color": COLORS[i]},
            )
        )
    if zero_line:
        fig.add_hline(y=0, line_dash="dot", line_color="#6B7280")
    fig = _style_figure(fig, title, yaxis_title)
    if percent:
        fig.update_yaxes(tickformat=".0%")
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    correlations = returns.corr()
    text = np.vectorize(lambda value: f"{value:.2f}")(correlations.to_numpy())
    fig = go.Figure(
        go.Heatmap(
            z=correlations.to_numpy(),
            x=correlations.columns,
            y=correlations.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[(0.0, "#B91C1C"), (0.5, "#F9FAFB"), (1.0, "#1D4ED8")],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    fig = _style_figure(fig, "Return Correlation Matrix")
    fig.update_layout(height=650, hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return fig


def monthly_return_matrix(returns: pd.Series) -> pd.DataFrame:
    """Create year-by-month compounded returns plus a compounded YTD column."""

    values = returns.dropna().astype(float)
    monthly = values.groupby([values.index.year, values.index.month]).apply(
        lambda x: (1.0 + x).prod() - 1.0
    )
    matrix = monthly.unstack(level=1).reindex(columns=range(1, 13))
    matrix.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    matrix["YTD"] = values.groupby(values.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    matrix.index.name = "Year"
    return matrix


def monthly_return_heatmap(returns: pd.Series) -> go.Figure:
    matrix = monthly_return_matrix(returns)
    text = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    bound = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 0.05)
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index.astype(str),
            zmin=-bound,
            zmax=bound,
            zmid=0,
            colorscale=[(0.0, "#B91C1C"), (0.5, "#F9FAFB"), (1.0, "#047857")],
            text=text.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
            colorbar={"title": "Return", "tickformat": ".0%"},
        )
    )
    fig = _style_figure(fig, f"Monthly Returns — {returns.name}")
    fig.update_layout(hovermode="closest", height=max(430, 32 * len(matrix) + 180))
    fig.update_yaxes(autorange="reversed")
    return fig


def risk_return_scatter(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (name, row) in enumerate(metrics.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[row["Annualized Volatility"]],
                y=[row["CAGR"]],
                text=[name],
                name=name,
                mode="markers+text",
                textposition="top center",
                marker={
                    "size": 11 + 5 * max(float(row.get("Sharpe Ratio", 0.0)), 0.0),
                    "color": COLORS[i % len(COLORS)],
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=("%{text}<br>CAGR: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>"),
            )
        )
    fig = _style_figure(fig, "Risk–Return Map", "CAGR")
    fig.update_xaxes(title="Annualized volatility", tickformat=".0%", showgrid=True)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def contribution_figure(
    return_contributions: pd.Series,
    risk_contributions: pd.DataFrame,
) -> go.Figure:
    names = list(dict.fromkeys([*return_contributions.index, *risk_contributions.index]))
    return_values = return_contributions.reindex(names).fillna(0.0)
    risk_values = risk_contributions["Percent of Total Risk"].reindex(names).fillna(0.0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=return_values,
            name="Linked total-return contribution",
            marker_color="#2563EB",
            hovertemplate="%{y:.2%}<extra>%{fullData.name}</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=risk_values,
            name="Share of portfolio volatility",
            marker_color="#D97706",
            hovertemplate="%{y:.2%}<extra>%{fullData.name}</extra>",
        )
    )
    fig = _style_figure(fig, "Return and Risk Attribution", "Contribution")
    fig.update_layout(barmode="group", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    return fig


def metric_table_figure(metrics: pd.DataFrame) -> go.Figure:
    selected = [
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Beta",
        "Jensen Alpha",
    ]
    percent_fields = {
        "CAGR",
        "Annualized Volatility",
        "Max Drawdown",
        "Jensen Alpha",
    }
    cells: list[list[str]] = [metrics.index.tolist()]
    for field in selected:
        if field in percent_fields:
            cells.append([f"{value:.2%}" if pd.notna(value) else "—" for value in metrics[field]])
        else:
            cells.append([f"{value:.2f}" if pd.notna(value) else "—" for value in metrics[field]])
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Series", *selected],
                    "fill_color": "#111827",
                    "font": {"color": "white", "size": 12},
                    "align": "left",
                    "height": 34,
                },
                cells={
                    "values": cells,
                    "fill_color": [["#F9FAFB", "white"] * (len(metrics) // 2 + 1)],
                    "align": "left",
                    "height": 30,
                    "font": {"color": "#1F2937", "size": 11},
                },
            )
        ]
    )
    fig.update_layout(
        title={"text": "Performance Scorecard", "x": 0.01},
        height=230 + 30 * len(metrics),
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def build_figure_set(
    evaluation_returns: pd.DataFrame,
    rolling: Mapping[str, pd.DataFrame],
    metrics: pd.DataFrame,
    return_contributions: pd.Series,
    risk_contributions: pd.DataFrame,
) -> dict[str, go.Figure]:
    """Build the complete dashboard consistently in one call."""

    key_series = [column for column in ["Portfolio", "Benchmark"] if column in evaluation_returns]
    return {
        "scorecard": metric_table_figure(metrics),
        "growth": growth_of_dollar_figure(evaluation_returns),
        "drawdown": drawdown_figure(evaluation_returns[key_series]),
        "risk_return": risk_return_scatter(metrics),
        "rolling_sharpe": rolling_line_figure(
            rolling["sharpe"][key_series], "Rolling 12-Month Sharpe Ratio", "Sharpe ratio"
        ),
        "rolling_volatility": rolling_line_figure(
            rolling["volatility"][key_series],
            "Rolling 12-Month Volatility",
            "Annualized volatility",
            percent=True,
            zero_line=False,
        ),
        "rolling_beta": rolling_line_figure(
            rolling["beta"][key_series], "Rolling 12-Month Beta", "Beta"
        ),
        "correlation": correlation_heatmap(evaluation_returns),
        "monthly_returns": monthly_return_heatmap(evaluation_returns["Portfolio"]),
        "attribution": contribution_figure(return_contributions, risk_contributions),
    }
