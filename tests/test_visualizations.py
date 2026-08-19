import numpy as np
import pandas as pd
import plotly.graph_objects as go

from multi_asset_terminal.visualizations import (
    correlation_heatmap,
    growth_of_dollar_figure,
    monthly_return_heatmap,
    monthly_return_matrix,
)


def test_core_visualizations_build_without_mutating_returns():
    rng = np.random.default_rng(5)
    index = pd.date_range("2022-01-03", periods=520, freq="B")
    returns = pd.DataFrame(
        {
            "Portfolio": rng.normal(0.0003, 0.008, len(index)),
            "Benchmark": rng.normal(0.0002, 0.01, len(index)),
        },
        index=index,
    )
    original = returns.copy()
    figures = [
        growth_of_dollar_figure(returns),
        correlation_heatmap(returns),
        monthly_return_heatmap(returns["Portfolio"]),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert "YTD" in monthly_return_matrix(returns["Portfolio"]).columns
    pd.testing.assert_frame_equal(returns, original)
