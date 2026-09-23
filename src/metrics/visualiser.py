"""
Dash-based live visualiser for FlashBalanceAI experiment metrics.

Displays four live-updating charts:
  1. Episode Reward (mean)
  2. Latency (P95 / P99)
  3. CPU Utilisation (mean)
  4. SLA Violation Rate

Usage:
    from metrics.visualiser import create_app
    app = create_app(collector)
    app.run(debug=True)

Requires: dash, plotly  (pip install dash)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metrics.collector import MetricsCollector

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go

    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False


def create_app(collector: "MetricsCollector") -> "dash.Dash":
    """
    Build and return a Dash application wired to the given MetricsCollector.

    Args:
        collector: A MetricsCollector whose `.episodes` list will be
                   read on every refresh interval.

    Returns:
        A configured Dash app instance (call `.run()` to start).
    """
    if not _DASH_AVAILABLE:
        raise ImportError(
            "Dash is not installed. Install it with:  pip install dash"
        )

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("FlashBalanceAI — Live Metrics Dashboard"),
            dcc.Interval(id="refresh", interval=2000, n_intervals=0),
            html.Div(
                [
                    dcc.Graph(id="reward-chart"),
                    dcc.Graph(id="latency-chart"),
                ],
                style={"display": "flex"},
            ),
            html.Div(
                [
                    dcc.Graph(id="cpu-chart"),
                    dcc.Graph(id="sla-chart"),
                ],
                style={"display": "flex"},
            ),
        ]
    )

    @app.callback(
        Output("reward-chart", "figure"),
        Input("refresh", "n_intervals"),
    )
    def update_reward(_n: int) -> go.Figure:
        episodes = collector.episodes
        x = list(range(1, len(episodes) + 1))
        y = [ep["reward_episode_mean"] for ep in episodes]
        return go.Figure(
            data=[go.Scatter(x=x, y=y, mode="lines+markers", name="Reward")],
            layout=go.Layout(title="Episode Reward (mean)", xaxis_title="Episode", yaxis_title="Reward"),
        )

    @app.callback(
        Output("latency-chart", "figure"),
        Input("refresh", "n_intervals"),
    )
    def update_latency(_n: int) -> go.Figure:
        episodes = collector.episodes
        x = list(range(1, len(episodes) + 1))
        p95 = [ep["p95_latency"] for ep in episodes]
        p99 = [ep["p99_latency"] for ep in episodes]
        return go.Figure(
            data=[
                go.Scatter(x=x, y=p95, mode="lines+markers", name="P95"),
                go.Scatter(x=x, y=p99, mode="lines+markers", name="P99"),
            ],
            layout=go.Layout(title="Latency (ms)", xaxis_title="Episode", yaxis_title="Latency (ms)"),
        )

    @app.callback(
        Output("cpu-chart", "figure"),
        Input("refresh", "n_intervals"),
    )
    def update_cpu(_n: int) -> go.Figure:
        episodes = collector.episodes
        x = list(range(1, len(episodes) + 1))
        y = [ep["cpu_utilisation_mean"] for ep in episodes]
        return go.Figure(
            data=[go.Scatter(x=x, y=y, mode="lines+markers", name="CPU")],
            layout=go.Layout(title="CPU Utilisation (mean)", xaxis_title="Episode", yaxis_title="CPU %"),
        )

    @app.callback(
        Output("sla-chart", "figure"),
        Input("refresh", "n_intervals"),
    )
    def update_sla(_n: int) -> go.Figure:
        episodes = collector.episodes
        x = list(range(1, len(episodes) + 1))
        y = [ep["sla_violation_rate"] for ep in episodes]
        return go.Figure(
            data=[go.Scatter(x=x, y=y, mode="lines+markers", name="SLA Violation Rate")],
            layout=go.Layout(title="SLA Violation Rate", xaxis_title="Episode", yaxis_title="Rate"),
        )

    return app
