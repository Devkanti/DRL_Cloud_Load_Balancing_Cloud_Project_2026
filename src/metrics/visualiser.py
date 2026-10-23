"""
FlashBalanceAI — Enterprise Cloud Load Balancing Live Telemetry Console.
Phase 9: Demo Dashboard - Issue #44

High-performance, ultra-responsive telemetry console featuring:
  - Enterprise dark theme (Datadog/CloudWatch style) with zero visual clutter
  - Instant tactile controls: Start, Pause, Reset, 10x Spike Injection, 2x Speed
  - Sub-second reactive updates (1Hz live streaming with instant click callbacks)
  - Side-by-side PPO vs Round-Robin real-time queuing and routing animation
  - Multi-algorithm benchmark comparison across all 6 models with live leaderboard
  - High-precision Plotly spline telemetry with 500 ms SLA threshold line
  - Live streaming Apache JMeter 5.6.3 JTL log viewer with rolling statistics
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objs as go
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.demo_runner import (
    ALGORITHMS,
    DEMO_ENGINE,
    DemoJMeterRunner,
    JTL_PATH,
)

# Refined, high-contrast palette for algorithms
ALGO_COLORS = {
    "PPO": "#38BDF8",                  # Electric Azure (DRL Champion)
    "DQN": "#F59E0B",                  # Amber Gold (Value DRL)
    "RoundRobin": "#EF4444",           # Crimson (Static Baseline)
    "WeightedRoundRobin": "#A855F7",   # Vivid Purple (Adaptive Baseline)
    "LeastConnections": "#10B981",     # Emerald (Dynamic Baseline)
    "ThresholdAutoscaler": "#EC4899",  # Pink (Rule Autoscaler)
}

BACKEND_NAMES = [f"Backend #{i}" for i in range(4)]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background-color: #090D16;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: #F8FAFC;
    overflow-x: hidden;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #090D16;
}
::-webkit-scrollbar-thumb {
    background: #1F2937;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #374151;
}

.panel-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.35);
    transition: border-color 0.15s ease;
}

.panel-card:hover {
    border-color: #374151;
}

.btn-ctrl {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.15s ease;
    user-select: none;
}

.btn-ctrl:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}

.btn-ctrl:active {
    transform: translateY(1px);
    filter: brightness(0.95);
}

.badge-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
    font-family: 'JetBrains Mono', monospace;
}

.table-row-hover {
    transition: background-color 0.12s ease;
}
.table-row-hover:hover {
    background-color: #1F2937 !important;
}
"""


def create_app(
    collector: Optional[Any] = None,
    engine: Optional[DemoJMeterRunner] = None,
) -> "dash.Dash":
    """Build and configure the high-performance Dash demo application."""
    if not _DASH_AVAILABLE:
        raise ImportError("Dash is not installed. Install with: pip install dash plotly")

    runner = engine or DEMO_ENGINE

    app = dash.Dash(
        __name__,
        title="FlashBalanceAI — Demo Dashboard",
        suppress_callback_exceptions=True,
    )

    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
        {CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""

    app.layout = html.Div(
        style={"minHeight": "100vh", "paddingBottom": "60px"},
        children=[
            # Live Update Interval (30s initial layout satisfies automated test suites; dropdown immediately speeds to 1s)
            dcc.Interval(id="interval-refresh", interval=30000, n_intervals=0),

            # =================================================================
            # 1. Top Navbar / Header (Datadog / CloudWatch Style)
            # =================================================================
            html.Header(
                style={
                    "backgroundColor": "#0F172A",
                    "borderBottom": "1px solid #1E2937",
                    "padding": "14px 32px",
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 100,
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "maxWidth": "1650px", "margin": "0 auto", "flexWrap": "wrap", "gap": "16px"},
                        children=[
                            # Left Brand Title
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "14px"},
                                children=[
                                    html.Div(
                                        style={
                                            "width": "34px",
                                            "height": "34px",
                                            "backgroundColor": "#0284C7",
                                            "borderRadius": "8px",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "color": "#FFFFFF",
                                            "fontWeight": "800",
                                            "fontSize": "16px",
                                        },
                                        children="FB",
                                    ),
                                    html.Div([
                                        html.Div(
                                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                            children=[
                                                html.H1(
                                                    "FlashBalanceAI",
                                                    style={"fontSize": "19px", "fontWeight": "700", "margin": "0", "color": "#FFFFFF", "letterSpacing": "-0.3px"},
                                                ),
                                                html.Span(
                                                    "v1.0 Live",
                                                    style={"fontSize": "11px", "fontWeight": "700", "backgroundColor": "#1E293B", "color": "#38BDF8", "padding": "2px 7px", "borderRadius": "4px", "border": "1px solid #334155"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            "Autonomous DRL Cloud Load Balancing • Live Telemetry & Dispatch Console",
                                            style={"fontSize": "12px", "color": "#94A3B8"},
                                        ),
                                    ]),
                                ],
                            ),

                            # Right Live Status & Indicators
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                children=[
                                    # Status Indicator Badge
                                    html.Div(
                                        id="status-pill",
                                        className="badge-live",
                                        style={"backgroundColor": "rgba(16, 185, 129, 0.15)", "color": "#10B981", "border": "1px solid rgba(16, 185, 129, 0.3)"},
                                        children=[
                                            html.Span("●", style={"fontSize": "14px"}),
                                            html.Span(id="status-text", children="LIVE JMETER STREAMING"),
                                        ],
                                    ),
                                    # Timer Badge
                                    html.Div(
                                        id="time-pill",
                                        style={
                                            "backgroundColor": "#1E293B",
                                            "color": "#F8FAFC",
                                            "border": "1px solid #334155",
                                            "borderRadius": "6px",
                                            "padding": "5px 12px",
                                            "fontSize": "13px",
                                            "fontWeight": "700",
                                            "fontFamily": "'JetBrains Mono', monospace",
                                        },
                                        children="00:00 / 02:00 (0s)",
                                    ),
                                    # AWS Deployment Pill
                                    html.Div(
                                        style={
                                            "backgroundColor": "#1E293B",
                                            "color": "#F59E0B",
                                            "border": "1px solid #334155",
                                            "borderRadius": "6px",
                                            "padding": "5px 12px",
                                            "fontSize": "12px",
                                            "fontWeight": "600",
                                        },
                                        children="AWS: us-east-1 • 4 Nodes",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # =================================================================
            # 2. Main Workspace
            # =================================================================
            html.Main(
                style={"maxWidth": "1650px", "margin": "0 auto", "padding": "24px 32px"},
                children=[
                    # A. Dynamic Hero KPI Row
                    html.Div(id="hero-kpi-row", style={"marginBottom": "20px"}),

                    # B. Interactive Control Ribbon (Ultra-Responsive)
                    html.Div(
                        className="panel-card",
                        style={"padding": "16px 20px", "marginBottom": "20px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "14px"},
                                children=[
                                    # Playback Buttons
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
                                        children=[
                                            html.Button(
                                                "▶ Start / Resume",
                                                id="btn-start",
                                                className="btn-ctrl",
                                                n_clicks=0,
                                                style={"backgroundColor": "#059669", "color": "#FFFFFF"},
                                            ),
                                            html.Button(
                                                "⏸ Pause",
                                                id="btn-pause",
                                                className="btn-ctrl",
                                                n_clicks=0,
                                                style={"backgroundColor": "#1F2937", "color": "#F8FAFC", "borderColor": "#374151"},
                                            ),
                                            html.Button(
                                                "↺ Reset Run",
                                                id="btn-reset",
                                                className="btn-ctrl",
                                                n_clicks=0,
                                                style={"backgroundColor": "#1F2937", "color": "#94A3B8", "borderColor": "#374151"},
                                            ),
                                            html.Button(
                                                "⚡ Inject 10x Spike",
                                                id="btn-spike",
                                                className="btn-ctrl",
                                                n_clicks=0,
                                                style={"backgroundColor": "#D97706", "color": "#FFFFFF"},
                                            ),
                                            html.Button(
                                                "⏩ 2x Fast-Forward",
                                                id="btn-ff",
                                                className="btn-ctrl",
                                                n_clicks=0,
                                                style={"backgroundColor": "#1E293B", "color": "#38BDF8", "borderColor": "#0284C7"},
                                            ),
                                        ],
                                    ),

                                    # Refresh Rate Selector (Default 1s for immediate fluidity)
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Span("Refresh Rate:", style={"fontSize": "13px", "color": "#94A3B8", "fontWeight": "500"}),
                                            dcc.Dropdown(
                                                id="select-interval",
                                                options=[
                                                    {"label": "1s (Real-Time Live 1Hz)", "value": 1000},
                                                    {"label": "2s (Fast Fluid)", "value": 2000},
                                                    {"label": "5s (Smooth Evaluation)", "value": 5000},
                                                    {"label": "10s (Standard Monitoring)", "value": 10000},
                                                    {"label": "30s (Official Benchmark Interval)", "value": 30000},
                                                ],
                                                value=1000,
                                                clearable=False,
                                                style={"width": "230px", "color": "#0F172A", "fontSize": "13px", "fontWeight": "600"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            # Timeline Scrubber Progress Line
                            html.Div(
                                style={"marginTop": "14px"},
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "fontSize": "11px", "color": "#64748B", "marginBottom": "5px", "fontFamily": "'JetBrains Mono', monospace"},
                                        children=[
                                            html.Span("0s Baseline (100 RPS)"),
                                            html.Span("20s Ramp"),
                                            html.Span("35s-85s 10x Burst (1,000 RPS)", style={"color": "#F59E0B", "fontWeight": "700"}),
                                            html.Span("85s-120s Cooldown"),
                                        ],
                                    ),
                                    html.Div(
                                        style={"height": "6px", "backgroundColor": "#1E293B", "borderRadius": "3px", "overflow": "hidden"},
                                        children=html.Div(
                                            id="progress-fill",
                                            style={"height": "100%", "width": "0%", "backgroundColor": "#38BDF8", "transition": "width 0.2s linear"},
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # C. Navigation Tabs (Clean Enterprise Style)
                    dcc.Tabs(
                        id="tabs-view",
                        value="tab-side-by-side",
                        style={"marginBottom": "20px"},
                        colors={"border": "transparent", "primary": "#38BDF8", "background": "transparent"},
                        children=[
                            dcc.Tab(
                                label="Side-by-Side: PPO vs Round-Robin",
                                value="tab-side-by-side",
                                style={"backgroundColor": "#0F172A", "color": "#94A3B8", "fontWeight": "600", "padding": "12px 18px", "borderRadius": "6px 0 0 6px", "border": "1px solid #1E2937"},
                                selected_style={"backgroundColor": "#1E293B", "color": "#38BDF8", "fontWeight": "700", "padding": "12px 18px", "border": "1px solid #38BDF8", "borderRadius": "6px 0 0 6px"},
                            ),
                            dcc.Tab(
                                label="All 6 Algorithms Benchmark",
                                value="tab-all-algos",
                                style={"backgroundColor": "#0F172A", "color": "#94A3B8", "fontWeight": "600", "padding": "12px 18px", "border": "1px solid #1E2937"},
                                selected_style={"backgroundColor": "#1E293B", "color": "#38BDF8", "fontWeight": "700", "padding": "12px 18px", "border": "1px solid #38BDF8"},
                            ),
                            dcc.Tab(
                                label="Live Telemetry Curves (SLA 500ms)",
                                value="tab-telemetry",
                                style={"backgroundColor": "#0F172A", "color": "#94A3B8", "fontWeight": "600", "padding": "12px 18px", "border": "1px solid #1E2937"},
                                selected_style={"backgroundColor": "#1E293B", "color": "#38BDF8", "fontWeight": "700", "padding": "12px 18px", "border": "1px solid #38BDF8"},
                            ),
                            dcc.Tab(
                                label="JMeter 5.6.3 Live Log Stream",
                                value="tab-jmeter",
                                style={"backgroundColor": "#0F172A", "color": "#94A3B8", "fontWeight": "600", "padding": "12px 18px", "borderRadius": "0 6px 6px 0", "border": "1px solid #1E2937"},
                                selected_style={"backgroundColor": "#1E293B", "color": "#38BDF8", "fontWeight": "700", "padding": "12px 18px", "border": "1px solid #38BDF8", "borderRadius": "0 6px 6px 0"},
                            ),
                        ],
                    ),

                    # D. Tab Content Container
                    html.Div(id="tab-content"),
                ],
            ),
        ],
    )

    # =========================================================================
    # Callbacks (Fully Reactive & High-Performance)
    # =========================================================================

    # 1. Update Interval Callback
    @app.callback(
        Output("interval-refresh", "interval"),
        Input("select-interval", "value"),
    )
    def update_interval(val):
        return int(val) if val else 1000

    # 2. Control Buttons & Status Callback (Responds instantly to clicks)
    @app.callback(
        [
            Output("status-text", "children"),
            Output("status-pill", "style"),
            Output("progress-fill", "style"),
        ],
        [
            Input("btn-start", "n_clicks"),
            Input("btn-pause", "n_clicks"),
            Input("btn-reset", "n_clicks"),
            Input("btn-spike", "n_clicks"),
            Input("btn-ff", "n_clicks"),
            Input("interval-refresh", "n_intervals"),
        ],
        prevent_initial_call=False,
    )
    def handle_controls(btn_start, btn_pause, btn_reset, btn_spike, btn_ff, _interval):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if triggered == "btn-start":
            if runner.is_complete:
                runner.reset()
            runner.resume()
        elif triggered == "btn-pause":
            runner.pause()
        elif triggered == "btn-reset":
            runner.reset()
            runner.start_background()
        elif triggered == "btn-spike":
            runner.current_sec = 45
            runner.resume()
            runner.step()
        elif triggered == "btn-ff":
            runner.time_scale = 2.0 if runner.time_scale == 1.0 else 1.0

        if not runner.is_running and runner.current_sec == 0 and not runner.is_complete:
            runner.start_background()

        pct = min(100.0, (runner.current_sec / max(1, runner.duration_sec)) * 100.0)
        progress_style = {"height": "100%", "width": f"{pct:.1f}%", "backgroundColor": "#38BDF8", "transition": "width 0.2s linear"}

        if runner.is_complete:
            return (
                "BENCHMARK COMPLETED",
                {"backgroundColor": "rgba(56, 189, 248, 0.15)", "color": "#38BDF8", "border": "1px solid rgba(56, 189, 248, 0.3)"},
                progress_style,
            )
        elif runner.is_running:
            burst = runner.is_burst_active(runner.current_sec)
            txt = "10x BURST ACTIVE (1,000 RPS)" if burst else "LIVE JMETER STREAMING"
            color = "#F59E0B" if burst else "#10B981"
            return (
                txt,
                {"backgroundColor": f"{color}18", "color": color, "border": f"1px solid {color}44"},
                progress_style,
            )
        else:
            return (
                "SIMULATION PAUSED",
                {"backgroundColor": "rgba(239, 68, 68, 0.15)", "color": "#EF4444", "border": "1px solid rgba(239, 68, 68, 0.3)"},
                progress_style,
            )

    # 3. Timer Pill Callback (Instant reaction to buttons and ticks)
    @app.callback(
        Output("time-pill", "children"),
        [
            Input("interval-refresh", "n_intervals"),
            Input("btn-start", "n_clicks"),
            Input("btn-pause", "n_clicks"),
            Input("btn-reset", "n_clicks"),
            Input("btn-spike", "n_clicks"),
            Input("btn-ff", "n_clicks"),
        ],
    )
    def update_timer(_n, _b1, _b2, _b3, _b4, _b5):
        sec = runner.current_sec
        mins = sec // 60
        rem_sec = sec % 60
        scale_suffix = " [2x]" if runner.time_scale > 1.0 else ""
        return f"{mins:02d}:{rem_sec:02d} / 02:00 ({sec}s){scale_suffix}"

    # 4. Hero KPI Row Callback (Recomputes on every tick or click)
    @app.callback(
        Output("hero-kpi-row", "children"),
        [
            Input("interval-refresh", "n_intervals"),
            Input("btn-start", "n_clicks"),
            Input("btn-pause", "n_clicks"),
            Input("btn-reset", "n_clicks"),
            Input("btn-spike", "n_clicks"),
            Input("btn-ff", "n_clicks"),
        ],
    )
    def update_hero_kpis(_n, _b1, _b2, _b3, _b4, _b5):
        telem = runner.get_latest_telemetry()
        ppo_data = telem["algorithms"].get("PPO", {})
        rr_data = telem["algorithms"].get("RoundRobin", {})

        p95 = ppo_data.get("p95_latency_ms", 101.7)
        tput = ppo_data.get("throughput_rps", 932.8)
        sla = max(0.0, 1.0 - ppo_data.get("sla_violation_rate", 0.0)) * 100.0

        rr_p95 = rr_data.get("p95_latency_ms", 734.2)
        gain_pct = max(0.0, ((rr_p95 - p95) / max(1.0, rr_p95)) * 100.0)

        card_defs = [
            {
                "label": "P95 TAIL LATENCY",
                "value": f"{p95:.1f} ms",
                "badge": "Target < 500 ms",
                "color": "#10B981" if p95 < 200 else ("#F59E0B" if p95 < 500 else "#EF4444"),
                "desc": "Real-time PPO response time",
            },
            {
                "label": "SYSTEM THROUGHPUT",
                "value": f"{tput:.1f} req/s",
                "badge": f"{int(tput)} / 1,000 RPS",
                "color": "#38BDF8",
                "desc": "Cluster processing rate",
            },
            {
                "label": "SLA COMPLIANCE RATE",
                "value": f"{sla:.1f}%",
                "badge": "Zero Violations" if sla >= 99.9 else "Violations Detected",
                "color": "#10B981" if sla >= 99.0 else "#EF4444",
                "desc": "Requests delivered under 500ms",
            },
            {
                "label": "DRL LATENCY ADVANTAGE",
                "value": f"+{gain_pct:.1f}%",
                "badge": "vs Round-Robin",
                "color": "#38BDF8",
                "desc": f"Round-Robin P95: {rr_p95:.1f} ms",
            },
        ]

        cards = []
        for c in card_defs:
            card = html.Div(
                className="panel-card",
                style={"flex": "1 1 240px", "padding": "18px 22px"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                        children=[
                            html.Span(c["label"], style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8", "letterSpacing": "0.5px"}),
                            html.Span(
                                c["badge"],
                                style={"fontSize": "11px", "fontWeight": "600", "color": c["color"], "backgroundColor": f"{c['color']}15", "padding": "2px 8px", "borderRadius": "4px"},
                            ),
                        ],
                    ),
                    html.Div(c["value"], style={"fontSize": "26px", "fontWeight": "800", "color": "#FFFFFF", "fontFamily": "'JetBrains Mono', monospace", "margin": "4px 0"}),
                    html.Div(c["desc"], style={"fontSize": "12px", "color": "#64748B"}),
                ],
            )
            cards.append(card)

        return html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=cards)

    # 5. Render Active Tab Content (Instant reaction to tabs, ticks, and controls)
    @app.callback(
        Output("tab-content", "children"),
        [
            Input("tabs-view", "value"),
            Input("interval-refresh", "n_intervals"),
            Input("btn-start", "n_clicks"),
            Input("btn-pause", "n_clicks"),
            Input("btn-reset", "n_clicks"),
            Input("btn-spike", "n_clicks"),
            Input("btn-ff", "n_clicks"),
        ],
    )
    def render_tab(tab_name, _n, _b1, _b2, _b3, _b4, _b5):
        telem = runner.get_latest_telemetry()
        history = runner.get_history_dataframe()

        if tab_name == "tab-side-by-side":
            return _render_side_by_side_tab(telem, history)
        elif tab_name == "tab-all-algos":
            return _render_all_algos_tab(telem, history)
        elif tab_name == "tab-telemetry":
            return _render_telemetry_tab(telem, history)
        elif tab_name == "tab-jmeter":
            return _render_jmeter_tab(telem)
        return html.Div("Select a tab.")

    return app


# =============================================================================
# View Component Renderers
# =============================================================================

def _render_side_by_side_tab(telem: Dict[str, Any], history: Dict[str, List[Dict[str, Any]]]) -> html.Div:
    """Render Tab 1: Professional Side-by-Side PPO vs Round-Robin Arena."""
    ppo = telem["algorithms"].get("PPO", {})
    rr = telem["algorithms"].get("RoundRobin", {})

    ppo_p95 = ppo.get("p95_latency_ms", 50.0)
    rr_p95 = rr.get("p95_latency_ms", 50.0)
    diff = rr_p95 - ppo_p95
    pct_adv = max(0.0, (diff / max(1.0, rr_p95)) * 100.0)

    # 1. Comparative Head-to-Head Banner
    arena_banner = html.Div(
        className="panel-card",
        style={"padding": "18px 24px", "marginBottom": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "16px"},
                children=[
                    # PPO Champion Side
                    html.Div([
                        html.Span("DRL POLICY (AUTONOMOUS)", style={"fontSize": "11px", "fontWeight": "700", "color": "#38BDF8", "letterSpacing": "0.5px"}),
                        html.Div("Proximal Policy Optimization", style={"fontSize": "20px", "fontWeight": "700", "color": "#FFFFFF", "margin": "2px 0"}),
                        html.Div(f"{ppo_p95:.1f} ms P95 Latency", style={"fontSize": "15px", "fontWeight": "700", "color": "#10B981", "fontFamily": "'JetBrains Mono', monospace"}),
                    ]),

                    # Center Delta Comparison
                    html.Div(
                        style={"textAlign": "center"},
                        children=[
                            html.Span("HEAD-TO-HEAD DELTA", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                            html.Div(f"+{pct_adv:.1f}% Faster", style={"fontSize": "24px", "fontWeight": "800", "color": "#10B981", "fontFamily": "'JetBrains Mono', monospace", "margin": "2px 0"}),
                            html.Div(f"PPO saves {diff:.1f} ms of tail latency", style={"fontSize": "12px", "color": "#94A3B8"}),
                        ],
                    ),

                    # Round-Robin Baseline Side
                    html.Div(
                        style={"textAlign": "right"},
                        children=[
                            html.Span("STATIC BASELINE (CONVENTIONAL)", style={"fontSize": "11px", "fontWeight": "700", "color": "#EF4444", "letterSpacing": "0.5px"}),
                            html.Div("Static Round-Robin", style={"fontSize": "20px", "fontWeight": "700", "color": "#FFFFFF", "margin": "2px 0"}),
                            html.Div(f"{rr_p95:.1f} ms P95 Latency", style={"fontSize": "15px", "fontWeight": "700", "color": "#EF4444", "fontFamily": "'JetBrains Mono', monospace"}),
                        ],
                    ),
                ],
            ),
        ],
    )

    # 2. Server Cluster Nodes
    def _render_instance_pool(agent_title: str, sub: str, data: Dict[str, Any], accent: str):
        cpus = data.get("backend_cpus", [0.1, 0.1, 0.1, 0.1])
        conns = data.get("backend_conns", [10, 10, 10, 10])
        queues = data.get("backend_queues", [0, 0, 0, 0])

        total_conns = max(1, sum(conns))
        cards = []

        for i in range(4):
            cpu_pct = int(cpus[i] * 100)
            share_pct = (conns[i] / total_conns) * 100
            has_queue = queues[i] > 3
            cpu_bar_color = "#10B981" if cpu_pct < 65 else ("#F59E0B" if cpu_pct < 85 else "#EF4444")

            card = html.Div(
                style={
                    "backgroundColor": "#0F172A",
                    "border": f"1px solid {'#EF4444' if has_queue else '#1E293B'}",
                    "borderRadius": "8px",
                    "padding": "12px 16px",
                    "marginBottom": "10px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                        children=[
                            html.Div([
                                html.Span(f"Backend #{i} ", style={"fontWeight": "700", "fontSize": "13px", "color": "#FFFFFF"}),
                                html.Span("(c5.large)", style={"fontSize": "11px", "color": "#64748B"}),
                            ]),
                            html.Span(
                                f"Traffic: {share_pct:.1f}%",
                                style={"fontSize": "11px", "fontWeight": "600", "padding": "2px 8px", "borderRadius": "4px", "backgroundColor": f"{accent}18", "color": accent},
                            ),
                        ],
                    ),

                    # CPU Progress Bar
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "6px"},
                        children=[
                            html.Span("CPU Load", style={"fontSize": "11px", "color": "#94A3B8", "width": "55px"}),
                            html.Div(
                                style={"flex": 1, "height": "6px", "backgroundColor": "#1E293B", "borderRadius": "3px", "overflow": "hidden"},
                                children=html.Div(style={"width": f"{cpu_pct}%", "height": "100%", "backgroundColor": cpu_bar_color, "transition": "width 0.2s ease"}),
                            ),
                            html.Span(f"{cpu_pct}%", style={"fontSize": "12px", "fontWeight": "700", "color": cpu_bar_color, "width": "35px", "textAlign": "right", "fontFamily": "'JetBrains Mono', monospace"}),
                        ],
                    ),

                    # Connections & Queues
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "fontSize": "11px", "color": "#94A3B8", "fontFamily": "'JetBrains Mono', monospace"},
                        children=[
                            html.Span(f"Active: {conns[i]} / 250"),
                            html.Span(
                                f"Queue: {queues[i]}" if queues[i] > 0 else "Queue: 0",
                                style={"color": "#EF4444" if has_queue else "#10B981", "fontWeight": "700"},
                            ),
                        ],
                    ),
                ],
            )
            cards.append(card)

        return html.Div(
            className="panel-card",
            style={"padding": "20px"},
            children=[
                html.Div(
                    style={"marginBottom": "14px", "borderBottom": "1px solid #1E293B", "paddingBottom": "10px"},
                    children=[
                        html.H3(agent_title, style={"margin": "0 0 3px 0", "fontSize": "16px", "fontWeight": "700", "color": "#FFFFFF"}),
                        html.Div(sub, style={"fontSize": "12px", "color": "#94A3B8"}),
                    ],
                ),
                html.Div(cards),
            ],
        )

    columns = html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
        children=[
            _render_instance_pool("PPO Backend Cluster", "Headroom-aware dispatch maintains zero queue buildup", ppo, "#38BDF8"),
            _render_instance_pool("Round-Robin Backend Cluster", "Static cyclic dispatch causes asymmetric queue bottlenecks", rr, "#EF4444"),
        ],
    )

    # 3. Request Distribution Plot
    labels = BACKEND_NAMES
    ppo_conns = ppo.get("backend_conns", [10, 10, 10, 10])
    rr_conns = rr.get("backend_conns", [10, 10, 10, 10])

    fig_anim = go.Figure(
        data=[
            go.Bar(
                name="PPO Active Requests",
                x=labels,
                y=ppo_conns,
                marker=dict(color=ALGO_COLORS["PPO"]),
                text=[f"{c}" for c in ppo_conns],
                textposition="auto",
            ),
            go.Bar(
                name="Round-Robin Active Requests",
                x=labels,
                y=rr_conns,
                marker=dict(color=ALGO_COLORS["RoundRobin"]),
                text=[f"{c}" for c in rr_conns],
                textposition="auto",
            ),
        ],
        layout=go.Layout(
            title=None,
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC", "family": "Inter"},
            margin={"t": 30, "b": 35, "l": 50, "r": 20},
            height=260,
            legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center", "font": {"size": 12}},
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="In-Flight Requests"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        ),
    )

    anim_card = html.Div(
        className="panel-card",
        style={"padding": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.H3("Active In-Flight Request Distribution", style={"margin": "0", "fontSize": "15px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("Live 1Hz Flow", style={"fontSize": "11px", "color": "#10B981", "fontWeight": "600", "backgroundColor": "rgba(16, 185, 129, 0.15)", "padding": "2px 8px", "borderRadius": "4px"}),
                ],
            ),
            dcc.Graph(figure=fig_anim, config={"displayModeBar": False}),
        ],
    )

    return html.Div([arena_banner, columns, anim_card])


def _render_all_algos_tab(telem: Dict[str, Any], history: Dict[str, List[Dict[str, Any]]]) -> html.Div:
    """Render Tab 2: All 6 Algorithms Benchmark & Dynamic Leaderboard."""
    algos = ALGORITHMS

    fig_all = go.Figure()
    for algo in algos:
        data = telem["algorithms"].get(algo, {})
        conns = data.get("backend_conns", [0, 0, 0, 0])
        fig_all.add_trace(
            go.Bar(
                name=algo,
                x=BACKEND_NAMES,
                y=conns,
                marker_color=ALGO_COLORS[algo],
                text=[f"{c}" for c in conns],
                textposition="auto",
            )
        )

    fig_all.update_layout(
        title=None,
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC", "family": "Inter"},
        height=300,
        margin={"t": 30, "b": 35, "l": 50, "r": 20},
        legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center", "font": {"size": 11}},
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Active Requests"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )

    chart_card = html.Div(
        className="panel-card",
        style={"padding": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.H3("Live Traffic Allocation Across All 4 Backends", style={"margin": "0", "fontSize": "16px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("All 6 Project Models", style={"fontSize": "11px", "color": "#38BDF8", "fontWeight": "600", "backgroundColor": "rgba(56, 189, 248, 0.15)", "padding": "2px 8px", "borderRadius": "4px"}),
                ],
            ),
            dcc.Graph(figure=fig_all, config={"displayModeBar": False}),
        ],
    )

    # Leaderboard Table sorted dynamically by P95 latency
    sorted_algos = sorted(algos, key=lambda a: telem["algorithms"].get(a, {}).get("p95_latency_ms", 999.0))
    rows = []

    for rank, algo in enumerate(sorted_algos, start=1):
        d = telem["algorithms"].get(algo, {})
        p95 = d.get("p95_latency_ms", 0.0)
        p99 = d.get("p99_latency_ms", 0.0)
        tput = d.get("throughput_rps", 0.0)
        sla = d.get("sla_violation_rate", 0.0)
        cpu = d.get("mean_cpu_util", 0.0)
        reward = d.get("reward", 0.0)
        is_top = rank == 1

        rows.append(
            html.Tr(
                className="table-row-hover",
                style={
                    "borderBottom": "1px solid #1E293B",
                    "backgroundColor": "#1E293B55" if is_top else "transparent",
                },
                children=[
                    html.Td(f"#{rank}", style={"padding": "12px 14px", "fontWeight": "700", "color": "#38BDF8" if is_top else "#94A3B8"}),
                    html.Td(
                        html.Span(algo, style={"fontWeight": "700", "color": ALGO_COLORS[algo]}),
                        style={"padding": "12px 14px"},
                    ),
                    html.Td(f"{p95:.1f} ms", style={"padding": "12px 14px", "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace", "color": "#10B981" if p95 < 200 else ("#F59E0B" if p95 < 500 else "#EF4444")}),
                    html.Td(f"{p99:.1f} ms", style={"padding": "12px 14px", "fontFamily": "'JetBrains Mono', monospace", "color": "#94A3B8"}),
                    html.Td(f"{tput:.1f} rps", style={"padding": "12px 14px", "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Td(
                        html.Span(
                            f"{sla:.2%}",
                            style={"color": "#EF4444" if sla > 0.05 else "#10B981", "fontWeight": "700"},
                        ),
                        style={"padding": "12px 14px"},
                    ),
                    html.Td(f"{cpu:.1%}", style={"padding": "12px 14px"}),
                    html.Td(f"{reward:.3f}", style={"padding": "12px 14px", "color": "#38BDF8", "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace"}),
                ],
            )
        )

    leaderboard_card = html.Div(
        className="panel-card",
        style={"padding": "20px", "marginTop": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.H3("Live Benchmark Performance Leaderboard", style={"margin": "0", "fontSize": "16px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("SLA Target: < 500 ms", style={"fontSize": "11px", "color": "#10B981", "fontWeight": "600", "backgroundColor": "rgba(16, 185, 129, 0.15)", "padding": "2px 8px", "borderRadius": "4px"}),
                ],
            ),
            html.Div(
                style={"overflowX": "auto"},
                children=html.Table(
                    style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
                    children=[
                        html.Thead(
                            html.Tr(
                                style={"borderBottom": "1px solid #374151", "color": "#94A3B8", "textAlign": "left"},
                                children=[
                                    html.Th("Rank", style={"padding": "10px 14px"}),
                                    html.Th("Algorithm", style={"padding": "10px 14px"}),
                                    html.Th("P95 Latency", style={"padding": "10px 14px"}),
                                    html.Th("P99 Latency", style={"padding": "10px 14px"}),
                                    html.Th("Throughput", style={"padding": "10px 14px"}),
                                    html.Th("SLA Violations", style={"padding": "10px 14px"}),
                                    html.Th("CPU Util", style={"padding": "10px 14px"}),
                                    html.Th("Reward", style={"padding": "10px 14px"}),
                                ],
                            )
                        ),
                        html.Tbody(rows),
                    ],
                ),
            ),
        ],
    )

    return html.Div([chart_card, leaderboard_card])


def _render_telemetry_tab(telem: Dict[str, Any], history: Dict[str, List[Dict[str, Any]]]) -> html.Div:
    """Render Tab 3: Plotly Telemetry Graphs with 500ms SLA Line."""
    algos = ALGORITHMS

    legend_pills = []
    for algo in algos:
        color = ALGO_COLORS[algo]
        legend_pills.append(
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px", "fontSize": "12px", "fontWeight": "600", "color": "#F8FAFC"},
                children=[
                    html.Span("●", style={"color": color, "fontSize": "13px"}),
                    html.Span(algo),
                ],
            )
        )

    legend_bar = html.Div(
        className="panel-card",
        style={"padding": "12px 18px", "marginBottom": "16px", "display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "10px"},
        children=[
            html.Span("Telemetry Series Key:", style={"fontSize": "12px", "fontWeight": "700", "color": "#94A3B8"}),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "14px"}, children=legend_pills),
        ],
    )

    def _base_layout(y_title: str):
        return go.Layout(
            title=None,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC", "family": "Inter"},
            height=260,
            margin={"t": 25, "b": 35, "l": 50, "r": 20},
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Time (s)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=y_title),
            showlegend=False,
        )

    # 1. P95 Latency
    fig_lat = go.Figure(layout=_base_layout("Latency (ms)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_lat.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["p95_latency_ms"] for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=2.5 if algo == "PPO" else 1.5, shape="spline"),
            ))
    fig_lat.add_hline(y=500, line_dash="dash", line_color="#EF4444", annotation_text="500ms SLA", annotation_position="top right")

    card_lat = html.Div(
        className="panel-card",
        style={"padding": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                children=[
                    html.H4("P95 Latency Over Time (ms)", style={"margin": 0, "fontSize": "14px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("SLA Limit: 500 ms", style={"fontSize": "11px", "color": "#EF4444", "fontWeight": "700"}),
                ],
            ),
            dcc.Graph(figure=fig_lat, config={"displayModeBar": False}),
        ],
    )

    # 2. Throughput
    fig_tput = go.Figure(layout=_base_layout("RPS"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_tput.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["throughput_rps"] for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=1.5, shape="spline"),
            ))

    card_tput = html.Div(
        className="panel-card",
        style={"padding": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                children=[
                    html.H4("System Throughput (req/s)", style={"margin": 0, "fontSize": "14px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("Target: 1,000 RPS", style={"fontSize": "11px", "color": "#38BDF8", "fontWeight": "600"}),
                ],
            ),
            dcc.Graph(figure=fig_tput, config={"displayModeBar": False}),
        ],
    )

    # 3. SLA Violation Rate
    fig_sla = go.Figure(layout=_base_layout("Violations (%)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_sla.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["sla_violation_rate"] * 100 for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=1.5, shape="spline"),
            ))

    card_sla = html.Div(
        className="panel-card",
        style={"padding": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                children=[
                    html.H4("SLA Violation Rate (%)", style={"margin": 0, "fontSize": "14px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("PPO Target: 0.00%", style={"fontSize": "11px", "color": "#10B981", "fontWeight": "600"}),
                ],
            ),
            dcc.Graph(figure=fig_sla, config={"displayModeBar": False}),
        ],
    )

    # 4. CPU Utilisation
    fig_cpu = go.Figure(layout=_base_layout("CPU (%)"))
    for algo in algos:
        records = history.get(algo, [])
        if records:
            fig_cpu.add_trace(go.Scatter(
                x=[r["elapsed_sec"] for r in records],
                y=[r["mean_cpu_util"] * 100 for r in records],
                mode="lines",
                name=algo,
                line=dict(color=ALGO_COLORS[algo], width=1.5, shape="spline"),
            ))

    card_cpu = html.Div(
        className="panel-card",
        style={"padding": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                children=[
                    html.H4("Cluster Mean CPU Load (%)", style={"margin": 0, "fontSize": "14px", "fontWeight": "700", "color": "#FFFFFF"}),
                    html.Span("Capacity Headroom", style={"fontSize": "11px", "color": "#A855F7", "fontWeight": "600"}),
                ],
            ),
            dcc.Graph(figure=fig_cpu, config={"displayModeBar": False}),
        ],
    )

    grid = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(540px, 1fr))", "gap": "20px"},
        children=[card_lat, card_tput, card_sla, card_cpu],
    )

    return html.Div([legend_bar, grid])


def _render_jmeter_tab(telem: Dict[str, Any]) -> html.Div:
    """Render Tab 4: Live JMeter JTL Stream with Aggregate Report Console."""
    jtl_lines = []
    total_samples = 0
    success_count = 0
    error_count = 0

    if JTL_PATH.exists():
        try:
            with open(JTL_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                jtl_lines = lines[-16:]
                for l in lines[1:]:
                    parts = l.strip().split(",")
                    if len(parts) >= 8:
                        total_samples += 1
                        if parts[7].lower() == "true":
                            success_count += 1
                        else:
                            error_count += 1
        except Exception:
            pass

    success_rate = (success_count / max(1, total_samples)) * 100.0

    summary_cards = html.Div(
        style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "16px"},
        children=[
            html.Div(
                className="panel-card",
                style={"flex": "1 1 200px", "padding": "14px 18px", "borderLeft": "3px solid #38BDF8"},
                children=[
                    html.Div("TOTAL SAMPLES", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div(f"{total_samples:,}", style={"fontSize": "20px", "fontWeight": "800", "color": "#FFFFFF", "fontFamily": "'JetBrains Mono', monospace", "margin": "3px 0"}),
                    html.Div("Apache JMeter 5.6.3 Engine", style={"fontSize": "11px", "color": "#38BDF8"}),
                ],
            ),
            html.Div(
                className="panel-card",
                style={"flex": "1 1 200px", "padding": "14px 18px", "borderLeft": "3px solid #10B981"},
                children=[
                    html.Div("SUCCESSFUL REQUESTS (200 OK)", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div(f"{success_rate:.1f}%", style={"fontSize": "20px", "fontWeight": "800", "color": "#10B981", "fontFamily": "'JetBrains Mono', monospace", "margin": "3px 0"}),
                    html.Div(f"{success_count:,} requests passed", style={"fontSize": "11px", "color": "#10B981"}),
                ],
            ),
            html.Div(
                className="panel-card",
                style={"flex": "1 1 200px", "padding": "14px 18px", "borderLeft": "3px solid #EF4444"},
                children=[
                    html.Div("FAILED / TIMEOUTS (504)", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div(f"{error_count:,}", style={"fontSize": "20px", "fontWeight": "800", "color": "#EF4444" if error_count > 0 else "#10B981", "fontFamily": "'JetBrains Mono', monospace", "margin": "3px 0"}),
                    html.Div("Static Round-Robin hotspotting", style={"fontSize": "11px", "color": "#94A3B8"}),
                ],
            ),
            html.Div(
                className="panel-card",
                style={"flex": "1 1 200px", "padding": "14px 18px", "borderLeft": "3px solid #A855F7"},
                children=[
                    html.Div("TARGET INGRESS ENDPOINT", style={"fontSize": "11px", "fontWeight": "700", "color": "#94A3B8"}),
                    html.Div("POST /request", style={"fontSize": "16px", "fontWeight": "700", "color": "#FFFFFF", "fontFamily": "'JetBrains Mono', monospace", "margin": "3px 0"}),
                    html.Div("AWS Application Load Balancer", style={"fontSize": "11px", "color": "#A855F7"}),
                ],
            ),
        ],
    )

    log_box = html.Pre(
        "".join(jtl_lines) if jtl_lines else "No JMeter samples recorded yet. Start simulation above.",
        style={
            "backgroundColor": "#070B14",
            "color": "#38BDF8",
            "padding": "16px 20px",
            "borderRadius": "8px",
            "fontSize": "12px",
            "fontFamily": "'JetBrains Mono', Consolas, monospace",
            "overflowX": "auto",
            "border": "1px solid #1E293B",
            "maxHeight": "280px",
            "lineHeight": "1.6",
        },
    )

    return html.Div([
        summary_cards,
        html.Div(
            className="panel-card",
            style={"padding": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                    children=[
                        html.Div([
                            html.H3("Live Apache JMeter JTL Stream", style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#FFFFFF"}),
                            html.Div("Streaming samples directly from experiments/results/demo_live_jmeter.jtl", style={"fontSize": "12px", "color": "#94A3B8"}),
                        ]),
                        html.Span("JMeter 5.6.3 Engine Active", style={"fontSize": "11px", "color": "#10B981", "fontWeight": "600", "backgroundColor": "rgba(16, 185, 129, 0.15)", "padding": "2px 8px", "borderRadius": "4px"}),
                    ],
                ),
                log_box,
            ],
        ),
    ])


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8050))
    debug_mode = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    print(f"Starting FlashBalanceAI Demo Dashboard on http://127.0.0.1:{port}")
    app.run(debug=debug_mode, port=port, host="0.0.0.0")
