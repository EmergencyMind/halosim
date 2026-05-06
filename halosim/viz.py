"""
viz.py — Plotly chart builders for HaloSim.

All functions return a plotly Figure object for use with st.plotly_chart().
No scipy dependency — Plotly violin handles KDE internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (accessible)
# ---------------------------------------------------------------------------

_BLUE = "#2563EB"
_ORANGE = "#EA580C"
_GREEN = "#16A34A"
_GREY = "#94A3B8"


# ---------------------------------------------------------------------------
# 1. Gap distribution — violin + box per metric
# ---------------------------------------------------------------------------

def plot_gap_distribution(results_df: pd.DataFrame) -> go.Figure:
    """
    Violin + box plots for gap_min, gap_median, gap_max across providers.
    """
    metrics = {
        "gap_min":    "Minimum gap",
        "gap_median": "Median gap",
        "gap_max":    "Maximum gap",
    }

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(metrics.values()),
        shared_yaxes=True,
    )

    for col_i, (col, label) in enumerate(metrics.items(), start=1):
        vals = results_df[col].dropna().values
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=vals,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=_BLUE,
                line_color=_BLUE,
                opacity=0.7,
                showlegend=False,
            ),
            row=1, col=col_i,
        )

    fig.update_layout(
        title="Gap distribution across providers (includes run-in and run-out periods)",
        yaxis_title="Days",
        height=420,
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#E2E8F0", rangemode="tozero")
    return fig


# ---------------------------------------------------------------------------
# 2. Baseline readiness timeseries (on-shift providers only)
# ---------------------------------------------------------------------------

def plot_readiness_baseline(
    prop_on_shift: np.ndarray,
    n_days: int,
    rolling_days: int = 30,
    start_date: str = "2024-01-01",
) -> go.Figure:
    """
    Readiness over time for on-shift providers with no training program.
    Shows what the exposure pattern alone produces.
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    smoothed = (
        pd.Series(prop_on_shift)
        .rolling(rolling_days, min_periods=1, center=True)
        .mean()
        .values * 100
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=smoothed,
        mode="lines",
        name="On-shift readiness",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.1)",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"On-shift readiness over time — exposure only ({rolling_days}-day rolling mean)",
        xaxis_title="Date",
        yaxis_title="% ready",
        yaxis=dict(range=[0, 105]),
        height=340,
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


# ---------------------------------------------------------------------------
# 3. Threshold sweep — % exceeding X-day gap threshold
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    results_df: pd.DataFrame,
    thresholds: list[int] | None = None,
    threshold: int = 90,
) -> go.Figure:
    """
    Line chart: % of providers whose maximum inter-exposure gap exceeds threshold,
    swept across a range of threshold values (7 to 365 days).
    """
    if thresholds is None:
        thresholds = list(range(7, 366, 7))

    max_gaps = results_df["gap_max"].fillna(9999).values
    n = len(max_gaps)

    pct = [100 * (max_gaps > t).sum() / n for t in thresholds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=pct,
        mode="lines",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.1)",
        name="% exceeding threshold",
    ))

    # Reference line at readiness threshold
    fig.add_vline(
        x=threshold, line_dash="dash", line_color=_ORANGE,
        annotation_text=f"{threshold} days", annotation_position="top right",
    )

    fig.update_layout(
        title="% of providers with maximum gap exceeding threshold",
        xaxis_title="Threshold (days)",
        yaxis_title="% of providers",
        yaxis=dict(range=[0, 105]),
        height=380,
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


# ---------------------------------------------------------------------------
# 3. Readiness time series — with vs without training
# ---------------------------------------------------------------------------

def plot_readiness_timeseries(
    prop_baseline: np.ndarray,
    prop_trained: np.ndarray,
    n_days: int,
    rolling_days: int = 30,
    start_date: str = "2024-01-01",
) -> go.Figure:
    """
    Line chart of proportion of on-shift providers who are 'ready',
    comparing baseline (no training) vs with training program.
    Includes optional rolling mean smoothing.
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    def smooth(arr):
        s = pd.Series(arr).rolling(rolling_days, min_periods=1, center=True).mean()
        return s.values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=smooth(prop_baseline) * 100,
        mode="lines",
        name="No training",
        line=dict(color=_GREY, width=1.5, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=smooth(prop_trained) * 100,
        mode="lines",
        name="With training",
        line=dict(color=_GREEN, width=2),
    ))

    fig.update_layout(
        title=f"Population readiness over time ({rolling_days}-day rolling mean, on-shift providers)",
        xaxis_title="Date",
        yaxis_title="% ready",
        yaxis=dict(range=[0, 105]),
        height=400,
        legend=dict(x=0.01, y=0.05),
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


# ---------------------------------------------------------------------------
# 4. Individual provider swimlane heatmap
# ---------------------------------------------------------------------------

def plot_individual_swimlanes(
    exposure_matrix: np.ndarray,
    providers: list[str],
    sample_n: int = 30,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> go.Figure:
    """
    Heatmap showing exposure events (white = no exposure, blue = exposed)
    for a random sample of providers. Rows = providers, columns = days.
    """
    rng = np.random.default_rng(seed)
    n = min(sample_n, len(providers))
    idx = rng.choice(len(providers), size=n, replace=False)
    idx = np.sort(idx)

    submat = exposure_matrix[idx].astype(float)
    sub_labels = [providers[i] for i in idx]

    dates = pd.date_range(start_date, periods=exposure_matrix.shape[1], freq="D")
    # Downsample columns for display if > 365 days
    if len(dates) > 365:
        step = len(dates) // 365
        submat = submat[:, ::step]
        dates = dates[::step]

    fig = go.Figure(go.Heatmap(
        z=submat,
        x=dates,
        y=sub_labels,
        colorscale=[[0, "white"], [1, _BLUE]],
        showscale=False,
        xgap=0,
        ygap=1,
    ))

    fig.update_layout(
        title=f"Exposure events — sample of {n} providers",
        xaxis_title="Date",
        yaxis_title="Provider",
        height=max(300, n * 18),
        margin=dict(t=60, b=40, l=100),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Training program comparison
# ---------------------------------------------------------------------------

_PROGRAM_COLORS = {
    "No training":       _GREY,
    "Monthly (28d)":     _GREEN,
    "Bi-monthly (56d)":  _BLUE,
    "Quarterly (84d)":   _ORANGE,
    "Custom":            "#7C3AED",
    "Targeted":          "#0891B2",
}
_PROGRAM_DASH = {
    "No training": "dot",
}


def plot_training_comparison(
    programs: dict[str, np.ndarray],
    n_days: int,
    rolling_days: int = 30,
    start_date: str = "2024-01-01",
    training_days: dict[str, list[int]] | None = None,
) -> go.Figure:
    """
    Overlay readiness timeseries for multiple training programs on one chart.
    programs: {label: proportion_ready_on_shift array}
    training_days: {label: [day_index, ...]} — draws thin vertical tick marks per program
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    def smooth(arr):
        return pd.Series(arr).rolling(rolling_days, min_periods=1, center=True).mean().values

    fig = go.Figure()
    for label, arr in programs.items():
        color = _PROGRAM_COLORS.get(label, _BLUE)
        dash = _PROGRAM_DASH.get(label, "solid")
        width = 1.5 if label == "No training" else 2
        fig.add_trace(go.Scatter(
            x=dates,
            y=smooth(arr) * 100,
            mode="lines",
            name=label,
            line=dict(color=color, width=width, dash=dash),
        ))

    # Training day tick marks — thin vertical lines per program
    if training_days:
        shapes = []
        for label, days in training_days.items():
            color = _PROGRAM_COLORS.get(label, _BLUE)
            for d in days:
                if 0 <= d < n_days:
                    shapes.append(dict(
                        type="line",
                        x0=dates[d], x1=dates[d],
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color=color, width=0.8),
                        opacity=0.25,
                    ))
        if shapes:
            fig.update_layout(shapes=shapes)

    fig.update_layout(
        title=f"Training program comparison ({rolling_days}-day rolling mean, on-shift providers)",
        xaxis_title="Date",
        yaxis_title="% ready",
        yaxis=dict(range=[0, 105]),
        height=420,
        legend=dict(x=0.01, y=0.05),
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


# ---------------------------------------------------------------------------
# 6. Exposure count histogram
# ---------------------------------------------------------------------------

def plot_exposure_count_histogram(results_df: pd.DataFrame) -> go.Figure:
    """Histogram of total HALO events encountered per provider."""
    vals = results_df["n_events"].dropna().values

    fig = go.Figure(go.Histogram(
        x=vals,
        nbinsx=min(50, int(vals.max()) + 1) if len(vals) else 10,
        marker_color=_BLUE,
        marker_line_color="white",
        marker_line_width=0.5,
        opacity=0.85,
        name="Providers",
    ))

    fig.add_vline(
        x=float(np.median(vals)) if len(vals) else 0,
        line_dash="dash",
        line_color=_ORANGE,
        annotation_text=f"Median: {np.median(vals):.1f}" if len(vals) else "",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Distribution of total HALO exposures per provider",
        xaxis_title="Number of exposures",
        yaxis_title="Number of providers",
        height=360,
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.05,
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig
