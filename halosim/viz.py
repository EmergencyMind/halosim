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
    "Monthly (30d)":     _GREEN,
    "Bi-monthly (60d)":  _BLUE,
    "Quarterly (91d)":   _ORANGE,
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


# ---------------------------------------------------------------------------
# 7. MC readiness band — baseline ± trained ribbon
# ---------------------------------------------------------------------------

def plot_mc_readiness_band(
    readiness_b: np.ndarray,
    readiness_t: np.ndarray | None = None,
    rolling_days: int = 30,
    p_low: int = 10,
    p_high: int = 90,
    start_date: str = "2024-01-01",
) -> go.Figure:
    """
    Shaded-ribbon readiness chart across MC seeds.
    readiness_b / readiness_t: shape (n_samples, n_days), values in [0, 1] with NaN.
    """
    n_samples, n_days = readiness_b.shape
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    def smooth_matrix(mat: np.ndarray) -> np.ndarray:
        out = np.empty_like(mat, dtype=float)
        for i, row in enumerate(mat):
            out[i] = (
                pd.Series(row)
                .rolling(rolling_days, min_periods=1, center=True)
                .mean()
                .values
            )
        return out * 100

    sm_b = smooth_matrix(readiness_b)
    lo_b = np.nanpercentile(sm_b, p_low, axis=0)
    hi_b = np.nanpercentile(sm_b, p_high, axis=0)
    med_b = np.nanpercentile(sm_b, 50, axis=0)

    fig = go.Figure()

    # Baseline ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([hi_b, lo_b[::-1]]),
        fill="toself",
        fillcolor="rgba(148,163,184,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name=f"No training (p{p_low}–p{p_high})",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=med_b,
        mode="lines",
        name="No training (median)",
        line=dict(color=_GREY, width=2),
    ))

    if readiness_t is not None:
        sm_t = smooth_matrix(readiness_t)
        lo_t = np.nanpercentile(sm_t, p_low, axis=0)
        hi_t = np.nanpercentile(sm_t, p_high, axis=0)
        med_t = np.nanpercentile(sm_t, 50, axis=0)

        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([hi_t, lo_t[::-1]]),
            fill="toself",
            fillcolor="rgba(22,163,74,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f"With training (p{p_low}–p{p_high})",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=med_t,
            mode="lines",
            name="With training (median)",
            line=dict(color=_GREEN, width=2),
        ))

    y_min = float(np.nanmin(lo_b))
    y_max = float(np.nanmax(hi_b if readiness_t is None else np.maximum(hi_b, hi_t)))
    pad = max((y_max - y_min) * 0.08, 2.0)
    y_lo = max(0.0, y_min - pad)
    y_hi = min(100.0, y_max + pad)

    title_suffix = f"{n_samples} MC run{'s' if n_samples != 1 else ''}"
    fig.update_layout(
        title=f"On-shift readiness — {rolling_days}-day rolling mean ({title_suffix})",
        xaxis_title="Date",
        yaxis_title="% ready",
        yaxis=dict(range=[y_lo, y_hi]),
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
# 8. MC scalar histogram
# ---------------------------------------------------------------------------

def plot_mc_histogram(
    values: np.ndarray,
    metric_name: str,
    unit: str = "",
    p_low: int = 10,
    p_high: int = 90,
) -> go.Figure:
    """Histogram of a per-seed scalar metric with median and percentile vlines."""
    vals = values[np.isfinite(values)]
    med = float(np.median(vals))
    lo  = float(np.percentile(vals, p_low))
    hi  = float(np.percentile(vals, p_high))

    fig = go.Figure(go.Histogram(
        x=vals,
        marker_color=_BLUE,
        marker_line_color="white",
        marker_line_width=0.5,
        opacity=0.85,
    ))
    fig.add_vline(x=med, line_dash="solid", line_color=_ORANGE,
                  annotation_text=f"Median: {med:.1f}{unit}",
                  annotation_position="top right")
    fig.add_vline(x=lo, line_dash="dash", line_color=_GREY,
                  annotation_text=f"p{p_low}",
                  annotation_position="top left")
    fig.add_vline(x=hi, line_dash="dash", line_color=_GREY,
                  annotation_text=f"p{p_high}",
                  annotation_position="top right")

    xlabel = f"{metric_name}{' (' + unit + ')' if unit else ''}"
    fig.update_layout(
        title=metric_name,
        xaxis_title=xlabel,
        yaxis_title="Runs",
        height=300,
        margin=dict(t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        bargap=0.05,
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


# ---------------------------------------------------------------------------
# 9. MC summary table
# ---------------------------------------------------------------------------

def build_mc_summary_df(mc_result: dict) -> pd.DataFrame:
    """Return a summary DataFrame: Metric / Median / p10 / p90."""
    thresh = mc_result["threshold"]
    rows = []

    for arr, label, unit, fmt in [
        (mc_result["pct_exceeding"],   f"% providers exceeding {thresh}-day threshold", "%",    "{:.1f}"),
        (mc_result["median_gap"],      "Median gap between exposures",                  " days", "{:.0f}"),
        (mc_result["median_n_events"], "Median exposures per provider",                 "",      "{:.1f}"),
    ]:
        med = np.median(arr)
        lo  = np.percentile(arr, 10)
        hi  = np.percentile(arr, 90)
        rows.append({
            "Metric":   label,
            "Median":   fmt.format(med) + unit,
            "p10":      fmt.format(lo)  + unit,
            "p90":      fmt.format(hi)  + unit,
        })

    if mc_result.get("lift") is not None:
        lift = mc_result["lift"]
        rows.append({
            "Metric":  "Training lift",
            "Median":  f"{np.median(lift):+.1f} pp",
            "p10":     f"{np.percentile(lift, 10):+.1f} pp",
            "p90":     f"{np.percentile(lift, 90):+.1f} pp",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. MC threshold sweep
# ---------------------------------------------------------------------------

def plot_mc_threshold_sweep(
    pct_by_threshold: np.ndarray,  # (n_samples, n_thresholds)
    thresholds: np.ndarray,
    threshold_marker: int | None = None,
    p_low: int = 10,
    p_high: int = 90,
) -> go.Figure:
    """
    Ribbon chart: for each threshold (x), % providers with max gap > threshold (y).
    Band = p_low–p_high across MC runs; solid line = median.
    Labels the x-values where the median curve crosses 90%, 50%, and 10%.
    """
    n_samples = pct_by_threshold.shape[0]
    lo  = np.percentile(pct_by_threshold, p_low,  axis=0)
    hi  = np.percentile(pct_by_threshold, p_high, axis=0)
    med = np.percentile(pct_by_threshold, 50,     axis=0)

    def _x_at_y(y_target: float) -> float | None:
        """Find the x-value (threshold) where the median curve first crosses y_target."""
        if med[0] < y_target or med[-1] > y_target:
            return None
        return float(np.interp(y_target, med[::-1], thresholds[::-1]))

    fig = go.Figure()

    # Ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([thresholds, thresholds[::-1]]),
        y=np.concatenate([hi, lo[::-1]]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"p{p_low}–p{p_high}",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=thresholds, y=med,
        mode="lines",
        name="Median",
        line=dict(color=_BLUE, width=2),
    ))

    # Circles at 90% and 10% crossings
    for y_lvl in [90, 10]:
        x_lvl = _x_at_y(float(y_lvl))
        if x_lvl is not None:
            fig.add_trace(go.Scatter(
                x=[x_lvl], y=[y_lvl],
                mode="markers+text",
                marker=dict(color=_ORANGE, size=9, symbol="circle"),
                text=[f"  {x_lvl:.0f}d: {y_lvl}%"],
                textposition="middle right",
                textfont=dict(size=11, color=_ORANGE),
                showlegend=False,
                hovertemplate=f"{x_lvl:.0f}d → {y_lvl}% of providers<extra></extra>",
            ))

    # Diamond at 50% (median) crossing
    x_at_50 = _x_at_y(50.0)
    if x_at_50 is not None:
        fig.add_trace(go.Scatter(
            x=[x_at_50], y=[50.0],
            mode="markers+text",
            marker=dict(color=_ORANGE, size=11, symbol="diamond"),
            text=[f"  {x_at_50:.0f}d: 50%"],
            textposition="middle right",
            textfont=dict(size=11, color=_ORANGE),
            showlegend=False,
            hovertemplate=f"{x_at_50:.0f}d → 50% of providers<extra></extra>",
        ))

    # Selected threshold vline (no extra marker — circles/diamond already label the curve)
    if threshold_marker is not None:
        fig.add_vline(
            x=threshold_marker,
            line_dash="dash", line_color=_ORANGE, line_width=1.5,
            annotation_text=f"{threshold_marker}d",
            annotation_position="top right",
            annotation_font=dict(color=_ORANGE, size=11),
        )

    title_suffix = f"{n_samples} simulation{'s' if n_samples != 1 else ''}"
    fig.update_layout(
        title=f"% providers with gap > threshold ({title_suffix})",
        xaxis_title="Maximum gap between HALO exposures (days)",
        yaxis_title="% providers exceeding gap",
        yaxis=dict(range=[0, 105]),
        height=400,
        legend=dict(x=0.65, y=0.95),
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig
