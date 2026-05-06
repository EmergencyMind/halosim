"""
report.py  -  PDF report generator for HaloSim.

Uses fpdf2 for layout and kaleido (via plotly) for chart-to-PNG conversion.
Returns a bytes object suitable for st.download_button().
"""

from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd

from fpdf import FPDF

from halosim.viz import (
    plot_threshold_sweep,
    plot_readiness_timeseries,
    plot_exposure_count_histogram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLUE_RGB   = (37, 99, 235)
_DARK_RGB   = (15, 23, 42)
_MUTED_RGB  = (100, 116, 139)
_RULE_RGB   = (226, 232, 240)
_GREEN_RGB  = (22, 163, 74)


def _fig_to_png(fig, width: int = 740, height: int = 360) -> bytes:
    """Render a Plotly figure to PNG bytes via kaleido."""
    return fig.to_image(format="png", width=width, height=height, scale=2)


def _pct_table(rdf: pd.DataFrame) -> list[dict]:
    """Compute percentile rows for the gap table."""
    pcts = [5, 25, 50, 75, 95]
    rows = []
    for p in pcts:
        rows.append({
            "Percentile": f"{p}th",
            "Max gap (days)": f"{np.percentile(rdf['gap_max'].dropna(), p):.0f}",
            "Median gap (days)": f"{np.percentile(rdf['gap_median'].dropna(), p):.0f}",
            "Exposures / provider": f"{np.percentile(rdf['n_events'].dropna(), p):.1f}",
        })
    return rows


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class _HaloReport(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_font("Helvetica", size=10)

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_MUTED_RGB)
        self.cell(0, 6, "HaloSim  -  HALO Event Exposure & Readiness Report", align="L")
        self.set_text_color(*_DARK_RGB)
        self.ln(1)
        self.set_draw_color(*_RULE_RGB)
        self.set_line_width(0.3)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_MUTED_RGB)
        self.multi_cell(
            0, 4,
            "Walker D et al. Code Blue blindspots: mapping nursing exposure to cardiac arrests. "
            "Resuscitation. 2026. PMID: 41633464. "
            "HaloSim: https://sfl-halosim.streamlit.app/",
            align="C",
        )

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*_BLUE_RGB)
        self.cell(0, 8, text, ln=True)
        self.set_draw_color(*_BLUE_RGB)
        self.set_line_width(0.5)
        self.line(18, self.get_y(), 192, self.get_y())
        self.set_text_color(*_DARK_RGB)
        self.set_draw_color(*_RULE_RGB)
        self.set_line_width(0.3)
        self.ln(4)

    def kv_row(self, label: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_MUTED_RGB)
        self.cell(52, 5, label, ln=False)
        self.set_font("Helvetica", size=9)
        self.set_text_color(*_DARK_RGB)
        self.cell(0, 5, value, ln=True)

    def metric_box(self, x: float, y: float, w: float, label: str, value: str, delta: str = ""):
        self.set_xy(x, y)
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(*_RULE_RGB)
        self.set_line_width(0.3)
        self.rect(x, y, w, 18, style="FD")
        self.set_xy(x + 3, y + 2)
        self.set_font("Helvetica", size=7)
        self.set_text_color(*_MUTED_RGB)
        self.cell(w - 6, 4, label, ln=True)
        self.set_x(x + 3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_DARK_RGB)
        self.cell(w - 6, 6, value, ln=True)
        if delta:
            self.set_x(x + 3)
            self.set_font("Helvetica", size=7)
            self.set_text_color(*_GREEN_RGB)
            self.cell(w - 6, 4, delta, ln=True)

    def data_table(self, headers: list[str], rows: list[dict], col_widths: list[float]):
        # Header row
        self.set_fill_color(248, 250, 252)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_MUTED_RGB)
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, h, border=1, fill=True, align="C")
        self.ln()
        # Data rows
        self.set_font("Helvetica", size=8)
        self.set_text_color(*_DARK_RGB)
        for row in rows:
            for h, w in zip(headers, col_widths):
                self.cell(w, 5, str(row.get(h, "")), border=1, align="C")
            self.ln()

    def callout(self, text: str):
        self.set_fill_color(239, 246, 255)
        self.set_draw_color(147, 197, 253)
        self.set_line_width(0.3)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*_DARK_RGB)
        self.multi_cell(0, 5, text, border=1, fill=True)
        self.ln(2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(
    sim_b,
    sim_t,
    params: dict,
    training_program_label: str = "None",
) -> bytes:
    """
    Build a PDF report and return raw bytes.

    params keys used: n_days, n_providers, seed, event_source, event_rate,
    readiness_model, readiness_threshold, simulation_date.
    """
    pdf = _HaloReport()

    # ── Page 1: Title + parameters + metrics + percentile table ──────────────
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_DARK_RGB)
    pdf.cell(0, 10, "HaloSim Exposure & Readiness Report", ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*_MUTED_RGB)
    pdf.cell(0, 5, f"Generated {params.get('simulation_date', date.today().isoformat())}  ·  "
             "https://sfl-halosim.streamlit.app/", ln=True)
    pdf.ln(4)

    # Parameters section
    pdf.section_title("Simulation Parameters")
    pdf.kv_row("Providers:", f"{params.get('n_providers', ' - '):,}")
    pdf.kv_row("Simulation window:", f"{params.get('n_days', ' - ')} days")
    pdf.kv_row("Event source:", str(params.get('event_source', ' - ')))
    if params.get('event_source') == "Generate (Poisson MC)":
        pdf.kv_row("Event rate:", f"{params.get('event_rate', 0.14):.2f} events/day "
                   f"(~{params.get('event_rate', 0.14) * 365:.0f}/year)")
    pdf.kv_row("Readiness model:", str(params.get('readiness_model', 'binary')).capitalize())
    pdf.kv_row("Readiness threshold:", f"{params.get('readiness_threshold', 90)} days")
    pdf.kv_row("Training program:", training_program_label)
    pdf.kv_row("Random seed:", str(params.get('seed', 42)))
    pdf.ln(4)

    # Exposure metrics
    rdf = sim_b.results_df
    n = len(rdf)
    n_zero = int((rdf["n_events"] == 0).sum())
    n_exceed = int(rdf["max_gap_exceeds_threshold"].sum())
    thresh = sim_b.readiness_threshold_days

    pdf.section_title("Exposure Analysis  -  Key Metrics")
    box_w = 41.5
    y0 = pdf.get_y()
    pdf.metric_box(18,        y0, box_w, "Providers simulated",       f"{n:,}")
    pdf.metric_box(18 + box_w + 2, y0, box_w, "Median exposures / provider",
                   f"{rdf['n_events'].median():.1f}")
    pdf.metric_box(18 + (box_w + 2) * 2, y0, box_w, f"% max gap > {thresh}d",
                   f"{100 * n_exceed / n:.1f}%")
    pdf.metric_box(18 + (box_w + 2) * 3, y0, box_w, "% zero exposures",
                   f"{100 * n_zero / n:.1f}%")
    pdf.set_y(y0 + 22)
    pdf.ln(2)

    # Percentile table
    pdf.section_title("Gap Statistics  -  Percentiles")
    headers = ["Percentile", "Max gap (days)", "Median gap (days)", "Exposures / provider"]
    col_widths = [28, 46, 46, 54]
    pdf.data_table(headers, _pct_table(rdf), col_widths)
    pdf.ln(4)

    # Interpretation callout
    pct_t = 100 * (rdf["gap_max"].fillna(9999) > thresh).mean()
    if pct_t >= 80:
        interp = (f"{pct_t:.0f}% of providers exceed the {thresh}-day gap threshold  -  consistent with "
                  "the paper's community hospital finding of 98% (PMID: 41633464). Consider whether "
                  "your training frequency is sufficient to compensate for infrequent live exposure.")
    elif pct_t >= 40:
        interp = (f"{pct_t:.0f}% of providers exceed the {thresh}-day gap threshold. "
                  "Your event rate or shift density differs from the paper's community hospital setting. "
                  "Evaluate whether current training frequency maintains adequate readiness.")
    else:
        interp = (f"{pct_t:.0f}% of providers exceed the {thresh}-day gap threshold  -  relatively low. "
                  "Your event rate or population size may be higher than a typical community hospital, "
                  "meaning live exposure alone may maintain meaningful readiness.")
    pdf.callout(f"Interpretation: {interp}")

    # ── Page 2: Threshold sweep chart ────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Threshold Sweep")
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(*_MUTED_RGB)
    pdf.cell(0, 5, "Percentage of providers whose maximum inter-exposure gap exceeds each threshold (7-365 days).", ln=True)
    pdf.set_text_color(*_DARK_RGB)
    pdf.ln(2)

    try:
        sweep_img = _fig_to_png(plot_threshold_sweep(rdf), width=740, height=380)
        pdf.image(io.BytesIO(sweep_img), x=18, y=None, w=174)
    except Exception:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "[Chart could not be rendered  -  open the app to view.]", ln=True)

    pdf.ln(4)

    try:
        hist_img = _fig_to_png(plot_exposure_count_histogram(rdf), width=740, height=340)
        pdf.section_title("Exposure Count Distribution")
        pdf.image(io.BytesIO(hist_img), x=18, y=None, w=174)
    except Exception:
        pass

    # ── Page 3: Training simulation ───────────────────────────────────────────
    if sim_t is not None and training_program_label != "None (exposure only)":
        pdf.add_page()
        pdf.section_title("Training Simulation")

        b_mean = float(np.nanmean(sim_b.proportion_ready_on_shift) * 100)
        t_mean = float(np.nanmean(sim_t.proportion_ready_on_shift) * 100)
        n_train = int(sim_t.training_matrix.sum())

        box_w2 = 55.5
        y1 = pdf.get_y()
        pdf.metric_box(18,           y1, box_w2, "Avg readiness  -  no training", f"{b_mean:.1f}%")
        pdf.metric_box(18 + box_w2 + 2, y1, box_w2, "Avg readiness  -  with training",
                       f"{t_mean:.1f}%", delta=f"{t_mean - b_mean:+.1f} pp")
        pdf.metric_box(18 + (box_w2 + 2) * 2, y1, box_w2, "Training events delivered",
                       f"{n_train:,}")
        pdf.set_y(y1 + 22)
        pdf.ln(4)

        try:
            ts_img = _fig_to_png(
                plot_readiness_timeseries(
                    sim_b.proportion_ready_on_shift,
                    sim_t.proportion_ready_on_shift,
                    n_days=sim_b.n_days,
                    rolling_days=30,
                ),
                width=740, height=400,
            )
            pdf.image(io.BytesIO(ts_img), x=18, y=None, w=174)
        except Exception:
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 5, "[Chart could not be rendered  -  open the app to view.]", ln=True)

        pdf.ln(4)
        improvement = t_mean - b_mean
        if abs(improvement) < 1:
            t_interp = ("Training had minimal effect on on-shift readiness. "
                        "The live event rate may be sufficient to maintain baseline readiness independently.")
        else:
            t_interp = (
                f"Training raised average on-shift readiness by {improvement:.1f} percentage points "
                f"({b_mean:.0f}% -> {t_mean:.0f}%) using the '{training_program_label}' program. "
                f"A total of {n_train:,} training events were delivered across the simulation window."
            )
        pdf.callout(f"Interpretation: {t_interp}")

    return bytes(pdf.output())
