"""
HaloSim — HALO Event Exposure & Training Simulation
Streamlit app entry point
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from halosim.events import generate_events, load_events_from_upload
from halosim.schedules import (
    SCHEDULE_TYPES,
    DEFAULT_SCHEDULE_TYPE,
    generate_schedule,
    generate_from_pattern,
    load_schedule_from_upload,
    MAX_PROVIDERS,
    WARN_PROVIDERS,
)
from halosim.simulation import Simulation
from halosim.viz import (
    plot_exposure_count_histogram,
    plot_gap_distribution,
    plot_individual_swimlanes,
    plot_readiness_timeseries,
    plot_threshold_sweep,
    plot_training_comparison,
)
from halosim.report import generate_pdf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent / "assets"
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="HaloSim",
    page_icon="https://raw.githubusercontent.com/EmergencyMind/halosim/master/assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
      html, body, p, li, td, th, label, button { font-family: 'DM Sans', sans-serif !important; }
      /* Sidebar: reduce top padding and divider margins only */
      section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.75rem !important;
      }
      section[data-testid="stSidebar"] hr { margin: 0.5rem 0 !important; }
      section[data-testid="stSidebar"] h3 {
        font-size: 0.9rem !important;
        margin-bottom: 0.25rem !important;
      }
      /* Metric cards */
      [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
      }
      /* Tab strip */
      [data-testid="stTabs"] button { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "n_days": 365,
        "n_providers": 200,
        "seed": 42,
        # events
        "event_source": "Generate (Poisson MC)",
        "event_rate": 0.14,
        "event_day_rate": 0.07,
        "event_night_rate": 0.07,
        "seasonal_amplitude": 0.0,
        "seasonal_phase": 0.0,
        "events_df": None,
        "events_warnings": [],
        "events_errors": [],
        # schedules
        "schedule_source": "Generate schedules",
        "schedule_type": DEFAULT_SCHEDULE_TYPE,
        "schedule_array": None,
        "schedule_providers": None,
        "schedule_warnings": [],
        "schedule_errors": [],
        # readiness
        "readiness_model": "binary",
        "readiness_threshold": 90,
        "readiness_half_life": 60,
        "ebbinghaus_b": 0.05,
        "step_partial": 0.5,
        "step_t2": 180,
        # training
        "training_program": "none",
        "training_interval": 28,
        "training_start": 0,
        "training_effect": "full",
        "training_equivalence": 1.0,
        "training_threshold": 0.5,
        "join_type": "simple",
        "complex_window": 4,
        # simulation outputs
        "sim_baseline": None,
        "sim_trained": None,
        "sim_ran": False,
        "_last_run_hash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _sim_hash() -> str:
    """Stable string fingerprint of all simulation-relevant session state."""
    s = st.session_state
    parts = [
        s.n_days, s.n_providers, s.seed,
        # events
        s.event_source, s.event_rate, s.event_day_rate, s.event_night_rate,
        s.seasonal_amplitude, s.seasonal_phase,
        # schedules
        s.schedule_source,
        s.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
        # uploaded data: use row-count / shape as proxy
        len(s.events_df) if s.events_df is not None else -1,
        str(s.schedule_array.shape) if s.schedule_array is not None else "none",
        # readiness
        s.readiness_model, s.readiness_threshold, s.readiness_half_life,
        s.ebbinghaus_b, s.step_t2, s.step_partial,
    ]
    return ":".join(str(p) for p in parts)


@st.cache_data(show_spinner=False, max_entries=16)
def _run_sim(
    n_days, providers_tuple, schedule, events_df, seed,
    readiness_model, readiness_threshold, readiness_half_life,
    ebbinghaus_b, step_t2, step_partial,
    training_program, training_interval, training_start,
    training_effect, training_equivalence, training_threshold,
) -> "Simulation":
    """Run one simulation and cache the result by all inputs."""
    sim = Simulation(
        n_days=n_days,
        providers=list(providers_tuple),
        schedule=schedule,
        events=events_df,
        seed=seed,
        readiness_model=readiness_model,
        readiness_threshold_days=readiness_threshold,
        readiness_half_life_days=readiness_half_life,
        ebbinghaus_b=ebbinghaus_b,
        step_t2_days=step_t2,
        step_partial_value=step_partial,
        training_program=training_program,
        training_interval_days=training_interval,
        training_start_day=training_start,
        training_effect=training_effect,
        training_equivalence=training_equivalence,
        training_target_threshold=training_threshold,
    )
    sim.run()
    return sim


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    _hdr_logo, _hdr_text = st.columns([1, 2.5])
    with _hdr_logo:
        st.image(str(ASSETS_DIR / "logo.png"), width=56)
    with _hdr_text:
        st.markdown("**HaloSim**")
        st.caption("HALO Event Exposure\n& Training Simulation")
    st.divider()
    st.subheader("Simulation settings")

    n_days = st.selectbox(
        "Simulation window",
        [90, 180, 365, 730],
        index=[90, 180, 365, 730].index(st.session_state.n_days)
        if st.session_state.n_days in [90, 180, 365, 730] else 2,
        format_func=lambda x: f"{x} days (~{x//365} yr)" if x >= 365 else f"{x} days (~{x//30} mo)",
    )
    st.session_state.n_days = n_days

    n_providers = st.number_input(
        "Number of providers",
        min_value=10,
        max_value=MAX_PROVIDERS,
        value=st.session_state.n_providers,
        step=10,
    )
    n_providers = int(n_providers)
    st.session_state.n_providers = n_providers
    if n_providers > WARN_PROVIDERS:
        st.warning(f"⚠️ {n_providers:,} providers — simulation may take a few seconds.")

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=st.session_state.seed,
        help="Set for reproducible results.",
    )
    st.session_state.seed = int(seed)

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.divider()
    st.caption("Built by [Sangfroid Labs](https://sangfroidlabs.com)")


# ---------------------------------------------------------------------------
# Result banner (above tabs — always visible regardless of active tab)
# ---------------------------------------------------------------------------

if st.session_state.sim_ran and st.session_state.sim_baseline is not None:
    if st.session_state._last_run_hash == _sim_hash():
        _sim = st.session_state.sim_baseline
        st.success(
            f"✓ {len(_sim.providers):,} providers × {_sim.n_days} days — "
            "results in the **Exposure Analysis** tab."
        )
    else:
        st.warning("Settings changed — click **▶ Run Simulation** to update results.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_events, tab_schedules, tab_exposure, tab_training = st.tabs(
    ["📅 Events", "👥 Schedules", "📊 Exposure Analysis", "🏋️ Training Simulation"]
)

# ── Tab 1: Events ──────────────────────────────────────────────────────────

with tab_events:
    st.header("HALO Event Configuration")

    source_options = ["Generate (Poisson MC)", "Upload CSV / Excel"]
    event_source = st.radio(
        "Event source",
        source_options,
        index=source_options.index(st.session_state.event_source),
        horizontal=True,
    )
    st.session_state.event_source = event_source

    if event_source == "Generate (Poisson MC)":
        rate = st.slider(
            "Event rate (events / day)",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.event_rate,
            step=0.01,
            help="0.14 events/day ≈ 4.3/month — matches cardiac arrest rate in Dworkis 2026",
        )
        st.session_state.event_rate = rate
        # Keep split rates in sync with total rate (overridden in expander if set separately)
        if st.session_state.event_day_rate + st.session_state.event_night_rate != rate:
            st.session_state.event_day_rate = round(rate / 2, 3)
            st.session_state.event_night_rate = round(rate / 2, 3)
        per_month = rate * 30.44
        st.caption(
            f"~**{per_month:.1f} events/month** &nbsp;·&nbsp; "
            f"{rate * n_days:.0f} expected over {n_days} days"
        )

        with st.expander("Advanced event settings"):
            st.caption("Split rates by shift type and add seasonal variation.")
            c1, c2 = st.columns(2)
            with c1:
                day_rate = st.slider("Day shift rate (events/day)", 0.01, 1.0,
                                     st.session_state.event_day_rate, 0.01,
                                     key="adv_day_rate")
                st.session_state.event_day_rate = day_rate
            with c2:
                night_rate = st.slider("Night shift rate (events/day)", 0.01, 1.0,
                                       st.session_state.event_night_rate, 0.01,
                                       key="adv_night_rate")
                st.session_state.event_night_rate = night_rate
            amp = st.slider("Seasonal amplitude (0 = flat)", 0.0, 0.9,
                            st.session_state.seasonal_amplitude, 0.05)
            phase = st.slider("Peak day of year", 0.0, 365.0,
                              st.session_state.seasonal_phase, 1.0)
            st.session_state.seasonal_amplitude = amp
            st.session_state.seasonal_phase = phase

        st.divider()
        with st.expander("Load sample data instead"):
            if st.button("Use sample_events.csv"):
                sample = DATA_DIR / "sample_events.csv"
                df = pd.read_csv(sample)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["day_idx"] = (
                    pd.to_datetime(df["date"]) - pd.Timestamp("2024-01-01")
                ).dt.days
                st.session_state.events_df = df[["day_idx", "date", "shift_type"]]
                st.session_state.event_source = "Upload CSV / Excel"
                st.rerun()

    else:  # upload
        uploaded = st.file_uploader(
            "Upload events file",
            type=["csv", "xlsx", "xls"],
            help="Required columns: date (YYYY-MM-DD), shift_type (day/night).",
        )
        if uploaded:
            raw = uploaded.read()
            df, errs = load_events_from_upload(
                raw, uploaded.name, n_days,
                allow_hour_col=False,
            )
            st.session_state.events_errors = errs
            if df is not None:
                st.session_state.events_df = df
        for e in st.session_state.events_errors:
            st.error(e)
        if st.session_state.events_df is not None:
            edf = st.session_state.events_df
            st.success(f"✓ {len(edf)} events loaded")
            st.dataframe(edf.head(10), use_container_width=True)

        with st.expander("Advanced event settings"):
            st.caption("Enable the hour column for complex shift-boundary join.")
            allow_hour_adv = st.checkbox("File includes 'hour' column (0–23)", value=False,
                                         key="allow_hour_adv")
            if allow_hour_adv and uploaded:
                raw = uploaded.read() if hasattr(uploaded, "read") else b""
                df2, errs2 = load_events_from_upload(raw, uploaded.name, n_days, allow_hour_col=True)
                if df2 is not None:
                    st.session_state.events_df = df2

        st.divider()
        join_opts = ["simple", "complex (requires hour column)"]
        join = st.selectbox(
            "Exposure join type",
            join_opts,
            index=0 if st.session_state.join_type == "simple" else 1,
            help="Complex join counts events within ±N hours of shift boundary.",
        )
        st.session_state.join_type = "simple" if join == "simple" else "complex"
        if st.session_state.join_type == "complex":
            win = st.slider("Window (hours)", 1, 8, st.session_state.complex_window)
            st.session_state.complex_window = win
            if st.session_state.events_df is not None and \
               "hour" not in st.session_state.events_df.columns:
                st.warning("Complex join requires an 'hour' column in the events file. "
                           "Falling back to simple join.")


# ── Tab 2: Schedules ───────────────────────────────────────────────────────

with tab_schedules:
    st.header("Provider Schedule Configuration")

    sched_options = ["Generate schedules", "Upload CSV / Excel"]

    # Migrate legacy session state value
    if st.session_state.schedule_source in ("Built-in templates", "Custom 28-day pattern"):
        st.session_state.schedule_source = "Generate schedules"

    sched_source = st.radio(
        "Schedule source",
        sched_options,
        index=min(
            sched_options.index(st.session_state.schedule_source)
            if st.session_state.schedule_source in sched_options else 0,
            len(sched_options) - 1,
        ),
        horizontal=True,
    )
    st.session_state.schedule_source = sched_source

    if sched_source == "Generate schedules":
        _type_descriptions = {
            "3/7 Day":   "3 randomly placed day shifts per 7-day week, rest off",
            "3/7 Night": "3 randomly placed night shifts per 7-day week, rest off",
            "4/7 Day":   "4 randomly placed day shifts per 7-day week, rest off",
            "4/7 Night": "4 randomly placed night shifts per 7-day week, rest off",
            "Progressive (day & night mix)":
                "3-4 shifts per week, each randomly assigned day or night",
            "Random":    "Each day drawn from empirical d/n/o weights (Dworkis 2026: 25% day, 23% night, 52% off)",
        }
        current_type = st.session_state.get("schedule_type", DEFAULT_SCHEDULE_TYPE)
        if current_type not in SCHEDULE_TYPES:
            current_type = DEFAULT_SCHEDULE_TYPE

        selected_type = st.selectbox(
            "Schedule type",
            SCHEDULE_TYPES,
            index=SCHEDULE_TYPES.index(current_type),
        )
        st.session_state.schedule_type = selected_type
        st.caption(_type_descriptions[selected_type])

        with st.expander("Advanced schedule settings"):
            st.caption("Define a fixed 28-character d/n/o pattern that tiles across the "
                       "simulation window. All providers share the same template.")
            pattern = st.text_input(
                "28-day pattern",
                value="dddoooodddoooodddoooodddoooo",
                max_chars=28,
                key="custom_pattern_input",
            )
            pattern = pattern.lower().strip()
            bad_chars = [c for c in pattern if c not in "dno"]
            if bad_chars:
                st.error(f"Invalid characters: {set(bad_chars)}. Use only d, n, o.")
            elif len(pattern) != 28:
                st.warning(f"Pattern is {len(pattern)} characters — needs to be exactly 28.")
            else:
                d_p = pattern.count("d") / 28 * 100
                n_p = pattern.count("n") / 28 * 100
                o_p = pattern.count("o") / 28 * 100
                st.caption(f"Day: {d_p:.0f}% | Night: {n_p:.0f}% | Off: {o_p:.0f}%")
                if st.button("Use this pattern", key="use_custom_pattern"):
                    st.session_state.schedule_type = pattern
                    st.session_state.schedule_source = "Custom 28-day pattern"
                    st.rerun()

    elif sched_source == "Upload CSV / Excel":
        uploaded_s = st.file_uploader(
            "Upload schedule file",
            type=["csv", "xlsx", "xls"],
            help="Required columns: provider_id, date (YYYY-MM-DD), "
                 "shift_type (day/night/off or d/n/o)",
            key="schedule_upload",
        )
        if uploaded_s:
            raw = uploaded_s.read()
            arr, providers, errs = load_schedule_from_upload(
                raw, uploaded_s.name, n_days
            )
            st.session_state.schedule_errors = errs
            if arr is not None:
                st.session_state.schedule_array = arr
                st.session_state.schedule_providers = providers

        for e in st.session_state.schedule_errors:
            if e.startswith("Warning:"):
                st.warning(e)
            else:
                st.error(e)
        if st.session_state.schedule_array is not None:
            arr = st.session_state.schedule_array
            st.success(
                f"✓ Schedule loaded: {arr.shape[0]} providers × {arr.shape[1]} days"
            )

        st.divider()
        with st.expander("Use pre-built sample schedule"):
            if st.button("Load sample_schedule.csv (20 providers)"):
                raw = (DATA_DIR / "sample_schedule.csv").read_bytes()
                arr, providers, errs = load_schedule_from_upload(
                    raw, "sample_schedule.csv", n_days
                )
                if arr is not None:
                    st.session_state.schedule_array = arr
                    st.session_state.schedule_providers = providers
                    st.rerun()


# ── Tab 3: Exposure Analysis ───────────────────────────────────────────────

with tab_exposure:
    st.header("Exposure Analysis")

    # Readiness model — always visible
    model_map = {
        "Binary threshold": "binary",
        "Exponential decay": "exponential",
        "Ebbinghaus forgetting curve": "ebbinghaus",
        "Two-threshold step": "step",
    }
    model_label = st.selectbox(
        "Readiness model",
        list(model_map.keys()),
        index=list(model_map.values()).index(st.session_state.readiness_model),
        help="How provider readiness decays between exposures.",
    )
    st.session_state.readiness_model = model_map[model_label]

    thresh = st.slider("Readiness threshold (days since last exposure)",
                       7, 730, st.session_state.readiness_threshold,
                       help="Provider is 'ready' if last exposure is within this window.")
    st.session_state.readiness_threshold = thresh

    with st.expander("Advanced readiness settings"):
        st.caption("Model-specific parameters. Only relevant when using a non-binary model.")
        if st.session_state.readiness_model == "exponential":
            hl = st.slider("Half-life (days)", 7, 365,
                           int(st.session_state.readiness_half_life))
            st.session_state.readiness_half_life = float(hl)
        elif st.session_state.readiness_model == "ebbinghaus":
            b = st.slider("Forgetting rate b", 0.001, 0.5,
                          st.session_state.ebbinghaus_b, 0.001, format="%.3f")
            st.session_state.ebbinghaus_b = b
        elif st.session_state.readiness_model == "step":
            c1, c2 = st.columns(2)
            with c1:
                t2 = st.slider("T2 — partial readiness ends (days)",
                               thresh + 1, 730,
                               max(st.session_state.step_t2, thresh + 1))
                st.session_state.step_t2 = t2
            with c2:
                pv = st.slider("Partial readiness value", 0.0, 1.0,
                               st.session_state.step_partial, 0.05)
                st.session_state.step_partial = pv
        else:
            st.caption("No additional parameters for the binary threshold model.")

    st.divider()

    if not st.session_state.sim_ran:
        st.info("Configure Events and Schedules above, then click **▶ Run Simulation** in the sidebar.")
    else:
        sim = st.session_state.sim_baseline
        if sim is None or sim.results_df is None:
            st.error("Simulation result is unavailable — check for errors above.")
        else:
            rdf = sim.results_df
            n = len(rdf)
            n_zero = (rdf["n_events"] == 0).sum()
            n_exceed = rdf["max_gap_exceeds_threshold"].sum()
            thresh = sim.readiness_threshold_days

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Providers simulated", f"{n:,}")
            c2.metric("Median exposures / provider",
                      f"{rdf['n_events'].median():.1f}")
            c3.metric(f"% with max gap > {thresh}d",
                      f"{100 * n_exceed / n:.1f}%")
            c4.metric("% with zero exposures",
                      f"{100 * n_zero / n:.1f}%")

            # Percentile table
            st.divider()
            st.subheader("Gap statistics — percentiles")
            _pcts = [5, 25, 50, 75, 95]
            _pct_df = pd.DataFrame({
                "Percentile": [f"{p}th" for p in _pcts],
                "Max gap (days)": [f"{np.percentile(rdf['gap_max'].dropna(), p):.0f}" for p in _pcts],
                "Median gap (days)": [f"{np.percentile(rdf['gap_median'].dropna(), p):.0f}" for p in _pcts],
                "Exposures / provider": [f"{np.percentile(rdf['n_events'].dropna(), p):.1f}" for p in _pcts],
            })
            st.dataframe(_pct_df, use_container_width=True, hide_index=True)

            st.divider()
            st.plotly_chart(plot_exposure_count_histogram(rdf),
                            use_container_width=True)
            st.plotly_chart(plot_gap_distribution(rdf),
                            use_container_width=True)
            st.plotly_chart(plot_threshold_sweep(rdf),
                            use_container_width=True)

            # Interpretation callout for threshold sweep
            _pct_90 = 100 * (rdf["gap_max"].fillna(9999) > 90).mean()
            if _pct_90 >= 80:
                _interp = (f"**{_pct_90:.0f}%** of providers exceed the 90-day gap benchmark — "
                           "consistent with the paper's community hospital finding of 98% "
                           "(Dworkis 2026). Training may be needed to compensate for "
                           "infrequent live exposure.")
            elif _pct_90 >= 40:
                _interp = (f"**{_pct_90:.0f}%** of providers exceed the 90-day gap benchmark. "
                           "Your event rate or shift density differs from the paper's community "
                           "hospital setting. Evaluate whether current training frequency "
                           "maintains adequate readiness.")
            else:
                _interp = (f"**{_pct_90:.0f}%** of providers exceed the 90-day gap benchmark — "
                           "relatively low. Your event rate may be higher than a typical community "
                           "hospital, suggesting live exposure alone contributes meaningfully to "
                           "readiness.")
            st.info(_interp)

            with st.expander("Individual provider swimlanes (random sample)"):
                n_swim = st.slider("Providers to display", 10, 80, 30, key="swimlane_n")
                st.plotly_chart(
                    plot_individual_swimlanes(
                        sim.exposure_matrix,
                        sim.providers,
                        sample_n=n_swim,
                        seed=sim.seed,
                    ),
                    use_container_width=True,
                )

            st.divider()
            _dl1, _dl2 = st.columns(2)
            with _dl1:
                csv = rdf.to_csv(index=False).encode()
                st.download_button(
                    "Download per-provider gap statistics (CSV)",
                    data=csv,
                    file_name="halosim_exposure_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with _dl2:
                if st.button("Generate PDF report", use_container_width=True):
                    with st.spinner("Rendering report…"):
                        try:
                            _prog_label = st.session_state.get("_training_prog_label",
                                                               "None (exposure only)")
                            _pdf_bytes = generate_pdf(
                                sim_b=st.session_state.sim_baseline,
                                sim_t=st.session_state.sim_trained,
                                params={
                                    "n_days": sim.n_days,
                                    "n_providers": len(sim.providers),
                                    "seed": sim.seed,
                                    "event_source": st.session_state.event_source,
                                    "event_rate": st.session_state.event_rate,
                                    "readiness_model": st.session_state.readiness_model,
                                    "readiness_threshold": st.session_state.readiness_threshold,
                                },
                                training_program_label=_prog_label,
                            )
                            st.download_button(
                                "Download PDF",
                                data=_pdf_bytes,
                                file_name=f"halosim_report_{sim.n_days}d_{len(sim.providers)}p.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as _e:
                            st.error(f"PDF generation failed: {_e}")


# ── Tab 4: Training Simulation ─────────────────────────────────────────────

with tab_training:
    st.header("Training Simulation")

    program_map = {
        "None (exposure only)": "none",
        "Monthly (every 28 days)": "monthly",
        "Bi-monthly (every 56 days)": "bimonthly",
        "Quarterly (every 84 days)": "quarterly",
    }

    prog_label = st.selectbox(
        "Training program",
        list(program_map.keys()),
        index=0,
    )
    st.session_state.training_program = program_map[prog_label]

    with st.expander("Advanced training settings"):
        st.caption("Custom schedules, targeted delivery, partial training effectiveness, "
                   "and chart smoothing.")

        adv_prog_map = {
            "Custom interval": "custom",
            "Targeted (train undertrained providers only)": "targeted",
        }
        adv_prog_label = st.selectbox(
            "Override with advanced program",
            ["— use selection above —"] + list(adv_prog_map.keys()),
            key="adv_prog_select",
        )
        if adv_prog_label != "— use selection above —":
            st.session_state.training_program = adv_prog_map[adv_prog_label]

        if st.session_state.training_program == "custom":
            c1, c2 = st.columns(2)
            with c1:
                ti = st.slider("Training interval (days)", 7, 365,
                               st.session_state.training_interval, key="adv_ti")
                st.session_state.training_interval = ti
            with c2:
                ts = st.slider("First training day", 0, 90,
                               st.session_state.training_start, key="adv_ts")
                st.session_state.training_start = ts

        if st.session_state.training_program == "targeted":
            targ_thresh = st.slider(
                "Train providers whose readiness is below (%)",
                10, 100, int(st.session_state.training_threshold * 100), 5,
                key="adv_targ",
            )
            st.session_state.training_threshold = targ_thresh / 100.0
            ti2 = st.slider("Minimum days between training sessions", 7, 180,
                            st.session_state.training_interval, key="adv_ti2")
            st.session_state.training_interval = ti2

        if st.session_state.training_program != "none":
            st.divider()
            effect_opts = ["Full reset (training = live exposure)", "Partial boost"]
            eff = st.radio("Training effectiveness", effect_opts, horizontal=True,
                           index=0 if st.session_state.training_effect == "full" else 1,
                           key="adv_effect")
            st.session_state.training_effect = "full" if eff == effect_opts[0] else "partial"
            if st.session_state.training_effect == "partial":
                eq = st.slider("Equivalence factor (1.0 = same as live exposure)",
                               0.1, 1.0, st.session_state.training_equivalence, 0.05,
                               key="adv_eq")
                st.session_state.training_equivalence = eq

        st.divider()
        roll = st.slider("Rolling mean window (days)", 1, 90, 30, key="roll_window")
        st.session_state["_roll_window"] = roll

    st.divider()

    if not st.session_state.sim_ran:
        st.info("Run the simulation first using the **▶ Run Simulation** button in the sidebar.")
    else:
        sim_b = st.session_state.sim_baseline
        sim_t = st.session_state.sim_trained
        if sim_b is None:
            st.error("No simulation results available.")
        else:
            # Summary metrics
            b_mean = np.nanmean(sim_b.proportion_ready_on_shift) * 100
            t_mean = (np.nanmean(sim_t.proportion_ready_on_shift) * 100
                      if sim_t else b_mean)
            n_train = int(sim_t.training_matrix.sum()) if sim_t else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg readiness without training",
                      f"{b_mean:.1f}%")
            c2.metric("Avg readiness with training",
                      f"{t_mean:.1f}%",
                      delta=f"{t_mean - b_mean:+.1f}%")
            c3.metric("Total training events delivered", f"{n_train:,}")

            st.divider()

            roll = st.session_state.get("_roll_window", 30)

            fig = plot_readiness_timeseries(
                sim_b.proportion_ready_on_shift,
                sim_t.proportion_ready_on_shift if sim_t else sim_b.proportion_ready_on_shift,
                n_days=sim_b.n_days,
                rolling_days=roll,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation callout
            _improve = t_mean - b_mean
            if abs(_improve) < 1:
                _t_interp = ("Training had minimal effect on on-shift readiness — "
                             "live exposure alone may be sufficient at this event rate.")
            elif _improve > 0:
                _t_interp = (
                    f"Training raised average on-shift readiness by **{_improve:.1f} pp** "
                    f"({b_mean:.0f}% → {t_mean:.0f}%). "
                    f"{n_train:,} training events were delivered across the simulation window."
                )
            else:
                _t_interp = (
                    f"Readiness with training ({t_mean:.0f}%) is similar to baseline "
                    f"({b_mean:.0f}%). Consider adjusting training frequency or model parameters."
                )
            st.info(_t_interp)

            # Training program comparison
            st.divider()
            with st.expander("Compare training programs"):
                _compare_options = {
                    "No training": "none",
                    "Monthly (28d)": "monthly",
                    "Bi-monthly (56d)": "bimonthly",
                    "Quarterly (84d)": "quarterly",
                }
                _selected = st.multiselect(
                    "Programs to compare",
                    list(_compare_options.keys()),
                    default=["No training", "Monthly (28d)", "Quarterly (84d)"],
                    key="compare_programs",
                )
                _roll2 = st.slider("Rolling mean (days)", 1, 90, roll, key="compare_roll")
                if _selected and st.session_state.sim_baseline is not None:
                    _sb = st.session_state.sim_baseline
                    _compare_data: dict[str, np.ndarray] = {}
                    _s = st.session_state
                    for _lbl in _selected:
                        _prog = _compare_options[_lbl]
                        _csim = _run_sim(
                            _sb.n_days, tuple(_sb.providers), _sb.schedule, _sb.events,
                            _sb.seed,
                            _s.readiness_model, _s.readiness_threshold,
                            _s.readiness_half_life, _s.ebbinghaus_b,
                            _s.step_t2, _s.step_partial,
                            _prog, 28, 0, "full", 1.0, 0.5,
                        )
                        _compare_data[_lbl] = _csim.proportion_ready_on_shift
                    st.plotly_chart(
                        plot_training_comparison(_compare_data, _sb.n_days, _roll2),
                        use_container_width=True,
                    )

            with st.expander("Also show: all providers (including off-shift)"):
                fig2 = plot_readiness_timeseries(
                    sim_b.proportion_ready_all,
                    sim_t.proportion_ready_all if sim_t else sim_b.proportion_ready_all,
                    n_days=sim_b.n_days,
                    rolling_days=roll,
                )
                st.caption("⚠️ This includes providers currently off-shift. "
                           "The on-shift metric above is the primary indicator.")
                st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Run simulation (triggered by sidebar button)
# ---------------------------------------------------------------------------

if run_btn:
    errors = []

    # 1. Build events
    if st.session_state.event_source == "Generate (Poisson MC)":
        _day_r = st.session_state.event_day_rate
        _ngt_r = st.session_state.event_night_rate
        _amp   = st.session_state.seasonal_amplitude
        # Use split rates if they differ from the simple half/half default
        if (_day_r != _ngt_r or _amp > 0):
            events_df, e_warn = generate_events(
                n_days=n_days,
                rate=_day_r + _ngt_r,
                seed=int(seed),
                day_rate=_day_r,
                night_rate=_ngt_r,
                seasonal_amplitude=_amp,
                seasonal_phase_days=st.session_state.seasonal_phase,
            )
        else:
            events_df, e_warn = generate_events(
                n_days=n_days,
                rate=st.session_state.event_rate,
                seed=int(seed),
            )
        if e_warn:
            for w in e_warn:
                st.sidebar.warning(w)
    else:
        events_df = st.session_state.events_df
        if events_df is None:
            errors.append("No events loaded. Upload a file or switch to 'Generate' mode.")

    # 2. Build schedule
    sched_source = st.session_state.schedule_source
    s_warn = []
    if sched_source == "Upload CSV / Excel" and st.session_state.schedule_array is not None:
        arr = st.session_state.schedule_array
        if arr.shape[1] >= n_days:
            schedule = arr[:n_providers, :n_days]
        else:
            schedule, s_warn = generate_schedule(
                n_providers=n_providers,
                n_days=n_days,
                schedule_type=st.session_state.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
                seed=int(seed),
            )
        providers_list = (st.session_state.schedule_providers or [])[:n_providers] \
            or [f"P{i+1:04d}" for i in range(n_providers)]
    elif sched_source == "Custom 28-day pattern":
        pattern = st.session_state.get("schedule_type", "dddoooodddoooodddoooodddoooo")
        schedule, s_warn = generate_from_pattern(
            n_providers=n_providers,
            n_days=n_days,
            pattern=pattern,
            seed=int(seed),
        )
        providers_list = [f"P{i+1:04d}" for i in range(n_providers)]
    else:
        schedule, s_warn = generate_schedule(
            n_providers=n_providers,
            n_days=n_days,
            schedule_type=st.session_state.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
            seed=int(seed),
        )
        providers_list = [f"P{i+1:04d}" for i in range(n_providers)]

    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.stop()

    # 3 & 4. Run simulations (results cached by _run_sim)
    training_prog = st.session_state.training_program
    _s = st.session_state
    _common = dict(
        n_days=n_days, providers_tuple=tuple(providers_list),
        schedule=schedule, events_df=events_df, seed=int(seed),
        readiness_model=_s.readiness_model,
        readiness_threshold=_s.readiness_threshold,
        readiness_half_life=_s.readiness_half_life,
        ebbinghaus_b=_s.ebbinghaus_b,
        step_t2=_s.step_t2, step_partial=_s.step_partial,
        training_interval=_s.training_interval,
        training_start=_s.training_start,
        training_effect=_s.training_effect,
        training_equivalence=_s.training_equivalence,
        training_threshold=_s.training_threshold,
    )

    sim_b = _run_sim(**_common, training_program="none")

    if training_prog != "none":
        sim_t = _run_sim(**_common, training_program=training_prog)
    else:
        sim_t = sim_b

    st.session_state.sim_baseline = sim_b
    st.session_state.sim_trained = sim_t
    st.session_state.sim_ran = True
    st.session_state._last_run_hash = _sim_hash()
    # Store training label for PDF report
    _prog_labels = {v: k for k, v in {
        "None (exposure only)": "none", "Monthly (every 28 days)": "monthly",
        "Bi-monthly (every 56 days)": "bimonthly", "Quarterly (every 84 days)": "quarterly",
        "Custom interval": "custom",
        "Targeted (train undertrained providers only)": "targeted",
    }.items()}
    st.session_state["_training_prog_label"] = _prog_labels.get(training_prog, training_prog)

    st.rerun()
