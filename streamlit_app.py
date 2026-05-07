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
    load_schedule_from_upload,
    MAX_PROVIDERS,
    WARN_PROVIDERS,
)
from halosim.simulation import Simulation
from halosim.viz import (
    plot_exposure_count_histogram,
    plot_gap_distribution,
    plot_readiness_baseline,
    plot_threshold_sweep,
    plot_training_comparison,
    plot_mc_readiness_band,
    plot_mc_histogram,
    build_mc_summary_df,
)

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
      section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.75rem !important;
      }
      section[data-testid="stSidebar"] hr { margin: 0.5rem 0 !important; }
      section[data-testid="stSidebar"] h3 {
        font-size: 0.9rem !important;
        margin-bottom: 0.25rem !important;
      }
      [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
      }
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
        "event_day_pct": 50,
        "events_df": None,
        "events_warnings": [],
        "events_errors": [],
        # schedules
        "schedule_source": "Generate schedules",
        "schedule_type": DEFAULT_SCHEDULE_TYPE,
        "schedule_day_pct": None,
        "schedule_night_pct": None,
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
        "training_interval": 30,
        "training_start": 14,
        "training_effect": "full",
        "training_equivalence": 1.0,
        "training_threshold": 0.5,
        "join_type": "simple",
        "complex_window": 4,
        # MC simulation outputs
        "mc_result": None,
        "mc_ran": False,
        "mc_n_samples": 50,
        "_last_mc_hash": None,
        # upload byte caches
        "events_upload_bytes": None,
        "events_upload_name": "",
        "schedule_upload_bytes": None,
        "schedule_upload_name": "",
        # auto-run flag for demo loader
        "_auto_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _sim_hash() -> str:
    """Stable string fingerprint of simulation inputs (excludes training and n_samples)."""
    s = st.session_state
    parts = [
        s.n_days, s.n_providers, s.seed,
        s.event_source, s.event_rate, s.get("event_day_pct", 50),
        s.schedule_source,
        s.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
        s.get("schedule_day_pct"), s.get("schedule_night_pct"),
        len(s.events_df) if s.events_df is not None else -1,
        str(s.schedule_array.shape) if s.schedule_array is not None else "none",
        s.readiness_model, s.readiness_threshold, s.readiness_half_life,
        s.ebbinghaus_b, s.step_t2, s.step_partial,
    ]
    return ":".join(str(p) for p in parts)


def _mc_hash() -> str:
    """Fingerprint covering all inputs including training and n_samples."""
    s = st.session_state
    return (
        _sim_hash()
        + f":{s.mc_n_samples}"
        + f":{s.training_program}:{s.training_interval}:{s.training_start}"
        + f":{s.training_effect}:{s.training_equivalence}:{s.training_threshold}"
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _run_sim(
    n_days, providers_tuple, schedule, events_df, seed,
    readiness_model, readiness_threshold, readiness_half_life,
    ebbinghaus_b, step_t2, step_partial,
    training_program, training_interval, training_start,
    training_effect, training_equivalence, training_threshold,
) -> "Simulation":
    """Run one simulation (used for the training comparison chart)."""
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


@st.cache_data(show_spinner=False, max_entries=4)
def _run_mc(
    n_days: int,
    providers_tuple: tuple,
    fixed_schedule,        # np.ndarray if uploaded, else None
    fixed_events_df,       # pd.DataFrame if uploaded, else None
    event_source: str,
    event_rate: float,
    event_day_pct: int,
    schedule_source: str,
    schedule_type: str,
    schedule_day_pct,      # int or None
    schedule_night_pct,    # int or None
    readiness_model: str,
    readiness_threshold: int,
    readiness_half_life: float,
    ebbinghaus_b: float,
    step_t2: int,
    step_partial: float,
    training_program: str,
    training_interval: int,
    training_start: int,
    training_effect: str,
    training_equivalence: float,
    training_threshold: float,
    n_samples: int,
    base_seed: int,
) -> dict:
    """
    Run n_samples paired simulations (baseline + trained) with seeds base_seed … base_seed+n_samples-1.
    Events and schedules are re-drawn each seed when generated; fixed when uploaded.
    Returns a dict of aggregated arrays — no Simulation objects.
    """
    n_providers = len(providers_tuple)

    readiness_b_list: list[np.ndarray] = []
    readiness_t_list: list[np.ndarray] = []
    pct_exceeding_list: list[float] = []
    median_gap_list: list[float] = []
    median_n_events_list: list[float] = []
    lift_list: list[float] = []

    ref_results_df = None
    ref_events_df = None
    ref_schedule = None
    ref_proportion_b = None
    ref_proportion_t = None
    ref_training_matrix = None

    _weights = (
        {"d": schedule_day_pct / 100,
         "n": schedule_night_pct / 100,
         "o": (100 - schedule_day_pct - schedule_night_pct) / 100}
        if schedule_day_pct is not None and schedule_night_pct is not None
        else None
    )

    for s in range(n_samples):
        cur_seed = base_seed + s

        # Events
        if fixed_events_df is None:
            d_pct = event_day_pct / 100
            if event_day_pct != 50:
                ev_s, _ = generate_events(
                    n_days=n_days, rate=event_rate, seed=cur_seed,
                    day_rate=event_rate * d_pct,
                    night_rate=event_rate * (1 - d_pct),
                )
            else:
                ev_s, _ = generate_events(n_days=n_days, rate=event_rate, seed=cur_seed)
        else:
            ev_s = fixed_events_df

        # Schedule
        if fixed_schedule is None:
            sched_s, _ = generate_schedule(
                n_providers=n_providers,
                n_days=n_days,
                schedule_type=schedule_type,
                seed=cur_seed,
                weights=_weights,
            )
        else:
            sched_s = fixed_schedule

        # Baseline simulation
        sim_b = Simulation(
            n_days=n_days,
            providers=list(providers_tuple),
            schedule=sched_s,
            events=ev_s,
            seed=cur_seed,
            readiness_model=readiness_model,
            readiness_threshold_days=readiness_threshold,
            readiness_half_life_days=readiness_half_life,
            ebbinghaus_b=ebbinghaus_b,
            step_t2_days=step_t2,
            step_partial_value=step_partial,
            training_program="none",
            training_interval_days=training_interval,
            training_start_day=training_start,
            training_effect=training_effect,
            training_equivalence=training_equivalence,
            training_target_threshold=training_threshold,
        )
        sim_b.run()

        # Training simulation (only if program is set)
        sim_t = None
        if training_program != "none":
            sim_t = Simulation(
                n_days=n_days,
                providers=list(providers_tuple),
                schedule=sched_s,
                events=ev_s,
                seed=cur_seed,
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
            sim_t.run()

        # Collect per-seed scalars
        readiness_b_list.append(sim_b.proportion_ready_on_shift.copy())
        pct_exceeding_list.append(100.0 * sim_b.results_df["max_gap_exceeds_threshold"].mean())
        median_gap_list.append(float(sim_b.results_df["gap_median"].dropna().median()))
        median_n_events_list.append(float(sim_b.results_df["n_events"].dropna().median()))

        if sim_t is not None:
            readiness_t_list.append(sim_t.proportion_ready_on_shift.copy())
            lift_list.append(
                (np.nanmean(sim_t.proportion_ready_on_shift)
                 - np.nanmean(sim_b.proportion_ready_on_shift)) * 100
            )

        # Keep reference run data (seed 0)
        if s == 0:
            ref_results_df = sim_b.results_df.copy()
            ref_events_df = sim_b.events.copy()
            ref_schedule = sim_b.schedule.copy()
            ref_proportion_b = sim_b.proportion_ready_on_shift.copy()
            ref_proportion_t = sim_t.proportion_ready_on_shift.copy() if sim_t else None
            ref_training_matrix = sim_t.training_matrix.copy() if sim_t else None

    return {
        "readiness_b":       np.array(readiness_b_list),
        "readiness_t":       np.array(readiness_t_list) if readiness_t_list else None,
        "pct_exceeding":     np.array(pct_exceeding_list),
        "median_gap":        np.array(median_gap_list),
        "median_n_events":   np.array(median_n_events_list),
        "lift":              np.array(lift_list) if lift_list else None,
        "n_days":            n_days,
        "n_samples":         n_samples,
        "threshold":         readiness_threshold,
        "training_program":  training_program,
        "ref_results_df":    ref_results_df,
        "ref_events_df":     ref_events_df,
        "ref_schedule":      ref_schedule,
        "ref_proportion_b":  ref_proportion_b,
        "ref_proportion_t":  ref_proportion_t,
        "ref_training_matrix": ref_training_matrix,
        "providers":         list(providers_tuple),
    }


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
        help="Set for reproducible results. MC runs use seed, seed+1, seed+2, …",
    )
    st.session_state.seed = int(seed)

    thresh_sidebar = st.number_input(
        "Critical threshold (days)",
        min_value=1,
        value=st.session_state.readiness_threshold,
        step=1,
        help="Maximum acceptable gap between HALO event exposures.",
    )
    thresh_sidebar = int(thresh_sidebar)
    if thresh_sidebar > n_days:
        st.error(
            f"Threshold ({thresh_sidebar} d) exceeds the simulation window ({n_days} d)."
        )
        thresh_sidebar = st.session_state.readiness_threshold
    else:
        st.session_state.readiness_threshold = thresh_sidebar
    st.session_state.readiness_model = "binary"

    # ── Training program ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Training program")

    _prog_display_map = {
        "None (exposure only)":       "none",
        "Monthly (every 30 days)":    "monthly",
        "Bi-monthly (every 60 days)": "bimonthly",
        "Quarterly (every 91 days)":  "quarterly",
        "Custom interval":            "custom",
    }
    _prog_labels = list(_prog_display_map.keys())
    _current_label = {v: k for k, v in _prog_display_map.items()}.get(
        st.session_state.training_program, "None (exposure only)"
    )
    _sel_label = st.selectbox(
        "Program",
        _prog_labels,
        index=_prog_labels.index(_current_label),
        key="training_prog_select",
        label_visibility="collapsed",
    )
    _sel_prog = _prog_display_map[_sel_label]
    st.session_state.training_program = _sel_prog

    if _sel_prog == "custom":
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            _ti = st.slider("Interval (days)", 7, 365,
                            st.session_state.training_interval, key="ti_custom")
            st.session_state.training_interval = _ti
        with _tc2:
            _ts = st.slider("First session (day)", 0, 90,
                            st.session_state.training_start, key="ts_custom")
            st.session_state.training_start = _ts

    if _sel_prog != "none":
        _effect_opts = ["Full reset", "Partial boost"]
        _eff = st.radio(
            "Effectiveness",
            _effect_opts,
            horizontal=True,
            index=0 if st.session_state.training_effect == "full" else 1,
            key="train_effect",
        )
        st.session_state.training_effect = "full" if _eff == "Full reset" else "partial"
        if st.session_state.training_effect == "partial":
            _eq = st.slider(
                "Equivalence factor",
                0.1, 1.0, st.session_state.training_equivalence, 0.05,
                help="1.0 = same as live exposure",
                key="train_eq",
            )
            st.session_state.training_equivalence = _eq

    # ── Run ────────────────────────────────────────────────────────────────
    st.divider()

    with st.expander("Advanced"):
        _n_samp = st.slider(
            "MC samples (N)",
            min_value=1,
            max_value=200,
            value=st.session_state.mc_n_samples,
            step=1,
            help="Each sample re-draws event timing and shift assignments from a different seed.",
        )
        st.session_state.mc_n_samples = _n_samp
        if _n_samp == 1:
            st.caption("N=1 — single-seed run (fastest, no distribution).")
        elif _n_samp > 100:
            st.caption(f"N={_n_samp} — may take 30–60 s for 200 providers × 365 days.")
        else:
            st.caption(f"N={_n_samp} — seeds {int(seed)} … {int(seed) + _n_samp - 1}.")

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.divider()
    st.caption("Built by [Sangfroid Labs](https://sangfroidlabs.com)")


# ---------------------------------------------------------------------------
# Result banner
# ---------------------------------------------------------------------------

if st.session_state.mc_ran and st.session_state.mc_result is not None:
    _mc = st.session_state.mc_result
    if st.session_state._last_mc_hash == _mc_hash():
        _n_samp_done = _mc["n_samples"]
        _prog_label = {v: k for k, v in _prog_display_map.items()}.get(
            _mc["training_program"], "None (exposure only)"
        )
        _run_note = f"{_n_samp_done} MC run{'s' if _n_samp_done != 1 else ''}"
        _train_note = f" · {_prog_label}" if _mc["training_program"] != "none" else ""
        st.success(
            f"✓ {len(_mc['providers']):,} providers × {_mc['n_days']} days"
            f" · {_run_note}{_train_note}"
            " — results in the tabs below."
        )
    else:
        st.warning("Settings changed — click **▶ Run Simulation** to update results.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_start, tab_events, tab_schedules, tab_exposure, tab_training = st.tabs(
    ["🟢 Start Here", "🔥 Events", "📅 Schedules", "📊 Exposure Analysis", "🏋️ Training Effects"]
)

# ── Tab 0: Start Here ──────────────────────────────────────────────────────

with tab_start:
    st.header("Welcome to HaloSim")
    st.markdown(
        "HaloSim models how often providers encounter **high-acuity low-occurrence (HALO) events** "
        "on shift, and simulates whether training programs can fill the readiness gaps that result "
        "from infrequent live exposure. Based on the methodology in **PMID: 41633464**."
    )

    st.divider()
    st.subheader("How to run a simulation")

    steps = [
        ("🔧 Sidebar", "Set your **simulation window** (e.g. 365 days), **number of providers**, "
         "**random seed**, and **critical threshold** (the maximum acceptable gap in days). "
         "Choose a **training program** (or leave as None). "
         "Under **Advanced**, set the number of MC samples (N=50 by default)."),
        ("🔥 Events tab", "Choose **Generate** to draw events from a Poisson model (set events/year), "
         "or **Upload** a CSV with columns `date` and `shift_type` (day/night)."),
        ("📅 Schedules tab", "Choose a **schedule type** (e.g. 3/7 Day) to randomly generate "
         "shift schedules, or **Upload** a CSV with columns `provider_id`, `date`, `shift_type`."),
        ("▶ Run Simulation", "Click **▶ Run Simulation** in the sidebar. HaloSim runs N simulations "
         "simultaneously — each with a different seed controlling event timing and shift assignments. "
         "Results are shown as distributions, not single estimates."),
        ("📊 Exposure Analysis tab", "View **gap statistics** as distributions across the N runs — "
         "how often providers exceed the threshold, median gaps, and readiness over time."),
        ("🏋️ Training Effects tab", "If a training program is selected, this tab shows the "
         "distribution of training lift across all MC runs — how much readiness improves, "
         "and how variable that improvement is."),
    ]

    for i, (label, desc) in enumerate(steps, 1):
        c_num, c_label, c_desc = st.columns([0.35, 1.2, 4])
        with c_num:
            st.markdown(f"### {i}")
        with c_label:
            st.markdown(f"**{label}**")
        with c_desc:
            st.markdown(desc)
        if i < len(steps):
            st.markdown("<hr style='margin:0.3rem 0; border-color:#E2E8F0'>",
                        unsafe_allow_html=True)

    st.divider()
    st.subheader("Key concepts")
    st.markdown("""
- **HALO event** — A high-acuity, low-occurrence event (e.g. cardiac arrest) that requires practiced skills but happens rarely enough that most providers go months between exposures.
- **Readiness threshold** — The maximum gap (in days) before a provider is considered under-exposed. Default: 90 days.
- **Gap** — The interval between consecutive exposures for a single provider, including lead-in and trail-out.
- **On-shift readiness** — The proportion of providers who are *currently on shift* and within their readiness threshold. Off-shift providers are excluded.
- **MC sampling** — Each run re-draws event timing and shift assignments from a different random seed, producing a distribution of outcomes rather than a single point estimate.
""")

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
        _lc1, _lc2 = st.columns(2)
        with _lc1:
            if st.button("Load sample events (48 events / year)", use_container_width=True):
                _raw = (DATA_DIR / "sample_events.csv").read_bytes()
                _df, _errs = load_events_from_upload(_raw, "sample_events.csv", n_days)
                if _df is not None:
                    st.session_state.events_df    = _df
                    st.session_state.event_source = "Upload CSV / Excel"
                    st.session_state.events_errors = _errs
                    st.rerun()
        with _lc2:
            if st.button("Load full demo scenario", use_container_width=True,
                         help="Loads sample events + sample schedule (20 providers) and runs the simulation."):
                _raw_ev = (DATA_DIR / "sample_events.csv").read_bytes()
                _df_ev, _ = load_events_from_upload(_raw_ev, "sample_events.csv", 365)
                _raw_sc = (DATA_DIR / "sample_schedule.csv").read_bytes()
                _arr, _provs, _ = load_schedule_from_upload(_raw_sc, "sample_schedule.csv", 365)
                if _df_ev is not None and _arr is not None:
                    st.session_state.events_df          = _df_ev
                    st.session_state.event_source       = "Upload CSV / Excel"
                    st.session_state.events_errors      = []
                    st.session_state.schedule_array     = _arr
                    st.session_state.schedule_providers = _provs
                    st.session_state.schedule_source    = "Upload CSV / Excel"
                    st.session_state.schedule_errors    = []
                    st.session_state.n_days             = 365
                    st.session_state.n_providers        = len(_provs)
                    st.session_state._auto_run          = True
                    st.rerun()

        st.divider()
        rate_per_year = st.slider(
            "Event rate (events / year)",
            min_value=1,
            max_value=365,
            value=round(st.session_state.event_rate * 365),
            step=1,
            help="~51/year matches cardiac arrest rate in PMID: 41633464",
        )
        rate = rate_per_year / 365.0
        st.session_state.event_rate = rate
        st.caption(
            f"~**{rate * 30.44:.1f} events/month** &nbsp;·&nbsp; "
            f"{rate * n_days:.0f} expected over {n_days} days"
        )

        with st.expander("Advanced event settings"):
            day_pct = st.slider(
                "% occurring on day shifts",
                min_value=0, max_value=100,
                value=st.session_state.get("event_day_pct", 50),
                step=5,
                help="Remaining % occurs on night shifts. Default 50/50.",
            )
            st.session_state.event_day_pct = day_pct
            st.caption(
                f"Day rate: {rate * day_pct / 100:.3f}/day &nbsp;·&nbsp; "
                f"Night rate: {rate * (100 - day_pct) / 100:.3f}/day"
            )

    else:
        st.caption("Expected format:")
        _sample_ev = pd.DataFrame({
            "date":       ["2024-01-03", "2024-01-11", "2024-01-19", "2024-02-01", "2024-02-14"],
            "shift_type": ["day",        "night",       "day",        "night",      "day"],
        })
        st.dataframe(_sample_ev, use_container_width=False, hide_index=True)
        st.caption("Optional: add an `hour` column (0–23) to enable shift-boundary join (Advanced).")

        uploaded = st.file_uploader("Upload events file", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            if uploaded.name != st.session_state.events_upload_name:
                st.session_state.events_upload_bytes = uploaded.read()
                st.session_state.events_upload_name  = uploaded.name
                st.session_state.events_errors       = []
                st.session_state.events_df           = None
            _ev_bytes = st.session_state.events_upload_bytes
            _ev_name  = st.session_state.events_upload_name
            df, errs = load_events_from_upload(_ev_bytes, _ev_name, n_days, allow_hour_col=False)
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
            st.caption("Enable the hour column and complex shift-boundary join.")
            allow_hour_adv = st.checkbox("File includes 'hour' column (0–23)", value=False,
                                         key="allow_hour_adv")
            if allow_hour_adv and st.session_state.events_upload_bytes:
                df2, errs2 = load_events_from_upload(
                    st.session_state.events_upload_bytes,
                    st.session_state.events_upload_name,
                    n_days, allow_hour_col=True,
                )
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
                    st.warning("Complex join requires an 'hour' column. Falling back to simple join.")


# ── Tab 2: Schedules ───────────────────────────────────────────────────────

with tab_schedules:
    st.header("Provider Schedule Configuration")

    sched_options = ["Generate schedules", "Upload CSV / Excel"]

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
            "Random":    "Each day drawn from empirical d/n/o weights (PMID: 41633464: 25% day, 23% night, 52% off)",
        }
        current_type = st.session_state.get("schedule_type", DEFAULT_SCHEDULE_TYPE)
        if current_type not in SCHEDULE_TYPES:
            current_type = DEFAULT_SCHEDULE_TYPE

        selected_type = st.radio(
            "Schedule type",
            SCHEDULE_TYPES,
            index=SCHEDULE_TYPES.index(current_type),
            captions=list(_type_descriptions.values()),
        )
        st.session_state.schedule_type = selected_type

        with st.expander("Advanced schedule settings"):
            st.caption(
                "Override with a custom shift distribution. Each provider gets a "
                "randomly generated schedule drawn from these percentages."
            )
            _d_default = st.session_state.get("schedule_day_pct") or 25
            _n_default = st.session_state.get("schedule_night_pct") or 23
            _d_pct = st.slider("% day shifts", 0, 100, _d_default, 5, key="sched_day_pct")
            _n_max = 100 - _d_pct
            if _n_max > 0:
                _n_pct = st.slider("% night shifts", 0, _n_max,
                                   min(_n_default, _n_max), 5, key="sched_night_pct")
            else:
                _n_pct = 0
                st.caption("Night: 0% (day shifts at 100%)")
            _o_pct = 100 - _d_pct - _n_pct
            st.caption(f"Day: {_d_pct}% &nbsp;·&nbsp; Night: {_n_pct}% &nbsp;·&nbsp; Off: {_o_pct}%")
            _use_custom = st.checkbox(
                "Use these percentages (overrides schedule type above)",
                value=(st.session_state.get("schedule_day_pct") is not None),
                key="use_custom_sched_weights",
            )
            if _use_custom:
                st.session_state.schedule_day_pct = _d_pct
                st.session_state.schedule_night_pct = _n_pct
            else:
                st.session_state.schedule_day_pct = None
                st.session_state.schedule_night_pct = None

    elif sched_source == "Upload CSV / Excel":
        st.caption("Expected format:")
        _sample_sched = pd.DataFrame({
            "provider_id": ["P0001", "P0001", "P0001", "P0002", "P0002"],
            "date":        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"],
            "shift_type":  ["day",        "off",         "night",      "night",      "off"],
        })
        st.dataframe(_sample_sched, use_container_width=False, hide_index=True)
        st.caption("One row per provider per day. `shift_type`: day / night / off (or d / n / o).")

        uploaded_s = st.file_uploader(
            "Upload schedule file",
            type=["csv", "xlsx", "xls"],
            key="schedule_upload",
        )
        if uploaded_s is not None:
            if uploaded_s.name != st.session_state.schedule_upload_name:
                st.session_state.schedule_upload_bytes = uploaded_s.read()
                st.session_state.schedule_upload_name  = uploaded_s.name
                st.session_state.schedule_errors       = []
                st.session_state.schedule_array        = None
                st.session_state.schedule_providers    = None
            _sc_bytes = st.session_state.schedule_upload_bytes
            _sc_name  = st.session_state.schedule_upload_name
            arr, providers, errs = load_schedule_from_upload(_sc_bytes, _sc_name, n_days)
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

    if not st.session_state.mc_ran or st.session_state.mc_result is None:
        st.info("Configure Events and Schedules above, then click **▶ Run Simulation** in the sidebar.")
    else:
        mc = st.session_state.mc_result
        rdf = mc["ref_results_df"]
        thresh = mc["threshold"]
        n_samp = mc["n_samples"]
        n = len(rdf)

        # Both sources uploaded → identical runs
        _both_fixed = (
            st.session_state.event_source == "Upload CSV / Excel"
            and st.session_state.schedule_source == "Upload CSV / Excel"
        )
        if _both_fixed and n_samp > 1:
            st.warning(
                "Both events and schedule are uploaded — all MC runs are identical. "
                "Results are a point estimate, not a distribution."
            )

        # Summary metrics (distributions across seeds)
        st.subheader("Key metrics across MC runs")
        _pct = mc["pct_exceeding"]
        _gap = mc["median_gap"]
        _nev = mc["median_n_events"]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"% exceeding {thresh}-day threshold",
            f"{np.median(_pct):.1f}%",
            help=f"Median across {n_samp} runs. Range (p10–p90): {np.percentile(_pct, 10):.1f}–{np.percentile(_pct, 90):.1f}%",
        )
        c2.metric(
            "Median gap between exposures",
            f"{np.median(_gap):.0f} days",
            help=f"Median across {n_samp} runs. Range (p10–p90): {np.percentile(_gap, 10):.0f}–{np.percentile(_gap, 90):.0f} days",
        )
        c3.metric(
            "Median exposures per provider",
            f"{np.median(_nev):.1f}",
            help=f"Median across {n_samp} runs. Range (p10–p90): {np.percentile(_nev, 10):.1f}–{np.percentile(_nev, 90):.1f}",
        )

        # Summary table
        st.divider()
        st.subheader("Distribution summary")
        st.dataframe(build_mc_summary_df(mc), use_container_width=True, hide_index=True)

        # Readiness band chart
        st.divider()
        st.subheader("On-shift readiness over time")
        st.caption(
            "Shaded ribbon = p10–p90 across MC runs. "
            "Solid line = median. "
            "Providers currently off-shift are excluded."
        )
        _roll = st.slider(
            "Rolling mean window (days)",
            1, 90, st.session_state.get("_roll_window_exp", 30),
            key="roll_exp",
        )
        st.session_state["_roll_window_exp"] = _roll
        st.plotly_chart(
            plot_mc_readiness_band(mc["readiness_b"], rolling_days=_roll),
            use_container_width=True,
        )

        # Distribution histograms
        st.divider()
        st.subheader("Distribution across MC runs")
        _h1, _h2, _h3 = st.columns(3)
        with _h1:
            st.plotly_chart(
                plot_mc_histogram(_pct, f"% exceeding {thresh}-day threshold", unit="%"),
                use_container_width=True,
            )
        with _h2:
            st.plotly_chart(
                plot_mc_histogram(_gap, "Median gap", unit=" days"),
                use_container_width=True,
            )
        with _h3:
            st.plotly_chart(
                plot_mc_histogram(_nev, "Median exposures / provider"),
                use_container_width=True,
            )

        # Reference-run charts (seed-0 single realisation)
        st.divider()
        st.subheader("Reference run detail (seed-0 realisation)")
        st.caption(
            "The charts below are from the reference run (seed 0). "
            "They show the distributional structure within a single realisation — "
            "the distribution charts above show how this varies across realizations."
        )
        st.plotly_chart(plot_exposure_count_histogram(rdf), use_container_width=True)
        st.plotly_chart(plot_gap_distribution(rdf), use_container_width=True)
        st.plotly_chart(plot_threshold_sweep(rdf, threshold=thresh), use_container_width=True)

        _pct_t = 100 * (rdf["gap_max"].fillna(9999) > thresh).mean()
        st.info(
            f"Reference run: **{_pct_t:.0f}%** of providers exceeded the {thresh}-day threshold. "
            f"Across all {n_samp} runs: {np.median(_pct):.0f}% (p10–p90: "
            f"{np.percentile(_pct, 10):.0f}–{np.percentile(_pct, 90):.0f}%)."
        )

        # Downloads
        st.divider()
        st.subheader("Downloads")
        _dl1, _dl2, _dl3 = st.columns(3)

        with _dl1:
            _rdf_rounded = rdf.copy()
            if "gap_mean" in _rdf_rounded.columns:
                _rdf_rounded["gap_mean"] = _rdf_rounded["gap_mean"].round(3)
            st.download_button(
                "📥 Exposure stats — ref run (CSV)",
                data=_rdf_rounded.to_csv(index=False).encode(),
                file_name="halosim_exposure_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with _dl2:
            _ev_dl = mc["ref_events_df"][["day_idx", "shift_type"]].copy()
            if "date" in mc["ref_events_df"].columns:
                _ev_dl.insert(0, "date", mc["ref_events_df"]["date"])
            st.download_button(
                "📥 Events — ref run (CSV)",
                data=_ev_dl.to_csv(index=False).encode(),
                file_name="halosim_events.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with _dl3:
            # MC scalar results CSV
            _mc_dl = pd.DataFrame({
                "seed": [int(seed) + s for s in range(n_samp)],
                "pct_exceeding_threshold": mc["pct_exceeding"].round(2),
                "median_gap_days": mc["median_gap"].round(1),
                "median_n_events": mc["median_n_events"].round(1),
            })
            if mc["lift"] is not None:
                _mc_dl["training_lift_pp"] = mc["lift"].round(2)
            st.download_button(
                "📥 MC scalar results (CSV)",
                data=_mc_dl.to_csv(index=False).encode(),
                file_name="halosim_mc_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Simulated events detail
        with st.expander("View simulated events (reference run)"):
            _ev = mc["ref_events_df"]
            _n_ev = len(_ev)
            _n_day_ev = int((_ev["shift_type"] == "day").sum())
            _n_night_ev = int((_ev["shift_type"] == "night").sum())
            _expected = round(st.session_state.event_rate * mc["n_days"])

            ec1, ec2, ec3 = st.columns(3)
            if st.session_state.event_source == "Generate (Poisson MC)":
                ec1.metric("Total events (seed-0)", _n_ev,
                           delta=f"{_n_ev - _expected:+d} vs expected {_expected}",
                           delta_color="off")
            else:
                ec1.metric("Total events (uploaded)", _n_ev)
            ec2.metric("Day shift", f"{_n_day_ev}  ({100*_n_day_ev//_n_ev if _n_ev else 0}%)")
            ec3.metric("Night shift", f"{_n_night_ev}  ({100*_n_night_ev//_n_ev if _n_ev else 0}%)")

            _ev2 = _ev.copy()
            _ev2["month"] = (_ev2["day_idx"] // 30).clip(upper=max(0, mc["n_days"] // 30 - 1))
            _monthly_day = _ev2[_ev2["shift_type"] == "day"].groupby("month").size().reset_index(name="count")
            _monthly_ngt = _ev2[_ev2["shift_type"] == "night"].groupby("month").size().reset_index(name="count")
            _all_months = list(range(mc["n_days"] // 30))
            _labels = [f"Mo {m+1}" for m in _all_months]
            _day_counts = [int(_monthly_day.set_index("month")["count"].get(m, 0)) for m in _all_months]
            _ngt_counts = [int(_monthly_ngt.set_index("month")["count"].get(m, 0)) for m in _all_months]
            import plotly.graph_objects as _go
            _fig_ev = _go.Figure([
                _go.Bar(x=_labels, y=_day_counts, name="Day", marker_color="#F59E0B", opacity=0.85),
                _go.Bar(x=_labels, y=_ngt_counts, name="Night", marker_color="#2563EB", opacity=0.85),
            ])
            _fig_ev.update_layout(
                title="Events per month (reference run)",
                barmode="stack",
                height=260, margin=dict(t=40, b=20, l=30, r=10),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", x=0.01, y=1.12),
            )
            _fig_ev.update_xaxes(gridcolor="#E2E8F0")
            _fig_ev.update_yaxes(gridcolor="#E2E8F0", title="Events")
            st.plotly_chart(_fig_ev, use_container_width=True)

        with st.expander("View simulated schedules (reference run)"):
            _sched = mc["ref_schedule"]
            _provs = mc["providers"]
            _n_p, _n_d = _sched.shape
            _day_ct = (_sched == "d").sum(axis=1)
            _ngt_ct = (_sched == "n").sum(axis=1)
            _off_ct = (_sched == "o").sum(axis=1)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Avg day shifts / provider",
                       f"{_day_ct.mean():.1f}  ({100*_day_ct.mean()/_n_d:.0f}%)")
            sc2.metric("Avg night shifts / provider",
                       f"{_ngt_ct.mean():.1f}  ({100*_ngt_ct.mean()/_n_d:.0f}%)")
            sc3.metric("Avg days off / provider",
                       f"{_off_ct.mean():.1f}  ({100*_off_ct.mean()/_n_d:.0f}%)")

            _rng_s = np.random.default_rng(int(seed))
            _s_idx = np.sort(_rng_s.choice(_n_p, size=min(10, _n_p), replace=False))
            _show_days = min(42, _n_d)
            _n_weeks = _show_days // 7
            _submat = _sched[np.ix_(_s_idx, np.arange(_show_days))]
            _enc = np.where(_submat == "d", 2, np.where(_submat == "n", 1, 0)).astype(float)
            _plabels = [_provs[i] for i in _s_idx]
            _n_samp_s = len(_s_idx)

            _fig_sched = _go.Figure(_go.Heatmap(
                z=_enc,
                x=list(range(1, _show_days + 1)),
                y=_plabels,
                colorscale=[[0, "#F1F5F9"], [0.5, "#2563EB"], [1.0, "#F59E0B"]],
                showscale=True,
                colorbar=dict(
                    tickvals=[0.33, 1.0, 1.67],
                    ticktext=["Off", "Night", "Day"],
                    thickness=12, len=0.6,
                ),
                zmin=0, zmax=2,
                xgap=1, ygap=2,
            ))
            for _wk in range(1, _n_weeks):
                _fig_sched.add_vline(
                    x=_wk * 7 + 0.5,
                    line_width=2, line_color="#334155", line_dash="solid",
                )
            _fig_sched.update_layout(
                title=f"Shift pattern — first {_n_weeks} weeks, sample of {_n_samp_s} providers (reference run)",
                height=max(280, _n_samp_s * 28),
                margin=dict(t=50, b=30, l=80, r=60),
                xaxis=dict(title="Day", tickmode="array",
                           tickvals=[w * 7 + 4 for w in range(_n_weeks)],
                           ticktext=[f"Wk {w+1}" for w in range(_n_weeks)]),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(_fig_sched, use_container_width=True)


# ── Tab 4: Training Effects ────────────────────────────────────────────────

with tab_training:
    st.header("Training Effects")

    if not st.session_state.mc_ran or st.session_state.mc_result is None:
        st.info("Configure Events and Schedules above, then click **▶ Run Simulation** in the sidebar.")
    else:
        mc = st.session_state.mc_result
        n_samp = mc["n_samples"]

        if mc["training_program"] == "none":
            st.info(
                "No training program selected. "
                "Choose a program in the sidebar and click **▶ Run Simulation** to see training effects."
            )
        else:
            _prog_label = {v: k for k, v in _prog_display_map.items()}.get(
                mc["training_program"], mc["training_program"]
            )

            # Key metrics
            _b_means = np.array([np.nanmean(row) * 100 for row in mc["readiness_b"]])
            _t_means = np.array([np.nanmean(row) * 100 for row in mc["readiness_t"]])
            _lifts   = mc["lift"]
            _days_b  = np.array([(row < 0.80).sum() for row in mc["readiness_b"]])
            _days_t  = np.array([(row < 0.80).sum() for row in mc["readiness_t"]])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Median readiness — no training",
                f"{np.median(_b_means):.1f}%",
                help=f"p10–p90: {np.percentile(_b_means, 10):.1f}–{np.percentile(_b_means, 90):.1f}%",
            )
            c2.metric(
                "Median readiness — with training",
                f"{np.median(_t_means):.1f}%",
                delta=f"{np.median(_lifts):+.1f} pp",
                help=f"p10–p90: {np.percentile(_t_means, 10):.1f}–{np.percentile(_t_means, 90):.1f}%",
            )
            c3.metric(
                "Days readiness <80% — no training",
                f"{np.median(_days_b):.0f}",
                help=f"Median across {n_samp} runs (p10–p90: {np.percentile(_days_b, 10):.0f}–{np.percentile(_days_b, 90):.0f})",
            )
            c4.metric(
                "Days readiness <80% — with training",
                f"{np.median(_days_t):.0f}",
                delta=f"{np.median(_days_t) - np.median(_days_b):+.0f} days",
                delta_color="inverse",
                help=f"Median across {n_samp} runs (p10–p90: {np.percentile(_days_t, 10):.0f}–{np.percentile(_days_t, 90):.0f})",
            )

            # Reference run training stats
            if mc["ref_training_matrix"] is not None:
                _tm = mc["ref_training_matrix"]
                _n_sess = int(_tm.any(axis=0).sum())
                _n_reached = int(_tm.any(axis=1).sum())
                _c5, _c6, _, _ = st.columns(4)
                _c5.metric("Training sessions — ref run", f"{_n_sess:,}",
                           help="Days with at least one provider trained (reference run, seed 0).")
                _c6.metric("Providers reached — ref run", f"{_n_reached:,}",
                           help="Unique providers trained at least once (reference run, seed 0).")

            st.divider()

            # Readiness band chart
            _roll = st.slider(
                "Rolling mean window (days)",
                1, 90, st.session_state.get("_roll_window", 30),
                key="roll_window",
            )
            st.session_state["_roll_window"] = _roll

            st.plotly_chart(
                plot_mc_readiness_band(
                    mc["readiness_b"], mc["readiness_t"],
                    rolling_days=_roll,
                ),
                use_container_width=True,
            )

            # Lift distribution histogram
            st.plotly_chart(
                plot_mc_histogram(_lifts, "Training lift", unit=" pp"),
                use_container_width=True,
            )

            # Interpretation
            _med_lift = float(np.median(_lifts))
            _p10_lift = float(np.percentile(_lifts, 10))
            _p90_lift = float(np.percentile(_lifts, 90))
            if abs(_med_lift) < 1:
                _t_interp = (
                    f"Training had minimal effect on median readiness across {n_samp} runs — "
                    "live exposure alone may be sufficient at this event rate."
                )
            elif _med_lift > 0:
                _t_interp = (
                    f"**{_prog_label}** raised median on-shift readiness by "
                    f"**{_med_lift:+.1f} pp** across {n_samp} MC runs "
                    f"(p10–p90: {_p10_lift:+.1f} to {_p90_lift:+.1f} pp). "
                    f"The width of this interval reflects how much the training benefit depends "
                    "on event timing and shift assignments."
                )
            else:
                _t_interp = (
                    f"Median readiness with training is similar to baseline "
                    f"({_med_lift:+.1f} pp). Consider adjusting training frequency."
                )
            st.info(_t_interp)

            st.divider()

            # Training program comparison (single-seed, uses _run_sim)
            _compare_options = {
                "No training":       "none",
                "Monthly (30d)":     "monthly",
                "Bi-monthly (60d)":  "bimonthly",
                "Quarterly (91d)":   "quarterly",
                "Custom":            "custom",
            }
            _active_label = {v: k for k, v in _compare_options.items()}.get(
                mc["training_program"], "No training"
            )
            _default_sel = ["No training"] + (
                [_active_label] if _active_label != "No training" else []
            )

            _use_my_settings = st.toggle(
                "Use my current Custom settings for comparison",
                value=True,
                key="compare_use_my_settings",
                help=(
                    "ON: Custom line uses your current interval / effectiveness settings.  "
                    "OFF: all programs use standardised defaults (monthly=30d, "
                    "bi-monthly=60d, quarterly=91d, custom=30d)."
                ),
            )

            _selected = st.multiselect(
                "Programs to compare (reference run)",
                list(_compare_options.keys()),
                default=_default_sel,
                key="compare_programs",
            )

            if _selected:
                _s = st.session_state
                _compare_data: dict[str, np.ndarray] = {}
                _training_days: dict[str, list[int]] = {}
                _std_intervals = {"monthly": 30, "bimonthly": 60, "quarterly": 91, "custom": 30}
                for _lbl in _selected:
                    _prog = _compare_options[_lbl]
                    _is_active = _use_my_settings and (_prog == mc["training_program"])
                    _interval = (_s.training_interval if _is_active else _std_intervals.get(_prog, 30))
                    _start    = (_s.training_start   if _is_active else 14)
                    _csim = _run_sim(
                        mc["n_days"], tuple(mc["providers"]),
                        mc["ref_schedule"], mc["ref_events_df"],
                        int(_s.seed),
                        _s.readiness_model, _s.readiness_threshold,
                        _s.readiness_half_life, _s.ebbinghaus_b,
                        _s.step_t2, _s.step_partial,
                        _prog, _interval, _start,
                        _s.training_effect      if _is_active else "full",
                        _s.training_equivalence if _is_active else 1.0,
                        _s.training_threshold   if _is_active else 0.5,
                    )
                    _compare_data[_lbl] = _csim.proportion_ready_on_shift
                    if _prog != "none":
                        _training_days[_lbl] = list(
                            np.arange(_start, mc["n_days"], _interval, dtype=int)
                        )
                st.caption(
                    "Comparison chart uses the reference run (seed 0). "
                    "Vertical lines mark training session days."
                )
                st.plotly_chart(
                    plot_training_comparison(
                        _compare_data, mc["n_days"], _roll,
                        training_days=_training_days,
                    ),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Run simulation (triggered by sidebar button or auto-run flag)
# ---------------------------------------------------------------------------

if run_btn or st.session_state.get("_auto_run", False):
    st.session_state._auto_run = False
    errors = []

    # 1. Build events
    if st.session_state.event_source == "Generate (Poisson MC)":
        events_df = None   # generated inside _run_mc per seed
        fixed_events_df = None
    else:
        fixed_events_df = st.session_state.events_df
        if fixed_events_df is None:
            errors.append("No events loaded. Upload a file or switch to 'Generate' mode.")
        events_df = fixed_events_df

    # 2. Build schedule
    sched_source = st.session_state.schedule_source
    if sched_source == "Upload CSV / Excel" and st.session_state.schedule_array is not None:
        arr = st.session_state.schedule_array
        if arr.shape[1] >= n_days:
            fixed_schedule = arr[:n_providers, :n_days]
        else:
            fixed_schedule = None  # uploaded but wrong length → fall back to generate
        providers_list = (st.session_state.schedule_providers or [])[:n_providers] \
            or [f"P{i+1:04d}" for i in range(n_providers)]
    else:
        fixed_schedule = None
        providers_list = [f"P{i+1:04d}" for i in range(n_providers)]

    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.stop()

    _s = st.session_state

    with st.spinner(f"Running {_s.mc_n_samples} simulation{'s' if _s.mc_n_samples != 1 else ''}…"):
        _mc = _run_mc(
            n_days=n_days,
            providers_tuple=tuple(providers_list),
            fixed_schedule=fixed_schedule,
            fixed_events_df=fixed_events_df,
            event_source=_s.event_source,
            event_rate=_s.event_rate,
            event_day_pct=_s.get("event_day_pct", 50),
            schedule_source=_s.schedule_source,
            schedule_type=_s.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
            schedule_day_pct=_s.get("schedule_day_pct"),
            schedule_night_pct=_s.get("schedule_night_pct"),
            readiness_model=_s.readiness_model,
            readiness_threshold=_s.readiness_threshold,
            readiness_half_life=_s.readiness_half_life,
            ebbinghaus_b=_s.ebbinghaus_b,
            step_t2=_s.step_t2,
            step_partial=_s.step_partial,
            training_program=_s.training_program,
            training_interval=_s.training_interval,
            training_start=_s.training_start,
            training_effect=_s.training_effect,
            training_equivalence=_s.training_equivalence,
            training_threshold=_s.training_threshold,
            n_samples=_s.mc_n_samples,
            base_seed=int(_s.seed),
        )

    st.session_state.mc_result = _mc
    st.session_state.mc_ran = True
    st.session_state._last_mc_hash = _mc_hash()
    st.rerun()
