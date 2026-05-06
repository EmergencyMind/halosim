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
    plot_readiness_baseline,
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
        "training_ran": False,
        "_last_run_hash": None,
        "_last_training_hash": None,
        # upload byte caches (prevent double-read bug)
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
    """Stable string fingerprint of main simulation inputs (excludes training)."""
    s = st.session_state
    parts = [
        s.n_days, s.n_providers, s.seed,
        # events
        s.event_source, s.event_rate, s.get("event_day_pct", 50),
        # schedules
        s.schedule_source,
        s.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
        s.get("schedule_day_pct"), s.get("schedule_night_pct"),
        # uploaded data: use row-count / shape as proxy
        len(s.events_df) if s.events_df is not None else -1,
        str(s.schedule_array.shape) if s.schedule_array is not None else "none",
        # readiness
        s.readiness_model, s.readiness_threshold, s.readiness_half_life,
        s.ebbinghaus_b, s.step_t2, s.step_partial,
    ]
    return ":".join(str(p) for p in parts)


def _training_hash() -> str:
    """Fingerprint of training-specific inputs on top of the main sim hash."""
    s = st.session_state
    parts = [
        _sim_hash(),
        s.training_program, s.training_interval, s.training_start,
        s.training_effect, s.training_equivalence, s.training_threshold,
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

    thresh_sidebar = st.number_input(
        "Critical threshold (days)",
        min_value=1,
        value=st.session_state.readiness_threshold,
        step=1,
        help="Maximum acceptable gap between HALO event exposures. "
             "Providers exceeding this are considered under-exposed.",
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
            f"✓ {len(_sim.providers):,} providers × {_sim.n_days} days"
            " — results in the **Exposure Analysis** tab."
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
         "**random seed**, and **critical threshold** (the maximum acceptable gap in days between "
         "exposures — default 90). These apply to all tabs."),
        ("🔥 Events tab", "Choose **Generate** to draw events from a Poisson model (set events/year), "
         "or **Upload** a CSV with columns `date` and `shift_type` (day/night). "
         "The event rate drives how often providers are exposed on shift."),
        ("📅 Schedules tab", "Choose a **schedule type** (e.g. 3/7 Day, Progressive) to randomly "
         "generate a shift schedule for each provider, or **Upload** a CSV with columns "
         "`provider_id`, `date`, and `shift_type` (day/night/off). "
         "Each provider gets an independently randomised schedule."),
        ("▶ Run Simulation", "Click **▶ Run Simulation** in the sidebar. Results are cached — "
         "re-run any time you change settings. The banner at the top will turn orange if "
         "settings have changed since the last run."),
        ("📊 Exposure Analysis tab", "View **gap statistics**: how many days pass between "
         "exposures for each provider, including the lead-in (day 0 → first exposure) and "
         "trail-out (last exposure → end of window). Adjust the **readiness threshold** to "
         "define the maximum acceptable gap."),
        ("🏋️ Training Effects tab", "After running the simulation, go to the Training Effects tab. "
         "Select a **training program** (None / Monthly / Bi-monthly / Quarterly / Custom / Targeted) "
         "and click **🏋️ Run Training Simulation**. Results include a readiness comparison chart — "
         "add more programs to the multiselect to overlay them."),
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
- **Readiness threshold** — The maximum gap (in days) before a provider is considered to have lost procedural readiness. Default: 90 days.
- **Gap** — The interval between consecutive exposures for a single provider. Includes lead-in (days before first exposure) and trail-out (days after last exposure to end of window).
- **On-shift readiness** — The proportion of providers who are *currently on shift* and within their readiness threshold. This is the primary metric — off-shift providers are excluded.
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
        # ── One-click sample / demo loaders ──────────────────────────────────
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
                # events
                _raw_ev = (DATA_DIR / "sample_events.csv").read_bytes()
                _df_ev, _ = load_events_from_upload(_raw_ev, "sample_events.csv", 365)
                # schedule
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
            help="Number of HALO events per year across the unit. ~51/year matches cardiac arrest rate in PMID: 41633464",
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

    else:  # upload
        st.caption("Expected format:")
        _sample_ev = pd.DataFrame({
            "date":       ["2024-01-03", "2024-01-11", "2024-01-19", "2024-02-01", "2024-02-14"],
            "shift_type": ["day",        "night",       "day",        "night",      "day"],
        })
        st.dataframe(_sample_ev, use_container_width=False, hide_index=True)
        st.caption("Optional: add an `hour` column (0–23) to enable shift-boundary join (Advanced).")

        uploaded = st.file_uploader(
            "Upload events file",
            type=["csv", "xlsx", "xls"],
        )
        # Read bytes once and cache — prevents empty re-read on subsequent reruns
        if uploaded is not None:
            if uploaded.name != st.session_state.events_upload_name:
                # New file: read and cache
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
                # Reuse cached bytes — no second read needed
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
        # Read bytes once and cache — prevents empty re-read on subsequent reruns
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
            c1, c2, c3 = st.columns(3)
            c1.metric("Median exposures / provider",
                      f"{rdf['n_events'].median():.1f}")
            c2.metric("Median days between exposures",
                      f"{rdf['gap_median'].dropna().median():.0f}")
            c3.metric(f"% exceeding {thresh}-day critical threshold",
                      f"{100 * n_exceed / n:.1f}%")

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
            st.plotly_chart(plot_threshold_sweep(rdf, threshold=thresh),
                            use_container_width=True)

            # Interpretation callout for threshold sweep
            _pct_t = 100 * (rdf["gap_max"].fillna(9999) > thresh).mean()
            st.info(f"**{_pct_t:.0f}%** of providers exceeded the {thresh}-day critical threshold.")

            # Readiness over time (baseline, no training)
            st.divider()
            st.subheader("On-shift readiness over time")
            st.caption(
                "This chart shows the proportion of providers who are **currently on shift** "
                "and within the readiness threshold — meaning they have had a live HALO exposure "
                f"within the last {thresh} days. Providers who are off-shift are excluded. "
                "This is the readiness picture produced by live exposure alone, with no training program."
            )
            st.plotly_chart(
                plot_readiness_baseline(sim.proportion_ready_on_shift, sim.n_days),
                use_container_width=True,
            )

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
            st.subheader("Downloads")
            _dl1, _dl2 = st.columns(2)

            # 1. Exposure statistics CSV (gap_mean rounded to 3 dp)
            with _dl1:
                _rdf_rounded = rdf.copy()
                if "gap_mean" in _rdf_rounded.columns:
                    _rdf_rounded["gap_mean"] = _rdf_rounded["gap_mean"].round(3)
                st.download_button(
                    "📥 Exposure statistics (CSV)",
                    data=_rdf_rounded.to_csv(index=False).encode(),
                    file_name="halosim_exposure_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # 2. Simulated events CSV
            with _dl2:
                _ev_dl = sim.events[["day_idx", "shift_type"]].copy()
                if "date" in sim.events.columns:
                    _ev_dl.insert(0, "date", sim.events["date"])
                st.download_button(
                    "📥 Simulated events (CSV)",
                    data=_ev_dl.to_csv(index=False).encode(),
                    file_name="halosim_events.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            _dl3, _dl4 = st.columns(2)

            # 3. Simulated schedules CSV (long format: provider_id, day, shift_type)
            with _dl3:
                _sched_rows = []
                for _pi, _pid in enumerate(sim.providers):
                    for _di in range(sim.n_days):
                        _sched_rows.append((_pid, _di + 1, sim.schedule[_pi, _di]))
                _sched_dl = pd.DataFrame(_sched_rows, columns=["provider_id", "day", "shift_type"])
                st.download_button(
                    "📥 Simulated schedules (CSV)",
                    data=_sched_dl.to_csv(index=False).encode(),
                    file_name="halosim_schedules.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # 4. PDF summary report
            with _dl4:
                if st.button("📄 Generate PDF report", use_container_width=True):
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

            # ── Simulated data detail ──────────────────────────────────────
            st.divider()
            with st.expander("View simulated events"):
                _ev = sim.events
                _n_ev = len(_ev)
                _n_day_ev = int((_ev["shift_type"] == "day").sum())
                _n_night_ev = int((_ev["shift_type"] == "night").sum())
                _expected = round(st.session_state.event_rate * sim.n_days)

                ec1, ec2, ec3 = st.columns(3)
                if st.session_state.event_source == "Generate (Poisson MC)":
                    ec1.metric("Total events (actual)", _n_ev,
                               delta=f"{_n_ev - _expected:+d} vs expected {_expected}",
                               delta_color="off")
                    st.caption(
                        f"Expected ~{_expected} events over {sim.n_days} days at the configured rate. "
                        "Actual count varies around this — Poisson sampling is stochastic, so differences "
                        "of 10–20% are normal. Re-run with a different seed to see the range."
                    )
                else:
                    ec1.metric("Total events (uploaded)", _n_ev)
                    st.caption(f"{_n_ev} events loaded from uploaded file over {sim.n_days} days.")
                ec2.metric("Day shift", f"{_n_day_ev}  ({100*_n_day_ev//_n_ev if _n_ev else 0}%)")
                ec3.metric("Night shift", f"{_n_night_ev}  ({100*_n_night_ev//_n_ev if _n_ev else 0}%)")

                _ev2 = _ev.copy()
                _ev2["month"] = (_ev2["day_idx"] // 30).clip(upper=max(0, sim.n_days // 30 - 1))
                _monthly_day = _ev2[_ev2["shift_type"] == "day"].groupby("month").size().reset_index(name="count")
                _monthly_ngt = _ev2[_ev2["shift_type"] == "night"].groupby("month").size().reset_index(name="count")
                _all_months = list(range(sim.n_days // 30))
                _labels = [f"Mo {m+1}" for m in _all_months]
                _day_counts = [int(_monthly_day.set_index("month")["count"].get(m, 0)) for m in _all_months]
                _ngt_counts = [int(_monthly_ngt.set_index("month")["count"].get(m, 0)) for m in _all_months]
                import plotly.graph_objects as _go
                _fig_ev = _go.Figure([
                    _go.Bar(x=_labels, y=_day_counts, name="Day", marker_color="#F59E0B", opacity=0.85),
                    _go.Bar(x=_labels, y=_ngt_counts, name="Night", marker_color="#2563EB", opacity=0.85),
                ])
                _fig_ev.update_layout(
                    title="Events per month",
                    barmode="stack",
                    height=260, margin=dict(t=40, b=20, l=30, r=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", x=0.01, y=1.12),
                )
                _fig_ev.update_xaxes(gridcolor="#E2E8F0")
                _fig_ev.update_yaxes(gridcolor="#E2E8F0", title="Events")
                st.plotly_chart(_fig_ev, use_container_width=True)

            with st.expander("View simulated schedules"):
                _sched = sim.schedule
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

                # Shift pattern heatmap — first 6 weeks, 10 random providers
                _rng_s = np.random.default_rng(sim.seed)
                _s_idx = np.sort(_rng_s.choice(_n_p, size=min(10, _n_p), replace=False))
                _show_days = min(42, _n_d)
                _n_weeks = _show_days // 7
                _submat = _sched[np.ix_(_s_idx, np.arange(_show_days))]
                _enc = np.where(_submat == "d", 2, np.where(_submat == "n", 1, 0)).astype(float)
                _plabels = [sim.providers[i] for i in _s_idx]
                _n_samp = len(_s_idx)

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
                    title=f"Shift pattern — first {_n_weeks} weeks, sample of {_n_samp} providers",
                    height=max(280, _n_samp * 28),
                    margin=dict(t=50, b=30, l=80, r=60),
                    xaxis=dict(title="Day", tickmode="array",
                               tickvals=[w * 7 + 4 for w in range(_n_weeks)],
                               ticktext=[f"Wk {w+1}" for w in range(_n_weeks)]),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(_fig_sched, use_container_width=True)


# ── Tab 4: Training Simulation ─────────────────────────────────────────────

with tab_training:
    st.header("Training Effects")

    if not st.session_state.sim_ran:
        st.info("Configure Events and Schedules above, then click **▶ Run Simulation** in the sidebar.")
        training_btn = False
    else:
        sim_b = st.session_state.sim_baseline
        if sim_b is None:
            st.error("No simulation results available.")
            training_btn = False
        else:
            _prog_display_map = {
                "None (exposure only)":       "none",
                "Monthly (every 28 days)":    "monthly",
                "Bi-monthly (every 56 days)": "bimonthly",
                "Quarterly (every 84 days)":  "quarterly",
                "Custom interval":            "custom",
            }
            _prog_labels = list(_prog_display_map.keys())
            _current_label = {v: k for k, v in _prog_display_map.items()}.get(
                st.session_state.training_program, "None (exposure only)"
            )
            _sel_label = st.selectbox(
                "Training program",
                _prog_labels,
                index=_prog_labels.index(_current_label),
                key="training_prog_select",
            )
            _sel_prog = _prog_display_map[_sel_label]
            st.session_state.training_program = _sel_prog

            if _sel_prog == "custom":
                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    _ti = st.slider("Training interval (days)", 7, 365,
                                    st.session_state.training_interval, key="ti_custom")
                    st.session_state.training_interval = _ti
                with _tc2:
                    _ts = st.slider("First training day", 0, 90,
                                    st.session_state.training_start, key="ts_custom")
                    st.session_state.training_start = _ts

            if _sel_prog != "none":
                _effect_opts = ["Full reset (training = live exposure)", "Partial boost"]
                _eff = st.radio("Training effectiveness", _effect_opts, horizontal=True,
                                index=0 if st.session_state.training_effect == "full" else 1,
                                key="train_effect")
                st.session_state.training_effect = "full" if _eff == _effect_opts[0] else "partial"
                if st.session_state.training_effect == "partial":
                    _eq = st.slider("Equivalence factor (1.0 = same as live exposure)",
                                    0.1, 1.0, st.session_state.training_equivalence, 0.05,
                                    key="train_eq")
                    st.session_state.training_equivalence = _eq

            training_btn = st.button("🏋️ Run Training Simulation", type="primary",
                                     use_container_width=True)

            if st.session_state.training_ran and \
               st.session_state._last_training_hash != _training_hash():
                st.warning("Training settings changed — click **🏋️ Run Training Simulation** to update.")

            # ── Results (only after training run) ─────────────────────────
            if st.session_state.training_ran and st.session_state.sim_trained is not None:
                sim_t = st.session_state.sim_trained
                st.divider()

                roll = st.slider("Rolling mean window (days)", 1, 90,
                                 st.session_state.get("_roll_window", 30), key="roll_window")
                st.session_state["_roll_window"] = roll

                b_mean = np.nanmean(sim_b.proportion_ready_on_shift) * 100
                t_mean = np.nanmean(sim_t.proportion_ready_on_shift) * 100
                tm = sim_t.training_matrix
                n_sessions = int(tm.any(axis=0).sum())
                n_reached  = int(tm.any(axis=1).sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg readiness without training", f"{b_mean:.1f}%")
                c2.metric("Avg readiness with training", f"{t_mean:.1f}%",
                          delta=f"{t_mean - b_mean:+.1f}%")
                c3.metric("Training sessions held", f"{n_sessions:,}",
                          help="Number of distinct days on which at least one provider was trained.")
                c4.metric("Providers reached", f"{n_reached:,}",
                          help="Number of unique providers who received at least one training session.")

                st.divider()

                _compare_options = {
                    "No training":       "none",
                    "Monthly (28d)":     "monthly",
                    "Bi-monthly (56d)":  "bimonthly",
                    "Quarterly (84d)":   "quarterly",
                    "Custom":            "custom",
                }
                _active_label = {v: k for k, v in _compare_options.items()}.get(
                    _sel_prog, "No training"
                )
                _default_sel = ["No training"] + (
                    [_active_label] if _active_label != "No training" else []
                )

                _use_my_settings = st.toggle(
                    "Use my current Custom/Targeted settings",
                    value=True,
                    key="compare_use_my_settings",
                    help=(
                        "ON: the Custom and Targeted lines reflect your current interval / "
                        "threshold settings above.  "
                        "OFF: all programs use standardised defaults (monthly=28d, "
                        "bi-monthly=56d, quarterly=84d, custom=28d, targeted=50% threshold) "
                        "so comparisons are directly apples-to-apples."
                    ),
                )
                if not _use_my_settings:
                    st.caption(
                        "Standardised mode — all programs use fixed defaults regardless of "
                        "the controls above."
                    )

                _selected = st.multiselect(
                    "Programs to compare",
                    list(_compare_options.keys()),
                    default=_default_sel,
                    key="compare_programs",
                )

                if _selected:
                    _s = st.session_state
                    _compare_data: dict[str, np.ndarray] = {}
                    for _lbl in _selected:
                        _prog = _compare_options[_lbl]
                        _is_active = _use_my_settings and (_prog == _sel_prog)
                        _csim = _run_sim(
                            sim_b.n_days, tuple(sim_b.providers), sim_b.schedule, sim_b.events,
                            sim_b.seed,
                            _s.readiness_model, _s.readiness_threshold,
                            _s.readiness_half_life, _s.ebbinghaus_b,
                            _s.step_t2, _s.step_partial,
                            _prog,
                            _s.training_interval    if _is_active else 28,
                            _s.training_start       if _is_active else 0,
                            _s.training_effect      if _is_active else "full",
                            _s.training_equivalence if _is_active else 1.0,
                            _s.training_threshold   if _is_active else 0.5,
                        )
                        _compare_data[_lbl] = _csim.proportion_ready_on_shift
                    st.plotly_chart(
                        plot_training_comparison(_compare_data, sim_b.n_days, roll),
                        use_container_width=True,
                    )

                _improve = t_mean - b_mean
                if abs(_improve) < 1:
                    _t_interp = ("Training had minimal effect on on-shift readiness — "
                                 "live exposure alone may be sufficient at this event rate.")
                elif _improve > 0:
                    _t_interp = (
                        f"Training raised average on-shift readiness by **{_improve:.1f} pp** "
                        f"({b_mean:.0f}% → {t_mean:.0f}%). "
                        f"{n_sessions:,} training sessions reached {n_reached:,} providers."
                    )
                else:
                    _t_interp = (
                        f"Readiness with training ({t_mean:.0f}%) is similar to baseline "
                        f"({b_mean:.0f}%). Consider adjusting training frequency or model parameters."
                    )
                st.info(_t_interp)

                with st.expander("Also show: all providers (including off-shift)"):
                    fig2 = plot_readiness_timeseries(
                        sim_b.proportion_ready_all,
                        sim_t.proportion_ready_all,
                        n_days=sim_b.n_days,
                        rolling_days=roll,
                    )
                    st.caption("⚠️ This includes providers currently off-shift. "
                               "The on-shift metric above is the primary indicator.")
                    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Run simulation (triggered by sidebar button)
# ---------------------------------------------------------------------------

if run_btn or st.session_state.get("_auto_run", False):
    st.session_state._auto_run = False   # consume flag immediately
    errors = []

    # 1. Build events
    if st.session_state.event_source == "Generate (Poisson MC)":
        _rate  = st.session_state.event_rate
        _d_pct = st.session_state.get("event_day_pct", 50) / 100
        _day_r = round(_rate * _d_pct, 4)
        _ngt_r = round(_rate * (1 - _d_pct), 4)
        if _d_pct != 0.5:
            events_df, e_warn = generate_events(
                n_days=n_days,
                rate=_rate,
                seed=int(seed),
                day_rate=_day_r,
                night_rate=_ngt_r,
            )
        else:
            events_df, e_warn = generate_events(
                n_days=n_days,
                rate=_rate,
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
    else:
        _d = st.session_state.get("schedule_day_pct")
        _n = st.session_state.get("schedule_night_pct")
        _weights = (
            {"d": _d / 100, "n": _n / 100, "o": (100 - _d - _n) / 100}
            if _d is not None and _n is not None
            else None
        )
        schedule, s_warn = generate_schedule(
            n_providers=n_providers,
            n_days=n_days,
            schedule_type=st.session_state.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
            seed=int(seed),
            weights=_weights,
        )
        providers_list = [f"P{i+1:04d}" for i in range(n_providers)]

    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.stop()

    # 3. Run baseline simulation (no training)
    _s = st.session_state
    sim_b = _run_sim(
        n_days=n_days, providers_tuple=tuple(providers_list),
        schedule=schedule, events_df=events_df, seed=int(seed),
        readiness_model=_s.readiness_model,
        readiness_threshold=_s.readiness_threshold,
        readiness_half_life=_s.readiness_half_life,
        ebbinghaus_b=_s.ebbinghaus_b,
        step_t2=_s.step_t2, step_partial=_s.step_partial,
        training_program="none",
        training_interval=28,
        training_start=0,
        training_effect="full",
        training_equivalence=1.0,
        training_threshold=0.5,
    )

    st.session_state.sim_baseline = sim_b
    st.session_state.sim_trained = None
    st.session_state.sim_ran = True
    st.session_state.training_ran = False
    st.session_state._last_run_hash = _sim_hash()
    st.rerun()


# ---------------------------------------------------------------------------
# Training simulation (triggered by Run Training button in Training tab)
# ---------------------------------------------------------------------------

if st.session_state.get("sim_ran") and training_btn:
    _sb = st.session_state.sim_baseline
    _s = st.session_state
    _sim_t = _run_sim(
        n_days=_sb.n_days,
        providers_tuple=tuple(_sb.providers),
        schedule=_sb.schedule,
        events_df=_sb.events,
        seed=_sb.seed,
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
    )
    st.session_state.sim_trained = _sim_t
    st.session_state.training_ran = True
    st.session_state._last_training_hash = _training_hash()
    st.rerun()
