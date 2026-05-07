"""
HaloSim — HALO Event Exposure & Training Simulation
"""

from __future__ import annotations

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
    plot_training_comparison,
    plot_mc_readiness_band,
    plot_mc_histogram,
    plot_mc_threshold_sweep,
    build_mc_summary_df,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent / "assets"
DATA_DIR   = Path(__file__).parent / "data"

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
# Constants
# ---------------------------------------------------------------------------

_PROG_MAP = {
    "None (exposure only)":       "none",
    "Monthly (every 30 days)":    "monthly",
    "Bi-monthly (every 60 days)": "bimonthly",
    "Quarterly (every 91 days)":  "quarterly",
}
_PROG_INTERVALS = {"none": 30, "monthly": 30, "bimonthly": 60, "quarterly": 91}
_TRAINING_START = 14   # fixed offset for all programs

# ---------------------------------------------------------------------------
# Session state
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
        "events_errors": [],
        # schedules
        "schedule_source": "Generate schedules",
        "schedule_type": DEFAULT_SCHEDULE_TYPE,
        "schedule_day_pct": None,
        "schedule_night_pct": None,
        "schedule_array": None,
        "schedule_providers": None,
        "schedule_errors": [],
        # readiness
        "readiness_model": "binary",
        "readiness_threshold": 90,
        # training
        "training_program": "none",
        "training_effect": "full",
        "training_equivalence": 1.0,
        # MC
        "mc_n_samples": 50,
        "mc_result": None,
        "mc_ran": False,
        "_last_mc_hash": None,
        # upload caches
        "events_upload_bytes": None,
        "events_upload_name": "",
        "schedule_upload_bytes": None,
        "schedule_upload_name": "",
        "_auto_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Migrate stale values from prior versions
    if st.session_state.training_program not in _PROG_MAP.values():
        st.session_state.training_program = "none"
    if st.session_state.schedule_source in ("Built-in templates", "Custom 28-day pattern"):
        st.session_state.schedule_source = "Generate schedules"


_init_state()


def _sim_hash() -> str:
    s = st.session_state
    parts = [
        s.n_days, s.n_providers, s.seed,
        s.event_source, s.event_rate, s.get("event_day_pct", 50),
        s.schedule_source, s.get("schedule_type", DEFAULT_SCHEDULE_TYPE),
        s.get("schedule_day_pct"), s.get("schedule_night_pct"),
        len(s.events_df) if s.events_df is not None else -1,
        str(s.schedule_array.shape) if s.schedule_array is not None else "none",
        s.readiness_model, s.readiness_threshold,
    ]
    return ":".join(str(p) for p in parts)


def _mc_hash() -> str:
    s = st.session_state
    return (
        _sim_hash()
        + f":{s.mc_n_samples}"
        + f":{s.training_program}:{s.training_effect}:{s.training_equivalence}"
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _run_sim(
    n_days, providers_tuple, schedule, events_df, seed,
    readiness_model, readiness_threshold,
    training_program, training_interval, training_start,
    training_effect, training_equivalence,
) -> "Simulation":
    sim = Simulation(
        n_days=n_days, providers=list(providers_tuple),
        schedule=schedule, events=events_df, seed=seed,
        readiness_model=readiness_model,
        readiness_threshold_days=readiness_threshold,
        training_program=training_program,
        training_interval_days=training_interval,
        training_start_day=training_start,
        training_effect=training_effect,
        training_equivalence=training_equivalence,
    )
    sim.run()
    return sim


@st.cache_data(show_spinner=False, max_entries=4)
def _run_mc(
    n_days, providers_tuple, fixed_schedule, fixed_events_df,
    event_source, event_rate, event_day_pct,
    schedule_source, schedule_type, schedule_day_pct, schedule_night_pct,
    readiness_model, readiness_threshold,
    training_program, training_interval, training_start,
    training_effect, training_equivalence,
    seeds_tuple,
) -> dict:
    n_providers = len(providers_tuple)
    n_samples   = len(seeds_tuple)
    _weights = (
        {"d": schedule_day_pct / 100,
         "n": schedule_night_pct / 100,
         "o": (100 - schedule_day_pct - schedule_night_pct) / 100}
        if schedule_day_pct is not None and schedule_night_pct is not None else None
    )

    readiness_b_list, readiness_t_list, lift_list = [], [], []
    pct_exc_list, med_gap_list, med_nev_list, pct_by_thr_list = [], [], [], []
    ref = {}
    _sweep_thresholds = np.arange(7, 366)

    for s, cur in enumerate(seeds_tuple):

        # Events
        if fixed_events_df is None:
            d = event_day_pct / 100
            kw = {"day_rate": event_rate * d, "night_rate": event_rate * (1 - d)} \
                 if event_day_pct != 50 else {}
            ev, _ = generate_events(n_days=n_days, rate=event_rate, seed=cur, **kw)
        else:
            ev = fixed_events_df

        # Schedule
        if fixed_schedule is None:
            sched, _ = generate_schedule(
                n_providers=n_providers, n_days=n_days,
                schedule_type=schedule_type, seed=cur, weights=_weights,
            )
        else:
            sched = fixed_schedule

        # Baseline
        sim_b = Simulation(
            n_days=n_days, providers=list(providers_tuple),
            schedule=sched, events=ev, seed=cur,
            readiness_model=readiness_model,
            readiness_threshold_days=readiness_threshold,
            training_program="none",
            training_interval_days=training_interval,
            training_start_day=training_start,
            training_effect=training_effect,
            training_equivalence=training_equivalence,
        )
        sim_b.run()

        # Trained
        sim_t = None
        if training_program != "none":
            sim_t = Simulation(
                n_days=n_days, providers=list(providers_tuple),
                schedule=sched, events=ev, seed=cur,
                readiness_model=readiness_model,
                readiness_threshold_days=readiness_threshold,
                training_program=training_program,
                training_interval_days=training_interval,
                training_start_day=training_start,
                training_effect=training_effect,
                training_equivalence=training_equivalence,
            )
            sim_t.run()

        readiness_b_list.append(sim_b.proportion_ready_on_shift.copy())
        pct_exc_list.append(100.0 * sim_b.results_df["max_gap_exceeds_threshold"].mean())
        med_gap_list.append(float(sim_b.results_df["gap_median"].dropna().median()))
        med_nev_list.append(float(sim_b.results_df["n_events"].dropna().median()))
        _gap_max = sim_b.results_df["gap_max"].fillna(9999).values
        pct_by_thr_list.append(
            np.array([100.0 * (_gap_max > t).mean() for t in _sweep_thresholds])
        )

        if sim_t is not None:
            readiness_t_list.append(sim_t.proportion_ready_on_shift.copy())
            lift_list.append(
                (np.nanmean(sim_t.proportion_ready_on_shift)
                 - np.nanmean(sim_b.proportion_ready_on_shift)) * 100
            )

        if s == 0:
            ref = {
                "results_df":    sim_b.results_df.copy(),
                "events_df":     sim_b.events.copy(),
                "schedule":      sim_b.schedule.copy(),
                "proportion_b":  sim_b.proportion_ready_on_shift.copy(),
                "proportion_t":  sim_t.proportion_ready_on_shift.copy() if sim_t else None,
                "training_mat":  sim_t.training_matrix.copy() if sim_t else None,
            }

    return {
        "readiness_b":     np.array(readiness_b_list),
        "readiness_t":     np.array(readiness_t_list) if readiness_t_list else None,
        "pct_exceeding":   np.array(pct_exc_list),
        "median_gap":      np.array(med_gap_list),
        "median_n_events": np.array(med_nev_list),
        "lift":            np.array(lift_list) if lift_list else None,
        "n_days":          n_days,
        "n_samples":       n_samples,
        "threshold":       readiness_threshold,
        "training_program": training_program,
        "providers":          list(providers_tuple),
        "seeds":              list(seeds_tuple),
        "pct_by_threshold":   np.array(pct_by_thr_list),
        "sweep_thresholds":   _sweep_thresholds,
        **{f"ref_{k}": v for k, v in ref.items()},
    }


# ---------------------------------------------------------------------------
# Sidebar — N runs + Run button only
# ---------------------------------------------------------------------------

with st.sidebar:
    _logo_col, _title_col = st.columns([1, 2.5])
    with _logo_col:
        st.image(str(ASSETS_DIR / "logo.png"), width=56)
    with _title_col:
        st.markdown("**HaloSim**")
        st.caption("HALO Event Exposure\n& Training Simulation")

    st.divider()

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if st.session_state.mc_ran and st.session_state._last_mc_hash != _mc_hash():
        st.warning("Settings changed — re-run to update.")

    st.divider()
    st.caption("Built by [Sangfroid Labs](https://sangfroidlabs.com)")


# ---------------------------------------------------------------------------
# Result banner
# ---------------------------------------------------------------------------

if st.session_state.mc_ran and st.session_state.mc_result is not None:
    _mc = st.session_state.mc_result
    if st.session_state._last_mc_hash == _mc_hash():
        _prog_label = {v: k for k, v in _PROG_MAP.items()}.get(
            _mc["training_program"], "None (exposure only)"
        )
        _train_note = f" · {_prog_label}" if _mc["training_program"] != "none" else ""
        st.success(
            f"✓  {len(_mc['providers']):,} providers × {_mc['n_days']} days"
            f" · {_mc['n_samples']} MC run{'s' if _mc['n_samples'] != 1 else ''}"
            f"{_train_note}"
        )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_params, tab_exposure, tab_training = st.tabs(
    ["⚙️ Model Parameters", "📊 Exposure", "🏋️ Training"]
)


# ── Tab 0: Model Parameters ────────────────────────────────────────────────

with tab_params:

    # ── Simulation ──────────────────────────────────────────────────────────
    st.subheader("Simulation")
    _p1, _p2 = st.columns(2)

    with _p1:
        n_days = st.selectbox(
            "Duration",
            [90, 180, 365, 730],
            index=[90, 180, 365, 730].index(st.session_state.n_days)
            if st.session_state.n_days in [90, 180, 365, 730] else 2,
            format_func=lambda x: f"{x} days (~{x // 365} yr)" if x >= 365
                                   else f"{x} days (~{x // 30} mo)",
        )
        st.session_state.n_days = n_days

        thresh = st.number_input(
            "Critical threshold (days)",
            min_value=1, value=st.session_state.readiness_threshold, step=1,
            help="Maximum acceptable gap between HALO exposures. Providers beyond this are under-exposed.",
        )
        thresh = int(thresh)
        if thresh > n_days:
            st.error(f"Threshold ({thresh} d) exceeds duration ({n_days} d).")
            thresh = st.session_state.readiness_threshold
        st.session_state.readiness_threshold = thresh
        st.session_state.readiness_model = "binary"

    with _p2:
        n_providers = int(st.number_input(
            "Number of providers",
            min_value=10, max_value=MAX_PROVIDERS,
            value=st.session_state.n_providers, step=10,
        ))
        st.session_state.n_providers = n_providers
        if n_providers > WARN_PROVIDERS:
            st.warning(f"⚠️ {n_providers:,} providers — may be slow.")

        _n_samp = st.slider(
            "Number of simulations",
            min_value=1, max_value=200,
            value=st.session_state.mc_n_samples,
            help="Each simulation draws independent random seeds for event timing and "
                 "shift assignments. N=1 gives a single result; N≥50 gives distributions.",
        )
        st.session_state.mc_n_samples = _n_samp


    # ── HALO Events ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("HALO Events")

    _ev_src_opts = ["Generate (Poisson MC)", "Upload CSV / Excel"]
    event_source = st.radio(
        "Source", _ev_src_opts,
        index=_ev_src_opts.index(st.session_state.event_source),
        horizontal=True, label_visibility="collapsed",
    )
    st.session_state.event_source = event_source

    if event_source == "Generate (Poisson MC)":
        _lc1, _lc2 = st.columns(2)
        with _lc1:
            if st.button("Load sample events (48 / year)", use_container_width=True):
                _raw = (DATA_DIR / "sample_events.csv").read_bytes()
                _df, _ = load_events_from_upload(_raw, "sample_events.csv", n_days)
                if _df is not None:
                    st.session_state.events_df    = _df
                    st.session_state.event_source = "Upload CSV / Excel"
                    st.rerun()
        with _lc2:
            if st.button("Load full demo (20 providers)", use_container_width=True,
                         help="Loads sample events + schedule and runs the simulation."):
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

        rate_per_year = st.slider(
            "Event rate (events / year)", 1, 365,
            value=round(st.session_state.event_rate * 365), step=1,
            help="~51/year matches cardiac arrest rate in PMID: 41633464",
        )
        st.session_state.event_rate = rate_per_year / 365.0
        st.caption(
            f"~**{st.session_state.event_rate * 30.44:.1f} events/month** &nbsp;·&nbsp; "
            f"{st.session_state.event_rate * n_days:.0f} expected over {n_days} days"
        )

        with st.expander("Advanced: day / night split"):
            day_pct = st.slider(
                "% on day shifts", 0, 100,
                value=st.session_state.get("event_day_pct", 50), step=5,
            )
            st.session_state.event_day_pct = day_pct
            st.caption(
                f"Day: {st.session_state.event_rate * day_pct / 100:.3f}/day &nbsp;·&nbsp; "
                f"Night: {st.session_state.event_rate * (100 - day_pct) / 100:.3f}/day"
            )

    else:  # Upload
        st.caption("Required columns: `date`, `shift_type` (day / night)")
        _sample_ev = pd.DataFrame({
            "date":       ["2024-01-03", "2024-01-11", "2024-02-01"],
            "shift_type": ["day",        "night",       "day"],
        })
        st.dataframe(_sample_ev, use_container_width=False, hide_index=True)

        uploaded_ev = st.file_uploader("Upload events file", type=["csv", "xlsx", "xls"],
                                       key="ev_upload")
        if uploaded_ev is not None:
            if uploaded_ev.name != st.session_state.events_upload_name:
                st.session_state.events_upload_bytes = uploaded_ev.read()
                st.session_state.events_upload_name  = uploaded_ev.name
                st.session_state.events_errors       = []
                st.session_state.events_df           = None
            df, errs = load_events_from_upload(
                st.session_state.events_upload_bytes,
                st.session_state.events_upload_name,
                n_days, allow_hour_col=False,
            )
            st.session_state.events_errors = errs
            if df is not None:
                st.session_state.events_df = df
        for e in st.session_state.events_errors:
            st.error(e)
        if st.session_state.events_df is not None:
            st.success(f"✓ {len(st.session_state.events_df)} events loaded")

    # ── Provider Schedules ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Provider Schedules")

    _sc_src_opts = ["Generate schedules", "Upload CSV / Excel"]
    sched_source = st.radio(
        "Source", _sc_src_opts,
        index=min(
            _sc_src_opts.index(st.session_state.schedule_source)
            if st.session_state.schedule_source in _sc_src_opts else 0,
            len(_sc_src_opts) - 1,
        ),
        horizontal=True, label_visibility="collapsed",
    )
    st.session_state.schedule_source = sched_source

    if sched_source == "Generate schedules":
        _type_captions = {
            "3/7 Day":   "3 day shifts per week, rest off",
            "3/7 Night": "3 night shifts per week, rest off",
            "4/7 Day":   "4 day shifts per week, rest off",
            "4/7 Night": "4 night shifts per week, rest off",
            "Progressive (day & night mix)": "3–4 shifts/week, mix of day and night",
            "Random":    "Each day drawn from empirical weights (25% day, 23% night, 52% off)",
        }
        cur_type = st.session_state.get("schedule_type", DEFAULT_SCHEDULE_TYPE)
        if cur_type not in SCHEDULE_TYPES:
            cur_type = DEFAULT_SCHEDULE_TYPE
        sel_type = st.radio(
            "Schedule type", SCHEDULE_TYPES,
            index=SCHEDULE_TYPES.index(cur_type),
            captions=list(_type_captions.values()),
        )
        st.session_state.schedule_type = sel_type

        with st.expander("Advanced: custom shift weights"):
            _d_def = st.session_state.get("schedule_day_pct") or 25
            _n_def = st.session_state.get("schedule_night_pct") or 23
            _d = st.slider("% day shifts", 0, 100, _d_def, 5, key="sc_d")
            _n_max = 100 - _d
            _n = st.slider("% night shifts", 0, _n_max, min(_n_def, _n_max), 5, key="sc_n") \
                 if _n_max > 0 else 0
            st.caption(f"Day {_d}% · Night {_n}% · Off {100 - _d - _n}%")
            if st.checkbox("Use these weights (overrides schedule type)",
                           value=(st.session_state.get("schedule_day_pct") is not None),
                           key="use_custom_wts"):
                st.session_state.schedule_day_pct = _d
                st.session_state.schedule_night_pct = _n
            else:
                st.session_state.schedule_day_pct = None
                st.session_state.schedule_night_pct = None

    else:  # Upload schedule
        st.caption("Required columns: `provider_id`, `date`, `shift_type` (day / night / off)")
        _sample_sc = pd.DataFrame({
            "provider_id": ["P0001", "P0001", "P0002"],
            "date":        ["2024-01-01", "2024-01-02", "2024-01-01"],
            "shift_type":  ["day",        "off",         "night"],
        })
        st.dataframe(_sample_sc, use_container_width=False, hide_index=True)

        uploaded_sc = st.file_uploader("Upload schedule file", type=["csv", "xlsx", "xls"],
                                       key="sc_upload")
        if uploaded_sc is not None:
            if uploaded_sc.name != st.session_state.schedule_upload_name:
                st.session_state.schedule_upload_bytes = uploaded_sc.read()
                st.session_state.schedule_upload_name  = uploaded_sc.name
                st.session_state.schedule_errors       = []
                st.session_state.schedule_array        = None
                st.session_state.schedule_providers    = None
            arr, provs, errs = load_schedule_from_upload(
                st.session_state.schedule_upload_bytes,
                st.session_state.schedule_upload_name,
                n_days,
            )
            st.session_state.schedule_errors = errs
            if arr is not None:
                st.session_state.schedule_array    = arr
                st.session_state.schedule_providers = provs
        for e in st.session_state.schedule_errors:
            st.warning(e) if e.startswith("Warning:") else st.error(e)
        if st.session_state.schedule_array is not None:
            _a = st.session_state.schedule_array
            st.success(f"✓ Schedule: {_a.shape[0]} providers × {_a.shape[1]} days")

        with st.expander("Use pre-built sample schedule"):
            if st.button("Load sample_schedule.csv (20 providers)"):
                raw = (DATA_DIR / "sample_schedule.csv").read_bytes()
                arr, provs, _ = load_schedule_from_upload(raw, "sample_schedule.csv", n_days)
                if arr is not None:
                    st.session_state.schedule_array    = arr
                    st.session_state.schedule_providers = provs
                    st.rerun()

    # ── Training Program ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Training Program")
    st.caption("Optional. Select a program to compare trained vs. untrained readiness.")

    _prog_labels = list(_PROG_MAP.keys())
    _cur_label = {v: k for k, v in _PROG_MAP.items()}.get(
        st.session_state.training_program, "None (exposure only)"
    )
    _sel_label = st.selectbox(
        "Program", _prog_labels,
        index=_prog_labels.index(_cur_label),
        label_visibility="collapsed",
    )
    st.session_state.training_program = _PROG_MAP[_sel_label]

    if st.session_state.training_program != "none":
        _eff_opts = ["Full reset (training = live exposure)", "Partial boost"]
        _eff = st.radio(
            "Effectiveness", _eff_opts, horizontal=True,
            index=0 if st.session_state.training_effect == "full" else 1,
        )
        st.session_state.training_effect = "full" if _eff == _eff_opts[0] else "partial"
        if st.session_state.training_effect == "partial":
            _eq = st.slider(
                "Equivalence factor (1.0 = same as live exposure)",
                0.1, 1.0, st.session_state.training_equivalence, 0.05,
            )
            st.session_state.training_equivalence = _eq

        _interval = _PROG_INTERVALS[st.session_state.training_program]
        st.caption(
            f"Sessions every **{_interval} days**, first on day **{_TRAINING_START}**. "
            f"~{365 / _interval:.1f} sessions per year."
        )


# ── Tab 1: Exposure ────────────────────────────────────────────────────────

with tab_exposure:
    st.header("Exposure Analysis")

    if not st.session_state.mc_ran or st.session_state.mc_result is None:
        st.info(
            "Set your parameters in **⚙️ Model Parameters**, set N in the sidebar, "
            "then click **▶ Run Simulation**."
        )
    else:
        mc    = st.session_state.mc_result
        rdf   = mc["ref_results_df"]
        thresh = mc["threshold"]
        n_samp = mc["n_samples"]

        if (st.session_state.event_source == "Upload CSV / Excel"
                and st.session_state.schedule_source == "Upload CSV / Excel"
                and n_samp > 1):
            st.warning(
                "Both events and schedule are uploaded — all MC runs are identical. "
                "Switch at least one source to Generate to produce a true distribution."
            )

        # Key metrics
        _pct = mc["pct_exceeding"]
        _gap = mc["median_gap"]
        _nev = mc["median_n_events"]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"% exceeding {thresh}-day threshold",
            f"{np.median(_pct):.1f}%",
            help=f"p10–p90: {np.percentile(_pct,10):.1f}–{np.percentile(_pct,90):.1f}%",
        )
        c2.metric(
            "Median inter-exposure gap",
            f"{np.median(_gap):.0f} days",
            help=f"p10–p90: {np.percentile(_gap,10):.0f}–{np.percentile(_gap,90):.0f} days",
        )
        c3.metric(
            "Median exposures / provider",
            f"{np.median(_nev):.1f}",
            help=f"p10–p90: {np.percentile(_nev,10):.1f}–{np.percentile(_nev,90):.1f}",
        )

        # Summary table
        st.divider()
        st.dataframe(build_mc_summary_df(mc), use_container_width=True, hide_index=True)

        # Readiness band
        st.divider()
        st.subheader("On-shift readiness over time")
        st.caption(
            "Shaded band = p10–p90 across all runs. Solid line = median. "
            "Off-shift providers excluded."
        )
        _roll_e = st.slider("Rolling mean (days)", 1, 90,
                            st.session_state.get("_roll_e", 30), key="roll_e")
        st.session_state["_roll_e"] = _roll_e
        st.plotly_chart(
            plot_mc_readiness_band(mc["readiness_b"], rolling_days=_roll_e),
            use_container_width=True,
        )

        # Histograms
        st.divider()
        st.subheader("Distribution across runs")
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

        # MC threshold sweep
        st.divider()
        st.subheader("Gap exceedance by threshold")
        st.caption(
            "Each threshold on the x-axis represents the maximum allowed gap between HALO "
            "exposures. The band shows how the % of under-exposed providers varies across "
            "all simulation runs. Solid line = median; shaded = p10–p90."
        )
        if "pct_by_threshold" in mc:
            st.plotly_chart(
                plot_mc_threshold_sweep(
                    mc["pct_by_threshold"], mc["sweep_thresholds"], threshold_marker=thresh,
                ),
                use_container_width=True,
            )

        # Downloads
        st.divider()
        st.subheader("Downloads")
        _d1, _d2, _d3 = st.columns(3)
        with _d1:
            _rdf2 = rdf.copy()
            if "gap_mean" in _rdf2.columns:
                _rdf2["gap_mean"] = _rdf2["gap_mean"].round(3)
            st.download_button(
                "📥 Exposure stats — ref run",
                data=_rdf2.to_csv(index=False).encode(),
                file_name="halosim_exposure.csv", mime="text/csv",
                use_container_width=True,
            )
        with _d2:
            _ev_dl = mc["ref_events_df"][["day_idx", "shift_type"]].copy()
            if "date" in mc["ref_events_df"].columns:
                _ev_dl.insert(0, "date", mc["ref_events_df"]["date"])
            st.download_button(
                "📥 Events — ref run",
                data=_ev_dl.to_csv(index=False).encode(),
                file_name="halosim_events.csv", mime="text/csv",
                use_container_width=True,
            )
        with _d3:
            _mc_dl = pd.DataFrame({
                "run":                     list(range(1, n_samp + 1)),
                "seed":                    mc["seeds"],
                "pct_exceeding_threshold": mc["pct_exceeding"].round(2),
                "median_gap_days":         mc["median_gap"].round(1),
                "median_n_events":         mc["median_n_events"].round(1),
            })
            if mc["lift"] is not None:
                _mc_dl["training_lift_pp"] = mc["lift"].round(2)
            st.download_button(
                "📥 MC scalar results",
                data=_mc_dl.to_csv(index=False).encode(),
                file_name="halosim_mc.csv", mime="text/csv",
                use_container_width=True,
            )

        # Simulated data expanders
        with st.expander("View simulated events (reference run)"):
            _ev = mc["ref_events_df"]
            _ne = len(_ev)
            _nd = int((_ev["shift_type"] == "day").sum())
            _nn = int((_ev["shift_type"] == "night").sum())
            _exp = round(st.session_state.event_rate * mc["n_days"])
            e1, e2, e3 = st.columns(3)
            if st.session_state.event_source == "Generate (Poisson MC)":
                e1.metric("Events (seed-0)", _ne, delta=f"{_ne-_exp:+d} vs {_exp} expected",
                          delta_color="off")
            else:
                e1.metric("Events (uploaded)", _ne)
            e2.metric("Day", f"{_nd} ({100*_nd//_ne if _ne else 0}%)")
            e3.metric("Night", f"{_nn} ({100*_nn//_ne if _ne else 0}%)")

        with st.expander("View simulated schedules (reference run)"):
            _sch = mc["ref_schedule"]
            _prv = mc["providers"]
            _np2, _nd2 = _sch.shape
            s1, s2, s3 = st.columns(3)
            _dc = (_sch == "d").sum(axis=1)
            _nc = (_sch == "n").sum(axis=1)
            _oc = (_sch == "o").sum(axis=1)
            s1.metric("Avg day shifts / provider", f"{_dc.mean():.1f} ({100*_dc.mean()/_nd2:.0f}%)")
            s2.metric("Avg night shifts / provider", f"{_nc.mean():.1f} ({100*_nc.mean()/_nd2:.0f}%)")
            s3.metric("Avg days off / provider", f"{_oc.mean():.1f} ({100*_oc.mean()/_nd2:.0f}%)")


# ── Tab 2: Training Effects ────────────────────────────────────────────────

with tab_training:
    st.header("Training Effects")

    if not st.session_state.mc_ran or st.session_state.mc_result is None:
        st.info(
            "Set your parameters in **⚙️ Model Parameters**, set N in the sidebar, "
            "then click **▶ Run Simulation**."
        )
    else:
        mc = st.session_state.mc_result
        n_samp = mc["n_samples"]

        if mc["training_program"] == "none":
            st.info(
                "No training program selected. "
                "Go to **⚙️ Model Parameters → Training Program** and choose a program, "
                "then re-run."
            )
        else:
            _prog_label = {v: k for k, v in _PROG_MAP.items()}.get(
                mc["training_program"], mc["training_program"]
            )

            # Compute per-seed summaries
            _b_means = np.array([np.nanmean(r) * 100 for r in mc["readiness_b"]])
            _t_means = np.array([np.nanmean(r) * 100 for r in mc["readiness_t"]])
            _lifts   = mc["lift"]
            _days_b  = np.array([(r < 0.80).sum() for r in mc["readiness_b"]])
            _days_t  = np.array([(r < 0.80).sum() for r in mc["readiness_t"]])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Readiness — no training",
                f"{np.median(_b_means):.1f}%",
                help=f"p10–p90: {np.percentile(_b_means,10):.1f}–{np.percentile(_b_means,90):.1f}%",
            )
            c2.metric(
                "Readiness — with training",
                f"{np.median(_t_means):.1f}%",
                delta=f"{np.median(_lifts):+.1f} pp",
                help=f"p10–p90: {np.percentile(_t_means,10):.1f}–{np.percentile(_t_means,90):.1f}%",
            )
            c3.metric(
                "Days <80% — no training",
                f"{np.median(_days_b):.0f}",
                help=f"p10–p90: {np.percentile(_days_b,10):.0f}–{np.percentile(_days_b,90):.0f}",
            )
            c4.metric(
                "Days <80% — with training",
                f"{np.median(_days_t):.0f}",
                delta=f"{np.median(_days_t)-np.median(_days_b):+.0f} days",
                delta_color="inverse",
                help=f"p10–p90: {np.percentile(_days_t,10):.0f}–{np.percentile(_days_t,90):.0f}",
            )

            if mc["ref_training_mat"] is not None:
                _tm = mc["ref_training_mat"]
                _c5, _c6, _, _ = st.columns(4)
                _c5.metric("Sessions — ref run", f"{int(_tm.any(axis=0).sum()):,}",
                           help="Days with at least one provider trained (seed-0).")
                _c6.metric("Providers reached — ref run", f"{int(_tm.any(axis=1).sum()):,}",
                           help="Unique providers trained at least once (seed-0).")

            # Readiness band chart
            st.divider()
            _roll_t = st.slider("Rolling mean (days)", 1, 90,
                                st.session_state.get("_roll_t", 30), key="roll_t")
            st.session_state["_roll_t"] = _roll_t
            st.plotly_chart(
                plot_mc_readiness_band(mc["readiness_b"], mc["readiness_t"],
                                       rolling_days=_roll_t),
                use_container_width=True,
            )

            # Lift histogram
            st.plotly_chart(
                plot_mc_histogram(_lifts, "Training lift", unit=" pp"),
                use_container_width=True,
            )

            # Interpretation
            _med = float(np.median(_lifts))
            _p10 = float(np.percentile(_lifts, 10))
            _p90 = float(np.percentile(_lifts, 90))
            if abs(_med) < 1:
                _interp = (
                    f"**{_prog_label}** had minimal effect on median readiness across "
                    f"{n_samp} runs — live exposure alone may be sufficient at this event rate."
                )
            elif _med > 0:
                _interp = (
                    f"**{_prog_label}** raised median readiness by **{_med:+.1f} pp** "
                    f"(p10–p90: {_p10:+.1f} to {_p90:+.1f} pp) across {n_samp} runs. "
                    f"The width of this range reflects how much the benefit depends on "
                    "when events happen to fall relative to training sessions."
                )
            else:
                _interp = (
                    f"Median readiness with training is similar to baseline ({_med:+.1f} pp). "
                    "Consider increasing training frequency."
                )
            st.info(_interp)

            # Program comparison (reference run)
            st.divider()
            st.subheader("Program comparison (reference run)")
            st.caption(
                "Overlay multiple programs using the reference run schedule and events. "
                "Vertical lines mark training days."
            )

            _cmp_opts = {
                "No training":       "none",
                "Monthly (30d)":     "monthly",
                "Bi-monthly (60d)":  "bimonthly",
                "Quarterly (91d)":   "quarterly",
            }
            _act_lbl = {v: k for k, v in _cmp_opts.items()}.get(
                mc["training_program"], "No training"
            )
            _selected = st.multiselect(
                "Programs to overlay",
                list(_cmp_opts.keys()),
                default=["No training"] + ([_act_lbl] if _act_lbl != "No training" else []),
                key="cmp_progs",
            )

            if _selected:
                _cmp_data: dict[str, np.ndarray] = {}
                _cmp_tdays: dict[str, list[int]] = {}
                for _lbl in _selected:
                    _prog = _cmp_opts[_lbl]
                    _ivl  = _PROG_INTERVALS.get(_prog, 30)
                    _csim = _run_sim(
                        mc["n_days"], tuple(mc["providers"]),
                        mc["ref_schedule"], mc["ref_events_df"],
                        int(st.session_state.seed),
                        st.session_state.readiness_model,
                        st.session_state.readiness_threshold,
                        _prog, _ivl, _TRAINING_START,
                        st.session_state.training_effect,
                        st.session_state.training_equivalence,
                    )
                    _cmp_data[_lbl] = _csim.proportion_ready_on_shift
                    if _prog != "none":
                        _cmp_tdays[_lbl] = list(
                            np.arange(_TRAINING_START, mc["n_days"], _ivl, dtype=int)
                        )
                st.plotly_chart(
                    plot_training_comparison(_cmp_data, mc["n_days"], _roll_t,
                                            training_days=_cmp_tdays),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Run block
# ---------------------------------------------------------------------------

if run_btn or st.session_state.get("_auto_run", False):
    st.session_state._auto_run = False
    _s = st.session_state
    errors = []

    # Events
    if _s.event_source == "Generate (Poisson MC)":
        fixed_events_df = None
    else:
        fixed_events_df = _s.events_df
        if fixed_events_df is None:
            errors.append("No events loaded — upload a file or switch to Generate.")

    # Schedule
    if _s.schedule_source == "Upload CSV / Excel" and _s.schedule_array is not None:
        _arr = _s.schedule_array
        fixed_schedule = _arr[:_s.n_providers, :_s.n_days] \
                         if _arr.shape[1] >= _s.n_days else None
        providers_list = (_s.schedule_providers or [])[:_s.n_providers] \
                         or [f"P{i+1:04d}" for i in range(_s.n_providers)]
    else:
        fixed_schedule = None
        providers_list = [f"P{i+1:04d}" for i in range(_s.n_providers)]

    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.stop()

    _training_interval = _PROG_INTERVALS.get(_s.training_program, 30)

    _fresh_seeds = tuple(int(x) for x in np.random.randint(1000, 10001, _s.mc_n_samples))

    with st.spinner(f"Running {_s.mc_n_samples} simulation{'s' if _s.mc_n_samples != 1 else ''}…"):
        _mc = _run_mc(
            n_days=_s.n_days,
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
            training_program=_s.training_program,
            training_interval=_training_interval,
            training_start=_TRAINING_START,
            training_effect=_s.training_effect,
            training_equivalence=_s.training_equivalence,
            seeds_tuple=_fresh_seeds,
        )

    st.session_state.mc_result = _mc
    st.session_state.mc_ran    = True
    st.session_state._last_mc_hash = _mc_hash()
    st.rerun()
