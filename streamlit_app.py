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
    TEMPLATES,
    generate_schedule,
    generate_synthetic_population,
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
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HaloSim",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "mode": "Basic",
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
        "schedule_source": "Built-in templates",
        "schedule_templates": ["3-on Day / 4-off"],
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔴 HaloSim")
    st.caption("HALO Event Exposure & Training Simulation")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Basic", "Advanced"],
        index=0 if st.session_state.mode == "Basic" else 1,
        horizontal=True,
        help="Basic: binary threshold, simple inputs. Advanced: full model controls.",
    )
    st.session_state.mode = mode
    advanced = mode == "Advanced"

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

    n_providers = st.slider(
        "Population size (providers)",
        min_value=10,
        max_value=MAX_PROVIDERS,
        value=st.session_state.n_providers,
        step=10,
    )
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
        if not advanced:
            rate = st.slider(
                "Event rate (events / day)",
                min_value=0.01,
                max_value=1.0,
                value=st.session_state.event_rate,
                step=0.01,
                help="0.14 ≈ ~50 events/year (matches cardiac arrest rate in Dworkis 2026)",
            )
            st.session_state.event_rate = rate
            st.caption(
                f"Expected events over {n_days} days: **{rate * n_days:.0f}** "
                f"(~{rate * 365:.0f}/year)"
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                day_rate = st.slider("Day shift rate (events/day)", 0.01, 1.0,
                                     st.session_state.event_day_rate, 0.01)
                st.session_state.event_day_rate = day_rate
            with c2:
                night_rate = st.slider("Night shift rate (events/day)", 0.01, 1.0,
                                       st.session_state.event_night_rate, 0.01)
                st.session_state.event_night_rate = night_rate

            with st.expander("Seasonal variation"):
                amp = st.slider("Amplitude (0 = flat)", 0.0, 0.9,
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
        allow_hour = advanced
        uploaded = st.file_uploader(
            "Upload events file",
            type=["csv", "xlsx", "xls"],
            help="Required columns: date (YYYY-MM-DD), shift_type (day/night). "
                 + ("Optional: hour (0–23) for complex join." if advanced else ""),
        )
        if uploaded:
            raw = uploaded.read()
            df, errs = load_events_from_upload(
                raw, uploaded.name, n_days,
                allow_hour_col=allow_hour,
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

    if advanced:
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

    sched_options = ["Built-in templates", "Upload CSV / Excel"]
    if advanced:
        sched_options.append("Custom 28-day pattern")

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

    if sched_source == "Built-in templates":
        selected = st.multiselect(
            "Templates to include (providers randomly assigned one)",
            list(TEMPLATES.keys()),
            default=st.session_state.schedule_templates,
        )
        if selected:
            st.session_state.schedule_templates = selected

        if advanced:
            with st.expander("Template details"):
                for name, pat in TEMPLATES.items():
                    d_pct = pat.count("d") / 28 * 100
                    n_pct = pat.count("n") / 28 * 100
                    o_pct = pat.count("o") / 28 * 100
                    st.caption(
                        f"**{name}** — Day: {d_pct:.0f}%, Night: {n_pct:.0f}%, "
                        f"Off: {o_pct:.0f}%"
                    )
                    st.code(pat, language=None)

        st.divider()
        with st.expander("Use pre-built sample schedule instead"):
            if st.button("Use sample_schedule.csv (20 providers)"):
                raw = (DATA_DIR / "sample_schedule.csv").read_bytes()
                arr, providers, errs = load_schedule_from_upload(
                    raw, "sample_schedule.csv", n_days
                )
                if arr is not None:
                    st.session_state.schedule_array = arr
                    st.session_state.schedule_providers = providers
                    st.session_state.schedule_source = "Upload CSV / Excel"
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

    elif sched_source == "Custom 28-day pattern":
        st.caption("Enter a 28-character pattern using d (day), n (night), o (off).")
        pattern = st.text_input(
            "28-day pattern",
            value="dddoooodddoooodddoooodddoooo",
            max_chars=28,
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
            st.session_state.schedule_templates = [pattern]


# ── Tab 3: Exposure Analysis ───────────────────────────────────────────────

with tab_exposure:
    st.header("Exposure Analysis")

    if advanced:
        st.subheader("Readiness model")
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
        )
        st.session_state.readiness_model = model_map[model_label]

        if st.session_state.readiness_model == "binary":
            thresh = st.slider("Ready if last exposure within (days)",
                               7, 730, st.session_state.readiness_threshold)
            st.session_state.readiness_threshold = thresh
        elif st.session_state.readiness_model == "exponential":
            hl = st.slider("Half-life (days)", 7, 365,
                           int(st.session_state.readiness_half_life))
            st.session_state.readiness_half_life = float(hl)
        elif st.session_state.readiness_model == "ebbinghaus":
            b = st.slider("Forgetting rate b", 0.001, 0.5,
                          st.session_state.ebbinghaus_b, 0.001,
                          format="%.3f")
            st.session_state.ebbinghaus_b = b
        elif st.session_state.readiness_model == "step":
            c1, c2, c3 = st.columns(3)
            with c1:
                t1 = st.slider("T1 — full ready (days)", 7, 365,
                                st.session_state.readiness_threshold)
                st.session_state.readiness_threshold = t1
            with c2:
                t2 = st.slider("T2 — partial ends (days)", t1 + 1, 730,
                                max(st.session_state.step_t2, t1 + 1))
                st.session_state.step_t2 = t2
            with c3:
                pv = st.slider("Partial readiness value", 0.0, 1.0,
                               st.session_state.step_partial, 0.05)
                st.session_state.step_partial = pv
    else:
        st.info(
            f"**Binary threshold model** — providers are considered 'ready' if they "
            f"have had a live HALO exposure within the last **{st.session_state.readiness_threshold} days**. "
            "Switch to Advanced mode to change the readiness model."
        )

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

            st.divider()
            st.plotly_chart(plot_exposure_count_histogram(rdf),
                            use_container_width=True)
            st.plotly_chart(plot_gap_distribution(rdf),
                            use_container_width=True)
            st.plotly_chart(plot_threshold_sweep(rdf),
                            use_container_width=True)

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
            with st.expander("Download results table"):
                csv = rdf.to_csv(index=False).encode()
                st.download_button(
                    "Download per-provider gap statistics (CSV)",
                    data=csv,
                    file_name="halosim_exposure_results.csv",
                    mime="text/csv",
                )


# ── Tab 4: Training Simulation ─────────────────────────────────────────────

with tab_training:
    st.header("Training Simulation")

    program_map = {
        "None (exposure only)": "none",
        "Monthly (every 28 days)": "monthly",
        "Bi-monthly (every 56 days)": "bimonthly",
        "Quarterly (every 84 days)": "quarterly",
    }
    if advanced:
        program_map["Custom interval"] = "custom"
        program_map["Targeted (train undertrained providers only)"] = "targeted"

    prog_label = st.selectbox(
        "Training program",
        list(program_map.keys()),
        index=0,
    )
    st.session_state.training_program = program_map[prog_label]

    if advanced and st.session_state.training_program == "custom":
        c1, c2 = st.columns(2)
        with c1:
            ti = st.slider("Training interval (days)", 7, 365,
                           st.session_state.training_interval)
            st.session_state.training_interval = ti
        with c2:
            ts = st.slider("First training day", 0, 90,
                           st.session_state.training_start)
            st.session_state.training_start = ts

    if advanced and st.session_state.training_program == "targeted":
        targ_thresh = st.slider(
            "Train providers whose readiness is below (%)",
            10, 100, int(st.session_state.training_threshold * 100), 5
        )
        st.session_state.training_threshold = targ_thresh / 100.0
        ti2 = st.slider("Minimum days between training sessions", 7, 180,
                        st.session_state.training_interval)
        st.session_state.training_interval = ti2

    if advanced and st.session_state.training_program != "none":
        st.subheader("Training effect")
        effect_opts = ["Full reset (training = live exposure)", "Partial boost"]
        eff = st.radio("Training effectiveness", effect_opts, horizontal=True,
                       index=0 if st.session_state.training_effect == "full" else 1)
        st.session_state.training_effect = "full" if eff == effect_opts[0] else "partial"
        if st.session_state.training_effect == "partial":
            eq = st.slider("Equivalence factor (1.0 = same as live exposure)",
                           0.1, 1.0, st.session_state.training_equivalence, 0.05)
            st.session_state.training_equivalence = eq

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

            roll = 30
            if advanced:
                roll = st.slider("Rolling mean window (days)", 1, 90, 30,
                                 key="roll_window")

            fig = plot_readiness_timeseries(
                sim_b.proportion_ready_on_shift,
                sim_t.proportion_ready_on_shift if sim_t else sim_b.proportion_ready_on_shift,
                n_days=sim_b.n_days,
                rolling_days=roll,
            )
            st.plotly_chart(fig, use_container_width=True)

            if advanced:
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
        if advanced:
            events_df, e_warn = generate_events(
                n_days=n_days,
                rate=st.session_state.event_day_rate + st.session_state.event_night_rate,
                seed=int(seed),
                day_rate=st.session_state.event_day_rate,
                night_rate=st.session_state.event_night_rate,
                seasonal_amplitude=st.session_state.seasonal_amplitude,
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
    if sched_source == "Upload CSV / Excel" and st.session_state.schedule_array is not None:
        schedule = st.session_state.schedule_array[:n_providers, :n_days] \
            if st.session_state.schedule_array.shape[1] >= n_days else None
        providers_list = st.session_state.schedule_providers[:n_providers]
        if schedule is None or schedule.shape[1] < n_days:
            # Rebuild if window longer than upload
            schedule, s_warn = generate_schedule(
                n_providers=n_providers,
                n_days=n_days,
                templates=st.session_state.schedule_templates or list(TEMPLATES.keys()),
                seed=int(seed),
            )
            providers_list = [f"P{i+1:04d}" for i in range(n_providers)]
    elif sched_source == "Custom 28-day pattern":
        templates_to_use = st.session_state.schedule_templates or ["dddoooodddoooodddoooodddoooo"]
        schedule, s_warn = generate_schedule(
            n_providers=n_providers,
            n_days=n_days,
            templates=templates_to_use,
            seed=int(seed),
        )
        providers_list = [f"P{i+1:04d}" for i in range(n_providers)]
    else:
        templates_to_use = st.session_state.schedule_templates or list(TEMPLATES.keys())
        schedule, s_warn = generate_schedule(
            n_providers=n_providers,
            n_days=n_days,
            templates=templates_to_use,
            seed=int(seed),
        )
        providers_list = [f"P{i+1:04d}" for i in range(n_providers)]

    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.stop()

    # 3. Run baseline simulation
    with st.spinner("Running simulation…"):
        sim_b = Simulation(
            n_days=n_days,
            providers=providers_list,
            schedule=schedule,
            events=events_df,
            seed=int(seed),
            readiness_model=st.session_state.readiness_model,
            readiness_threshold_days=st.session_state.readiness_threshold,
            readiness_half_life_days=st.session_state.readiness_half_life,
            ebbinghaus_b=st.session_state.ebbinghaus_b,
            step_t2_days=st.session_state.step_t2,
            step_partial_value=st.session_state.step_partial,
            training_program="none",
        )
        sim_b.run()

    # 4. Run training simulation if a program is selected
    training_prog = st.session_state.training_program
    if training_prog != "none":
        with st.spinner("Running training simulation…"):
            sim_t = Simulation(
                n_days=n_days,
                providers=providers_list,
                schedule=schedule,
                events=events_df,
                seed=int(seed),
                readiness_model=st.session_state.readiness_model,
                readiness_threshold_days=st.session_state.readiness_threshold,
                readiness_half_life_days=st.session_state.readiness_half_life,
                ebbinghaus_b=st.session_state.ebbinghaus_b,
                step_t2_days=st.session_state.step_t2,
                step_partial_value=st.session_state.step_partial,
                training_program=training_prog,
                training_interval_days=st.session_state.training_interval,
                training_start_day=st.session_state.training_start,
                training_effect=st.session_state.training_effect,
                training_equivalence=st.session_state.training_equivalence,
                training_target_threshold=st.session_state.training_threshold,
            )
            sim_t.run()
    else:
        sim_t = sim_b  # no training — same as baseline

    st.session_state.sim_baseline = sim_b
    st.session_state.sim_trained = sim_t
    st.session_state.sim_ran = True

    st.sidebar.success(
        f"✓ Done — {len(providers_list):,} providers × {n_days} days"
    )
    st.rerun()
