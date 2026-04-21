# HaloSim

**HALO Event Exposure & Training Simulation**

Live app: **https://sfl-halosim.streamlit.app/**

HaloSim models how often providers in a clinical setting actually encounter high-acuity
low-occurrence (HALO) events — cardiac arrests, airway emergencies, or any critical event
that is rare but consequential. It simulates a provider population, assigns shift schedules,
distributes events across those shifts, and computes each provider's exposure history. A second
simulation layer shows how training programs affect population readiness over time.

The methodology is based on:

> PMID: 41633464 — *Code Blue Blindspots: Quantifying Nursing Exposure to Cardiac Arrest in a
> Community Hospital.* Resuscitation, 2026.

---

## Using the app

### 1. Set sidebar parameters

Before running, configure in the **sidebar**:

- **Simulation window** — 90 / 180 / 365 / 730 days
- **Number of providers** — default 200 (max 5,000)
- **Random seed** — controls all Poisson sampling and schedule generation; set for reproducibility
- **Critical threshold (days)** — the maximum acceptable gap between HALO exposures; providers
  exceeding this are flagged as under-exposed (default 90 days)
- **Training program** — None / Monthly (28d) / Bi-monthly (56d) / Quarterly (84d) / Custom / Targeted

### 2. Configure events (Events tab)

**Generate (Poisson MC)** — draw events from a Poisson model:
- Set **events/year** (slider; ~51/year matches PMID: 41633464 community hospital rate)
- Advanced settings: adjust % occurring on **day vs. night shifts** (default 50/50)
- One-click loaders: **Load sample events** (48 events / year) or **Load full demo scenario**
  (events + schedule pre-loaded, simulation auto-runs)

**Upload CSV / Excel** — upload your own event data.

### 3. Configure schedules (Schedules tab)

**Generate schedules** — randomly generates an independent schedule for each provider:

| Schedule type | Description |
|---|---|
| 3/7 Day | 3 randomly placed day shifts per 7-day week, rest off |
| 3/7 Night | 3 randomly placed night shifts per 7-day week, rest off |
| 4/7 Day | 4 randomly placed day shifts per 7-day week, rest off |
| 4/7 Night | 4 randomly placed night shifts per 7-day week, rest off |
| Progressive (day & night mix) | 3–4 shifts per week, each randomly assigned day or night |
| Random | Each day drawn from empirical weights: 25% day, 23% night, 52% off (PMID: 41633464) |

Advanced schedule settings: override with custom % day / % night sliders.

**Upload CSV / Excel** — upload a pre-built schedule.

### 4. Run

Click **▶ Run Simulation** in the sidebar. Results are cached — re-run any time settings change.
The banner at the top turns **orange** if any setting has changed since the last run.

### 5. Exposure Analysis tab

- Summary metrics: median exposures / provider, median days between exposures, % exceeding threshold
- Percentile table (5th / 25th / 50th / 75th / 95th) for max gap, median gap, exposures / provider
- Charts: exposure count histogram, gap distribution (min / median / max), threshold sweep
- Individual provider swimlane heatmap
- Simulated events and schedule detail expanders (with monthly bar chart and heatmap)
- Downloads: exposure statistics CSV, simulated events CSV, simulated schedules CSV, PDF report

### 6. Training Effects tab

The training program is selected in the **sidebar** before running. This tab shows:
- Summary metrics: average readiness with vs. without training
- Comparison chart: overlay multiple programs on one chart
  - Pre-populated with "No training" + your selected program
  - Toggle between **"Use my current settings"** (Custom/Targeted reflect your controls)
    and **standardised defaults** (all programs use fixed reference intervals for apples-to-apples comparison)
- Additional controls appear only when relevant:
  - **Custom**: training interval slider + first training day
  - **Targeted**: readiness threshold + minimum interval between sessions
  - **Custom / Targeted / any non-None**: training effectiveness (full reset or partial boost)

---

## File formats for upload

### Events file (CSV or Excel)

| Column | Type | Notes |
|--------|------|-------|
| `date` | YYYY-MM-DD | Required |
| `shift_type` | `day` or `night` | Required |
| `hour` | integer 0–23 | Optional; enables shift-boundary join (Advanced) |

### Schedule file (CSV or Excel)

| Column | Type | Notes |
|--------|------|-------|
| `provider_id` | string | Required |
| `date` | YYYY-MM-DD | Required |
| `shift_type` | `day`, `night`, or `off` | Required (also accepts `d`, `n`, `o`) |

Missing dates for a provider default to `off`. Maximum 5,000 providers.

---

## Sample data

`data/sample_events.csv` — 48 synthetic events over 365 days (Poisson, seed 42)  
`data/sample_schedule.csv` — 20 providers × 365 days

Use **Load sample events** or **Load full demo scenario** (Events tab, Generate mode) to load
these with one click. "Load full demo scenario" also pre-loads the schedule and auto-runs the
simulation.

---

## Reproducibility

Set the **Random seed** in the sidebar to reproduce any result. The seed controls all Poisson
event generation and schedule randomisation. Results are identical across runs with the same seed.

---

## Limitations

- Poisson-generated events will not exactly reproduce real-data results (e.g., the 98%
  exceeding-90-days figure from PMID: 41633464) because real event timing differs from Poisson.
  Upload your actual event file to reproduce the paper's results.
- Instances have ~1 GB RAM. Population sizes above ~2,000 providers over 730 days may be slow.
- Targeted training uses a sequential day-loop and is slower than vectorised modes;
  expect a few extra seconds for large populations.
- Readiness is modelled as a binary threshold (ready / not ready). The gap between exposures
  determines whether a provider remains within threshold.

---

## Citation

If you use HaloSim in research, please cite:

> PMID: 41633464 — *Code Blue Blindspots: Quantifying Nursing Exposure to Cardiac Arrest in a
> Community Hospital.* Resuscitation, 2026.  
> HaloSim: https://sfl-halosim.streamlit.app/ (https://github.com/EmergencyMind/halosim)

---

## License

MIT
