# HaloSim

**HALO Event Exposure & Training Simulation**

Live app: **https://sfl-halosim.streamlit.app/**

HaloSim models how often providers in a clinical setting actually encounter high-acuity
low-occurrence (HALO) events — cardiac arrests, airway emergencies, or any critical event
that is rare but consequential. It simulates a provider population, assigns shift schedules,
distributes events across those shifts, and computes each provider's exposure history. A second
simulation layer shows how training programs affect population readiness over time.

The methodology is based on:

> PMID: 41633464 — *Code Blue blindspots: mapping nursing exposure to cardiac arrests.*
> Resuscitation. 2026.

---

## Using the app

### 1. Configure your model (⚙️ Model Parameters tab)

All simulation parameters are set in the **⚙️ Model Parameters** tab, organized into four sections:

**Simulation (1/4)**
- **Duration** — 90 / 180 / 365 / 730 days
- **Number of providers** — default 200 (max 5,000)
- **Critical threshold (days)** — maximum acceptable gap between HALO exposures; providers
  exceeding this are flagged as under-exposed (default 90 days)
- **Number of simulations** — each run draws independent random seeds for event timing and
  shift assignments; 50 is a good default

**HALO Events (2/4)**
- **Generate (Poisson MC)** — draw events from a Poisson model; set events per year
  (~51/year matches the community hospital rate in PMID: 41633464)
  - Advanced: adjust % occurring on day vs. night shifts
- **Upload CSV / Excel** — upload your own event log

**Provider Schedules (3/4)**
- **Generate schedules** — pick a shift pattern from a button grid:

| Schedule type | Description |
|---|---|
| 3/7 Day | 3 day shifts per week, rest off |
| 3/7 Night | 3 night shifts per week, rest off |
| 4/7 Day | 4 day shifts per week, rest off |
| 4/7 Night | 4 night shifts per week, rest off |
| Progressive (day & night mix) | 3–4 shifts/week, mix of day and night |
| Random | Each day drawn from empirical weights: 25% day, 23% night, 52% off (PMID: 41633464) |

- Advanced schedule settings: override with custom % day / % night sliders
- **Upload CSV / Excel** — upload a pre-built schedule

**Training Program (4/4)**

Choose one program via the 2×2 button grid:
- **None (exposure only)** — model exposure gap without any training intervention
- **Monthly (every 30 days)**
- **Bi-monthly (every 60 days)**
- **Quarterly (every 91 days)**

Training sessions start on day 14 and repeat at the selected interval. All providers on shift on a
training day receive a full readiness reset.

---

### 2. Run

Click **▶ Run Simulation** above the tabs. A progress bar appears next to the button while
simulations run. A green banner confirms completion; an orange warning appears if any setting
has changed since the last run.

---

### 3. Exposure Analysis (📊 Exposure tab)

Summary metrics (median across all MC runs, with p10–p90 in tooltip):
- **% exceeding threshold** — share of providers whose maximum gap exceeds the critical threshold
- **Median inter-exposure gap** — median days between consecutive HALO exposures
- **Median exposures / provider**

Charts:
- **Providers with gap > threshold** — threshold sweep showing % of providers exceeding each gap
  duration; solid line = median, shaded band = p10–p90
- **On-shift readiness over time** — rolling proportion of on-shift providers within the readiness
  threshold; solid line = median, shaded band = p10–p90
- Chart options expander: adjust rolling mean window (1–90 days)

---

### 4. Training Effects (🏋️ Training tab)

Only active when a training program is selected. Shows:

Summary metrics (2×2 grid):
- **Training sessions** — median number of sessions held across runs
- **Providers trained** — median providers receiving at least one training session
- **Change in % > threshold gap** — shift in the share exceeding the critical threshold
  (trained − baseline; negative = improvement)
- **On-shift readiness** — median percentage-point increase vs. no training

Charts:
- **Providers with effective gap > threshold** — baseline (blue) vs. trained (green);
  effective gap counts both HALO exposures and training sessions as resets
- **On-shift readiness over time** — baseline vs. trained comparison band

---

### 5. Download Results (⬇️ Download Results tab)

Available after running a simulation:

- **📄 PDF report** — formatted report with simulation parameters, exposure metrics, and
  charts; includes a training effects section if a program was selected
- **📦 Raw data (JSON)** — schedules, events, and training assignments for the reference run
  (seed 0); raw simulation inputs only, no analysis

---

## Sidebar

The sidebar contains:
- **Instructions** — brief in-app usage guide
- **About** — overview of HALO events and the project
- Link to [Sangfroid Labs](https://sangfroid-labs.netlify.app/)

---

## File formats for upload

### Events file (CSV or Excel)

| Column | Type | Notes |
|--------|------|-------|
| `date` | YYYY-MM-DD | Required |
| `shift_type` | `day` or `night` | Required |

### Schedule file (CSV or Excel)

| Column | Type | Notes |
|--------|------|-------|
| `provider_id` | string | Required |
| `date` | YYYY-MM-DD | Required |
| `shift_type` | `day`, `night`, or `off` | Required (also accepts `d`, `n`, `o`) |

Missing dates for a provider default to `off`. Maximum 5,000 providers.

---

## Reproducibility

The **Number of simulations** setting controls how many independent MC runs are drawn. Each run
uses a different random seed for event timing and schedule generation. Results are stable across
runs with sufficient N (≥50 recommended). Individual run seeds are recorded in the raw JSON export.

---

## Limitations

- Poisson-generated events will not exactly reproduce real-data results (e.g., the 98%
  exceeding-90-days figure from PMID: 41633464) because real event timing differs from Poisson.
  Upload your actual event file to reproduce the paper's results.
- Instances have ~1 GB RAM. Population sizes above ~2,000 providers over 730 days may be slow.

---

## Citation

If you use HaloSim in research, please cite:

> Walker D, Dworkis D. *HaloSim: HALO Event Exposure & Training Simulation* [software]. Sangfroid Labs; 2026. https://sfl-halosim.streamlit.app/

---

## License

MIT
