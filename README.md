# HaloSim

**HALO Event Exposure & Training Simulation**

Live app: **https://sfl-halosim.streamlit.app/**

HaloSim models how often providers in a clinical setting actually encounter high-acuity
low-occurrence (HALO) events — cardiac arrests, airway emergencies, or any critical event
that is rare but consequential. It simulates a provider population, assigns shift schedules,
distributes events across those shifts, and computes each provider's exposure history. A second
simulation layer shows how training programs affect population readiness over time.

The methodology is based on:

> Dworkis DA. *Code Blue Blindspots: Quantifying Nursing Exposure to Cardiac Arrest in a
> Community Hospital.* Resuscitation, 2026.

---

## Using the app

### Basic mode

1. **Events tab** — set a Poisson event rate (events/day). Rate 0.14 ≈ ~50 cardiac arrest-equivalents/year.
2. **Schedules tab** — pick one or more shift templates; the app randomly assigns each provider a template.
3. Click **▶ Run Simulation** in the sidebar.
4. **Exposure Analysis tab** — view exposure distributions, threshold sweep chart, and per-provider summary.
5. **Training Simulation tab** — select a training frequency (monthly, quarterly, etc.) and compare population readiness with vs. without training.

### Advanced mode

Toggle **Advanced** in the sidebar to unlock:

- Separate day/night event rates and seasonal variation
- Custom 28-day schedule patterns (enter any d/n/o string)
- CSV/Excel upload for both events and schedules
- Complex exposure join (requires an `hour` column in your events file)
- All four readiness decay models (binary, exponential, Ebbinghaus, two-threshold step)
- Custom training intervals, partial training boost, and targeted training (trains only under-ready providers)

---

## File formats for upload

### Events file (CSV or Excel)

| Column | Type | Notes |
|--------|------|-------|
| `date` | YYYY-MM-DD | Required |
| `shift_type` | `day` or `night` | Required |
| `hour` | integer 0–23 | Optional; enables complex join in Advanced mode |

### Schedule file (CSV or Excel)

| Column | Type | Notes |
|--------|------|-------|
| `provider_id` | string | Required |
| `date` | YYYY-MM-DD | Required |
| `shift_type` | `day`, `night`, or `off` | Required (also accepts `d`, `n`, `o`) |

Missing dates for a provider default to `off`. Maximum 5,000 providers.

---

## Built-in schedule templates

All templates are 28-day repeating cycles (`d` = day shift, `n` = night, `o` = off):

| Template | 28-day pattern |
|----------|---------------|
| 3-on Day / 4-off | `dddoooodddoooo...` |
| 3-on Night / 4-off | `nnnoooonnnoooo...` |
| 4-on Day / 3-off | `ddddoooddddooo...` |
| 4-on Night / 3-off | `nnnnooonnnnooo...` |
| Rotating Day→Night | Day and night shifts alternating |
| Progressive (nurse-style) | Mixed d/n/o per day (matches paper's Line A–I logic) |

---

## Sample data

`data/sample_events.csv` — 48 synthetic events over 365 days (Poisson, seed 42, ~0.14/day)  
`data/sample_schedule.csv` — 20 providers × 365 days, mixed 3/7 and 4/7 templates

These are available via the **Load sample data** buttons inside the Events and Schedules tabs.

---

## Reproducibility

Set the **Random seed** in the sidebar to reproduce any result. The seed controls all
Poisson event generation and schedule assignment. Results are identical across runs with the
same seed.

---

## Limitations

- Poisson-generated events will not exactly reproduce real-data results (e.g., the 98%
  exceeding-90-days figure from Dworkis 2026) because real event timing is clustered differently
  than Poisson. To reproduce the paper exactly, upload the actual event data.
- Instances have ~1 GB RAM. Population sizes above ~2,000 providers over 730 days may be slow.
- Targeted training is a sequential day-loop and is slower than vectorized modes;
  expect a few extra seconds for large populations.

---

## Citation

If you use HaloSim in research, please cite:

> Dworkis DA. *Code Blue Blindspots: Quantifying Nursing Exposure to Cardiac Arrest in a
> Community Hospital.* Resuscitation, 2026.  
> HaloSim: https://sfl-halosim.streamlit.app/ (https://github.com/EmergencyMind/halosim)

---

## License

MIT
