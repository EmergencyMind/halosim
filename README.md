# HaloSim

**HALO Event Exposure & Training Simulation**

HaloSim models the distribution of exposure to high-acuity low-occurrence (HALO) events within a
provider population, and simulates the effect of training programs on population readiness over
time. It generalizes the methodology from:

> Dworkis DA. *Code Blue Blindspots: Quantifying Nursing Exposure to Cardiac Arrest in a
> Community Hospital.* Resuscitation, 2026.

## Quick start (local)

```bash
git clone https://github.com/<your-username>/halosim.git
cd halosim
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to a **public** GitHub repo (e.g. `github.com/<you>/halosim`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — your app is live in ~2 minutes at a public URL

No server, no cost, no configuration.

## Using the app

### Basic mode
- **Events tab:** set a Poisson rate (events/day); 0.14 ≈ ~50 cardiac arrest-equivalents/year
- **Schedules tab:** pick one or more shift templates; the app randomly assigns each provider a template
- **Click ▶ Run Simulation** in the sidebar
- **Exposure Analysis tab:** view gap distributions, threshold sweep, and per-provider summary
- **Training Simulation tab:** select a training frequency and see its effect on population readiness

### Advanced mode
Unlocks: separate day/night event rates, seasonal variation, custom 28-day schedule patterns,
CSV/Excel upload for both events and schedules, complex join (requires hour column), all four
readiness decay models, custom training intervals, targeted training, and partial training boost.

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

## Sample data

`data/sample_events.csv` — 48 synthetic events over 365 days (Poisson, seed 42, ~0.14/day)  
`data/sample_schedule.csv` — 20 providers × 365 days, mixed 3/7 and 4/7 templates

## Reproducibility

Set the **Random seed** in the sidebar to reproduce any result. The seed controls all
Poisson event generation and schedule assignment. Results are identical across runs with the
same seed.

## Limitations

- Poisson-generated events will not exactly reproduce real-data results (e.g., the 98%
  exceeding-90-days figure from Dworkis 2026) because real event timing is clustered differently
  than Poisson; to reproduce the paper exactly, upload the actual event data.
- Free Streamlit Community Cloud instances have ~1GB RAM. Population sizes above ~2,000 providers
  over 730 days may be slow or timeout.
- The targeted training simulation is a sequential day-loop and is slower than vectorized modes;
  expect a few extra seconds for large populations.

## Citation

If you use HaloSim in research, please cite:

> Dworkis DA. *Code Blue Blindspots: Quantifying Nursing Exposure to Cardiac Arrest in a
> Community Hospital.* Resuscitation, 2026.  
> HaloSim: https://github.com/<your-username>/halosim

## License

MIT
