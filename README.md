# TMR12 – RV Gunnerus operating state from ship motion, week 44 2023

Can we tell from the ship's own motion what it is doing: in harbour, on DP or in transit? To validate real operations against simulations you have to compare like with like, so every part of a voyage needs the right operating state. This project explores the RV Gunnerus motion logs (30 Oct, 31 Oct, 2 Nov, 3 Nov 2023), puts them next to the E39 wave buoys outside Ålesund (A, B, C, D, F) and in Halsafjorden (G) for the sea state, and builds models that find the operating state from the motion alone.

## The project in 9 steps

The app (`streamlit run app.py`) tells the same story, one step per page. Each page has the question, one main plot, a takeaway, and a collapsed "Show details" section.

1. **The question:** can the ship's own motion tell us whether it is in harbour, on DP or in transit, and why that matters for comparing real operations with simulations.
2. **One day at sea:** what does the data look like? The ship sails from the quay to a few stations and back, and moves most at the stations.
3. **Can we trust the data?** The ship logs are complete. Limits: GNSS height updates only ~10×/s, it has a noise floor of ~0.15 m (4σ), and buoys C, D and F have no wave data in November.
4. **How the ship moves:** almost still at the quay, a little in transit, most at the stations. Each state moves differently, which is what the models use.
5. **The waves at the buoys:** small waves. Exposed buoy D had about 0.6–0.7 m, sheltered A and B about 0.2 m.
6. **Ship vs buoy:** more waves give more heave (r ≈ 0.73), but this is tangled up with the site (D vs A) and with distance to the buoy.
7. **Heading vs waves:** where the buoy was close (31 Oct, D), the ship rolled most with waves from the side.
8. **Machine learning: operating state:** an unsupervised clustering of 1-minute windows (`python cluster.py`), scored against the correct timeline (the rule-based phases from step 2) and compared with a supervised decision tree. Both get most minutes right; calm DP minutes look like harbour.
9. **What we learned:** mostly yes, a model can label the operating state from motion alone, plus the limits and next steps.

## Run

```bash
python -m venv .venv                  # once (Python 3.11)
.venv\Scripts\activate                # Windows PowerShell/cmd; Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt       # pinned, tested versions

python main.py            # builds data/processed/*.parquet (~1-6 min per day the first time) + reports/*.png
streamlit run app.py      # the explorer
python tests/test_data.py # self-checks
```

In VS Code, pick `.venv` via *Python: Select Interpreter*.

## Layout

```
app.py              Streamlit UI: the 9 steps
main.py             CLI: processed data + PNG reports
tmr12/data.py       loading, 10 s summaries, phases, 10-min blocks, heading blocks, buoys, quality checks
tmr12/cluster.py    operating state (Harbour / DP / Transit) from motion: GMM clustering + decision tree
cluster.py          CLI: runs the clustering, writes reports/cluster/
tmr12/maps.py       folium map
tmr12/report.py     static PNG overview
data/raw/ship/      30/, 31/, 02/, 03/  ->  <hh>/<YYYYMMDDhhmmss>.csv
data/raw/buoys/     E39 buoy subset (+ cut_buoys.py that made it)
data/processed/     generated, safe to delete
reports/            generated PNGs
docs/               test program; ml_choices.md = every ML choice and why
tests/              self-checks
```

To add a day, drop its folder into `data/raw/ship/`. The date is read from the file names. Add a one-line description for it to `PLAN` in `tmr12/data.py`.

## Buoy data

Source: Norwegian Meteorological Institute, E39 buoys (NLOD / CC BY 3.0 NO): <https://thredds.met.no/thredds/catalog/obs/buoy-svv-e39/2023/.html>

| Buoy | Location | Months here |
|---|---|---|
| A, B, C | Sulafjorden | Oct, Nov (C: Nov raw only) |
| D | Breisundet | Oct (Nov raw only) |
| F | Vartdalsfjorden | Oct |
| G | Halsafjorden (near Kristiansund) | Oct, Nov |
| L4 | Sulafjorden | Oct raw only |

"Raw only" files hold the buoy's own motion (heave, pitch, roll, compass at 2 Hz) but no wave parameters (Hm0, Tp, Mdir), so the app does not use them.

To add a buoy, download its monthly `*_wave.nc` into `data/raw/buoys/`, then run:

```bash
pip install xarray netCDF4                          # only needed for this step
python data/raw/buoys/cut_buoys.py data/raw/buoys data/raw/buoys
```

This cuts each new file to the trial windows (`*_trial.nc`) and rebuilds `buoy_wave_params_trial.csv` from all cut files. The app picks the new buoy up automatically.

## Ship CSV columns (`;`, no header, 100 Hz, one file = 10 s)

| # | Meaning |
|---|---|
| 0 | unix time [s] |
| 1, 2 | latitude, longitude [deg] |
| 3 | height [cm]\* (GNSS; tide + heave) |
| 4 | unknown integer |
| 5, 6 | north / east velocity [cm/s]\* |
| 7 | down velocity [cm/s]\* |
| 8, 9, 10 | roll, pitch, heading [deg] |
| 11–13 | roll / pitch / yaw rate [deg/s]\* |

\* Inferred from the data, not documented. Check against the DIPAI/Miros documentation.

## Conventions

- **Times:** shown in local time (CET). Buoy records are in UTC, and each timestamp is the **end** of its 10-minute period. Ship 10-minute blocks use the same grid.
- **Wave direction:** buoy `Mdir`/`ThTp` give the direction the waves come **from**, in degrees true.
- **Relative wave heading:** `(wave_from − ship_heading)`, folded to ±180°. 0 = head seas, +90 = waves from starboard, −90 = from port, ±180 = following seas.
- **Ship "heave 4σ":** 4 × std of the GNSS height, comparable in spirit to the buoy's Hm0. It is the ship's response, not the sea surface, and about 0.1 m of it is GNSS noise.
- **Stations:** stops of at least 10 min away from the start/end quay. **Steady-heading blocks:** heading within 3° for at least 5 min. The thresholds are at the top of `tmr12/data.py`.

## Known data gaps

- **30 Oct:** the plan said buoys D and B, but the ship actually stopped at D (Station 1) and A (Stations 2–3).
- **2 Nov:** buoy D (where the ship sat) has only raw files, no wave parameters. The nearest buoy with data is A, about 7 km away.
- **3 Nov:** the ship was off Kristiansund. The nearest buoy is G, about 24 km from Station 2 and 65 km from Station 1. G sits inside Halsafjorden (short fjord waves, Hm0 ≈ 0.3 m), so it is only a rough reference for the open-sea conditions the ship saw.
