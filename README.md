# TMR12 – RV Gunnerus wave-radar trial, week 44 2023

Explore the ship motion logs from RV Gunnerus (31 Oct, 2 Nov, 3 Nov 2023) on a map, and compare them with the E39 wave buoys outside Ålesund.

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

## App pages

| Page | What it shows |
|---|---|
| **Day overview** | The track on a real map (OpenStreetMap, satellite, OpenSeaMap sea marks), stations and buoys. Below it: speed, heading, roll/pitch and heave over a chosen time window, plus raw 100 Hz signals for any 2-minute window. |
| **Ship vs buoys** | Buoy Hm0/Tp next to ship heave and roll/pitch in 10-minute blocks, the ship–buoy distance, and a per-station table. |
| **Heading vs waves** | Ship heading against the buoy's wave direction, and the resulting relative wave heading. Includes a table of steady-heading blocks (which test-plan angle each block really was) and a plot of roll vs relative heading across all days. |

## Layout

```
app.py              Streamlit UI
main.py             CLI: processed data + PNG reports
tmr12/data.py       loading, 10 s summaries, phases, 10-min blocks, heading blocks, buoys
tmr12/maps.py       folium map
tmr12/report.py     static PNG overview
data/raw/ship/      31/, 02/, 03/  ->  <hh>/<YYYYMMDDhhmmss>.csv
data/raw/buoys/     E39 buoy subset (+ cut_buoys.py that made it)
data/processed/     generated, safe to delete
reports/            generated PNGs
docs/               test program
tests/              self-checks
```

To add a day, drop its folder into `data/raw/ship/`. The date is read from the file names.

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

- **2 Nov:** buoy D (where the ship sat) has only raw files, no wave parameters. The nearest buoy with data is A, about 7 km away.
- **3 Nov:** the ship was near Kristiansund, about 100 km from the buoys, so the buoy comparison is not meaningful.

Buoy data from the Norwegian Meteorological Institute (NLOD / CC BY 3.0 NO).
