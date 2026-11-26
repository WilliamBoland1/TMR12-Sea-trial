"""Run the whole EDA: python eda.py

Writes PNG figures and CSV tables to reports/eda/.
Ship data: the 10 s summaries in data/processed/ (in git) are used for every day. The raw 100 Hz files
(data/raw/ship/<day folder>/ as in the README, not in git) are used where they exist. To read the raw
files from another folder (for example the OneDrive "DP data" folder), set SHIP_DIR below.
"""
from pathlib import Path

import pandas as pd

from tmr12 import data, eda

SHIP_DIR = None   # e.g. Path('/Users/<you>/Library/CloudStorage/OneDrive-NTNU/TMR12/DP data')

if __name__ == '__main__':
    import os
    folder = SHIP_DIR or os.environ.get('TMR12_SHIP_DIR')
    if folder:
        data.RAW_SHIP = Path(folder)

    b = data.load_buoys()
    raw_dirs = eda.available_days()   # {day: raw folder, or None when only the 10 s summary exists}

    # 1) Load every day: 10 s summaries, phases (Quay / Transit / Station n)
    days, phases = {}, {}
    for day, raw in raw_dirs.items():
        s = data.load_day(day)
        p = data.phases(s)
        days[day] = eda.add_phase_column(s, p).assign(day=day)
        phases[day] = p
        print(f'{day}: {len(s)} files, raw 100 Hz data: {"yes" if raw else "no"}')
        print(p[['phase', 'start', 'end', 'minutes']].round(1).to_string(index=False), '\n')
    s_all = pd.concat(days.values(), ignore_index=True)

    # 2) Tables
    eda.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({day: eda.ship_checks(days[day], raw_dirs[day]) for day in days}).to_csv(
        eda.OUT / 'ship_quality.csv')
    data.buoy_checks(b).to_csv(eda.OUT / 'buoy_quality.csv', index=False)
    eda.summary_stats(s_all).to_csv(eda.OUT / 'summary_stats.csv')
    # Steady-heading blocks; with raw files the motion is computed from the 100 Hz data of each whole block
    spectra = {}
    blocks = pd.concat([eda.dp_blocks(days[d], phases[d], b, eda.raw_files(raw_dirs[d]) if raw_dirs[d] else None,
                                      spectra).assign(day=d)
                        for d in days], ignore_index=True)
    blocks.round(3).to_csv(eda.OUT / 'dp_heading_blocks.csv', index=False)

    # Duplicates and outliers (Tukey's 1.5 IQR rule on the 1-minute values, per phase type)
    minutes = pd.concat([eda.minute_stats(s) for s in days.values()], ignore_index=True)
    eda.duplicates(days, b).to_csv(eda.OUT / 'duplicates.csv', index=False)
    eda.iqr_outliers(minutes).to_csv(eda.OUT / 'outliers_iqr.csv', index=False)
    print(eda.duplicates(days, b).to_string(index=False))
    print(eda.iqr_outliers(minutes).to_string(index=False))
    print(f'{len(blocks)} steady-heading blocks at the DP stations')

    # 3) Figures
    print(eda.fig_completeness(days))
    print(eda.fig_tracks(days, b))
    for day in days:
        print(eda.fig_day_timeseries(day, days[day], phases[day]))
    print(eda.fig_univariate(s_all))
    buoy_days = sorted(set(b.time.dt.tz_convert(data.TZ).dt.strftime('%Y-%m-%d')) & set(phases))
    print(eda.fig_buoys(b, {d: phases[d] for d in buoy_days}))
    print(eda.fig_footprints(s_all, blocks))
    print(eda.fig_vs_heading(blocks))
    print(eda.fig_correlation(blocks))
    print(eda.fig_spectra(blocks, spectra))
    print(eda.fig_missing(blocks))
