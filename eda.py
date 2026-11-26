"""Run the whole EDA: python eda.py

Writes PNG figures and CSV tables to reports/eda/.
Ship data: data/raw/ship/<day folder>/ as in the README. To read the ship data from another folder
(for example the OneDrive "DP data" folder), set SHIP_DIR below.
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

    # 1) Load every day: 10 s summaries, phases (Quay / Transit / Station n)
    days, phases = {}, {}
    for day in data.days():
        s = data.load_day(day)
        p = data.phases(s)
        days[day] = eda.add_phase_column(s, p).assign(day=day)
        phases[day] = p
        print(f'{day}: {len(s)} files')
        print(p[['phase', 'start', 'end', 'minutes']].round(1).to_string(index=False), '\n')
    s_all = pd.concat(days.values(), ignore_index=True)

    # 2) Tables
    eda.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({day: data.ship_checks(days[day], data.days()[day]) for day in days}).to_csv(
        eda.OUT / 'ship_quality.csv')
    data.buoy_checks(b).to_csv(eda.OUT / 'buoy_quality.csv', index=False)
    eda.summary_stats(s_all).to_csv(eda.OUT / 'summary_stats.csv')
    # Steady-heading blocks; the motion is also computed from the raw 100 Hz data of each whole block
    spectra = {}
    blocks = pd.concat([eda.dp_blocks(days[d], phases[d], b, eda.raw_files(data.days()[d]), spectra).assign(day=d)
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
    print(eda.fig_buoys(b, phases))
    print(eda.fig_footprints(s_all, blocks))
    print(eda.fig_vs_heading(blocks))
    print(eda.fig_correlation(blocks))
    print(eda.fig_spectra(blocks, spectra))
    print(eda.fig_missing(blocks))
