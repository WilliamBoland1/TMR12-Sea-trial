"""Automatic data profiling (fg-data-profiling, earlier ydata-profiling): python profile_data.py

Makes one HTML report per table in reports/profiling/: type, missing values, duplicates, distribution,
correlations and warnings for every column. Install: pip install fg-data-profiling
"""
import os
from pathlib import Path

import pandas as pd
from data_profiling import ProfileReport

from tmr12 import cluster, data, eda

SHIP_DIR = None   # e.g. Path('/Users/<you>/Downloads/DP data')
OUT = data.ROOT / 'reports' / 'profiling'

if __name__ == '__main__':
    folder = SHIP_DIR or os.environ.get('TMR12_SHIP_DIR')
    if folder:
        data.RAW_SHIP = Path(folder)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for day in eda.available_days():
        s = data.load_day(day)
        rows.append(eda.add_phase_column(s, data.phases(s)).assign(day=day))
    w = cluster.windows(pd.concat(rows, ignore_index=True))
    tables = {'ship_1min': w.assign(time=w.time.dt.tz_localize(None)),   # the report wants naive time stamps
              'buoys_10min': data.load_buoys().assign(time=lambda b: b.time.dt.tz_localize(None))}
    for name, df in tables.items():
        ProfileReport(df, title=f'RV Gunnerus trial 30 Oct - 3 Nov 2023: {name}', minimal=True).to_file(OUT / f'{name}.html')
        print(OUT / f'{name}.html')
