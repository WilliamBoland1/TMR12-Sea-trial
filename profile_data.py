"""Automatic data profiling (fg-data-profiling, earlier ydata-profiling): python profile_data.py

Makes one HTML report per table in reports/profiling/: type, missing values, duplicates, distribution,
correlations and warnings for every column. Run eda.py first (it writes the heading-block table).
Install: pip install fg-data-profiling
"""
import os
from pathlib import Path

import pandas as pd
from data_profiling import ProfileReport

from tmr12 import data, eda

SHIP_DIR = None   # e.g. Path('/Users/<you>/Downloads/DP data')
OUT = data.ROOT / 'reports' / 'profiling'

if __name__ == '__main__':
    folder = SHIP_DIR or os.environ.get('TMR12_SHIP_DIR')
    if folder:
        data.RAW_SHIP = Path(folder)
    OUT.mkdir(parents=True, exist_ok=True)

    minutes = []
    for day in eda.available_days():
        s = data.load_day(day)
        m = eda.minute_stats(eda.add_phase_column(s, data.phases(s)))
        minutes.append(m.assign(time=m.time.dt.tz_localize(None)))   # the report wants naive time stamps
    tables = {'ship_1min': pd.concat(minutes, ignore_index=True),
              'buoys_10min': data.load_buoys().assign(time=lambda b: b.time.dt.tz_localize(None)),
              'heading_blocks': pd.read_csv(eda.OUT / 'dp_heading_blocks.csv')}
    for name, df in tables.items():
        ProfileReport(df, title=f'RV Gunnerus trial 30 Oct - 3 Nov 2023: {name}', minimal=True).to_file(OUT / f'{name}.html')
        print(OUT / f'{name}.html')
