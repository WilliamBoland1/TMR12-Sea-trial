"""Make the data-visualisation figures: python vis.py  (run eda.py first)

Writes PNG figures to reports/vis/. Ship data folder: set SHIP_DIR as in eda.py.
"""
from pathlib import Path

import pandas as pd

from tmr12 import data, eda, vis

SHIP_DIR = None   # e.g. Path('/Users/<you>/Downloads/DP data')

if __name__ == '__main__':
    import os
    folder = SHIP_DIR or os.environ.get('TMR12_SHIP_DIR')
    if folder:
        data.RAW_SHIP = Path(folder)

    b = data.load_buoys()
    blocks = vis.load_blocks()
    days = {}
    for day in data.days():
        s = data.load_day(day)
        days[day] = eda.add_phase_column(s, data.phases(s))

    print(vis.fig_motion_rose(blocks))
    print(vis.fig_heave_vs_buoy(blocks, b))
    print(vis.fig_two_time_scales(blocks, days))
