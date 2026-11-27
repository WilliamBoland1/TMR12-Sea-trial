"""Make the data-visualisation figures: python vis.py

Writes PNG figures to reports/vis/. Uses the same 1-minute windows and labels as the model (tmr12/cluster.py).
Only the 10 s summaries in data/processed/ and the buoy file are needed, not the raw 100 Hz files.
"""
import pandas as pd

from tmr12 import cluster, data, eda, vis

if __name__ == '__main__':
    b = data.load_buoys()
    rows, phases = [], {}
    for day in eda.available_days():
        s = data.load_day(day)
        phases[day] = data.phases(s)
        rows.append(eda.add_phase_column(s, phases[day]).assign(day=day))
    w = cluster.windows(pd.concat(rows, ignore_index=True))
    w = w[w.correct != 'Unknown']

    print(vis.fig_time_budget(w))
    print(vis.fig_decision_space(w, phases))
    print(vis.fig_dp_vs_sea_state(w, phases, b))
    print(vis.fig_parallel(w))
