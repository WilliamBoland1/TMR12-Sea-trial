"""Find the vessel's operating state (Harbour / DP / Transit) from its motion alone: python cluster.py

Unsupervised: k-means groups 1-minute windows of ship motion into three clusters.
The reference timeline (rule-based phases) is used only to score the result.
For comparison, a supervised random forest is trained on it (leave one day out). Writes figures and tables to reports/cluster/.
Ship data folder: set SHIP_DIR as in eda.py.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tmr12 import cluster, data, forest

SHIP_DIR = None
OUT = Path(__file__).resolve().parent / 'reports' / 'cluster'


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return OUT / name


if __name__ == '__main__':
    folder = SHIP_DIR or os.environ.get('TMR12_SHIP_DIR')
    if folder:
        data.RAW_SHIP = Path(folder)

    w = cluster.load_windows()
    print(f'{len(w)} one-minute windows from {w.day.nunique()} days')

    out = cluster.fit(w)
    out, rf = forest.fit_forest(out)
    scores, per_state, table = cluster.check(out)
    print('Performance against the reference timeline:')
    print(scores.round(3))
    print(per_state.round(3))
    print(table)
    checks = forest.feature_checks(out)
    print('Random forest without a feature (leave one day out, not smoothed):')
    print(checks.round(3))

    OUT.mkdir(parents=True, exist_ok=True)
    checks.to_csv(OUT / 'feature_checks.csv', lineterminator='\n')
    table.to_csv(OUT / 'confusion_vs_correct.csv')
    scores.to_csv(OUT / 'performance.csv')
    per_state.to_csv(OUT / 'performance_per_state.csv')
    out.round(4).to_csv(OUT / 'states_per_minute.csv', index=False)
    print(save(cluster.fig_timeline(out), 'c1_timeline.png'))
    print(save(cluster.fig_features(out), 'c2_features_per_state.png'))
    print(save(forest.fig_importance(rf), 'c3_forest_importance.png'))
