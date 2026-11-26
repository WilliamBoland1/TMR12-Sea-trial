"""Find the vessel's operating state (Harbour / DP / Transit) from its motion alone: python cluster.py

Unsupervised: a Gaussian mixture model groups 1-minute windows of ship motion into three clusters.
The correct timeline (rule-based phases) is used only to score the result.
For comparison, a supervised decision tree is trained on it (leave one day out). Writes figures and tables to reports/cluster/.
Ship data folder: set SHIP_DIR as in eda.py.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import export_text

from tmr12 import cluster, data

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
    out, tree = cluster.fit_tree(out)
    print('Decision tree fitted on all days:')
    print(export_text(tree, feature_names=cluster.FEATURES, decimals=2))
    scores, per_state, table = cluster.check(out)
    print('Performance against the correct timeline:')
    print(scores.round(3))
    print(per_state.round(3))
    print(f'Windows where the GMM is unsure (p < {cluster.UNSURE:g}): {(out.prob < cluster.UNSURE).mean():.1%}')
    print(table)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / 'confusion_vs_correct.csv')
    scores.to_csv(OUT / 'performance.csv')
    per_state.to_csv(OUT / 'performance_per_state.csv')
    out.round(4).to_csv(OUT / 'states_per_minute.csv', index=False)
    print(save(cluster.fig_timeline(out), 'c1_timeline.png'))
    print(save(cluster.fig_features(out), 'c2_features_per_state.png'))
    print(save(cluster.fig_tree(tree), 'c3_decision_tree.png'))
