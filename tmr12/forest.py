"""Supervised comparison for the clustering: a random forest learns the reference state (Harbour / DP / Transit)
from the same five motion features as the k-means clustering in cluster.py."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tmr12 import eda
from tmr12.cluster import FEATURES, smooth

N_TREES = 200


def new_forest():
    return RandomForestClassifier(N_TREES, min_samples_leaf=5, n_jobs=-1, random_state=0)


def predict_lodo(out, features=FEATURES, shuffle=None):
    """Each day is predicted by a forest trained on the other days only (leave one day out), so it is
    always tested on a day it never saw. shuffle: a feature whose values are shuffled on the test day."""
    known = out.correct != 'Unknown'
    rng = np.random.default_rng(0)
    pred = pd.Series('', index=out.index)
    for day in out.day.unique():
        train, test = known & (out.day != day), out.day == day
        x = out.loc[test, features].copy()
        if shuffle:
            x[shuffle] = rng.permutation(x[shuffle].values)
        pred[test] = new_forest().fit(out.loc[train, features], out.correct[train]).predict(x)
    return pred


def fit_forest(out):
    """Leave-one-day-out predictions (predict_lodo). Adds: forest, forest_smooth. Also returns one forest fitted on
    all days."""
    known = out.correct != 'Unknown'
    out = out.assign(forest=predict_lodo(out))
    out['forest_smooth'] = smooth(out.forest, out.day)
    return out, new_forest().fit(out.loc[known, FEATURES], out.correct[known])


def feature_checks(out):
    """How much the forest needs each feature: leave-one-day-out accuracy (not smoothed) when a feature is shuffled
    on the test day (the trained trees lose it) or left out when training (the forest can use the others instead)."""
    known = out.correct != 'Unknown'
    without = lambda *fs: [f for f in FEATURES if f not in fs]
    checks = {'all five features': predict_lodo(out),
              'roll shuffled on the test day': predict_lodo(out, shuffle='roll_std'),
              'without roll': predict_lodo(out, without('roll_std')),
              'without pitch': predict_lodo(out, without('pitch_std')),
              'without roll and pitch': predict_lodo(out, without('roll_std', 'pitch_std')),
              'without speed': predict_lodo(out, without('speed_kn'))}
    return pd.Series({k: (p[known] == out.correct[known]).mean() for k, p in checks.items()}, name='accuracy')


def fig_importance(forest):
    """How much each feature helps the forest separate the states (mean decrease in impurity, sums to 1)."""
    imp = pd.Series(forest.feature_importances_, index=[eda.FEATURE_LABELS[f] for f in FEATURES]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(imp.index, imp.values, color=eda.GREY)
    ax.set_xlabel('Feature importance')
    fig.tight_layout()
    return fig
