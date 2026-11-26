"""Supervised comparison for the clustering: a random forest learns the correct state (Harbour / DP / Transit)
from the same five motion features as the GMM in cluster.py."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tmr12.cluster import COLOURS, FEATURES, UNITS, smooth

N_TREES = 200


def new_forest():
    return RandomForestClassifier(N_TREES, min_samples_leaf=5, n_jobs=-1, random_state=0)


def fit_forest(out):
    """Each day is predicted by a forest trained on the other days only (leave one day out), so it is
    always tested on a day it never saw. Adds: forest, forest_smooth. Also returns one forest fitted on all days."""
    known = out.correct != 'Unknown'
    pred = pd.Series('', index=out.index)
    for day in out.day.unique():
        train = known & (out.day != day)
        pred[out.day == day] = new_forest().fit(out.loc[train, FEATURES], out.correct[train]).predict(
            out.loc[out.day == day, FEATURES])
    out = out.assign(forest=pred)
    out['forest_smooth'] = smooth(out.forest, out.day)
    return out, new_forest().fit(out.loc[known, FEATURES], out.correct[known])


def fig_importance(forest):
    """How much each feature helps the forest separate the states (mean decrease in impurity, sums to 1)."""
    imp = pd.Series(forest.feature_importances_, index=[f'{f} [{UNITS[f]}]' for f in FEATURES]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(imp.index, imp.values, color=COLOURS['DP'])
    ax.set_xlabel('Feature importance')
    fig.tight_layout()
    return fig
