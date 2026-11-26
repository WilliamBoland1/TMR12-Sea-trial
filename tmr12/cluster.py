"""Unsupervised operating-state detection: Harbour, Transit or DP, found from the ship's own motion.

No land or distance rules. Each 1-minute window is described by five numbers from the ship logs, and a
Gaussian mixture model (GMM) groups similar windows into three clusters. The correct timeline (the phases
from data.phases(): quay = Harbour, station = DP, transit) is used only afterwards, to score the result.
For comparison, a supervised decision tree learns the correct timeline from the same five features.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tmr12 import data, eda

WINDOW = '1min'      # one sample = one minute of data
SMOOTH_MIN = 5       # a state lasts minutes: take the most common cluster over this many windows
N_STATES = 3
UNSURE = 0.9         # GMM certainty below this counts as unsure
TREE_DEPTH = 3       # small enough to read the tree as a few rules
FEATURES = ['speed_kn', 'roll_std', 'pitch_std', 'heave_hs_m', 'turn_deg']
UNITS = {'speed_kn': 'kn', 'roll_std': 'deg', 'pitch_std': 'deg', 'heave_hs_m': 'm', 'turn_deg': 'deg'}
STATES = ['Harbour', 'DP', 'Transit']
PHASE_STATE = {'Quay': 'Harbour', 'Station': 'DP', 'Transit': 'Transit'}   # data.phases() kind -> state name
COLOURS = {'Harbour': '#8c8c8c', 'DP': '#1f77b4', 'Transit': '#ff7f0e'}


def heading_change(h):
    """Heading change from first to last sample in the window [deg], handling the 0/360 wrap."""
    return np.degrees(np.abs(np.angle(np.exp(1j * np.radians(h.iloc[-1] - h.iloc[0])))))


def windows(s):
    """10 s summaries of one or more days (with 'day' and 'kind' columns) -> one row per minute."""
    key = [s.day, s.time.dt.floor(WINDOW).rename('time')]
    return s.groupby(key).agg(
        speed_kn=('speed_kn', 'median'), roll_std=('roll_std', 'mean'), pitch_std=('pitch_std', 'mean'),
        heave_hs_m=('heave_hs_m', 'mean'), turn_deg=('heading', heading_change),
        correct=('kind', lambda k: PHASE_STATE.get(k.mode()[0], 'Unknown'))).reset_index()


def load_windows():
    """Every day's 10 s rows with their correct state, cut into 1-minute windows."""
    rows = []
    for day in data.days():
        s = data.load_day(day)
        rows.append(eda.add_phase_column(s, data.phases(s)).assign(day=day))
    return windows(pd.concat(rows, ignore_index=True))


def prepare(w):
    """Log-transform (all five are skewed, with long tails) and standardise (GMM compares distances)."""
    x = w[FEATURES].copy()
    x['turn_deg'] += 1                      # heading change is often 0, so use log(1 + x)
    return StandardScaler().fit_transform(np.log(x.clip(lower=1e-3)))


def name_clusters(w, labels):
    """Give clusters a meaning after fitting: fastest = Transit, calmest of the rest = Harbour, last = DP."""
    med = w.assign(c=labels).groupby('c')[['speed_kn', 'roll_std']].median()
    transit = med.speed_kn.idxmax()
    harbour = med.drop(transit).roll_std.idxmin()
    return {transit: 'Transit', harbour: 'Harbour', **{c: 'DP' for c in med.index if c not in (transit, harbour)}}


def smooth(states, days):
    """Most common state over SMOOTH_MIN minutes, per day (removes one-minute flickers)."""
    codes = pd.Series(pd.Categorical(states, categories=STATES).codes, index=states.index)
    sm = codes.groupby(days).transform(
        lambda c: c.rolling(SMOOTH_MIN, center=True, min_periods=1).apply(lambda v: np.bincount(v.astype(int)).argmax()))
    return sm.astype(int).map(dict(enumerate(STATES)))


def fit(w, seed=0):
    """Cluster the windows. Adds: state (raw), state_smooth, prob (GMM certainty of its pick, 0-1)."""
    z = prepare(w)
    gmm = GaussianMixture(N_STATES, n_init=5, random_state=seed).fit(z)
    labels = gmm.predict(z)
    names = name_clusters(w, labels)
    out = w.assign(state=pd.Series(labels).map(names).values, prob=gmm.predict_proba(z).max(axis=1))
    out['state_smooth'] = smooth(out.state, out.day)
    return out


def new_tree():
    return DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=0)


def fit_tree(out):
    """Supervised comparison: a decision tree learns the correct state from the same five features.
    Each day is predicted by a tree trained on the other days only (leave one day out), so it is
    always tested on a day it never saw. Adds: tree, tree_smooth. Also returns one tree fitted on all days, to show."""
    known = out.correct != 'Unknown'
    pred = pd.Series('', index=out.index)
    for day in out.day.unique():
        train = known & (out.day != day)
        pred[out.day == day] = new_tree().fit(out.loc[train, FEATURES], out.correct[train]).predict(
            out.loc[out.day == day, FEATURES])
    out = out.assign(tree=pred)
    out['tree_smooth'] = smooth(out.tree, out.day)
    return out, new_tree().fit(out.loc[known, FEATURES], out.correct[known])


VERSIONS = {'Clustering': 'state', 'Clustering, smoothed': 'state_smooth',
            'Decision tree': 'tree', 'Decision tree, smoothed': 'tree_smooth'}


def check(out):
    """Score the clustering (and the decision tree) against the correct timeline.
    scores: per model version, accuracy (share of minutes right) and adjusted Rand index (1 = identical).
    per_state: for the smoothed clustering, recall (share of the correct minutes the model found), precision (share
    of the model's minutes that are right), F1 (the two combined) and the number of correct minutes.
    table: correct state (rows) vs cluster (columns), in minutes."""
    known = out[out.correct != 'Unknown']
    scores = pd.DataFrame({name: {'accuracy': accuracy_score(known.correct, known[col]),
                                  'ARI': adjusted_rand_score(known.correct, known[col])}
                           for name, col in VERSIONS.items() if col in known}).T
    p, r, f, n = precision_recall_fscore_support(known.correct, known.state_smooth, labels=STATES, zero_division=0)
    per_state = pd.DataFrame({'recall': r, 'precision': p, 'F1': f, 'minutes': n}, index=STATES)
    table = pd.crosstab(known.correct, known.state_smooth, rownames=['Correct timeline'], colnames=['Cluster'])
    return scores, per_state, table.reindex(index=STATES, columns=STATES, fill_value=0)


# ---------- figures ----------

def fig_timeline(out):
    """Per day: colour strips over local time, the correct timeline on top, then the decision tree (if fitted)
    and the clustering at the bottom. Black ticks mark minutes where the GMM is unsure of its pick."""
    rows = [(c, label) for c, label in [('correct', 'Correct timeline'), ('tree_smooth', 'Decision tree'),
                                        ('state_smooth', 'Clustering')] if c in out][::-1]   # bottom row first
    days = list(out.day.unique())
    fig, axes = plt.subplots(len(days), 1, figsize=(12, 0.6 * len(rows) * len(days) + 0.4), squeeze=False)
    for ax, day in zip(axes[:, 0], days):
        d = out[out.day == day]
        t = d.time.dt.tz_convert(data.TZ).dt.tz_localize(None)
        for y, (col, _) in enumerate(rows):
            ax.scatter(t, [y] * len(d), c=d[col].map(COLOURS).fillna('white'), marker='|', s=400, linewidths=2)
        unsure = d.prob < UNSURE
        ax.scatter(t[unsure], [-0.55] * unsure.sum(), marker='|', s=40, color='black')
        ax.set_yticks(range(len(rows)), [label for _, label in rows])
        ax.set_ylim(-0.8, len(rows) - 0.5)
        ax.set_title(day, loc='left', fontsize=10)
        ax.grid(False)
    handles = [plt.Line2D([], [], color=c, lw=6, label=s) for s, c in COLOURS.items()]
    handles.append(plt.Line2D([], [], color='black', marker='|', ls='', label=f'GMM unsure (p < {UNSURE:g})'))
    axes[0, 0].legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1.15), ncol=4, fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel('Local time')
    fig.tight_layout()
    return fig


def fig_features(out):
    """What each cluster looks like: distribution of the five inputs per state (log scale)."""
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(3 * len(FEATURES), 3.5))
    for ax, f in zip(axes, FEATURES):
        vals = [out.loc[out.state_smooth == st, f].clip(lower=1e-3) for st in COLOURS]
        bp = ax.boxplot(vals, labels=list(COLOURS), patch_artist=True, showfliers=False,
                        medianprops={'color': 'black'})
        for patch, c in zip(bp['boxes'], COLOURS.values()):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_yscale('log')
        ax.set_title(f'{f} [{UNITS[f]}]', fontsize=10)
    fig.suptitle('Inputs per cluster', x=0.01, ha='left')
    fig.tight_layout()
    return fig


def fig_tree(tree):
    """The decision tree fitted on all days: each box is a yes/no question on one feature, read top to bottom."""
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_tree(tree, feature_names=FEATURES, class_names=list(tree.classes_), impurity=False, rounded=True,
              fontsize=8, ax=ax)
    for box in ax.texts:   # colour each box by the state it predicts
        state = next((s for s in COLOURS if f'class = {s}' in box.get_text()), None)
        if state and box.get_bbox_patch():
            box.get_bbox_patch().set(facecolor=COLOURS[state], alpha=0.35)
    fig.tight_layout()
    return fig
