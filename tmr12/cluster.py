"""Unsupervised operating-state detection: Harbour, Transit or DP, found from the ship's own motion.

No land or distance rules. Each 1-minute window is described by five numbers from the ship logs, and a
k-means groups similar windows into three clusters. The reference timeline (the phases
from data.phases(): quay = Harbour, station = DP, transit) is used only afterwards, to score the result.
For comparison, a supervised random forest (forest.py) learns the reference timeline from the same five features.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tmr12 import data, eda

WINDOW = '1min'      # one sample = one minute of data
SMOOTH_MIN = 5       # a state lasts minutes: take the most common cluster over this many windows
N_STATES = 3
FEATURES = ['speed_kn', 'roll_std', 'pitch_std', 'heave_hs_m', 'turn_deg']
STATES = ['Harbour', 'DP', 'Transit']
PHASE_STATE = {'Quay': 'Harbour', 'Station': 'DP', 'Transit': 'Transit'}   # data.phases() kind -> state name
COLOURS = eda.MODE_COLOURS   # same mode colours as the EDA and vis figures


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
    """Every day's 10 s rows with their reference state, cut into 1-minute windows."""
    rows = []
    for day in eda.available_days():   # days in data/processed/ (in git) or data/raw/ship/
        s = data.load_day(day)
        rows.append(eda.add_phase_column(s, data.phases(s)).assign(day=day))
    return windows(pd.concat(rows, ignore_index=True))


def prepare(w):
    """Log-transform (all five are skewed, with long tails) and standardise (k-means compares distances)."""
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
    """Cluster the windows. Adds: state (raw), state_smooth."""
    z = prepare(w)
    labels = KMeans(N_STATES, n_init=10, random_state=seed).fit(z).labels_
    names = name_clusters(w, labels)
    out = w.assign(state=pd.Series(labels).map(names).values)
    out['state_smooth'] = smooth(out.state, out.day)
    return out


VERSIONS = {'Clustering': 'state', 'Clustering, smoothed': 'state_smooth',
            'Random forest': 'forest', 'Random forest, smoothed': 'forest_smooth'}


def check(out):
    """Score the clustering (and the random forest) against the reference timeline.
    scores: per model version, accuracy (share of minutes right) and adjusted Rand index (1 = identical).
    per_state: for the smoothed clustering, recall (share of the reference minutes the model found), precision (share
    of the model's minutes that are right), F1 (the two combined) and the number of reference minutes.
    table: reference state (rows) vs cluster (columns), in minutes."""
    known = out[out.correct != 'Unknown']
    scores = pd.DataFrame({name: {'accuracy': accuracy_score(known.correct, known[col]),
                                  'ARI': adjusted_rand_score(known.correct, known[col])}
                           for name, col in VERSIONS.items() if col in known}).T
    p, r, f, n = precision_recall_fscore_support(known.correct, known.state_smooth, labels=STATES, zero_division=0)
    per_state = pd.DataFrame({'recall': r, 'precision': p, 'F1': f, 'minutes': n}, index=STATES)
    table = pd.crosstab(known.correct, known.state_smooth, rownames=['Reference timeline'], colnames=['Cluster'])
    return scores, per_state, table.reindex(index=STATES, columns=STATES, fill_value=0)


# ---------- figures ----------

def fig_timeline(out):
    """Per day: colour strips over local time, the reference timeline on top, then the random forest (if fitted)
    and the clustering at the bottom."""
    rows = [(c, label) for c, label in [('correct', 'Reference timeline'), ('forest_smooth', 'Random forest'),
                                        ('state_smooth', 'Clustering')] if c in out][::-1]   # bottom row first
    days = list(out.day.unique())
    fig, axes = plt.subplots(len(days), 1, figsize=(12, 0.6 * len(rows) * len(days) + 0.4), squeeze=False)
    for ax, day in zip(axes[:, 0], days):
        d = out[out.day == day]
        t = d.time.dt.tz_convert(data.TZ).dt.tz_localize(None)
        for y, (col, _) in enumerate(rows):
            ax.scatter(t, [y] * len(d), c=d[col].map(COLOURS).fillna('white'), marker='|', s=400, linewidths=2)
        ax.set_yticks(range(len(rows)), [label for _, label in rows])
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.set_title(day, loc='left', fontsize=10)
        ax.grid(False)
    handles = [plt.Line2D([], [], color=c, lw=6, label=s) for s, c in COLOURS.items()]
    axes[0, 0].legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1.15), ncol=3, fontsize=8, frameon=False)
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
        ax.set_title(eda.FEATURE_LABELS[f], fontsize=10)
    fig.suptitle('Inputs per cluster', x=0.01, ha='left')
    fig.tight_layout()
    return fig

