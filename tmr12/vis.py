"""Data visualisation for the operating-mode problem (Harbour / DP / Transit), RV Gunnerus, 30 Oct - 3 Nov 2023.

The EDA figures (tmr12/eda.py) are for exploring. These four are made to communicate the main points:
  1. Time budget: how many hours the ship spent in each mode per day (what a mode detector is used for).
  2. Decision space: speed against roll per minute, coloured by mode, with the 3 kn transit rule.
  3. DP motion against sea state: roll and pitch at each DP station against the buoy Hm0, with spread bars,
     next to the harbour range: how calm the sea must be before DP looks like harbour.
  4. Parallel coordinates of the five features: the profile of each mode at a glance (course lab method).

Uses the same 1-minute windows and labels as the model (cluster.windows). Figures go to reports/vis/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmr12 import data, eda

OUT = data.ROOT / 'reports' / 'vis'
MODES = list(eda.MODE_COLOURS)   # Harbour, DP, Transit


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def short_day(day):
    t = pd.Timestamp(day)
    return f'{t.day} {t:%b}'   # not '%-d': that only works on Linux/macOS


def fig_time_budget(w):
    """Hours in each mode per day, as stacked horizontal bars with the hours written in each segment."""
    hours = pd.crosstab(w.day, w.correct).reindex(columns=MODES, fill_value=0) / 60
    hours.loc['all days'] = hours.sum() / len(hours)   # average day
    fig, ax = plt.subplots(figsize=(8, 3.2))
    left = np.zeros(len(hours))
    labels = [short_day(d) if d != 'all days' else 'average day' for d in hours.index]
    for mode in MODES:
        ax.barh(labels, hours[mode], left=left, color=eda.MODE_COLOURS[mode], edgecolor='white', lw=2, label=mode)
        for y, (x0, h) in enumerate(zip(left, hours[mode])):
            if h >= 1:   # no label in segments too narrow for text
                ax.text(x0 + h / 2, y, f'{h:.1f} h', ha='center', va='center', fontsize=8,
                        color='white' if mode != 'Harbour' else 'black')
        left += hours[mode].values
    ax.invert_yaxis()
    ax.set(xlabel='hours', xlim=(0, 24), xticks=range(0, 25, 3))
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    return save(fig, 'v1_time_budget.png')


def fig_decision_space(w, phases_by_day):
    """Speed against roll σ per minute (log axes), coloured by mode. Dashed line: the 3 kn rule behind the
    transit label. The DP minutes of the calm station (31 Oct S2) are ringed: they lie in the harbour cloud."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for mode in MODES:   # harbour first, so the smaller classes are drawn on top
        g = w[w.correct == mode]
        ax.scatter(eda.positive(g.speed_kn), eda.positive(g.roll_std), s=6, alpha=0.4, lw=0,
                   color=eda.MODE_COLOURS[mode], label=f'{mode} ({len(g)} min)')
    calm = next(m for day, st, m in eda.station_minutes(w, phases_by_day)
                if day == '2023-10-31' and st.phase == 'Station 2')
    ax.scatter(eda.positive(w.speed_kn[calm]), eda.positive(w.roll_std[calm]), s=14, facecolor='none',
               edgecolor='black', lw=0.6, label='DP at 31 Oct S2 (calm)')
    ax.axvline(data.MOVING_KN, color=eda.GREY, ls='--', lw=1)
    ax.text(data.MOVING_KN * 1.08, 0.03, 'transit rule: 3 kn', transform=ax.get_xaxis_transform(), fontsize=8,
            color=eda.GREY, rotation=90, va='bottom')
    ax.set(xscale='log', yscale='log', xlabel='speed [kn] (median per minute)', ylabel='roll σ [°] (per minute)')
    leg = ax.legend(frameon=False, loc='upper left', fontsize=8)
    for h in leg.legend_handles:
        h.set_alpha(1)
    return save(fig, 'v2_decision_space.png')


def fig_dp_vs_sea_state(w, phases_by_day, b):
    """Roll σ and pitch σ at each DP station against the buoy Hm0. Point: median; vertical bar: 25-75 % of the
    minutes; horizontal bar: lowest to highest 10-min Hm0 during the station. Grey band: 5-95 % of the harbour
    minutes. Dashed line: median at 31 Oct S2, which had no buoy within MAX_BUOY_KM."""
    harbour = w[w.correct == 'Harbour']
    rows = []
    for day, st, m in eda.station_minutes(w, phases_by_day):
        short, km = eda.nearest_buoy(st.lat, st.lon, b)
        hm0 = b[(b.short == short) & (b.time > st.start) & (b.time <= st.end + pd.Timedelta('10min'))].Hm0
        rows.append({'day': day, 'station': st.phase, 'buoy': short if km <= eda.MAX_BUOY_KM else 'none',
                     'hm0': hm0.mean(), 'hm0_lo': hm0.min(), 'hm0_hi': hm0.max(), 'minutes': int(m.sum()),
                     **{f'{c}_{q}': w.loc[m, c].quantile(q / 100) for c in ['roll_std', 'pitch_std'] for q in (25, 50, 75)}})
    st = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label in zip(axes, ['roll_std', 'pitch_std'], ['roll σ [°]', 'pitch σ [°]']):
        lo, hi = harbour[col].quantile([0.05, 0.95])
        ax.axhspan(lo, hi, color=eda.GREY, alpha=0.25, lw=0, label='harbour, 5-95 %')
        for short, colour in eda.BUOY_COLOURS.items():
            g = st[st.buoy == short]
            ax.errorbar(g.hm0, g[f'{col}_50'], xerr=[g.hm0 - g.hm0_lo, g.hm0_hi - g.hm0],
                        yerr=[g[f'{col}_50'] - g[f'{col}_25'], g[f'{col}_75'] - g[f'{col}_50']],
                        fmt='o', ms=6, color=colour, elinewidth=1, capsize=2, label=f'DP station at buoy {short}')
            for i, r in enumerate(g.itertuples()):   # labels alternately above and below, so neighbours do not overlap
                ax.annotate(f"{short_day(r.day)} {r.station.replace('Station ', 'S')}", (r.hm0, getattr(r, f'{col}_50')),
                            xytext=(5, 3 if i % 2 == 0 else -10), textcoords='offset points', fontsize=7)
        calm = st[(st.day == '2023-10-31') & (st.station == 'Station 2')].iloc[0]
        ax.axhline(calm[f'{col}_50'], color='black', ls='--', lw=1, label='31 Oct S2 (no buoy, calm)')
        ax.set(yscale='log', xlim=(0, 0.8), xlabel='buoy $H_{m0}$ [m]', ylabel=label)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    return save(fig, 'v3_dp_vs_sea_state.png')


def fig_parallel(w, n=150):
    """Parallel coordinates (pandas.plotting) of the five features: one line per minute, n random minutes per
    mode, log10 values scaled from 0 (lowest) to 1 (highest)."""
    from pandas.plotting import parallel_coordinates
    d = pd.concat([w[w.correct == m].sample(min(n, (w.correct == m).sum()), random_state=0) for m in MODES])
    x = np.log10(eda.positive(d[list(eda.FEATURE_LABELS)]))
    x = (x - x.min()) / (x.max() - x.min())
    x['mode'] = d.correct.values
    fig, ax = plt.subplots(figsize=(9, 4))
    parallel_coordinates(x, 'mode', ax=ax, color=[eda.MODE_COLOURS[m] for m in MODES], lw=0.8, alpha=0.35)
    ax.set_xticks(range(len(eda.FEATURE_LABELS)), ['speed', 'roll σ', 'pitch σ', 'heave 4σ', 'heading change'])
    ax.set_ylabel('log value, scaled (0 = lowest, 1 = highest)')
    leg = ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    for h in leg.legend_handles:
        h.set_alpha(1)
        h.set_linewidth(2)
    return save(fig, 'v4_parallel_coordinates.png')
