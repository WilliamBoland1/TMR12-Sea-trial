"""Exploratory data analysis (EDA) of the RV Gunnerus DP trials, 30-31 Oct 2023.

Focus: how the ship behaves while it keeps position on DP at the stations, and how that
depends on the waves and wind measured by the nearest E39 buoy.

Every plot function takes already-loaded data, saves one PNG in reports/eda/ and returns its path.
The heavy lifting (loading, phases, heading blocks) is reused from tmr12/data.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # write PNG files, no window
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from tmr12 import data

OUT = data.ROOT / 'reports' / 'eda'

# Colours (validated categorical palette, fixed order) and a neutral grey for "not in focus"
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
PHASE_COLOURS = {'Quay': GREY, 'Transit': BLUE, 'Station': ORANGE}
BUOY_COLOURS = {'D': BLUE, 'A': ORANGE}

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8a8984', 'axes.titlesize': 10, 'axes.titleweight': 'bold'})

M_PER_DEG_LAT = 111_320   # metres per degree latitude (good enough for a few hundred metres)
MAX_BUOY_KM = 3           # a buoy further away than this does not describe the waves at the ship


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def phase_kind(phase):
    """'Station 2' -> 'Station' (so all stations share one colour)."""
    return phase.split()[0]


def add_phase_column(s, p):
    """Label every 10 s row with the phase it belongs to (Quay / Transit / Station n)."""
    s = s.copy()
    s['phase'] = ''
    for r in p.itertuples():
        s.loc[(s.time >= r.start) & (s.time < r.end), 'phase'] = r.phase
    s['kind'] = s.phase.map(phase_kind)
    s['heave_hs_m'] = 4 * s.height_std / 100   # heave "4 sigma" per 10 s file, like a wave Hs
    return s


def local_xy(lat, lon, lat0, lon0):
    """North / east offset [m] from a reference point (flat-earth, fine for < 1 km)."""
    north = (lat - lat0) * M_PER_DEG_LAT
    east = (lon - lon0) * M_PER_DEG_LAT * np.cos(np.radians(lat0))
    return north, east


# ---------- tables ----------

def nearest_buoy(lat, lon, b):
    """Short name ('A', 'D', ...) of the buoy closest to a position, and the distance [km]."""
    d = data.buoy_distances(lat, lon, b)
    return d.index[0].split()[1], d.iloc[0]


def dp_blocks(s, p, b):
    """One row per steady-heading block at a DP station: ship motion, footprint, waves and wind.

    The heading blocks come from data.heading_blocks, using the buoy nearest to each station.
    Added here: footprint size (from the 10 s mean positions) and wind from the buoy.
    Stations further than MAX_BUOY_KM from every buoy keep their ship values, but get no waves/wind."""
    rows = []
    for st in p[p.phase.str.startswith('Station')].itertuples():
        short, km = nearest_buoy(st.lat, st.lon, b)
        buoy = b[b.short == short].buoy.iloc[0]
        hb = data.heading_blocks(s, p, b, buoy)
        if hb.empty:
            continue
        for blk in hb[hb.station == st.phase].itertuples():
            g = s[(s.time >= blk.start) & (s.time < blk.end)]
            north, east = local_xy(g.lat, g.lon, g.lat.mean(), g.lon.mean())
            radius = np.hypot(north, east)
            w = b[(b.buoy == buoy) & (b.time > blk.start) & (b.time <= blk.end + pd.Timedelta('10min'))]
            wind_from = data.circmean(w.WindDirection) if len(w) else np.nan
            row = blk._asdict()
            if km > MAX_BUOY_KM:   # too far away: drop the buoy values for this block
                short, w, wind_from = 'none', w.iloc[0:0], np.nan
                row.update({k: np.nan for k in ['Hm0', 'Tp', 'Mdir', 'ThTp', 'rel_Mdir', 'rel_ThTp']})
            rows.append({**row, 'buoy': short, 'buoy_km': km,
                         'footprint_r95_m': np.percentile(radius, 95),   # 95 % of positions inside this circle
                         'footprint_std_m': np.sqrt(north.var() + east.var()),
                         'heading_std_deg': np.degrees(np.std(np.radians(data.angle_diff(g.heading, blk.heading)))),
                         'wind_ms': w.WindSpeed.mean(), 'wind_from': wind_from,
                         'rel_wind': data.rel_heading(wind_from, blk.heading)})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop(columns=['Index'])
    out['abs_rel_wave'] = out.rel_Mdir.abs()   # 0 = head seas, 90 = beam, 180 = following
    out['abs_rel_wind'] = out.rel_wind.abs()
    return out


def summary_stats(s):
    """Univariate summary (count, mean, std, quartiles) of the main ship signals, per phase type."""
    cols = ['speed_kn', 'roll_std', 'pitch_std', 'heave_hs_m', 'roll_mean', 'pitch_mean']
    return s.groupby('kind')[cols].describe().T.round(3)


# ---------- figures ----------

def fig_completeness(days):
    """Data observability: how many 10 s files arrived per hour (expected 360) and where the gaps are."""
    fig, axes = plt.subplots(len(days), 1, figsize=(9, 2.2 * len(days)), sharey=True, squeeze=False)
    for ax, (day, s) in zip(axes[:, 0], days.items()):
        per_hour = s.set_index('time').n.resample('1h').count()   # labelled by the start of the hour
        ax.bar(per_hour.index, per_hour.values, width=0.9 / 24, align='edge', color=BLUE)
        ax.axhline(360, color=GREY, lw=1, ls='--')
        gaps = s[s.time.diff().dt.total_seconds() > 11]
        for t in gaps.time:
            ax.axvline(t, color=ORANGE, lw=0.8)
        ax.set_title(f'{day}: {len(s)} files, {len(gaps)} gaps longer than 11 s (orange lines)', loc='left')
        ax.set_ylabel('files per hour')
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M', tz=data.TZ))
    axes[-1, 0].set_xlabel('local time (CET)')
    fig.tight_layout()
    return save(fig, '1_data_completeness.png')


def fig_tracks(days, b):
    """Where the ship went each day, coloured by phase, with the buoy positions."""
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 4.2), squeeze=False)
    pos = data.buoy_positions(b)
    for ax, (day, s) in zip(axes[0], days.items()):
        for kind, colour in PHASE_COLOURS.items():
            g = s[s.kind == kind]
            ax.scatter(g.lon, g.lat, s=10 if kind == 'Station' else 3, color=colour, label=kind)
        for name, r in pos.iterrows():
            short = name.split()[1]
            ax.scatter(r.longitude, r.latitude, marker='^', s=70, facecolor='none', edgecolor='black', zorder=3)
            ax.annotate(f'buoy {short}', (r.longitude, r.latitude), xytext=(5, 5), textcoords='offset points')
        ax.set_aspect(1 / np.cos(np.radians(62.45)))   # so 1 km north looks as long as 1 km east
        ax.set_title(day, loc='left')
        ax.set_xlabel('longitude [°E]')
        ax.set_ylabel('latitude [°N]')
    axes[0, 0].legend(markerscale=2, frameon=False, loc='upper left')
    fig.tight_layout()
    return save(fig, '2_tracks.png')


def fig_day_timeseries(day, s, p):
    """One day at a glance: speed, heading, roll and heave over time, stations shaded."""
    signals = [('speed_kn', 'speed [kn]'), ('heading', 'heading [°]'),
               ('roll_std', 'roll σ [°]'), ('heave_hs_m', 'heave 4σ [m]')]
    fig, axes = plt.subplots(len(signals), 1, figsize=(9, 7), sharex=True)
    for ax, (col, label) in zip(axes, signals):
        ax.plot(s.time, s[col], lw=0.6, color=BLUE)
        ax.set_ylabel(label)
        for r in p[p.phase.str.startswith('Station')].itertuples():
            ax.axvspan(r.start, r.end, color=ORANGE, alpha=0.12, lw=0)
    for r in p[p.phase.str.startswith('Station')].itertuples():
        axes[0].annotate(r.phase.replace('Station ', 'S'), (r.start, axes[0].get_ylim()[1]), va='top', fontsize=8)
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M', tz=data.TZ))
    axes[-1].set_xlabel('local time (CET)')
    fig.tight_layout()
    return save(fig, f'3_timeseries_{day}.png')


def fig_univariate(s_all):
    """Distribution of the ship motion per phase type: histograms (top) and box plots (bottom)."""
    cols = [('roll_std', 'roll σ per 10 s [°]'), ('pitch_std', 'pitch σ per 10 s [°]'),
            ('heave_hs_m', 'heave 4σ per 10 s [m]')]
    fig, axes = plt.subplots(2, len(cols), figsize=(11, 6))
    for j, (col, label) in enumerate(cols):
        hi = s_all[col].quantile(0.995)   # cut the 0.5 % largest so the bulk is visible
        bins = np.linspace(0, hi, 40)
        for kind, colour in PHASE_COLOURS.items():
            v = s_all.loc[s_all.kind == kind, col]
            axes[0, j].hist(v.clip(upper=hi), bins=bins, density=True, histtype='step', lw=1.5,
                            color=colour, label=f'{kind} (n={len(v)})')
        axes[0, j].set_xlabel(label)
        kinds = list(PHASE_COLOURS)
        bp = axes[1, j].boxplot([s_all.loc[s_all.kind == k, col] for k in kinds],
                                showfliers=False, patch_artist=True, widths=0.5,
                                medianprops=dict(color='black'))
        axes[1, j].set_xticks(range(1, len(kinds) + 1), kinds)
        for patch, k in zip(bp['boxes'], kinds):
            patch.set(facecolor=PHASE_COLOURS[k], alpha=0.35, edgecolor=PHASE_COLOURS[k])
        axes[1, j].set_ylabel(label)
    axes[0, 0].set_ylabel('density')
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    return save(fig, '4_univariate_by_phase.png')


def fig_buoys(b, phases_by_day):
    """Waves and wind at buoys D and A during the two trial days, ship stations shaded."""
    signals = [('Hm0', 'Hm0 [m]'), ('Tp', 'Tp [s]'), ('Mdir', 'wave dir. from [°]'),
               ('WindSpeed', 'wind [m/s]'), ('WindDirection', 'wind dir. from [°]')]
    fig, axes = plt.subplots(len(signals), len(phases_by_day), figsize=(11, 9), sharex='col', squeeze=False)
    for j, (day, p) in enumerate(phases_by_day.items()):
        lo, hi = p.start.min() - pd.Timedelta('1h'), p.end.max() + pd.Timedelta('1h')
        for i, (col, label) in enumerate(signals):
            ax = axes[i, j]
            for short, colour in BUOY_COLOURS.items():
                g = b[(b.short == short) & (b.time >= lo) & (b.time <= hi)]
                y = g[col]
                style = dict(lw=1.2)
                if 'dir' in label:   # directions: dots, and keep values near north together (e.g. 355 and 5)
                    y = (y - 180) % 360 + 180 if col == 'Mdir' else y
                    style = dict(marker='.', ls='none', ms=3)
                ax.plot(g.time, y, color=colour, label=f'buoy {short}', **style)
            for r in p[p.phase.str.startswith('Station')].itertuples():
                ax.axvspan(r.start, r.end, color=GREY, alpha=0.15, lw=0)
            if j == 0:
                ax.set_ylabel(label)
        axes[0, j].set_title(day, loc='left')
        axes[2, j].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v % 360:.0f}'))
        axes[-1, j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M', tz=data.TZ))
        axes[-1, j].set_xlabel('local time (CET)')
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    return save(fig, '5_buoy_conditions.png')


def fig_footprints(s_all, blocks):
    """DP footprint: position (10 s means) inside the steady-heading blocks of each station, around the
    station mean, coloured by the relative wave heading of the block (grey = no buoy within MAX_BUOY_KM).
    Only block data is used, so the approach to the station and the turns between headings are left out.
    Circle = radius that holds 95 % of the positions (R95)."""
    stations = blocks[['day', 'station']].drop_duplicates().values.tolist()
    ncols = min(3, len(stations))
    nrows = int(np.ceil(len(stations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat[len(stations):]:
        ax.set_visible(False)   # empty panels in the last row
    cmap = plt.get_cmap('twilight')   # cyclic: -180° and +180° get the same colour
    for ax, (day, station) in zip(axes.flat, stations):
        st_blocks = blocks[(blocks.day == day) & (blocks.station == station)]
        parts = []
        for blk in st_blocks.itertuples():
            g = s_all[(s_all.day == day) & (s_all.time >= blk.start) & (s_all.time < blk.end)]
            parts.append(g.assign(rel=blk.rel_Mdir))
        g_all = pd.concat(parts)
        north, east = local_xy(g_all.lat, g_all.lon, g_all.lat.mean(), g_all.lon.mean())
        colours = [GREY if np.isnan(r) else cmap((r + 180) / 360) for r in g_all.rel]
        ax.scatter(east, north, s=3, c=colours)
        r95 = np.percentile(np.hypot(north, east), 95)
        ax.add_patch(plt.Circle((0, 0), r95, fill=False, color=GREY, ls='--'))
        lim = max(r95 * 1.3, 3)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal', xlabel='east [m]')
        short = f"{pd.Timestamp(day).day} {pd.Timestamp(day):%b} {station.replace('Station ', 'S')}"
        ax.set_title(f'{short}: R95 = {r95:.1f} m\n({len(st_blocks)} heading blocks)', loc='left', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('north [m]')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-180, 180))
    fig.colorbar(sm, ax=axes, shrink=0.6, label='relative wave heading [°]\n0 = head seas, ±90 = beam')
    return save(fig, '6_dp_footprint.png')


def fig_vs_heading(blocks):
    """Bivariate: footprint, roll and pitch per heading block vs the relative wave heading."""
    cols = [('footprint_r95_m', 'footprint R95 [m]'), ('roll_std', 'roll σ [°]'), ('pitch_std', 'pitch σ [°]')]
    blocks = blocks[blocks.buoy != 'none']
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 3.4))
    for ax, (col, label) in zip(axes, cols):
        for short, colour in BUOY_COLOURS.items():
            g = blocks[blocks.buoy == short]
            ax.scatter(g.abs_rel_wave, g[col], s=g.minutes * 1.5, color=colour, alpha=0.8,
                       edgecolor='white', lw=0.5, label=f'at buoy {short}')
        ax.set(xlabel='|relative wave heading| [°]  (0 = head, 90 = beam)', ylabel=label, xlim=(-5, 185))
        ax.set_xticks([0, 45, 90, 135, 180])
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return save(fig, '7_motion_vs_wave_heading.png')


def fig_correlation(blocks):
    """Multivariate: correlation matrix of the heading-block variables."""
    cols = {'Hm0': 'Hm0', 'Tp': 'Tp', 'wind_ms': 'wind speed', 'abs_rel_wave': '|rel. wave dir.|',
            'abs_rel_wind': '|rel. wind dir.|', 'roll_std': 'roll σ', 'pitch_std': 'pitch σ',
            'heave_hs_m': 'heave 4σ', 'footprint_r95_m': 'footprint R95', 'heading_std_deg': 'heading σ',
            'dist_km': 'dist. to buoy'}
    blocks = blocks[blocks.buoy != 'none']
    c = blocks[list(cols)].rename(columns=cols).corr(method='spearman')   # rank correlation: robust, few points
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(c)), c.columns)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f'{c.iat[i, j]:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(c.iat[i, j]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, label='Spearman correlation')
    fig.tight_layout()
    return save(fig, '8_correlation.png')
