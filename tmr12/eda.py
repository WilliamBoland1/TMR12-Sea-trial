"""Exploratory data analysis (EDA) of the RV Gunnerus DP trials, 30 Oct - 3 Nov 2023.

Focus: how the ship behaves while it keeps position on DP at the stations, and how that
depends on the waves and wind measured by the nearest E39 buoy.

Every plot function takes already-loaded data, saves one PNG in reports/eda/ and returns its path.
The heavy lifting (loading, phases, heading blocks) is reused from tmr12/data.py.

Two kinds of ship data are used:
  - the 10 s summaries in data/processed/ (in git, all days): position, heading, speed, and n/mean/std
    of roll, pitch and height per 10 s file;
  - the raw 100 Hz files (not in git, only where available): needed for spectra and filtered heave.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # write PNG files, no window
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import signal

from tmr12 import data

OUT = data.ROOT / 'reports' / 'eda'

# Colours (validated categorical palette, fixed order) and a neutral grey for "not in focus"
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
PHASE_COLOURS = {'Quay': GREY, 'Transit': BLUE, 'Station': ORANGE}
BUOY_COLOURS = {'D': BLUE, 'A': ORANGE}
DAY_COLOURS = {'2023-10-30': BLUE, '2023-10-31': ORANGE, '2023-11-02': AQUA, '2023-11-03': GREY}

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8a8984', 'axes.titlesize': 10, 'axes.titleweight': 'bold'})

M_PER_DEG_LAT = 111_320   # metres per degree latitude (good enough for a few hundred metres)
MAX_BUOY_KM = 3           # a buoy further away than this does not describe the waves at the ship
FS = 100                  # raw sample rate [Hz]
HIGHPASS_S = 30           # keep motions faster than this period [s]: waves, not tide or slow GNSS drift


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def available_days():
    """{'YYYY-MM-DD': raw day folder or None} for every day with a 10 s summary or raw files.
    Days that are only in data/processed/ (the parquet files in git) get None: no raw 100 Hz data."""
    raw = data.days() if data.RAW_SHIP.exists() else {}
    processed = {f.stem.replace('ship_10s_', '') for f in data.PROCESSED.glob('ship_10s_*.parquet')}
    return {day: raw.get(day) for day in sorted(set(raw) | processed)}


def ship_checks(s, day_dir=None):
    """data.ship_checks for one day. Without the raw files the number of CSV columns cannot be checked."""
    if day_dir is not None:
        return data.ship_checks(s, day_dir)
    dt = s.time.diff().dt.total_seconds()
    expected = round((s.time.iloc[-1] - s.time.iloc[0]).total_seconds() / 10) + 1
    return {'files': len(s), 'expected': expected, 'complete_pct': 100 * len(s) / expected,
            'min_samples_per_file': int(s.n.min()), 'max_step_s': dt.max(), 'gaps_over_11s': int((dt > 11).sum()),
            'nan_values': int(s.isna().sum().sum()), 'out_of_range': data.out_of_range(s, data.SHIP_LIMITS),
            'csv_columns': np.nan}


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


def minute_stats(s):
    """1-minute windows (6 files = 6000 samples) for the overview figures.
    Roll and pitch: std over the whole minute (data.pooled), so each value covers several wave periods,
    unlike one 10 s file that holds only one or two. Heave: see the comment below; the exact wave
    heave per block comes from the raw data (raw_block_motion)."""
    rows = []
    for t, g in s.groupby(s.time.dt.floor('1min')):
        if len(g) < 6:
            continue
        rows.append({'time': t, 'speed_kn': g.speed_kn.mean(), 'heading': data.circmean(g.heading),
                     'roll_std': data.pooled(g, 'roll'), 'pitch_std': data.pooled(g, 'pitch'),
                     # heave: only the variation inside each 10 s file, so slow GNSS drift is left out
                     'heave_hs_m': 4 * data.within(g, 'height') / 100, 'phase': g.phase.mode().iloc[0]})
    m = pd.DataFrame(rows)
    m['kind'] = m.phase.map(phase_kind)
    return m


# ---------- raw 100 Hz data ----------

def raw_files(day_dir):
    """All raw 10 s files of one day, indexed by their start time (file names are in local time)."""
    files = sorted(Path(day_dir).glob('*/*.csv'))
    t = pd.to_datetime([f.stem for f in files], format='%Y%m%d%H%M%S')
    t = t.tz_localize(data.TZ, ambiguous='NaT', nonexistent='NaT')   # the DST night gives NaT
    return pd.Series(files, index=t)[t.notna()]


def raw_block(files, start, end):
    """Glue the raw files of one block into one continuous 100 Hz time series."""
    sel = files[(files.index >= start - pd.Timedelta('15s')) & (files.index < end)]
    d = pd.concat([pd.read_csv(f, sep=';', header=None, names=data.COLS) for f in sel])
    d = d[(d.t >= start.timestamp()) & (d.t < end.timestamp())]
    return d.sort_values('t').reset_index(drop=True)


def wave_part(x):
    """Remove the mean, trend and everything slower than HIGHPASS_S (tide, GNSS drift, slow list)."""
    sos = signal.butter(4, 1 / HIGHPASS_S, 'highpass', fs=FS, output='sos')
    return signal.sosfiltfilt(sos, signal.detrend(np.asarray(x, dtype=float)))


def spectrum(x):
    """Power spectral density (Welch, 120 s segments) and the peak period between 2 and 25 s."""
    f, psd = signal.welch(x, fs=FS, nperseg=120 * FS)
    ok = (f > 1 / 25) & (f < 1 / 2)
    return f, psd, 1 / f[ok][psd[ok].argmax()]


def raw_block_motion(d):
    """Wave-induced motion of one block from the raw 100 Hz data: std, heave 4σ and peak periods."""
    out, spectra = {}, {}
    for col, name in [('roll', 'roll'), ('pitch', 'pitch'), ('height_cm', 'heave')]:
        x = wave_part(d[col] / 100 if col == 'height_cm' else d[col])   # height: cm -> m
        f, psd, tp = spectrum(x)
        spectra[name] = (f, psd)
        out[f'{name}_tp_s'] = tp
        if name == 'heave':
            out['heave_hs_wave'] = 4 * x.std()
        else:
            out[f'{name}_std_wave'] = x.std()
    return out, spectra


# ---------- tables ----------

def nearest_buoy(lat, lon, b):
    """Short name ('A', 'D', ...) of the buoy closest to a position, and the distance [km]."""
    d = data.buoy_distances(lat, lon, b)
    return d.index[0].split()[1], d.iloc[0]


def dp_blocks(s, p, b, files=None, spectra=None):
    """One row per steady-heading block at a DP station: ship motion, footprint, waves and wind.

    The heading blocks come from data.heading_blocks, using the buoy nearest to each station.
    Added here: footprint size (from the 10 s mean positions) and wind from the buoy.
    Stations further than MAX_BUOY_KM from every buoy keep their ship values, but get no waves/wind.
    With the raw files (files = raw_files(day dir)) the wave-induced motion is computed from the
    100 Hz data of the whole block; spectra (a dict) then collects the spectra per block.
    Without raw files the motion comes from the 10 s summaries: roll and pitch σ pooled over the block
    (exact), heave 4σ from the variation inside each 10 s file (within 5 % of the raw value at buoy D
    on 30-31 Oct, see the EDA section). motion_source says which one was used."""
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
            if files is not None:
                motion, spec = raw_block_motion(raw_block(files, blk.start, blk.end))
                row.update(motion, motion_source='raw 100 Hz')
                if spectra is not None:
                    spectra[(blk.start, blk.end)] = spec
            else:
                row.update(roll_std_wave=blk.roll_std, pitch_std_wave=blk.pitch_std,
                           heave_hs_wave=blk.heave_hs_m, motion_source='10 s files')
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


def duplicates(days, b):
    """Duplicate records (same time stamp twice) in the ship summaries and the buoy data."""
    rows = [{'dataset': f'ship {day}', 'records': len(s), 'duplicates': int(s.t.duplicated().sum())}
            for day, s in days.items()]
    for name, g in b.groupby('buoy'):
        rows.append({'dataset': name, 'records': len(g), 'duplicates': int(g.time.duplicated().sum())})
    return pd.DataFrame(rows)


def iqr_outliers(m, cols=('roll_std', 'pitch_std', 'heave_hs_m')):
    """Tukey's rule: a value is an outlier if it lies more than 1.5 IQR outside the quartiles.
    Applied per phase type, since quay, transit and station have very different normal ranges."""
    rows = []
    for kind, g in m.groupby('kind'):
        for col in cols:
            q1, q3 = g[col].quantile([0.25, 0.75])
            hi = q3 + 1.5 * (q3 - q1)
            n = int((g[col] > hi).sum() + (g[col] < q1 - 1.5 * (q3 - q1)).sum())
            rows.append({'phase': kind, 'variable': col, 'upper_fence': round(hi, 3),
                         'outliers': n, 'percent': round(100 * n / len(g), 1)})
    return pd.DataFrame(rows)


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
    ncols = min(2, len(days))
    nrows = int(np.ceil(len(days) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flat[len(days):]:
        ax.set_visible(False)
    pos = data.buoy_positions(b)
    for ax, (day, s) in zip(axes.flat, days.items()):
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
    """One day at a glance: speed, heading, roll and heave over time (1-minute values), stations shaded."""
    s = minute_stats(s)
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
    """Distribution of the ship motion per phase type (1-minute values): histograms and box plots."""
    s_all = pd.concat([minute_stats(g) for _, g in s_all.groupby('day')], ignore_index=True)
    cols = [('roll_std', 'roll σ per minute [°]'), ('pitch_std', 'pitch σ per minute [°]'),
            ('heave_hs_m', 'heave 4σ per minute [m]')]
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
    """Waves and wind at buoys D and A on the trial days that have buoy data, ship stations shaded."""
    signals = [('Hm0', 'Hm0 [m]'), ('Tp', 'Tp [s]'), ('Mdir', 'wave dir. from [°]'),
               ('WindSpeed', 'wind [m/s]'), ('WindDirection', 'wind dir. from [°]')]
    fig, axes = plt.subplots(len(signals), len(phases_by_day), figsize=(4 * len(phases_by_day), 9), sharex='col',
                             squeeze=False)
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
    """Bivariate: footprint, roll, pitch and heave per heading block vs the relative wave heading.
    Colour = day, circle = buoy D, triangle = buoy A. Marker size = block length."""
    cols = [('footprint_r95_m', 'footprint R95 [m]'), ('roll_std_wave', 'roll σ [°]'),
            ('pitch_std_wave', 'pitch σ [°]'), ('heave_hs_wave', 'heave 4σ [m]')]
    blocks = blocks[blocks.buoy != 'none']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, cols):
        for (short, day), g in blocks.groupby(['buoy', 'day']):
            ax.scatter(g.abs_rel_wave, g[col], s=g.minutes * 1.5, color=DAY_COLOURS[day], alpha=0.8,
                       marker='o' if short == 'D' else '^', edgecolor='white', lw=0.5,
                       label=f"{pd.Timestamp(day):%-d %b}, buoy {short}")
        ax.set(xlabel='|relative wave heading| [°] (0 = head, 90 = beam)', ylabel=label, xlim=(-5, 185))
        ax.set_xticks([0, 45, 90, 135, 180])
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return save(fig, '7_motion_vs_wave_heading.png')


def fig_correlation(blocks):
    """Multivariate: correlation matrix of the heading-block variables."""
    cols = {'Hm0': 'Hm0', 'Tp': 'Tp', 'wind_ms': 'wind speed', 'abs_rel_wave': '|rel. wave dir.|',
            'abs_rel_wind': '|rel. wind dir.|', 'roll_std_wave': 'roll σ', 'pitch_std_wave': 'pitch σ',
            'heave_hs_wave': 'heave 4σ', 'heave_tp_s': 'heave peak period', 'footprint_r95_m': 'footprint R95',
            'heading_std_deg': 'heading σ',
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


def fig_spectra(blocks, spectra):
    """Roll, pitch and heave spectra of the heading blocks at buoy D that have raw 100 Hz data,
    coloured by the relative wave heading. Dashed line: mean buoy peak period Tp."""
    blocks = blocks[(blocks.buoy == 'D') & (blocks.motion_source == 'raw 100 Hz')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    cmap, norm = plt.get_cmap('viridis'), plt.Normalize(0, 180)
    for blk in blocks.itertuples():
        spec = spectra[(blk.start, blk.end)]
        for ax, name in zip(axes, ['roll', 'pitch', 'heave']):
            f, psd = spec[name]
            ok = (f > 1 / 25) & (f < 0.5)
            ax.plot(f[ok], psd[ok], lw=1, color=cmap(norm(blk.abs_rel_wave)))
    units = {'roll': '°²/Hz', 'pitch': '°²/Hz', 'heave': 'm²/Hz'}
    for ax, name in zip(axes, ['roll', 'pitch', 'heave']):
        ax.axvline(1 / blocks.Tp.mean(), color=GREY, ls='--', lw=1)
        ax.set(xlabel='frequency [Hz]', ylabel=f'{name} PSD [{units[name]}]')
        ax.set_title(name, loc='left')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes, shrink=0.8, label='|relative wave heading| [°]')
    return save(fig, '9_motion_spectra.png')


def fig_missing(blocks):
    """Missing values in the heading-block table: one row per block, one column per variable.
    Blocks with no buoy within MAX_BUOY_KM have no wave or wind data (missing by design, not at random).
    Blocks without raw 100 Hz files have no peak period (heave_tp_s needs the spectrum)."""
    cols = ['roll_std_wave', 'pitch_std_wave', 'heave_hs_wave', 'heave_tp_s', 'footprint_r95_m', 'Hm0', 'Tp',
            'Mdir', 'wind_ms', 'wind_from']
    miss = blocks[cols].isna().values
    fig, ax = plt.subplots(figsize=(6, max(4.5, 0.11 * len(blocks))))
    ax.imshow(miss, aspect='auto', cmap=matplotlib.colors.ListedColormap(['#e4e3df', ORANGE]), interpolation='none')
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha='right')
    labels = [f"{pd.Timestamp(d).day} {pd.Timestamp(d):%b} {st.replace('Station ', 'S')}"
              for d, st in zip(blocks.day, blocks.station)]
    ax.set_yticks(range(len(blocks)), labels, fontsize=6)
    ax.grid(False)
    ax.set_ylabel('heading block')
    ax.set_title(f'orange = missing ({miss.any(axis=1).sum()} of {len(blocks)} blocks)', loc='left', fontsize=9)
    return save(fig, '10_missing_values.png')
