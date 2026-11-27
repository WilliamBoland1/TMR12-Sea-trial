"""Exploratory data analysis (EDA) of the RV Gunnerus DP trials, 30 Oct - 3 Nov 2023.

Focus: how the three operating modes (Harbour, DP, Transit) differ in the data, as input to the
machine-learning model that finds the mode from the ship's own motion (tmr12/cluster.py).
The features per minute are taken from cluster.windows, so the EDA looks at exactly what the model sees.
The E39 buoys are used as context: the sea state explains when DP looks like lying in harbour.

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
PHASE_COLOURS = {'Quay': GREY, 'Station': BLUE, 'Transit': ORANGE}
MODE = {'Quay': 'Harbour', 'Station': 'DP', 'Transit': 'Transit'}   # data.phases() kind -> operating mode
MODE_COLOURS = {'Harbour': GREY, 'DP': BLUE, 'Transit': ORANGE}
FEATURE_LABELS = {'speed_kn': 'speed [kn]', 'roll_std': 'roll σ [°]', 'pitch_std': 'pitch σ [°]',
                  'heave_hs_m': 'heave 4σ [m]', 'turn_deg': 'heading change [°/min]'}   # cluster.FEATURES
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
    """Where the ship went each day, coloured by operating mode, with the buoy positions."""
    ncols = min(2, len(days))
    nrows = int(np.ceil(len(days) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flat[len(days):]:
        ax.set_visible(False)
    pos = data.buoy_positions(b)
    for ax, (day, s) in zip(axes.flat, days.items()):
        for kind, colour in PHASE_COLOURS.items():
            g = s[s.kind == kind]
            ax.scatter(g.lon, g.lat, s=10 if kind == 'Station' else 3, color=colour, label=MODE[kind])
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
            ax.axvspan(r.start, r.end, color=BLUE, alpha=0.12, lw=0)
    for r in p[p.phase.str.startswith('Station')].itertuples():
        axes[0].annotate(r.phase.replace('Station ', 'S'), (r.start, axes[0].get_ylim()[1]), va='top', fontsize=8)
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M', tz=data.TZ))
    axes[-1].set_xlabel('local time (CET)')
    fig.tight_layout()
    return save(fig, f'3_timeseries_{day}.png')

# ---------- operating mode (Harbour / DP / Transit) ----------

def mode_balance(w):
    """Minutes per operating mode and day: how many examples of each class the model gets."""
    t = pd.crosstab(w.day, w.correct, margins=True, margins_name='all')
    return t.reindex(columns=[*MODE_COLOURS, 'all'], fill_value=0)


def positive(x):
    """Log axes cannot show 0 (e.g. no heading change in a minute): put zeros at 0.001."""
    return x.clip(lower=1e-3)


def fig_features_by_mode(w):
    """The five model features per operating mode (1-minute windows, all days): histograms and box plots,
    on log axes since the values span several orders of magnitude."""
    fig, axes = plt.subplots(2, len(FEATURE_LABELS), figsize=(15, 5.5))
    for j, (col, label) in enumerate(FEATURE_LABELS.items()):
        v = positive(w[col])
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 40)
        for mode, colour in MODE_COLOURS.items():
            x = v[w.correct == mode]
            axes[0, j].hist(x, bins=bins, histtype='step', lw=1.5, color=colour, label=f'{mode} (n={len(x)})')
        axes[0, j].set(xscale='log', xlabel=label)
        bp = axes[1, j].boxplot([v[w.correct == m] for m in MODE_COLOURS], showfliers=False, patch_artist=True,
                                widths=0.5, medianprops=dict(color='black'))
        for patch, colour in zip(bp['boxes'], MODE_COLOURS.values()):
            patch.set(facecolor=colour, alpha=0.35, edgecolor=colour)
        axes[1, j].set_xticks(range(1, len(MODE_COLOURS) + 1), list(MODE_COLOURS))
        axes[1, j].set(yscale='log', ylabel=label)
    axes[0, 0].set_ylabel('minutes')
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save(fig, '4_features_by_mode.png')


def fig_feature_pairs(w, n=2000):
    """Pair plot (seaborn) of the five features on log axes, coloured by mode.
    A random sample of n minutes keeps it readable; each mode keeps its share."""
    import seaborn as sns
    d = w.sample(min(n, len(w)), random_state=0)
    x = np.log10(positive(d[list(FEATURE_LABELS)])).rename(columns=lambda c: 'log10 ' + FEATURE_LABELS[c])
    g = sns.pairplot(x.assign(mode=d.correct.values), hue='mode', palette=MODE_COLOURS, hue_order=list(MODE_COLOURS),
                     corner=True, diag_kind='hist', height=1.8, diag_kws={'element': 'step', 'fill': False},
                     plot_kws={'s': 8, 'alpha': 0.5, 'edgecolor': 'none'})
    path = OUT / '5_feature_pairs.png'
    OUT.mkdir(parents=True, exist_ok=True)
    g.savefig(path, dpi=150)
    plt.close(g.figure)
    return path


def fig_feature_correlation(w):
    """Spearman correlation between the five features (all minutes): which features carry the same information."""
    c = w[list(FEATURE_LABELS)].rename(columns=FEATURE_LABELS).corr(method='spearman')
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(c)), c.columns)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f'{c.iat[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(c.iat[i, j]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, label='Spearman correlation')
    fig.tight_layout()
    return save(fig, '6_feature_correlation.png')


MOTION = ['roll_std', 'pitch_std', 'heave_hs_m']   # the wave-induced motion features


def harbour_like(w, q=0.95):
    """True for minutes where roll, pitch and heave are all at or below the q-quantile of the harbour minutes,
    i.e. the ship moves as little as when it lies at the quay."""
    hi = w.loc[w.correct == 'Harbour', MOTION].quantile(q)
    return (w[MOTION] <= hi).all(axis=1)


def station_minutes(w, phases_by_day):
    """(day, station row of data.phases, mask of its minutes in w) for every DP station."""
    for day, p in phases_by_day.items():
        for st in p[p.phase.str.startswith('Station')].itertuples():
            yield day, st, (w.day == day) & (w.time >= st.start) & (w.time < st.end)


def station_table(w, phases_by_day, b):
    """One row per DP station: minutes, median roll σ and heave 4σ, the share of minutes that look like
    harbour (harbour_like), and the mean Hm0 of the nearest buoy (only if it is within MAX_BUOY_KM)."""
    like = harbour_like(w)
    rows = []
    for day, st, m in station_minutes(w, phases_by_day):
        short, km = nearest_buoy(st.lat, st.lon, b)
        bb = b[(b.short == short) & (b.time > st.start) & (b.time <= st.end + pd.Timedelta('10min'))]
        rows.append({'day': day, 'station': st.phase, 'minutes': int(m.sum()),
                     'roll_std': w.loc[m, 'roll_std'].median(), 'heave_hs_m': w.loc[m, 'heave_hs_m'].median(),
                     'harbour_like_pct': 100 * like[m].mean(),
                     'buoy': short if km <= MAX_BUOY_KM else 'none', 'buoy_km': km,
                     'Hm0': bb.Hm0.mean() if km <= MAX_BUOY_KM else np.nan})
    return pd.DataFrame(rows)


def fig_dp_stations(w, phases_by_day, b):
    """Why DP sometimes looks like harbour: roll σ and heave 4σ per minute at each DP station, next to the
    harbour range (grey band: 5-95 % of the harbour minutes). Labels: nearest buoy and its Hm0.
    The number above each box is the share of minutes whose roll, pitch and heave all look like harbour."""
    tab = station_table(w, phases_by_day, b)
    harbour = w[w.correct == 'Harbour']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ['roll_std', 'heave_hs_m']):
        lo, hi = harbour[col].quantile([0.05, 0.95])
        ax.axhspan(lo, hi, color=GREY, alpha=0.25, lw=0, label='harbour, 5-95 %')
        values = [positive(w.loc[m, col]) for _, _, m in station_minutes(w, phases_by_day)]
        bp = ax.boxplot(values, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black'))
        for patch in bp['boxes']:
            patch.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
        labels = [f"{pd.Timestamp(r.day):%-d %b} {r.station.replace('Station ', 'S')}\n"
                  + (f'{r.buoy}, Hm0 {r.Hm0:.2f} m' if r.buoy != 'none' else 'no buoy')
                  for r in tab.itertuples()]
        ax.set_xticks(range(1, len(tab) + 1), labels, rotation=45, ha='right', fontsize=7)
        ax.set(yscale='log', ylabel=FEATURE_LABELS[col])
        for i, r in enumerate(tab.itertuples(), start=1):
            ax.annotate(f'{r.harbour_like_pct:.0f}%', (i, 1), xycoords=('data', 'axes fraction'),
                        ha='center', va='bottom', fontsize=7)
    axes[0].legend(frameon=False, loc='lower left')
    fig.tight_layout()
    return save(fig, '7_dp_stations_vs_harbour.png')


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
