"""Data layer: RV Gunnerus ship logs and E39 buoy wave parameters.

Ship:  data/raw/ship/<folder>/<hh>/<YYYYMMDDhhmmss>.csv  (10 s at 100 Hz, ';', no header, file name = local time)
Buoys: data/raw/buoys/buoy_wave_params_trial.csv        (10-min records, UTC, time = END of the period)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_SHIP = ROOT / 'data' / 'raw' / 'ship'
BUOY_CSV = ROOT / 'data' / 'raw' / 'buoys' / 'buoy_wave_params_trial.csv'
PROCESSED = ROOT / 'data' / 'processed'
TZ = 'Europe/Oslo'

COLS = ['t', 'lat', 'lon',
        'height_cm',            # inferred: GNSS height [cm] (slow tide-like drift + heave)
        'c4',                   # unknown integer
        'vn_cms', 've_cms',     # inferred: north/east velocity [cm/s] (matches heading in transit)
        'vd_cms',               # inferred: down velocity [cm/s] (anti-correlates with d(height)/dt)
        'roll', 'pitch', 'heading',            # [deg]
        'roll_rate', 'pitch_rate', 'yaw_rate']  # inferred: [deg/s]

PLAN = {'2023-10-30': 'Test plan day 1: buoys D and B', '2023-10-31': 'Test plan day 2: buoy A or C',
'2023-11-02': 'Test plan day 4: strong waves, many headings',
        '2023-11-03': 'Test plan day 5: Hustadvika, heading for Trondheim'}

# Tuning knobs (checked against 30 Oct / 31 Oct / 2 Nov / 3 Nov; adjust if new data behaves differently)
MOVING_KN = 3           # 5-min median speed above this = transit
MIN_STOP_MIN = 10       # shorter stops count as transit
STEADY_DEG = 3          # heading change over 30 s below this = steady heading
MIN_BLOCK_MIN = 5       # shortest steady-heading block kept
MERGE_DEG, MERGE_GAP_MIN = 4, 2   # merge neighbouring blocks with ~same heading and a short gap
NOMINAL = [0, 15, -15, 30, -30, 60, -60, 90, -90, 180]   # test-plan headings relative to waves
WAVE = ['Hm0', 'Hm0a', 'Hm0b', 'Tp', 'Tm02', 'Mdir', 'ThTp', 'SprTp']   # buoy wave parameters

# Plausible ranges for the data-quality checks
SHIP_LIMITS = {'roll_mean': (-30, 30), 'pitch_mean': (-15, 15), 'speed_kn': (0, 15),
               'lat': (61, 64.5), 'lon': (4, 10), 'heading': (0, 360)}
BUOY_LIMITS = {'Hm0': (0, 10), 'Hm0a': (0, 10), 'Hm0b': (0, 10), 'Tp': (1, 25), 'Tm02': (1, 25),
               'Mdir': (0, 360), 'ThTp': (0, 360), 'SprTp': (0, 90)}


# ---------- ship ----------

def days():
    """{'YYYY-MM-DD': folder} for every ship day folder; the date comes from the file names."""
    out = {}
    for d in RAW_SHIP.iterdir():
        first = next(d.glob('*/*.csv'), None)
        if first:
            out[f'{first.stem[:4]}-{first.stem[4:6]}-{first.stem[6:8]}'] = d
    return dict(sorted(out.items()))


def summarize(day_dir):
    """One row per 10 s file: position, heading, speed, and n/mean/std of roll, pitch, height."""
    rows = []
    for f in sorted(Path(day_dir).glob('*/*.csv')):
        d = pd.read_csv(f, sep=';', header=None, names=COLS)
        rows.append({'t': d.t.iloc[0], 'n': len(d), 'lat': d.lat.mean(), 'lon': d.lon.mean(),
                     'heading': d.heading.iloc[len(d) // 2],  # heading wraps at 360: middle sample, not mean
                     'speed_kn': np.hypot(d.vn_cms, d.ve_cms).mean() / 100 * 1.94384,
                     'roll_mean': d.roll.mean(), 'roll_std': d.roll.std(),
                     'pitch_mean': d.pitch.mean(), 'pitch_std': d.pitch.std(),
                     'height_mean': d.height_cm.mean(), 'height_std': d.height_cm.std()})  # height in cm
    s = pd.DataFrame(rows)
    s['time'] = pd.to_datetime(s.t, unit='s', utc=True).dt.tz_convert(TZ)
    return s


def load_day(day):
    """10 s summary for a day ('YYYY-MM-DD'); built from the raw CSVs (~75 s) the first time."""
    f = PROCESSED / f'ship_10s_{day}.parquet'
    if not f.exists():
        PROCESSED.mkdir(parents=True, exist_ok=True)
        summarize(days()[day]).to_parquet(f)
    return pd.read_parquet(f)


def read_raw(day_dir, start, end):
    """Raw 100 Hz rows between two tz-aware timestamps. Keep windows short (one file per 10 s)."""
    lo, hi = [(t.tz_convert(TZ).tz_localize(None)).strftime('%Y%m%d%H%M%S')
              for t in (start - pd.Timedelta('20s'), end + pd.Timedelta('20s'))]
    files = [f for f in sorted(Path(day_dir).glob('*/*.csv')) if lo <= f.stem <= hi]
    d = pd.concat([pd.read_csv(f, sep=';', header=None, names=COLS) for f in files], ignore_index=True)
    d['time'] = pd.to_datetime(d.t, unit='s', utc=True).dt.tz_convert(TZ)
    return d[(d.time >= start) & (d.time < end)]


# ---------- statistics ----------

def circmean(deg):
    r = np.radians(np.asarray(deg, dtype=float))
    return np.degrees(np.arctan2(np.nanmean(np.sin(r)), np.nanmean(np.cos(r)))) % 360


def angle_diff(a, b):
    """a - b folded to [-180, 180)."""
    return (a - b + 180) % 360 - 180


def rel_heading(wave_from, heading):
    """Wave direction relative to the bow: 0 = head seas, +90 = from starboard, -90 = from port, ±180 = following."""
    return angle_diff(wave_from, heading)


def nearest_nominal(rel):
    return min(NOMINAL, key=lambda a: abs(angle_diff(rel, a)))


def pooled(g, col):
    """Exact std of all raw samples in g, from the per-file n / mean / std."""
    n, m, sd = g.n, g[f'{col}_mean'], g[f'{col}_std']
    N = n.sum()
    M = (n * m).sum() / N
    return np.sqrt((((n - 1) * sd ** 2).sum() + (n * (m - M) ** 2).sum()) / (N - 1))


def within(g, col):
    """Std of the variation inside each 10 s file only: drops slow drift (GNSS wander, tide).
    ponytail: 10 s windows underestimate swell periods above ~10 s; high-pass the raw signal if that matters."""
    return np.sqrt(((g.n - 1) * g[f'{col}_std'] ** 2).sum() / (g.n - 1).sum())


def stats(g):
    """Summary of any slice of 10 s rows (a phase, a 10-min block, a heading block)."""
    return pd.Series({'start': g.time.iloc[0], 'end': g.time.iloc[-1] + pd.Timedelta(seconds=10),
                      'minutes': len(g) / 6, 'lat': g.lat.mean(), 'lon': g.lon.mean(),
                      'heading': circmean(g.heading), 'speed_kn': g.speed_kn.mean(),
                      'roll_std': pooled(g, 'roll'), 'pitch_std': pooled(g, 'pitch'),
                      'heave_hs_m': 4 * within(g, 'height') / 100})  # 4*sigma, like a wave Hs


def phases(s):
    """Split the day into Quay / Transit / Station blocks from speed (5 min median, > MOVING_KN = moving).
    A stop that starts or ends the day is 'Quay'; every other stop is a 'Station'."""
    moving = s.speed_kn.rolling(30, center=True, min_periods=1).median() > MOVING_KN
    moving |= moving.groupby((moving != moving.shift()).cumsum()).transform('size') < MIN_STOP_MIN * 6
    groups = [g for _, g in s.groupby((moving != moving.shift()).cumsum())]
    rows = []
    for k, g in enumerate(groups):
        phase = 'Transit' if moving[g.index[0]] else 'Quay' if k in (0, len(groups) - 1) else 'Station'
        rows.append({'phase': phase, **stats(g)})
    p = pd.DataFrame(rows)
    st = p.phase == 'Station'
    p.loc[st, 'phase'] = [f'Station {i + 1}' for i in range(st.sum())]
    return p


def blocks(s):
    """10-min blocks labelled by their END time in UTC, the same grid as the buoy records."""
    key = s.time.dt.tz_convert('UTC').dt.ceil('10min').rename('time')
    return s.groupby(key).apply(stats).infer_objects().reset_index()


# ---------- buoys ----------

def load_buoys():
    b = pd.read_csv(BUOY_CSV, parse_dates=['time'])
    b['time'] = b.time.dt.tz_localize('UTC')
    b['short'] = b.buoy.str.split().str[1]   # 'E39 D Breisundet' -> 'D'
    return b


def dist_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def buoy_positions(b):
    return b.groupby('buoy')[['latitude', 'longitude']].median()


def buoy_distances(lat, lon, b):
    """Distance [km] from a point to every buoy in b, nearest first."""
    pos = buoy_positions(b)
    return dist_km(lat, lon, pos.latitude, pos.longitude).sort_values()


def day_buoys(s, b):
    """Buoy records that fall inside the ship day."""
    return b[(b.time >= s.time.min()) & (b.time <= s.time.max() + pd.Timedelta('10min'))]


def with_buoy(blk, b, buoy):
    """10-min ship blocks joined with one buoy's records (only times where the buoy has data)."""
    bb = b[b.buoy == buoy][['time', *WAVE, 'latitude', 'longitude']]
    m = blk.merge(bb, on='time')
    m['dist_km'] = dist_km(m.lat, m.lon, m.latitude, m.longitude)
    return m


def nearest_buoy_blocks(blk, b):
    """10-min ship blocks, each joined with the nearest buoy that has data at that time."""
    if b.empty:   # a day with no buoy records (3 Nov): no blocks to pair
        return blk.iloc[:0]
    m = pd.concat([with_buoy(blk, b, n).assign(buoy=n) for n in b.buoy.unique()], ignore_index=True)
    return m.sort_values('dist_km').drop_duplicates('time').sort_values('time').reset_index(drop=True)


def station_table(s, p, b, buoy):
    """Per station: ship motion (exact, from 10 s rows) next to the buoy's wave parameters."""
    bb = b[b.buoy == buoy]
    pos = buoy_positions(bb).iloc[0]
    rows = []
    for r in p[p.phase.str.startswith('Station')].itertuples():
        w = bb[(bb.time > r.start) & (bb.time <= r.end + pd.Timedelta('10min'))]
        rows.append({'station': r.phase, 'start': r.start, 'end': r.end, 'minutes': r.minutes,
                     'dist_km': dist_km(r.lat, r.lon, pos.latitude, pos.longitude),
                     'buoy_Hm0_m': w.Hm0.mean(), 'buoy_Tp_s': w.Tp.mean(), 'ship_heave_hs_m': r.heave_hs_m,
                     'roll_std': r.roll_std, 'pitch_std': r.pitch_std})
    return pd.DataFrame(rows)


def heading_blocks(s, p, b, buoy):
    """Steady-heading blocks inside each Station, with the buoy's wave direction and the relative heading."""
    bb = b[b.buoy == buoy].set_index('time')
    rows = []
    for r in p[p.phase.str.startswith('Station')].itertuples():
        g = s[(s.time >= r.start) & (s.time < r.end)]
        steady = angle_diff(g.heading, g.heading.shift(3)).abs() < STEADY_DEG   # vs 30 s earlier
        runs = [x for _, x in g[steady].groupby((steady != steady.shift()).cumsum()[steady])]
        merged = [x for x in runs if len(x) >= 6]   # drop < 1 min transients
        while True:   # repeat: one merge can bring a block close enough to its other neighbour
            out = []
            for x in merged:
                if (out and x.time.iloc[0] - out[-1].time.iloc[-1] < pd.Timedelta(minutes=MERGE_GAP_MIN)
                        and abs(angle_diff(circmean(x.heading), circmean(out[-1].heading))) < MERGE_DEG):
                    out[-1] = pd.concat([out[-1], x])
                else:
                    out.append(x)
            if len(out) == len(merged):
                break
            merged = out
        for x in merged:
            if len(x) < MIN_BLOCK_MIN * 6:
                continue
            st = stats(x)
            w = bb[bb.index.isin(x.time.dt.tz_convert('UTC').dt.ceil('10min').unique())]
            row = {'station': r.phase, **st, 'Hm0': w.Hm0.mean(), 'Tp': w.Tp.mean(),
                   'Mdir': circmean(w.Mdir) if len(w) else np.nan, 'ThTp': circmean(w.ThTp) if len(w) else np.nan,
                   'dist_km': dist_km(st.lat, st.lon, w.latitude.median(), w.longitude.median())}
            row['rel_Mdir'] = rel_heading(row['Mdir'], st.heading)
            row['rel_ThTp'] = rel_heading(row['ThTp'], st.heading)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------- data quality ----------

def out_of_range(df, limits):
    """Number of values outside their {column: (lo, hi)} range (NaN not counted)."""
    return int(sum(((df[c] < lo) | (df[c] > hi)).sum() for c, (lo, hi) in limits.items()))


def ship_checks(s, day_dir):
    """Observability checks for one ship day of 10 s rows: volume, gaps, schema, value ranges."""
    dt = s.time.diff().dt.total_seconds()
    expected = round((s.time.iloc[-1] - s.time.iloc[0]).total_seconds() / 10) + 1
    first = next(Path(day_dir).glob('*/*.csv'))
    return {'files': len(s), 'expected': expected, 'complete_pct': 100 * len(s) / expected,
            'min_samples_per_file': int(s.n.min()), 'max_step_s': dt.max(), 'gaps_over_11s': int((dt > 11).sum()),
            'nan_values': int(s.isna().sum().sum()), 'out_of_range': out_of_range(s, SHIP_LIMITS),
            'csv_columns': pd.read_csv(first, sep=';', header=None, nrows=5).shape[1]}


def update_rates(raw):
    """Changes per second of each raw column (100 Hz rows). ~100 = a true 100 Hz signal, ~10 = a 10 Hz sensor
    repeated between updates. A calm signal also changes less often because of its value resolution."""
    return (raw[COLS[1:]].diff().iloc[1:] != 0).sum() / (len(raw) / 100)


def buoy_checks(b):
    """Per buoy and trial day: records vs expected (10 min grid), largest gap, out-of-range values,
    and how far the buoy drifted from its median position (mooring watch circle)."""
    b = b.assign(day=b.time.dt.tz_convert(TZ).dt.strftime('%Y-%m-%d'))
    span = b.groupby('day').time.agg(['min', 'max'])
    expected = ((span['max'] - span['min']) / pd.Timedelta('10min')).astype(int) + 1
    pos = buoy_positions(b)
    rows = []
    for buoy in pos.index:
        p = pos.loc[buoy]
        for day in span.index:
            g = b[(b.buoy == buoy) & (b.day == day)]
            rows.append({'buoy': buoy, 'day': day, 'records': len(g), 'expected': expected[day],
                         'coverage_pct': 100 * len(g) / expected[day],
                         'max_gap_min': g.time.diff().max() / pd.Timedelta('1min'),
                         'out_of_range': out_of_range(g, BUOY_LIMITS),
                         'mooring_spread_m': 1000 * dist_km(g.latitude, g.longitude, p.latitude, p.longitude).max()})
    return pd.DataFrame(rows)
