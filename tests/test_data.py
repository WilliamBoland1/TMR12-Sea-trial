"""Self-checks for the non-trivial logic in tmr12.data.  Run: python tests/test_data.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tmr12 import data  # noqa: E402


def fake_day(speed, heading):
    """10 s rows like data.summarize() produces, from per-row speed and heading arrays."""
    n = len(speed)
    t = pd.date_range('2023-10-31 06:00', periods=n, freq='10s', tz='UTC')
    return pd.DataFrame({'time': t.tz_convert(data.TZ), 'n': 1000, 'lat': 62.446, 'lon': 5.927,
                         'heading': heading, 'speed_kn': speed, 'roll_mean': 0.0, 'roll_std': 1.0,
                         'pitch_mean': 0.0, 'pitch_std': 0.5, 'height_mean': 0.0, 'height_std': 10.0})


def test_pooled_std_is_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000) + 3 * np.sin(np.arange(10_000) / 300)   # slow drift between files
    chunks = x.reshape(10, 1000)
    g = pd.DataFrame({'n': 1000, 'roll_mean': chunks.mean(1), 'roll_std': chunks.std(1, ddof=1)})
    assert np.isclose(data.pooled(g, 'roll'), x.std(ddof=1))


def test_rel_heading():
    assert data.rel_heading(90, 0) == 90          # waves from starboard
    assert data.rel_heading(10, 350) == 20        # wrap across north
    assert data.rel_heading(350, 10) == -20       # from port
    assert abs(data.rel_heading(180, 0)) == 180   # following seas
    assert data.nearest_nominal(-17) == -15 and data.nearest_nominal(172) == 180


def test_phases():
    speed = np.r_[np.zeros(200), np.full(60, 10), np.zeros(200), np.full(60, 10), np.zeros(200)]
    p = data.phases(fake_day(speed, np.zeros(len(speed))))
    assert list(p.phase) == ['Quay', 'Transit', 'Station 1', 'Transit', 'Quay'], list(p.phase)
    assert np.allclose(p.heave_hs_m, 0.4)   # 4 * 10 cm within-file std


def test_heading_blocks():
    # station: 10 min at heading 0, 1 min turning, 10 min at heading 15; waves from 0 deg
    heading = np.r_[np.zeros(60), np.linspace(0, 15, 6), np.full(60, 15.0)]
    speed = np.r_[np.full(40, 10), np.zeros(len(heading)), np.full(40, 10)]
    s = fake_day(speed, np.r_[np.zeros(40), heading, np.full(40, 15.0)])
    p = data.phases(s)
    times = pd.date_range('2023-10-31 06:00', periods=12, freq='10min', tz='UTC')
    b = pd.DataFrame({'time': times, 'buoy': 'E39 D Breisundet', 'Hm0': 0.5, 'Tp': 6.0, 'Mdir': 0.0,
                      'ThTp': 0.0, 'latitude': 62.446, 'longitude': 5.927})
    hb = data.heading_blocks(s, p, b, 'E39 D Breisundet')
    assert len(hb) == 2, hb
    assert np.allclose(hb.rel_Mdir.round(), [0, -15]), hb.rel_Mdir.tolist()   # 15 deg starboard turn -> waves on port bow


def buoy_rows(name, times, lat, lon):
    return pd.DataFrame({'time': times, 'buoy': name, 'Hm0': 0.5, 'Hm0a': 0.1, 'Hm0b': 0.4, 'Tp': 6.0,
                         'Tm02': 3.0, 'Mdir': 0.0, 'ThTp': 0.0, 'SprTp': 30.0, 'latitude': lat, 'longitude': lon})


def test_nearest_buoy_blocks():
    t = pd.date_range('2023-10-31 06:10', periods=3, freq='10min', tz='UTC')
    blk = pd.DataFrame({'time': t, 'lat': 62.446, 'lon': 5.927})
    b = pd.concat([buoy_rows('E39 D Breisundet', t[:2], 62.446, 5.927),    # at the ship, but misses the 3rd block
                   buoy_rows('E39 A Sulafjorden', t, 62.427, 6.044)])
    m = data.nearest_buoy_blocks(blk, b)
    assert list(m.buoy) == ['E39 D Breisundet'] * 2 + ['E39 A Sulafjorden'], list(m.buoy)
    assert data.nearest_buoy_blocks(blk, b.iloc[:0]).empty   # a day without buoy records


def test_quality_checks():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / '00').mkdir()
        (Path(d) / '00' / 'x.csv').write_text(';'.join(['0'] * len(data.COLS)) + '\n')
        s = fake_day(np.zeros(100), np.zeros(100)).drop(index=50).reset_index(drop=True)   # one missing file
        s.loc[10, 'roll_mean'] = 50
        c = data.ship_checks(s, d)
    assert (c['files'], c['expected'], c['gaps_over_11s'], c['out_of_range'], c['csv_columns']) == (99, 100, 1, 1, 14), c
    t = pd.date_range('2023-10-31 06:00', periods=6, freq='10min', tz='UTC')
    q = data.buoy_checks(pd.concat([buoy_rows('A', t, 62.4, 6.0), buoy_rows('B', t[:3], 62.5, 6.0),
                                    buoy_rows('A', t + pd.Timedelta('2D'), 62.4, 6.0)]))
    q = q.set_index(['buoy', 'day'])
    assert q.loc[('B', '2023-11-02'), 'records'] == 0 and q.loc[('B', '2023-10-31'), 'coverage_pct'] == 50, q


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
