"""Data visualisation for the DP part of the project (30-31 Oct 2023).

Three figures, chosen for the project thesis on DP capability and DP footprints:
  1. Motion rose: roll, pitch and heave against the relative wave heading, drawn like a DP capability plot.
  2. Ship heave against buoy Hm0 with uncertainty bars: how far the ship can be used as a wave sensor.
  3. Two time scales on DP: slow position excursions (what DP controls) next to fast wave motion (what it does not).
  4. Pair plot and 5. parallel coordinates of the heading blocks: the multidimensional plots from the course lab.

Reads reports/eda/dp_heading_blocks.csv, so run eda.py first. Figures go to reports/vis/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmr12 import data, eda

OUT = data.ROOT / 'reports' / 'vis'
NOISE_HS_M = 0.11   # heave 4σ measured at the quay: GNSS noise floor (EDA, 1-minute windows)
DAY_COLOURS = {'2023-10-30': eda.BLUE, '2023-10-31': eda.ORANGE}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def load_blocks():
    """Heading blocks from the EDA, with times as timestamps."""
    blocks = pd.read_csv(eda.OUT / 'dp_heading_blocks.csv')
    for col in ['start', 'end']:
        blocks[col] = pd.to_datetime(blocks[col], utc=True).dt.tz_convert(data.TZ)
    return blocks


def fig_motion_rose(blocks):
    """Polar plots of roll σ, pitch σ and heave 4σ against the relative wave heading (buoy D only).
    0° = head seas at the top, waves from starboard to the right, like a DP capability plot."""
    blocks = blocks[blocks.buoy == 'D']
    cols = [('roll_std_wave', 'roll σ [°]'), ('pitch_std_wave', 'pitch σ [°]'), ('heave_hs_wave', 'heave 4σ [m]')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), subplot_kw={'projection': 'polar'})
    for ax, (col, label) in zip(axes, cols):
        for day, colour in DAY_COLOURS.items():
            g = blocks[blocks.day == day]
            ax.scatter(np.radians(g.rel_Mdir), g[col], s=g.minutes * 3, color=colour, alpha=0.85,
                       edgecolor='white', lw=0.5, label=pd.Timestamp(day).strftime('%-d %b'))
        ax.set_theta_zero_location('N')   # head seas at the top
        ax.set_theta_direction(-1)         # clockwise: waves from starboard to the right
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                          ['head', '45', 'beam\n(stb)', '135', 'following', '225', 'beam\n(port)', '315'])
        ax.set_rlim(0, blocks[col].max() * 1.1)
        ax.set_title(label, pad=18)
    axes[0].legend(loc='lower left', bbox_to_anchor=(-0.25, -0.15), frameon=False, title='buoy D')
    fig.tight_layout()
    return save(fig, 'v1_motion_rose.png')


def fig_heave_vs_buoy(blocks, b):
    """Ship heave 4σ against buoy Hm0 per heading block, with uncertainty bars.
    Buoy bar: lowest to highest 10-min Hm0 during the block (the sea state changes in time and space).
    Ship bar: statistical uncertainty of a std estimated from N waves, about σ/sqrt(2N), N = duration / Tp.
    Grey band: GNSS noise floor measured at the quay."""
    blocks = blocks[blocks.buoy != 'none'].copy()
    lo, hi = [], []
    for blk in blocks.itertuples():
        name = [n for n in b.buoy.unique() if n.split()[1] == blk.buoy][0]
        w = b[(b.buoy == name) & (b.time > blk.start) & (b.time <= blk.end + pd.Timedelta('10min'))]
        lo.append(w.Hm0.min())
        hi.append(w.Hm0.max())
    blocks['hm0_lo'], blocks['hm0_hi'] = lo, hi
    n_waves = blocks.minutes * 60 / blocks.Tp
    blocks['ship_err'] = blocks.heave_hs_wave / np.sqrt(2 * n_waves)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.axhspan(0, NOISE_HS_M, color=eda.GREY, alpha=0.2, lw=0, label=f'GNSS noise floor ({NOISE_HS_M} m)')
    ax.plot([0, 1], [0, 1], color=eda.GREY, ls='--', lw=1, label='ship = buoy')
    for short, colour in eda.BUOY_COLOURS.items():
        g = blocks[blocks.buoy == short]
        ax.errorbar(g.Hm0, g.heave_hs_wave, xerr=[g.Hm0 - g.hm0_lo, g.hm0_hi - g.Hm0], yerr=g.ship_err,
                    fmt='o', ms=6, color=colour, ecolor=colour, elinewidth=1, capsize=2, label=f'buoy {short}')
    ax.set(xlim=(0, 0.85), ylim=(0, 0.85), aspect='equal',
           xlabel='buoy $H_{m0}$ [m]', ylabel='ship heave 4σ [m]')
    ax.legend(frameon=False, loc='upper left')
    return save(fig, 'v2_heave_vs_buoy.png')


def fig_two_time_scales(blocks, days):
    """Two blocks at buoy D (beam seas and following seas), each with:
    left: position excursion from the block mean (10 s means, the slow motion DP works against);
    right: two minutes of raw 100 Hz roll and heave (the wave motion DP cannot remove)."""
    d = blocks[(blocks.buoy == 'D') & (blocks.day == '2023-10-31')]
    picks = [d.loc[d.abs_rel_wave.sub(90).abs().idxmin()], d.loc[d.abs_rel_wave.idxmax()]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [1.4, 1]})
    for row, blk in zip(axes, picks):
        s = days[blk.day]
        g = s[(s.time >= blk.start) & (s.time < blk.end)]
        north, east = eda.local_xy(g.lat, g.lon, g.lat.mean(), g.lon.mean())
        minutes = (g.time - blk.start).dt.total_seconds() / 60
        row[0].plot(minutes, north, color=eda.BLUE, lw=1.2, label='north')
        row[0].plot(minutes, east, color=eda.ORANGE, lw=1.2, label='east')
        row[0].set(xlabel='minutes into block', ylabel='excursion [m]', ylim=(-1.2, 1.2))
        row[0].set_title(f'heading {blk.heading:.0f}°, waves {blk.abs_rel_wave:.0f}° off the bow: '
                         f'R95 = {blk.footprint_r95_m:.2f} m', loc='left', fontsize=9)

        raw = eda.raw_block(eda.raw_files(data.days()[blk.day]), blk.start, blk.start + pd.Timedelta('2min'))
        t = raw.t - raw.t.iloc[0]
        roll = eda.wave_part(raw.roll)
        heave = eda.wave_part(raw.height_cm / 100)
        row[1].plot(t, roll, color=eda.BLUE, lw=0.8, label='roll [°]')
        row[1].plot(t, heave * 10, color=eda.AQUA, lw=0.8, label='heave [dm]')
        row[1].set(xlabel='seconds', ylim=(-8, 8))
        row[1].set_title(f'roll σ = {blk.roll_std_wave:.1f}°, heave 4σ = {blk.heave_hs_wave:.2f} m',
                         loc='left', fontsize=9)
    axes[0, 0].legend(frameon=False, ncol=2, loc='upper right')
    axes[0, 1].legend(frameon=False, ncol=2, loc='upper right')
    fig.tight_layout()
    return save(fig, 'v3_two_time_scales.png')


WAVE_SECTOR = {'head': (0, 45), 'beam': (45, 135), 'following': (135, 181)}


def sector(abs_rel):
    """Relative wave direction as a class: head (0-45°), beam (45-135°) or following (135-180°) seas."""
    for name, (lo, hi) in WAVE_SECTOR.items():
        if lo <= abs_rel < hi:
            return name


def fig_pairplot(blocks):
    """Pair plot (seaborn) of the main block variables, coloured by wave sector: every pair of variables
    as a scatter plot, and each variable's distribution on the diagonal."""
    import seaborn as sns
    cols = {'Hm0': 'Hm0 [m]', 'roll_std_wave': 'roll σ [°]', 'pitch_std_wave': 'pitch σ [°]',
            'heave_hs_wave': 'heave 4σ [m]', 'footprint_r95_m': 'R95 [m]'}
    d = blocks[blocks.buoy != 'none'].copy()
    d['waves'] = d.abs_rel_wave.map(sector)
    d = d[list(cols) + ['waves']].rename(columns=cols)
    palette = {'head': eda.BLUE, 'beam': eda.ORANGE, 'following': eda.AQUA}
    g = sns.pairplot(d, hue='waves', palette=palette, hue_order=list(palette), corner=True,
                     diag_kind='hist', plot_kws={'s': 30, 'edgecolor': 'white'}, height=1.7)
    path = OUT / 'v4_pairplot.png'
    OUT.mkdir(parents=True, exist_ok=True)
    g.savefig(path, dpi=150)
    plt.close(g.figure)
    return path


def fig_parallel(blocks):
    """Parallel coordinates (pandas.plotting): one line per heading block through all variables,
    each scaled to 0-1, coloured by wave sector."""
    from pandas.plotting import parallel_coordinates
    cols = ['Hm0', 'Tp', 'wind_ms', 'roll_std_wave', 'pitch_std_wave', 'heave_hs_wave', 'footprint_r95_m']
    d = blocks[blocks.buoy != 'none'].copy()
    scaled = (d[cols] - d[cols].min()) / (d[cols].max() - d[cols].min())
    scaled['waves'] = d.abs_rel_wave.map(sector)
    palette = {'head': eda.BLUE, 'beam': eda.ORANGE, 'following': eda.AQUA}
    order = [w for w in palette if w in set(scaled.waves)]
    scaled = scaled.sort_values('waves', key=lambda w: w.map(order.index))   # classes are coloured in this order
    fig, ax = plt.subplots(figsize=(10, 4))
    parallel_coordinates(scaled, 'waves', ax=ax, color=[palette[w] for w in order], lw=1.5, alpha=0.8)
    ax.set_xticks(range(len(cols)), ['Hm0', 'Tp', 'wind', 'roll σ', 'pitch σ', 'heave 4σ', 'R95'])
    ax.set_ylabel('scaled value (0 = lowest, 1 = highest)')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    return save(fig, 'v5_parallel_coordinates.png')
