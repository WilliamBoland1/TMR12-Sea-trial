"""Static one-page PNG overview of a ship day (matplotlib)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmr12.data import PLAN, ROOT  # noqa: E402

BLUE, ORANGE, AQUA, INK, INK2, BG = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#fcfcfb'

LEGEND = """Columns in each CSV (';', no header)
 0      unix time, 100 Hz
 1  2   latitude, longitude [deg]
 3      height [cm]*  (GNSS, tide + heave)
 4      unknown integer
 5  6   north / east velocity [cm/s]*
 7      down velocity [cm/s]*
 8  9   roll, pitch [deg]
 10     heading [deg]
 11-13  roll / pitch / yaw rate [deg/s]*
* inferred from the data, not documented"""


def overview_png(s, p, day, folder):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    grid = fig.add_gridspec(4, 2, width_ratios=[1, 1.5], hspace=0.3, wspace=0.12)
    fig.suptitle(f'RV Gunnerus, {s.time.iloc[len(s) // 2]:%a %d %b %Y}: what is in folder {folder}/  '
                 f'({len(s)} files x 10 s @ 100 Hz = {s.n.sum():,} rows)',
                 color=INK, fontsize=14, x=0.01, ha='left')
    fig.text(0.01, 0.935, PLAN.get(day, ''), color=INK2, fontsize=11)

    # Map: grey track, numbered badges (one per quay location)
    ax = fig.add_subplot(grid[:2, 0])
    ax.plot(s.lon, s.lat, color='#9e9d98', lw=1.5)
    stops = p[p.phase != 'Transit']
    stops = stops[~((stops.phase == 'Quay') & stops[['lat', 'lon']].round(2).duplicated())]
    for r in stops.itertuples():
        ax.text(r.lon, r.lat, 'Q' if r.phase == 'Quay' else r.phase.split()[-1], ha='center', va='center',
                color=BG, fontsize=9, weight='bold', bbox=dict(boxstyle='circle,pad=0.3', fc=INK, ec=BG, lw=1.5))
    ax.set(title='Ship track  (Q = quay, 1 2 = stations)', xlabel='Longitude [°E]', ylabel='Latitude [°N]')
    ax.set_aspect(1 / np.cos(np.radians(s.lat.mean())))  # true-ish shape at this latitude
    ax.margins(0.12, 0.25)
    axes = [ax]

    fig.add_subplot(grid[2:, 0]).axis('off')
    fig.axes[-1].text(0.02, 0.95, LEGEND, family='monospace', va='top', color=INK2, fontsize=10.5)

    panels = [('Speed [kn]', [('speed_kn', BLUE, None)]),
              ('Heading [°]', [('heading', BLUE, None)]),
              ('Roll & pitch\n(10 s std) [°]', [('roll_std', BLUE, 'Roll'), ('pitch_std', ORANGE, 'Pitch')]),
              ('Heave\n(10 s std) [cm]', [('height_std', AQUA, None)])]
    for i, (label, series) in enumerate(panels):
        ax = fig.add_subplot(grid[i, 1], sharex=axes[1] if i else None)
        axes.append(ax)
        for r in p[p.phase.str.startswith('Station')].itertuples():
            ax.axvspan(r.start, r.end, color='#ecebe6', lw=0)
            if i == 0:
                ax.text(r.start + (r.end - r.start) / 2, 1.02, r.phase, transform=ax.get_xaxis_transform(),
                        ha='center', color=INK2)
        for col, color, name in series:
            if col == 'heading':  # dots, so the 360->0 wrap doesn't draw vertical lines
                ax.plot(s.time, s[col], '.', color=color, ms=1.5)
                ax.set_yticks([0, 90, 180, 270, 360])
            else:
                ax.plot(s.time, s[col], color=color, lw=1, label=name)
        if series[0][2]:
            ax.legend(loc='upper right', frameon=False)
        ax.set_ylabel(label, color=INK2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=s.time.dt.tz))
        ax.tick_params(labelbottom=i == len(panels) - 1)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(color='#e6e5e0', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Local time (CET)')
    out = ROOT / 'reports' / f'{day}_overview.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
