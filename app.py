"""TMR12 wave-radar data explorer, told as 9 steps.  Run: streamlit run app.py"""
import warnings
from datetime import timedelta

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from tmr12 import cluster, data, maps

st.set_page_config(page_title='TMR12 wave radar data', page_icon='🌊', layout='wide')
alt.data_transformers.disable_max_rows()
warnings.filterwarnings('ignore', message='Automatically deduplicated selection')   # intended: one shared x-zoom
BLUE, ORANGE, AQUA, BUOY, GREY, SHADE = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#9e9d98', '#ecebe6'
RED, MID, INK, SURFACE = '#e34948', '#f0efec', '#0b0b0b', '#fcfcfb'
BUOY_COLORS = {'A': BLUE, 'B': ORANGE, 'C': AQUA, 'D': '#eda100', 'F': '#e87ba4', 'G': '#008300'}  # lines only
PHASES = ['Quay', 'Transit', 'Station']
DAYS = data.days()
MAX_KM = 10.0   # main plots leave out 10-min blocks further than this from their buoy

SHIP_VARS = {'speed_kn': 'Speed [kn]', 'roll_std': 'Roll std [°]', 'pitch_std': 'Pitch std [°]',
             'height_std': 'Heave std [cm]', 'roll_mean': 'Roll mean (list) [°]', 'pitch_mean': 'Pitch mean (trim) [°]',
             'height_mean': 'GNSS height mean [cm]'}
BUOY_PARAMS = {'Hm0': 'Hm0, significant wave height [m]', 'Hm0a': 'Hm0a, low-frequency band, swell [m]',
               'Hm0b': 'Hm0b, high-frequency band, wind sea [m]', 'Tp': 'Tp, peak period [s]',
               'Tm02': 'Tm02, mean period [s]', 'Mdir': 'Mdir, mean direction, from [° true]',
               'ThTp': 'ThTp, direction at the peak, from [° true]', 'SprTp': 'SprTp, directional spread at the peak [°]'}
RESPONSE = {'heave_hs_m': 'Ship heave 4σ [m]', 'roll_std': 'Ship roll σ [°]', 'pitch_std': 'Ship pitch σ [°]'}


# ---------- cached loading ----------

def local(t):
    return t.dt.tz_convert(data.TZ).dt.tz_localize(None).dt.floor('s')   # naive wall-clock time for widgets/charts


@st.cache_data(show_spinner='Reading raw ship data (about a minute the first time per day)...')
def load_day(day):
    s = data.load_day(day)
    s['local'] = local(s.time)
    return s


@st.cache_data
def load_phases(day):
    p = data.phases(load_day(day))
    return p.assign(start_l=local(p.start), end_l=local(p.end))


@st.cache_data
def load_buoys():
    return data.load_buoys()


@st.cache_data
def load_blocks(day):
    blk = data.blocks(load_day(day))
    blk['local'] = local(blk.time)
    return blk


@st.cache_data
def buoy_choice(day):
    """Per buoy, nearest first: distance [km] to the closest station of the day, which station, has data today."""
    s, p, b = load_day(day), load_phases(day), load_buoys()
    ref = p[p.phase.str.startswith('Station')]
    ref = ref if len(ref) else p
    rows = []
    for name, q in data.buoy_positions(b).iterrows():
        d = data.dist_km(ref.lat, ref.lon, q.latitude, q.longitude).values
        rows.append({'buoy': name, 'km': d.min(), 'station': ref.phase.iloc[d.argmin()]})
    out = pd.DataFrame(rows).set_index('buoy').sort_values('km')
    out['has_data'] = out.index.isin(data.day_buoys(s, b).buoy)
    return out


@st.cache_data
def load_heading_blocks(day, buoy):
    s = load_day(day)
    return data.heading_blocks(s, load_phases(day), data.day_buoys(s, load_buoys()), buoy)


@st.cache_data
def load_all():
    """Every day's 10 s rows, with day and phase (Station / Transit / Quay)."""
    out = []
    for d in DAYS:
        s = load_day(d).assign(day=d)
        for r in load_phases(d).itertuples():
            s.loc[(s.time >= r.start) & (s.time < r.end), 'phase'] = r.phase.split()[0]
        out.append(s)
    return pd.concat(out, ignore_index=True)


@st.cache_data
def load_buoy_blocks():
    """Every day's 10-min ship blocks, each joined with the nearest buoy that has data at that time."""
    b = load_buoys()
    m = pd.concat([data.nearest_buoy_blocks(load_blocks(d), data.day_buoys(load_day(d), b)).assign(day=d)
                   for d in DAYS], ignore_index=True)
    return m.assign(short=m.buoy.str.split().str[1])


@st.cache_data(show_spinner='Clustering one-minute windows...')
def run_cluster():
    """Step 8: every 1-minute window with its cluster, and its scores against the correct timeline."""
    out, tree = cluster.fit_tree(cluster.fit(cluster.load_windows()))
    return out, tree, *cluster.check(out)


@st.cache_data
def load_raw(day, start):
    raw = data.read_raw(DAYS[day], start, start + pd.Timedelta('2min'))
    raw['local'] = local(raw.time)
    return raw


# ---------- chart helpers ----------

def panel(df, series, y_title, colors, points=False, height=140, y=None, fmt='%H:%M'):
    """One time-series panel; series = {column: label}."""
    names = list(series.values())
    d = df[['local', *series]].rename(columns=series)
    c = alt.Chart(d).transform_fold(names, as_=['series', 'value'])
    c = c.mark_circle(size=8, opacity=0.8) if points else c.mark_line(strokeWidth=1.5)
    return c.encode(
        x=alt.X('local:T', title=None, axis=alt.Axis(format=fmt)),
        y=y or alt.Y('value:Q', title=y_title),
        color=alt.Color('series:N', scale=alt.Scale(domain=names, range=colors),
                        legend=alt.Legend(orient='top', title=None) if len(names) > 1 else None),
        tooltip=[alt.Tooltip('local:T', format='%H:%M:%S', title='time'), alt.Tooltip('series:N', title=' '),
                 alt.Tooltip('value:Q', format='.2f')],
    ).properties(height=height, title=alt.Title(y_title, anchor='start', fontSize=12) if len(names) == 1 else '')


def stack(bands, *panels):
    """Panels stacked with shared, zoomable x; bands (start_l, end_l, phase) shaded grey."""
    shade = alt.Chart(bands[['start_l', 'end_l', 'phase']]).mark_rect(color=SHADE).encode(
        x='start_l:T', x2='end_l:T', tooltip='phase:N')
    zoom = alt.selection_interval(bind='scales', encodings=['x'])
    # color independent: a shared scale merges every panel's series into one legend and repaints them
    return alt.vconcat(*[alt.layer(shade, c.add_params(zoom)) for c in panels]).resolve_scale(x='shared',
                                                                                             color='independent')


def corr_heatmap(df, cols):
    """Pearson r between columns ({column: label}), lower triangle: blue = negative, grey = none, red = positive."""
    order = list(cols.values())
    pos = {k: i for i, k in enumerate(order)}
    c = df[list(cols)].rename(columns=cols).corr().stack().reset_index()
    c.columns = ['a', 'b', 'r']
    c = c[c.a.map(pos) > c.b.map(pos)]
    base = alt.Chart(c).encode(x=alt.X('b:N', sort=order[:-1], title=None, axis=alt.Axis(labelAngle=-35, labelLimit=220)),
                               y=alt.Y('a:N', sort=order[1:], title=None, axis=alt.Axis(labelLimit=220)))
    cells = base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color('r:Q', title='r',   # lab: straight blue-grey-red, no magenta detour
                        scale=alt.Scale(domain=[-1, 0, 1], range=[BLUE, MID, RED], interpolate='lab')),
        tooltip=[alt.Tooltip('a:N', title=' '), alt.Tooltip('b:N', title='vs'), alt.Tooltip('r:Q', format='.2f')])
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text('r:Q', format='.2f'),
        color=alt.when('abs(datum.r) > 0.6').then(alt.value('white')).otherwise(alt.value(INK)))
    return (cells + text).properties(width=alt.Step(64), height=alt.Step(30))


def ship_buoy_scatter(m, resp='heave_hs_m'):
    """Buoy Hm0 vs a ship response per 10-min block: one panel per day, coloured by buoy.
    For heave (same unit and recipe as Hm0) a dashed line marks 'ship = buoy'."""
    heave = resp == 'heave_hs_m'
    hi = float(max(m.Hm0.max(), m[resp].max() if heave else 0)) * 1.05
    dom = sorted(m.short.unique())
    one = alt.Chart(pd.DataFrame({'x': [0, hi]})).mark_line(color=GREY, strokeDash=[4, 4]).encode(x='x:Q', y='x:Q')
    charts = []
    for day, g in m.groupby('day'):
        pts = alt.Chart(g[['day', 'local', 'short', 'dist_km', 'speed_kn', 'Hm0', resp]]).mark_circle(
            size=60, opacity=0.85, stroke=SURFACE, strokeWidth=1).encode(
            x=alt.X('Hm0:Q', title='Buoy Hm0 [m]', scale=alt.Scale(domain=[0, hi])),
            y=alt.Y(f'{resp}:Q', title=RESPONSE[resp], scale=alt.Scale(domain=[0, hi]) if heave else alt.Scale()),
            color=alt.Color('short:N', title='Buoy', legend=alt.Legend(orient='top'),
                            scale=alt.Scale(domain=dom, range=[BUOY_COLORS.get(k, GREY) for k in dom])),
            tooltip=['day', alt.Tooltip('local:T', format='%H:%M', title='block end'),
                     alt.Tooltip('short:N', title='buoy'), alt.Tooltip('dist_km:Q', format='.1f'),
                     alt.Tooltip('speed_kn:Q', format='.1f'), alt.Tooltip('Hm0:Q', format='.2f'),
                     alt.Tooltip(f'{resp}:Q', format='.2f')])
        charts.append((alt.layer(one, pts) if heave else pts).properties(
            width=240, height=240, title=alt.Title(f'{pd.Timestamp(day):%a %d %b}', anchor='start', fontSize=12)))
    return alt.hconcat(*charts)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)   # one new matplotlib figure per rerun, so close it


def show_table(df):
    df = df.copy()
    for c in ('start', 'end'):
        if c in df:
            df[c] = df[c].dt.strftime('%H:%M')
    st.dataframe(df.round(2), hide_index=True)


def pick_buoy(day):
    c = buoy_choice(day)
    ok = c[c.has_data]
    if ok.empty:
        st.info('No buoy wave data for this day.')
        return None
    if not c.has_data.iloc[0]:
        st.info(f'The nearest buoy, {c.index[0]} ({c.km.iloc[0]:.1f} km from {c.station.iloc[0]}), has no wave '
                'parameters this day, so only buoys further away can be compared.')
    buoy = st.selectbox('Buoy', ok.index, key=f'buoy-{day}',
                        format_func=lambda n: f'{n}  ({ok.km[n]:.1f} km from {ok.station[n]})')
    if ok.km[buoy] > 2:
        st.warning(f'Every station is at least {ok.km[buoy]:.0f} km from {buoy}: '
                   'its waves (especially direction) may differ from what the ship saw.')
    return buoy


def day_caption(day):
    return f'{pd.Timestamp(day):%A %d %B %Y} · {data.PLAN.get(day, "")} · data/raw/ship/{DAYS[day].name}/'


# ---------- story helpers: every step = question, short text, main plot, takeaway, details, next ----------

def step(n, question, what, caption=''):
    st.title(f'{n}. {STEPS[n - 1][1]}')
    if caption:
        st.caption(caption)
    st.subheader(question)
    st.markdown(what)


def takeaway(text):
    st.success(f'**Takeaway:** {text}', icon='💡')


def next_step(n):
    if n < len(PAGES):
        st.divider()
        st.page_link(PAGES[n], label=f'Next: {PAGES[n].title}', icon='➡️')


# ---------- the 9 steps ----------

def page_intro():
    step(1, "Can we tell from the ship's own motion what it is doing: in harbour, on DP or in transit?",
         'In week 44 of 2023 the research vessel **RV Gunnerus** tested a wave radar off Ålesund and Kristiansund. '
         'The whole time, it logged its own motion (position, heading, roll, pitch and height) **100 times a '
         'second**. In the fjords nearby, the **E39 wave buoys** measured the waves every 10 minutes.\n\n'
         '**Why it matters:** to validate real operations against simulations, you have to compare like with like. '
         'A ship lying in harbour, holding position on dynamic positioning (DP) or in transit behaves differently '
         'and is simulated differently, so every part of the log needs the right operating state. If a model can '
         'find that state from the motion log alone, the comparison can be done without labelling by hand.')
    a, b, m = load_all(), load_buoys(), load_buoy_blocks()
    c = st.columns(4)
    c[0].metric('Trial days', len(DAYS), border=True)
    c[1].metric('Ship motion samples', f'{a.n.sum() / 1e6:.1f} million', border=True)
    c[2].metric('Wave buoys', b.buoy.nunique(), border=True)
    c[3].metric('10-min ship–buoy pairs', len(m), border=True)
    st.subheader('The answer, one step at a time')
    st.markdown('\n'.join(f'{i}. **{t}**: {why}' for i, (_, t, _, why) in enumerate(STEPS, 1) if i > 1))
    next_step(1)


def page_day():
    day = st.session_state.day
    s, p, b = load_day(day), load_phases(day), load_buoys()
    step(2, 'What does one day of ship data look like?',
         'Pick a day in the sidebar. The map shows where the ship went, the plots how it moved. The ship writes one '
         'file every 10 seconds; each point below sums up one file: the average speed, the heading, and how much '
         'roll, pitch and height varied inside those 10 s (the standard deviation, σ).',
         caption=day_caption(day))

    active = p[p.phase != 'Quay']
    first, last = s.local.min().to_pydatetime(), s.local.max().to_pydatetime()
    lo = max(first, (active.start_l.min() - pd.Timedelta('30min')).to_pydatetime()) if len(active) else first
    hi = min(last, (active.end_l.max() + pd.Timedelta('30min')).to_pydatetime()) if len(active) else last
    t0, t1 = st.slider('Time window (local time)', first, last, (lo, hi), step=timedelta(minutes=5), format='HH:mm')
    w = s[(s.local >= t0) & (s.local <= t1)]

    left, right = st.columns([1, 1.25])
    with left:
        st_folium(maps.track_map(w, p, b), height=560, use_container_width=True, returned_objects=[],
                  key=f'map-{day}-{t0}-{t1}')
        st.caption('Q = quay, 1 2 3 = stations, violet rings = buoys. Hover for details.')
    with right:
        stations = p[p.phase.str.startswith('Station')]
        st.altair_chart(stack(
            stations[(stations.end_l >= t0) & (stations.start_l <= t1)],
            panel(w, {'speed_kn': 'Speed'}, 'Speed [kn]', [BLUE]),
            panel(w, {'heading': 'Heading'}, 'Heading [°]', [BLUE], points=True,
                  y=alt.Y('value:Q', title='Heading [°]', scale=alt.Scale(domain=[0, 360]),
                          axis=alt.Axis(values=[0, 90, 180, 270, 360]))),
            panel(w, {'roll_std': 'Roll', 'pitch_std': 'Pitch'}, 'Roll & pitch, 10 s std [°]', [BLUE, ORANGE]),
            panel(w, {'height_std': 'Heave'}, 'Heave, 10 s std of GNSS height [cm]', [AQUA]),
        ), width='stretch')
        st.caption('Grey bands = stations. Drag to pan, scroll to zoom the time axis.')

    takeaway('At the quay (Q) the ship lies still. It sails out to a few **stations** (numbers on the map, grey bands '
             'in the plots), stops there for between a quarter of an hour and several hours, and sails back. Away from '
             'the quay the motion jumps, and it is usually largest at the stations, where the ship lies still in the '
             'waves. Those station periods are what we compare with the buoys.')

    with st.expander('Show details: phase table and raw 100 Hz signals'):
        st.markdown('**Phases of the day** (statistics over every raw sample in the phase)')
        show_table(p[['phase', 'start', 'end', 'minutes', 'roll_std', 'pitch_std', 'heave_hs_m']])
        st.markdown('**Raw 100 Hz signals, every CSV column** (2-minute window)')
        default = (stations.start_l.iloc[0] + pd.Timedelta('10min')) if len(stations) else s.local.iloc[len(s) // 2]
        t = st.slider('Window start', first, last - timedelta(minutes=2), default.floor('min').to_pydatetime(),
                      step=timedelta(seconds=10), format='HH:mm:ss')
        raw = load_raw(day, pd.Timestamp(t).tz_localize(data.TZ))
        if raw.empty:
            st.info('No raw data in this window.')
        else:
            sec = '%H:%M:%S'
            st.altair_chart(alt.vconcat(
                panel(raw, {'roll': 'Roll', 'pitch': 'Pitch'}, 'Roll & pitch [°]', [BLUE, ORANGE], fmt=sec),
                panel(raw, {'height_cm': 'Height'}, 'GNSS height [cm]*', [AQUA], fmt=sec,
                      y=alt.Y('value:Q', title='GNSS height [cm]*', scale=alt.Scale(zero=False))),
                panel(raw, {'heading': 'Heading'}, 'Heading [°]', [BLUE], points=True, fmt=sec,
                      y=alt.Y('value:Q', title='Heading [°]', scale=alt.Scale(zero=False))),
                panel(raw, {'vn_cms': 'North', 've_cms': 'East', 'vd_cms': 'Down'}, 'Velocity [cm/s]*',
                      [BLUE, ORANGE, AQUA], fmt=sec),
                panel(raw, {'roll_rate': 'Roll', 'pitch_rate': 'Pitch', 'yaw_rate': 'Yaw'}, 'Rotation rate [°/s]*',
                      [BLUE, ORANGE, AQUA], fmt=sec),
                panel(raw, {'c4': 'Column 4'}, 'Column 4, unknown integer', [GREY], fmt=sec),
            ).resolve_scale(x='shared', color='independent'), width='stretch')
            st.caption('Position (columns 1–2) is on the map. * Meaning inferred from the data, not documented.')
    next_step(2)


def page_quality():
    step(3, 'Can we trust the data?',
         'Before reading anything into the data we check it: is every file there, is anything missing or impossible, '
         "and what are the sensors' limits? This is data observability.")
    a = load_all()
    rows = []
    for d in DAYS:
        c = data.ship_checks(load_day(d), DAYS[d])
        c['quay_heave_4σ_m'] = 4 * data.within(a[(a.day == d) & (a.phase == 'Quay')], 'height') / 100
        ok = (c['complete_pct'] == 100 and c['min_samples_per_file'] == 1000 and c['gaps_over_11s'] == 0
              and c['nan_values'] == 0 and c['out_of_range'] == 0 and c['csv_columns'] == len(data.COLS))
        rows.append({'day': d, 'status': '✅ OK' if ok else '⚠️ check', **c})
    ship = pd.DataFrame(rows)
    st.markdown('**Ship logs, one row per day**')
    show_table(ship[['day', 'status', 'files', 'complete_pct', 'gaps_over_11s', 'out_of_range', 'quay_heave_4σ_m']])
    st.caption('quay_heave_4σ_m = the "wave height" the ship reports while tied up, where there are no waves: '
               'the GNSS noise floor.')

    st.markdown('**Buoy wave records per day** (darker = more complete)')
    q = data.buoy_checks(load_buoys())
    q['label'] = (q.records.astype(str) + '/' + q.expected.astype(str)).where(q.records > 0, 'none')
    base = alt.Chart(q).encode(x=alt.X('day:N', title=None, axis=alt.Axis(labelAngle=0, orient='top')),
                               y=alt.Y('buoy:N', title=None))
    st.altair_chart((base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color('coverage_pct:Q', title='coverage %',
                        scale=alt.Scale(domain=[0, 100], range=['#cde2fb', '#1c5cab'])),
        tooltip=['buoy', 'day', 'records', 'expected', alt.Tooltip('max_gap_min:Q', format='.0f'), 'out_of_range'],
    ) + base.mark_text(fontSize=12).encode(
        text='label:N', color=alt.when('datum.coverage_pct > 50').then(alt.value('white')).otherwise(alt.value(INK)))
    ).properties(width=alt.Step(110), height=alt.Step(30)))

    st.markdown(
        '**Known gaps**\n'
        '- **30 Oct:** the plan said buoys D and B; the ship actually stopped at D (Station 1) and A (Stations 2–3).\n'
        '- **2 Nov:** buoy D, where the ship sat, has no wave data; the nearest buoy with data is A, about 7 km away.\n'
        '- **3 Nov:** the ship was off Kristiansund; the nearest buoy, G, is 24–65 km away inside Halsafjorden.')
    takeaway('The ship logs are complete: every 10 s file is there with all 1000 samples, with no gaps and no '
             'impossible values. Three limits to keep in mind: (1) the GNSS height only updates about 10 times a '
             'second, not 100; (2) even tied up at the quay the ship "heaves" about 0.15 m (4σ), which is GNSS noise; '
             '(3) buoys C, D and F have no wave data in November, so 2 and 3 Nov can only be compared with buoys '
             'several km away.')

    with st.expander('Show details: every check, sensor update rates, lineage'):
        show_table(ship)
        st.caption('expected = files a 10 s cadence gives over the logged span · every file should hold 1000 samples '
                   f'(10 s × 100 Hz) in {len(data.COLS)} columns · out_of_range uses the limits at the top of '
                   'tmr12/data.py.')
        rates = {}
        for d in DAYS:
            st1 = load_phases(d).query('phase == "Station 1"')
            if len(st1):
                rates[d] = data.update_rates(load_raw(d, (st1.start.iloc[0] + pd.Timedelta('10min')).floor('min')))
        st.markdown('**Effective update rate per raw column** (changes per second, 2 minutes into Station 1)')
        show_table(pd.DataFrame(rates).T.rename_axis('day').reset_index())
        st.caption('Rows are logged at 100 Hz, but not every sensor delivers 100 Hz. Roll, pitch and the rotation rates '
                   'change about 95 times a second (true 100 Hz). The GNSS columns (position, height, velocities, '
                   'column 4) change about 10 times a second on 30 Oct–2 Nov and about 25 on 3 Nov: the GNSS output '
                   'rate differs between days, and its values are repeated in between. Heave 4σ comes from GNSS height, '
                   'so it rests on that rate (fine for waves of 0.1–1 Hz).')
        st.markdown('**Buoy checks**')
        show_table(q.drop(columns='label'))
        st.caption('"none" = only raw motion files exist for that buoy and month (no Hm0/Tp/Mdir). '
                   'mooring_spread_m = farthest the buoy sat from its median position. Positions are stored with 3 '
                   'decimals (about 110 m), so spreads at that level are rounding, not drift.')
        st.markdown(
            '**Undocumented columns:** ship CSV columns 3–7 and 11–13 are interpreted from the data (see README).\n\n'
            '**Lineage.** Ship: `data/raw/ship/<day>/<hh>/*.csv` (100 Hz, 10 s per file) → '
            '`data/processed/ship_10s_<day>.parquet` (one row per file) → phases, 10-min blocks, heading blocks → pages. '
            'Buoys: MET Norway THREDDS `*_wave.nc` → `cut_buoys.py` (trial windows) → `buoy_wave_params_trial.csv` '
            '(10-min records, UTC, time = end of period) → pages.')
    next_step(3)


def page_ship():
    step(4, 'How much does the ship move, and when?',
         'Every 10 s file gives one value per variable; here are all days and all files. The boxes show how those '
         'values spread per day, split into three phases: tied up at the quay, sailing in transit, and stopped at a '
         'station. Pick another variable to compare.')
    a = load_all()
    var = st.selectbox('Variable', list(SHIP_VARS), index=list(SHIP_VARS).index('roll_std'),
                       format_func=SHIP_VARS.get, key='eda-var')
    st.altair_chart(alt.Chart(a[['day', 'phase', var]]).mark_boxplot(
        color=BLUE, size=28, outliers={'size': 4, 'opacity': 0.3}).encode(
        x=alt.X('day:N', title=None, axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y(f'{var}:Q', title=SHIP_VARS[var]),
    ).properties(width=300, height=260).facet(column=alt.Column('phase:N', title=None, sort=PHASES)))
    st.caption('Box = middle 50 % of the 10 s values, line = median, whiskers = 1.5 × IQR, dots = outliers.')

    med = a.groupby('phase').roll_std.median()
    top = a[a.phase == 'Station'].groupby('day').roll_std.median().idxmax()
    takeaway(f'At the quay the ship is almost still (typical roll σ {med.Quay:.2f}°). In transit it moves a little '
             f'({med.Transit:.2f}°), and at the stations most ({med.Station:.2f}°): stopped, it lies in the waves and '
             f'rolls. Station motion also changes from day to day with the sea; it was highest on '
             f'{pd.Timestamp(top):%d %b}. Heave at the quay is not zero (about 2 cm): that is GNSS noise, not waves.')

    with st.expander('Show details: statistics and correlations'):
        st.markdown(f'**{SHIP_VARS[var]} per day and phase**')
        show_table(a.groupby(['day', 'phase'])[var].describe().reset_index())
        st.markdown('**Correlation between ship variables** (transit and stations; the still quay would inflate them)')
        st.altair_chart(corr_heatmap(a[a.phase != 'Quay'], SHIP_VARS), width='content')
        st.caption('Pearson r. Std columns measure motion inside each 10 s; means show slow offsets (list, trim, tide).')
    next_step(4)


def page_waves():
    step(5, 'How big were the waves?',
         'The E39 buoys are moored in the fjords and report the sea state every 10 minutes. The key number is '
         '**Hm0, the significant wave height**: roughly the average height of the highest third of the waves.')
    b = load_buoys()
    b = b.assign(local=local(b.time))
    b['day'] = b.local.dt.strftime('%Y-%m-%d')
    param = st.selectbox('Parameter', list(BUOY_PARAMS), format_func=BUOY_PARAMS.get, key='eda-param')
    direction = param in ('Mdir', 'ThTp')
    d = b[['local', 'day', 'buoy', 'short', param]]
    c = alt.Chart(d)
    c = c.mark_circle(size=14, opacity=0.8) if direction else c.mark_line(strokeWidth=1.5)
    st.altair_chart(c.encode(
        x=alt.X('local:T', title=None, axis=alt.Axis(format='%H:%M')),
        y=alt.Y(f'{param}:Q', title=BUOY_PARAMS[param],
                scale=alt.Scale(domain=[0, 360]) if direction else alt.Scale(zero=True),
                axis=alt.Axis(values=[0, 90, 180, 270, 360]) if direction else alt.Axis()),
        color=alt.Color('short:N', title='Buoy', legend=alt.Legend(orient='top'),
                        scale=alt.Scale(domain=list(BUOY_COLORS), range=list(BUOY_COLORS.values()))),
        tooltip=['buoy', alt.Tooltip('local:T', format='%d %b %H:%M', title='time'),
                 alt.Tooltip(f'{param}:Q', format='.2f')],
    ).properties(width=240, height=220).facet(column=alt.Column('day:N', title=None)).resolve_scale(x='independent'))
    st.caption('A, B, C = Sulafjorden · D = Breisundet · F = Vartdalsfjorden · G = Halsafjorden (near Kristiansund). '
               'C, D and F have no wave data in November.')
    takeaway('The waves were small during the trial: Hm0 stayed below 0.8 m at every buoy. The exposed buoy **D** in '
             'Breisundet had the biggest waves (about 0.6–0.7 m), the sheltered buoys A and B in Sulafjorden about '
             '0.2 m. Keep this difference between sites in mind: it comes back in step 6.')

    with st.expander('Show details: distributions, wave climate, correlations'):
        left, right = st.columns(2)
        with left:
            st.markdown(f'**{param} per buoy**')
            stats = b.groupby('buoy')[param].describe()
            if direction:
                stats['circular_mean'] = b.groupby('buoy')[param].apply(data.circmean)
                st.caption('Directions wrap at 360°, so there is no boxplot and the plain mean/std below are not '
                           'meaningful: use circular_mean.')
            else:
                st.altair_chart(alt.Chart(d).mark_boxplot(color=BUOY, size=28, outliers={'size': 4, 'opacity': 0.4}).encode(
                    x=alt.X('short:N', title='Buoy', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(f'{param}:Q', title=BUOY_PARAMS[param])).properties(height=260), width='stretch')
            show_table(stats.reset_index())
        with right:
            st.markdown('**Wave climate: Hm0 vs Tp**')
            st.altair_chart(alt.Chart(b[['buoy', 'short', 'local', 'Hm0', 'Tp']]).mark_circle(
                size=24, opacity=0.7, color=BUOY).encode(
                x=alt.X('Tp:Q', title='Tp [s]'), y=alt.Y('Hm0:Q', title='Hm0 [m]'),
                tooltip=['buoy', alt.Tooltip('local:T', format='%d %b %H:%M', title='time'),
                         alt.Tooltip('Hm0:Q', format='.2f'), alt.Tooltip('Tp:Q', format='.1f')],
            ).properties(width=150, height=130).facet(facet=alt.Facet('short:N', title=None), columns=3))
            st.caption('Short Tp with small Hm0 = local wind sea; long Tp = swell reaching into the fjord.')
        st.markdown('**Correlation between wave parameters** (all buoys and days; directions left out, they wrap at 360°)')
        st.altair_chart(corr_heatmap(b, {k: k for k in data.WAVE if k not in ('Mdir', 'ThTp')}), width='content')
        st.caption('Each record covers the 10 minutes ending at its time.')
    next_step(5)


def page_ship_vs_buoy():
    day = st.session_state.day
    step(6, 'Does the ship move more when the waves are bigger?',
         'We cut the ship data into the same 10-minute blocks the buoys use, and pair each block with the nearest buoy '
         f'that has data (within {MAX_KM:.0f} km). For the ship we use **heave 4σ**: four times the spread of its '
         "up-and-down motion, the same recipe as the buoy's Hm0. If the ship were a perfect wave buoy, every point "
         'would sit on the dashed line.')
    m = load_buoy_blocks()
    near = m[m.dist_km <= MAX_KM]
    st.altair_chart(ship_buoy_scatter(near))
    takeaway(f'Roughly, yes: blocks with bigger waves give more heave (correlation r = '
             f'{near.heave_hs_m.corr(near.Hm0):.2f}). But look at the colours: all the big-wave points are at buoy D '
             'and all the small ones at A, so part of the link is simply *where* the ship was. On 2 Nov the only buoy '
             "with data, A, was 6–7 km away, and the ship heaved about twice what A's waves suggest: the buoy was not "
             "measuring the ship's sea.")

    with st.expander('Show details: other ship responses, correlations, one day in detail'):
        c1, c2 = st.columns(2)
        max_km = c1.slider('Only blocks within this distance of their buoy [km]', 0.5, 50.0, MAX_KM, 0.5, key='eda-km')
        resp = c2.selectbox('Ship response', list(RESPONSE), format_func=RESPONSE.get, key='eda-resp')
        mm = m[m.dist_km <= max_km]
        if mm.empty:
            st.info('No blocks within that distance.')
        else:
            st.altair_chart(ship_buoy_scatter(mm, resp))
            st.caption(f'{len(mm)} ten-minute blocks, each compared with the nearest buoy that has data at that time.')
            st.markdown('**Correlation, ship vs buoy**')
            st.altair_chart(corr_heatmap(mm, {'Hm0': 'Buoy Hm0', 'Hm0a': 'Buoy Hm0a', 'Hm0b': 'Buoy Hm0b',
                                              'Tp': 'Buoy Tp', 'Tm02': 'Buoy Tm02', 'SprTp': 'Buoy SprTp',
                                              'heave_hs_m': 'Ship heave 4σ', 'roll_std': 'Ship roll σ',
                                              'pitch_std': 'Ship pitch σ', 'speed_kn': 'Ship speed',
                                              'dist_km': 'Distance'}), width='content')
            st.caption('Read "Distance" as a site effect: the high-Hm0 blocks are all at the exposed buoy D (visited at '
                       'close range), the low ones at sheltered buoy A (up to 8 km away).')

        st.markdown('**One day in detail** (change the day in the sidebar)')
        st.caption(day_caption(day))
        buoy = pick_buoy(day)
        if buoy is not None:
            s, p = load_day(day), load_phases(day)
            today = data.day_buoys(s, load_buoys())
            mb = data.with_buoy(load_blocks(day), today, buoy)
            short = buoy.split()[1]
            stations = p[p.phase.str.startswith('Station')]
            st.altair_chart(stack(
                stations[(stations.end_l >= mb.local.min()) & (stations.start_l <= mb.local.max())],
                panel(mb, {'Hm0': f'Buoy {short} Hm0', 'heave_hs_m': 'Ship heave 4σ'}, 'Wave / heave height [m]',
                      [BUOY, AQUA]),
                panel(mb, {'roll_std': 'Roll σ', 'pitch_std': 'Pitch σ'}, 'Ship roll & pitch, 10 min std [°]',
                      [BLUE, ORANGE]),
                panel(mb, {'Tp': 'Tp'}, f'Buoy {short} peak period Tp [s]', [BUOY]),
                panel(mb, {'dist_km': 'Distance'}, f'Ship to buoy {short} [km]', [GREY]),
            ), width='stretch')
            st.caption('10-minute blocks on the buoy time grid; grey bands = stations. Ship heave 4σ is computed from '
                       "the GNSS height: it is the ship's response, not the sea surface.")
            show_table(data.station_table(s, p, today, buoy))
    next_step(6)


def page_heading():
    day = st.session_state.day
    step(7, 'Does it matter which way the waves hit the ship?',
         'The test plan turned the ship to different headings relative to the waves. We find the periods where the '
         "heading stayed steady for at least 5 minutes, and compare the ship's heading with the wave direction (Mdir) "
         'from the buoy. **0°** = waves straight on the bow (head seas), **±90°** = from the side, **±180°** = from '
         f'behind (following seas). Each point is one steady period, within {MAX_KM:.0f} km of its buoy.')
    ticks = sorted(data.NOMINAL + [-180])
    rows = []
    for d in DAYS:
        c = buoy_choice(d)
        ok = c[c.has_data]
        if len(ok):
            rows.append(load_heading_blocks(d, ok.index[0]).assign(day=d, buoy=ok.index[0]))
    allb = pd.concat(rows) if rows else pd.DataFrame()
    allb = allb[allb.dist_km <= MAX_KM] if len(allb) else allb
    if allb.empty:
        st.info('No steady-heading blocks near a buoy.')
    else:
        # one panel per day (shared axes): more than 3 day colours in one scatter can't be told apart
        st.altair_chart(alt.Chart(allb).mark_circle(size=70, opacity=0.85, color=BLUE, stroke=SURFACE,
                                                    strokeWidth=1).encode(
            x=alt.X('rel_Mdir:Q', title='Waves relative to the bow [°]  (+ = from starboard)',
                    scale=alt.Scale(domain=[-180, 180]), axis=alt.Axis(values=ticks)),
            y=alt.Y('roll_std:Q', title='Roll std over the block [°]'),
            tooltip=['day', 'station', alt.Tooltip('start:T', format='%H:%M'), alt.Tooltip('minutes:Q', format='.0f'),
                     alt.Tooltip('heading:Q', format='.0f'), alt.Tooltip('rel_Mdir:Q', format='.0f'),
                     alt.Tooltip('roll_std:Q', format='.2f'), 'buoy', alt.Tooltip('dist_km:Q', format='.1f')],
        ).properties(height=260, width=340).facet(facet=alt.Facet('day:N', title=None), columns=3))
    takeaway('Only one station gives a clean answer: 31 Oct at buoy D, with the ship less than 1 km from the buoy. '
             'There the ship rolled about 1.7–1.9° with head seas, 2.6–3.0° with waves from the side, and 1.0–1.5° '
             'with waves from behind, as you would expect for a ship. On the other days the buoy was mostly several '
             'km away, so its wave direction is a poor guide and no clear pattern shows.')

    with st.expander('Show details: one station in detail'):
        st.caption(day_caption(day) + ' · change the day in the sidebar')
        s, p = load_day(day), load_phases(day)
        stations = p[p.phase.str.startswith('Station')]
        buoy = pick_buoy(day)
        if buoy is not None and len(stations):
            c1, c2 = st.columns(2)
            station = c1.selectbox('Station', stations.phase)
            src = c2.radio('Wave direction from the buoy', ['Mdir', 'ThTp'], horizontal=True,
                           help='Mdir = mean wave direction, ThTp = direction at the spectral peak. Both are the '
                                'direction the waves come FROM, in degrees true.')
            r = stations[stations.phase == station].iloc[0]
            km = data.buoy_distances(r.lat, r.lon, load_buoys())[buoy]
            if km > 2:
                st.warning(f'{station} is {km:.0f} km from {buoy}: relative headings here are only indicative.')
            g = s[(s.time >= r.start) & (s.time < r.end)].copy()
            wave = data.day_buoys(s, load_buoys()).query('buoy == @buoy').set_index('time')[src]
            g['wave'] = wave.reindex(g.time.dt.tz_convert('UTC').dt.ceil('10min')).values
            g['rel'] = data.rel_heading(g.wave, g.heading)
            hb = load_heading_blocks(day, buoy)
            hb = hb[hb.station == station] if len(hb) else hb
            if len(hb):
                hb = hb.assign(nominal=hb[f'rel_{src}'].map(data.nearest_nominal), start_l=local(hb.start),
                               end_l=local(hb.end))
                bands = hb.assign(phase=[f'steady block, waves at {v:+.0f}° rel. bow' for v in hb[f'rel_{src}']])
            else:
                bands = pd.DataFrame({'start_l': pd.to_datetime([]), 'end_l': pd.to_datetime([]),
                                      'phase': pd.Series(dtype=str)})
            rules = alt.Chart(pd.DataFrame({'value': ticks})).mark_rule(color=GREY, strokeDash=[3, 3],
                                                                        opacity=0.6).encode(y='value:Q')
            st.altair_chart(stack(
                bands,
                panel(g, {'heading': 'Ship heading', 'wave': f'Waves from ({src}, buoy {buoy.split()[1]})'},
                      'Direction [° true]', [BLUE, BUOY], points=True, height=180,
                      y=alt.Y('value:Q', title='Direction [° true]', scale=alt.Scale(domain=[0, 360]),
                              axis=alt.Axis(values=[0, 90, 180, 270, 360]))),
                alt.layer(rules, panel(g, {'rel': 'Relative'},
                                       'Waves relative to the bow [°]  (0 = head seas, + = from starboard)',
                                       [BLUE], points=True, height=220,
                                       y=alt.Y('value:Q', title='[°]', scale=alt.Scale(domain=[-180, 180]),
                                               axis=alt.Axis(values=ticks)))),
            ), width='stretch')
            st.caption('Grey bands = detected steady-heading blocks (hover for the relative wave heading).')
            if hb.empty:
                st.info('No steady-heading blocks of 5 min or more in this station.')
            else:
                show_table(hb[['start', 'end', 'minutes', 'heading', src, f'rel_{src}', 'nominal', 'roll_std',
                               'pitch_std', 'heave_hs_m', 'Hm0', 'Tp', 'dist_km']])
                st.caption(f'rel_{src} = where the waves come from relative to the bow; "nominal" = closest test-plan '
                           'angle (0/15/30/60/90° starboard or port, 180°).')
    next_step(7)


def page_states():
    day = st.session_state.day
    out, tree, scores, per_state, table = run_cluster()
    acc = scores.accuracy
    step(8, "Can a model tell from the ship's motion whether it is in harbour, on DP or in transit?",
         'Here the ship is not told where it is. Each minute of the log is described by five numbers from its own '
         'motion: speed, roll σ, pitch σ, heave 4σ and heading change. A Gaussian mixture model groups similar '
         f'minutes into {cluster.N_STATES} states without any labels. The **correct timeline** (the phases from '
         'step 2: quay = harbour, station = DP, transit) is used only afterwards, to score the result. '
         'For comparison, a supervised **decision tree** learns the correct timeline from the same five numbers.',
         caption=day_caption(day))
    show_fig(cluster.fig_timeline(out[out.day == day]))
    st.caption(f'Top = correct timeline, middle = decision tree, bottom = clustering (both as the most common '
               f'state over {cluster.SMOOTH_MIN} min). Black ticks = minutes where the clustering is less than '
               f'{cluster.UNSURE:.0%} sure.')

    c = st.columns(4)
    c[0].metric('Minutes right, all days', f"{acc['Clustering, smoothed']:.1%}", border=True)
    for col, s in zip(c[1:], cluster.STATES):
        col.metric(f'{s} minutes found', f"{per_state.loc[s, 'recall']:.1%}", border=True,
                   help=f"{table.loc[s, s]} of {per_state.loc[s, 'minutes']} correct {s} minutes")
    off = table.stack()
    off = off[[a != b for a, b in off.index]]
    (true, pred), n = off.idxmax(), off.max()
    tree_miss = ((out.correct == true) & (out.tree_smooth == pred)).sum()
    takeaway(f"Mostly yes. The clustering gets {acc['Clustering, smoothed']:.1%} of all minutes right "
             f"without being told where the ship is. The biggest error: {n} {true} minutes called {pred}. "
             f"The supervised decision tree, which is told the answers for the other days, gets "
             f"{acc['Decision tree, smoothed']:.1%}, and it also calls {tree_miss} {true} minutes {pred}.")

    with st.expander('Show details: scores, confusion table, inputs per state, the decision tree'):
        st.markdown('**Overall scores** (raw = minute by minute, smoothed = most common state over '
                    f'{cluster.SMOOTH_MIN} min)')
        st.dataframe(scores.round(2))
        st.caption('Accuracy = share of minutes with the right state. ARI = adjusted Rand index, how well the two '
                   'timelines group the minutes the same way, whatever the names (1 = identical, 0 = random).')
        st.markdown('**Clustering per state** (smoothed)')
        st.dataframe(per_state.round(2))
        st.caption('Recall = share of the correct minutes of this state that the model found. Precision = share of '
                   'the minutes the model gave this state that are right. F1 = the two combined. Minutes = how many '
                   'minutes of this state the correct timeline has.')
        st.markdown('**Correct timeline (rows) vs clustering (columns)**, minutes')
        st.dataframe(table)
        show_fig(cluster.fig_features(out))
        st.markdown('Speed is needed to separate transit from DP.\n\n'
                    'In calm, sheltered water DP and harbour look the same from motion alone, so the model calls '
                    'such DP minutes harbour (for example 31 Oct, station 2).')
        st.markdown('**The decision tree** (supervised)')
        show_fig(cluster.fig_tree(tree))
        st.caption(f'Each day is predicted by a tree trained on the correct timeline of the other '
                   f'{out.day.nunique() - 1} days only, so it is always tested on a day it never saw. The tree shown '
                   f'here is fitted on all days. Read it top to bottom: each box asks one yes/no question (left = '
                   f'yes), and the colour is the state it predicts. Its first question is about '
                   f'{cluster.FEATURES[tree.tree_.feature[0]]}.')
    next_step(8)


def page_conclusion():
    out, tree, scores, per_state, table = run_cluster()
    acc = scores.accuracy
    step(9, 'So, can a model tell what the ship is doing?',
         f"**Mostly, yes.** From its own motion alone, an unsupervised clustering labels "
         f"{acc['Clustering, smoothed']:.1%} of the minutes with the right operating state, and a supervised "
         f"decision tree {acc['Decision tree, smoothed']:.1%}. A timeline like this tells you which part of a real "
         'operation to compare with which simulation.')
    st.markdown(
        '#### What each step showed\n'
        '1. **The data is usable, with known limits** (step 3): the ship logs are complete, and the sensor limits are '
        'known.\n'
        '2. **Each state moves differently** (step 4): almost still in harbour, a little in transit, most on DP at '
        'the stations. That difference is what the models use.\n'
        '3. **The sea state matters** (steps 5 to 7): the ship moves more in bigger waves and with waves from the '
        'side, so what a DP minute looks like depends on the weather that day.\n'
        f"4. **A model can label the states** (step 8): the clustering finds {per_state.loc['Harbour', 'recall']:.1%} "
        f"of the harbour minutes, {per_state.loc['DP', 'recall']:.1%} of the DP minutes and "
        f"{per_state.loc['Transit', 'recall']:.1%} of the transit minutes. Most misses are calm DP minutes called "
        'harbour, and the decision tree makes the same mistake.\n\n'
        '#### Limits\n'
        '- The correct timeline is itself made by rules (speed, and the first and last stop of the day is the quay), '
        "not taken from the DP system's own log.\n"
        '- In calm, sheltered water, DP and harbour look the same from motion alone.\n'
        f'- Only {len(DAYS)} days from one ship.\n'
        '- Several ship CSV columns are undocumented; their meaning is inferred from the data.\n\n'
        '#### Next steps\n'
        "- Check the timeline against the DP system's own log or the bridge log.\n"
        '- Test the model on more days and on other ships.\n'
        '- Tell calm DP from harbour: add a signal such as thruster use, if it is logged, or let the model know that '
        'states last for a while (a hidden Markov model).\n'
        '- Use the timeline to pick the matching simulation case for each part of a voyage.')


# ---------- app ----------

STEPS = [  # (page, title, icon, what it answers): the story, in order
    (page_intro, 'The question', '❓', ''),
    (page_day, 'One day at sea', '🗺️', 'what the ship data looks like for a single day'),
    (page_quality, 'Can we trust the data?', '✅', 'checks for gaps, bad values and sensor limits'),
    (page_ship, 'How the ship moves', '🚢', 'roll, pitch and heave in each operating state'),
    (page_waves, 'The waves at the buoys', '🌊', 'the sea state the ship worked in'),
    (page_ship_vs_buoy, 'Ship vs buoy', '⚖️', 'does the ship move more when the waves are bigger?'),
    (page_heading, 'Heading vs waves', '🧭', 'does it matter which way the waves hit the ship?'),
    (page_states, 'Machine learning: operating state', '🤖', 'can a model tell harbour, DP and transit apart?'),
    (page_conclusion, 'What we learned', '🏁', 'the answer, and its limits'),
]
PAGES = [st.Page(f, title=f'{i}. {t}', icon=icon, default=i == 1) for i, (f, t, icon, _) in enumerate(STEPS, 1)]

st.sidebar.selectbox('Day (used in steps 2, 6, 7, 8)', list(DAYS), key='day',
                     format_func=lambda d: f'{pd.Timestamp(d):%a %d %b %Y}')
nav = st.navigation(PAGES)
st.sidebar.caption('Buoy data from the Norwegian Meteorological Institute (NLOD / CC BY 3.0 NO). '
                   'Map tiles © OpenStreetMap contributors, OpenSeaMap, Esri.')
nav.run()
