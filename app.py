"""TMR12 wave-radar data explorer.  Run: streamlit run app.py"""
import warnings
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from tmr12 import data, maps

st.set_page_config(page_title='TMR12 wave radar data', page_icon='🌊', layout='wide')
alt.data_transformers.disable_max_rows()
warnings.filterwarnings('ignore', message='Automatically deduplicated selection')   # intended: one shared x-zoom
BLUE, ORANGE, AQUA, BUOY, GREY, SHADE = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#9e9d98', '#ecebe6'
DAY_COLORS = [BLUE, ORANGE, AQUA]   # categorical slots 1-3: validated for scatter (all-pairs) use
DAYS = data.days()


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
def load_raw(day, start):
    raw = data.read_raw(DAYS[day], start, start + pd.Timedelta('2min'))
    raw['local'] = local(raw.time)
    return raw


# ---------- chart helpers ----------

def panel(df, series, y_title, colors, points=False, height=140, y=None):
    """One time-series panel; series = {column: label}."""
    names = list(series.values())
    d = df[['local', *series]].rename(columns=series)
    c = alt.Chart(d).transform_fold(names, as_=['series', 'value'])
    c = c.mark_circle(size=8, opacity=0.8) if points else c.mark_line(strokeWidth=1.5)
    return c.encode(
        x=alt.X('local:T', title=None, axis=alt.Axis(format='%H:%M')),
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
    return alt.vconcat(*[alt.layer(shade, c.add_params(zoom)) for c in panels]).resolve_scale(x='shared')


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


def header(day, title):
    st.title(title)
    st.caption(f'{pd.Timestamp(day):%A %d %B %Y} · {data.PLAN.get(day, "")} · data/raw/ship/{DAYS[day].name}/')


# ---------- pages ----------

def page_day():
    day = st.session_state.day
    s, p, b = load_day(day), load_phases(day), load_buoys()
    header(day, 'Day overview')

    active = p[p.phase != 'Quay']
    first, last = s.local.min().to_pydatetime(), s.local.max().to_pydatetime()
    lo = max(first, (active.start_l.min() - pd.Timedelta('30min')).to_pydatetime()) if len(active) else first
    hi = min(last, (active.end_l.max() + pd.Timedelta('30min')).to_pydatetime()) if len(active) else last
    t0, t1 = st.slider('Time window (local time)', first, last, (lo, hi), step=timedelta(minutes=5), format='HH:mm')
    w = s[(s.local >= t0) & (s.local <= t1)]

    left, right = st.columns([1, 1.25])
    with left:
        st_folium(maps.track_map(w, p, b), height=520, use_container_width=True, returned_objects=[],
                  key=f'map-{day}-{t0}-{t1}')
        show_table(p[['phase', 'start', 'end', 'minutes', 'roll_std', 'pitch_std', 'heave_hs_m']])
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

    with st.expander('Raw 100 Hz signals (2-minute window)'):
        stations = p[p.phase.str.startswith('Station')]
        default = (stations.start_l.iloc[0] + pd.Timedelta('10min')) if len(stations) else s.local.iloc[len(s) // 2]
        t = st.slider('Window start', first, last - timedelta(minutes=2), default.floor('min').to_pydatetime(),
                      step=timedelta(seconds=10), format='HH:mm:ss')
        raw = load_raw(day, pd.Timestamp(t).tz_localize(data.TZ))
        if raw.empty:
            st.info('No raw data in this window.')
        else:
            st.altair_chart(alt.vconcat(
                panel(raw, {'roll': 'Roll', 'pitch': 'Pitch'}, 'Roll & pitch [°]', [BLUE, ORANGE]),
                panel(raw, {'height_cm': 'Height'}, 'GNSS height [cm]', [AQUA]),
            ).resolve_scale(x='shared'), width='stretch')


def page_buoys():
    day = st.session_state.day
    s, p, b = load_day(day), load_phases(day), load_buoys()
    header(day, 'Ship vs buoys')
    buoy = pick_buoy(day)
    if buoy is None:
        return
    today = data.day_buoys(s, b)
    m = data.with_buoy(load_blocks(day), today, buoy)
    short = buoy.split()[1]
    stations = p[p.phase.str.startswith('Station')]
    st.altair_chart(stack(
        stations[(stations.end_l >= m.local.min()) & (stations.start_l <= m.local.max())],
        panel(m, {'Hm0': f'Buoy {short} Hm0', 'heave_hs_m': 'Ship heave 4σ'}, 'Wave / heave height [m]', [BUOY, AQUA]),
        panel(m, {'roll_std': 'Roll σ', 'pitch_std': 'Pitch σ'}, 'Ship roll & pitch, 10 min std [°]', [BLUE, ORANGE]),
        panel(m, {'Tp': 'Tp'}, f'Buoy {short} peak period Tp [s]', [BUOY]),
        panel(m, {'dist_km': 'Distance'}, f'Ship to buoy {short} [km]', [GREY]),
    ), width='stretch')
    st.caption('10-minute blocks on the buoy time grid. Ship heave 4σ is computed from the GNSS height (column 3): '
               'it is the ship\'s response, not the sea surface, and about 0.1 m of it is GNSS noise (see the quay).')
    st.subheader('Per station')
    show_table(data.station_table(s, p, today, buoy))


def page_heading():
    day = st.session_state.day
    s, p = load_day(day), load_phases(day)
    header(day, 'Heading vs waves')
    stations = p[p.phase.str.startswith('Station')]
    buoy = pick_buoy(day)
    if buoy is None or stations.empty:
        return
    c1, c2 = st.columns(2)
    station = c1.selectbox('Station', stations.phase)
    src = c2.radio('Wave direction from the buoy', ['Mdir', 'ThTp'], horizontal=True,
                   help='Mdir = mean wave direction, ThTp = direction at the spectral peak. Both are the direction '
                        'the waves come FROM, in degrees true.')
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
        bands = pd.DataFrame({'start_l': pd.to_datetime([]), 'end_l': pd.to_datetime([]), 'phase': pd.Series(dtype=str)})
    ticks = sorted(data.NOMINAL + [-180])
    rules = alt.Chart(pd.DataFrame({'value': ticks})).mark_rule(color=GREY, strokeDash=[3, 3], opacity=0.6).encode(
        y='value:Q')
    st.altair_chart(stack(
        bands,
        panel(g, {'heading': 'Ship heading', 'wave': f'Waves from ({src}, buoy {buoy.split()[1]})'},
              'Direction [° true]', [BLUE, BUOY], points=True, height=180,
              y=alt.Y('value:Q', title='Direction [° true]', scale=alt.Scale(domain=[0, 360]),
                      axis=alt.Axis(values=[0, 90, 180, 270, 360]))),
        alt.layer(rules, panel(g, {'rel': 'Relative'}, 'Waves relative to the bow [°]  (0 = head seas, + = from starboard)',
                               [BLUE], points=True, height=220,
                               y=alt.Y('value:Q', title='[°]', scale=alt.Scale(domain=[-180, 180]),
                                       axis=alt.Axis(values=ticks)))),
    ), width='stretch')

    st.caption('Grey bands = detected steady-heading blocks (hover for the relative wave heading).')
    st.subheader('Steady-heading blocks')
    if hb.empty:
        st.info('No steady-heading blocks of 5 min or more in this station.')
    else:
        show_table(hb[['start', 'end', 'minutes', 'heading', src, f'rel_{src}', 'nominal', 'roll_std', 'pitch_std',
                       'heave_hs_m', 'Hm0', 'Tp', 'dist_km']])
        st.caption(f'rel_{src} = where the waves come from relative to the bow; "nominal" = closest test-plan angle '
                   '(0/15/30/60/90° starboard or port, 180°).')

    st.subheader('Roll vs relative wave heading, all days')
    max_km = st.slider('Only blocks within this distance of their buoy [km]', 0.5, 50.0, 10.0, 0.5)
    rows = []
    for d in DAYS:
        c = buoy_choice(d)
        ok = c[c.has_data]
        if len(ok):
            rows.append(load_heading_blocks(d, ok.index[0]).assign(day=d, buoy=ok.index[0]))
    allb = pd.concat(rows) if rows else pd.DataFrame()
    allb = allb[allb.dist_km <= max_km] if len(allb) else allb
    if allb.empty:
        st.info('No blocks within that distance.')
        return
    st.altair_chart(alt.Chart(allb).mark_circle(size=70, opacity=0.85, stroke='#fcfcfb', strokeWidth=1).encode(
        x=alt.X(f'rel_{src}:Q', title='Waves relative to the bow [°]  (+ = from starboard)',
                scale=alt.Scale(domain=[-180, 180]), axis=alt.Axis(values=ticks)),
        y=alt.Y('roll_std:Q', title='Roll std over the block [°]'),
        color=alt.Color('day:N', scale=alt.Scale(domain=list(DAYS), range=DAY_COLORS[:len(DAYS)]),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['day', 'station', alt.Tooltip('start:T', format='%H:%M'), alt.Tooltip('minutes:Q', format='.0f'),
                 alt.Tooltip('heading:Q', format='.0f'), alt.Tooltip(f'rel_{src}:Q', format='.0f'),
                 alt.Tooltip('roll_std:Q', format='.2f'), 'buoy', alt.Tooltip('dist_km:Q', format='.1f')],
    ).properties(height=320), width='stretch')
    st.caption('Each point is one steady-heading block, compared with the nearest buoy that has data that day.')


# ---------- app ----------

st.sidebar.selectbox('Day', list(DAYS), key='day', format_func=lambda d: f'{pd.Timestamp(d):%a %d %b %Y}')
nav = st.navigation([st.Page(page_day, title='Day overview', icon='🗺️'),
                     st.Page(page_buoys, title='Ship vs buoys', icon='🌊'),
                     st.Page(page_heading, title='Heading vs waves', icon='🧭')])
st.sidebar.caption('Buoy data from the Norwegian Meteorological Institute (NLOD / CC BY 3.0 NO). '
                   'Map tiles © OpenStreetMap contributors, OpenSeaMap, Esri.')
nav.run()
