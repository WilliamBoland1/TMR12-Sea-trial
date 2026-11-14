"""Folium map: ship track on a real basemap (Ålesund harbour, sea marks), stations and buoys."""
import folium
from folium.features import DivIcon

from tmr12.data import buoy_positions, circmean, day_buoys, dist_km

SATELLITE = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
SEAMARKS = 'https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png'
NEAR_KM = 15   # buoys closer than this to the track are included in the initial view


def badge(text, fill, ink, ring):
    return DivIcon(icon_size=(24, 24), icon_anchor=(12, 12), html=(
        f'<div style="background:{fill};color:{ink};border:2px solid {ring};border-radius:50%;width:24px;'
        f'height:24px;box-sizing:border-box;line-height:20px;text-align:center;font:bold 12px sans-serif">{text}</div>'))


def track_map(s, p, b, track_color='#2a78d6'):
    """s: 10 s rows to draw, p: phases of the day, b: all buoy records."""
    m = folium.Map(tiles='OpenStreetMap', control_scale=True)
    folium.TileLayer(SATELLITE, attr='Esri World Imagery', name='Satellite').add_to(m)
    folium.TileLayer(SEAMARKS, attr='OpenSeaMap', name='Sea marks', overlay=True).add_to(m)

    pts = s.iloc[::6][['lat', 'lon']].values.tolist()   # 1 point per minute
    folium.PolyLine(pts, color=track_color, weight=3, tooltip='Ship track').add_to(m)

    for r in p[p.phase != 'Transit'].itertuples():
        label = 'Q' if r.phase == 'Quay' else r.phase.split()[-1]
        tip = (f'<b>{r.phase}</b><br>{r.start:%H:%M}–{r.end:%H:%M}<br>'
               f'roll σ {r.roll_std:.2f}°, heave 4σ {r.heave_hs_m:.2f} m')
        folium.Marker([r.lat, r.lon], tooltip=tip, icon=badge(label, '#0b0b0b', '#fcfcfb', '#fcfcfb')).add_to(m)

    today = day_buoys(s, b)
    bounds = [[s.lat.min(), s.lon.min()], [s.lat.max(), s.lon.max()]]
    for name, pos in buoy_positions(b).iterrows():
        w = today[today.buoy == name]
        tip = (f'<b>Buoy {name}</b><br>Hm0 {w.Hm0.mean():.2f} m (max {w.Hm0.max():.2f})<br>Tp {w.Tp.mean():.1f} s'
               f'<br>Mdir {circmean(w.Mdir):.0f}° (from)' if len(w) else f'<b>Buoy {name}</b><br>no wave data this day')
        folium.Marker([pos.latitude, pos.longitude], tooltip=tip,
                      icon=badge(name.split()[1], '#fcfcfb', '#0b0b0b', '#6250d6')).add_to(m)  # buoy = violet
        if dist_km(s.lat, s.lon, pos.latitude, pos.longitude).min() < NEAR_KM:
            bounds += [[pos.latitude, pos.longitude]]

    folium.LayerControl(collapsed=False).add_to(m)
    lats, lons = zip(*bounds)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    return m
