"""Download E39 wave and wind data from MET Norway's THREDDS server (public web source).

The data are read with plain HTTP requests through the OPeNDAP 'ascii' service, so only the variables and
the time window we need are downloaded (a few kB instead of the full monthly files).
Data: Statens vegvesen / MET Norway, licence NLOD / CC BY 4.0.
"""
import re

import numpy as np
import pandas as pd
import requests

BASE = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39'
BUOYS = {'D': 'D_Breisundet', 'A': 'A_Sulafjorden', 'B': 'B_Sulafjorden', 'C': 'C_Sulafjorden'}
WAVE = ['Hm0', 'Hm0a', 'Hm0b', 'Tp', 'Tm02', 'Mdir', 'ThTp', 'SprTp']
WIND = ['WindSpeed', 'WindGust', 'WindDirection']


def parse_ascii(text):
    """Parse an OPeNDAP ascii answer into {variable: numpy array}.
    The answer has a header, a line of dashes, then blocks like 'Hm0[3]' followed by '0.5, 0.51, 0.52'."""
    body = text.split('-' * 10, 1)[-1]
    out = {}
    for name, values in re.findall(r'^([\w.]+)\[\d+\]\n([^\[]+?)(?=\n\s*\n|\Z)', body.strip() + '\n\n', re.M):
        out[name.split('.')[-1]] = np.array([float(v) for v in values.replace('\n', ' ').split(',')])
    return out


def get(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return parse_ascii(r.text)


def file_url(buoy, month, kind):
    """kind = 'wave' or 'wind'; month = pd.Timestamp of the first day of the month."""
    return f'{BASE}/{month:%Y}/{month:%m}/{month:%Y%m}_E39_{BUOYS[buoy]}_{kind}.nc.ascii'


def fetch(buoy, kind, start, end):
    """One buoy and one file kind ('wave' or 'wind') between start and end (UTC timestamps)."""
    month = start.replace(day=1, hour=0, minute=0, second=0)
    url = file_url(buoy, month, kind)
    t = get(url + '?time')['time']
    t = np.round(t / 60) * 60   # the ascii service prints 8 digits; the records are on whole minutes
    i0 = int(np.argmax(t >= start.timestamp()))
    i1 = int(np.nonzero(t <= end.timestamp())[0][-1])
    names = ['time', 'latitude', 'longitude'] + (WAVE if kind == 'wave' else WIND)
    query = ','.join(f'{n}%5B{i0}:1:{i1}%5D' for n in names)   # %5B and %5D are [ and ]
    d = get(f'{url}?{query}')
    df = pd.DataFrame({n: d[n] for n in names})
    df['time'] = pd.to_datetime(np.round(df.time / 60) * 60, unit='s')   # UTC, end of 10-min period
    return df


def buoy_table(buoys, start, end):
    """Wave and wind records for several buoys in the same format as data/raw/buoys/*.csv."""
    parts = []
    for short in buoys:
        wave = fetch(short, 'wave', start, end)
        wind = fetch(short, 'wind', start, end).drop(columns=['latitude', 'longitude'])
        df = wave.merge(wind, on='time', how='left')
        df.insert(1, 'buoy', 'E39 ' + BUOYS[short].replace('_', ' '))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
