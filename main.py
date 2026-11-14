"""CLI: build the processed 10 s summaries (data/processed) and PNG reports (reports/).

python main.py                 # every day under data/raw/ship
python main.py 2023-10-31      # one day
App: streamlit run app.py
"""
import sys

from tmr12 import data, report

if __name__ == '__main__':
    folders = data.days()
    for day in sys.argv[1:] or folders:
        s = data.load_day(day)
        assert s.time.is_monotonic_increasing, f'{day}: files out of order'
        p = data.phases(s)
        print(f'\n== {day} ({len(s)} files) -> {report.overview_png(s, p, day, folders[day].name)} ==')
        print(p[['phase', 'start', 'end', 'minutes', 'lat', 'lon', 'roll_std', 'heave_hs_m']].round(3)
              .to_string(index=False))
