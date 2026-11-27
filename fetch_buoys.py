"""Download the E39 buoy data for the trial days from MET Norway: python fetch_buoys.py [--force]

Writes data/raw/buoys/buoy_wave_params_trial.csv (wave + wind, 10-min records, UTC).
An existing file is kept unless --force is given, so a larger buoy file (more days/buoys) is not overwritten.
"""
import sys

import pandas as pd

from tmr12 import data, fetch

BUOYS = ['D', 'A']   # the buoys within 3 km of the DP stations
# Trial days with the ship near the buoys (on 3 Nov it was 45-117 km away), 05:00-18:50 UTC each day
DAYS = ['2023-10-30', '2023-10-31', '2023-11-02']


if __name__ == '__main__':
    if data.BUOY_CSV.exists() and '--force' not in sys.argv:
        sys.exit(f'{data.BUOY_CSV} exists already. Run with --force to download it again.')
    # one request per day, since the server keeps one file per month
    table = pd.concat([fetch.buoy_table(BUOYS, pd.Timestamp(f'{d} 05:00'), pd.Timestamp(f'{d} 18:50'))
                       for d in DAYS], ignore_index=True)
    data.BUOY_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(data.BUOY_CSV, index=False)
    print(table.groupby('buoy').agg(records=('time', 'size'), first=('time', 'min'), last=('time', 'max')))
