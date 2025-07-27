"""
PV power geenration using latitude and longitude data
"""

from __future__ import annotations

# === Imports =================================================================
from datetime import datetime
import requests

import pandas as pd
import matplotlib.pyplot as plt


from pvlib.iotools import get_cams
from pvlib.temperature import sapm_cell

plt.rc('text', usetex=False)


# === 1) Fetch irradiance & temperature =======================================

def fetch_data(lat: float, lon: float,
               start: datetime, end: datetime,
               email: str, identifier: str) -> pd.DataFrame:
    """Download hourly GHI (CAMS) and 2 m air temperature (ERA5 via open-meteo).

    Returns a DataFrame with columns ['ghi', 'temp_air'] indexed by UTC timestamps.
    """
    ghi = get_cams(lat, lon, start=start, end=end,
                   email=email, identifier=identifier,
                   time_step='1h', map_variables=True)[0]['ghi']

    url = (
        'https://archive-api.open-meteo.com/v1/era5'
        f'?latitude={lat}&longitude={lon}'
        f'&start_date={start.date()}&end_date={end.date()}'
        '&hourly=temperature_2m'
    )
    js = requests.get(url).json()['hourly']
    times = pd.to_datetime(js['time'], utc=True)
    temp_air = pd.Series(js['temperature_2m'], index=times, name='temp_air')

    df = pd.DataFrame({'ghi': ghi}).join(temp_air, how='inner')
    return df

# === 2) Simple DC PV model ====================================================

def compute_pv_power(df: pd.DataFrame, area: float) -> pd.DataFrame:
    """Add cell temperature and PV power estimate to df.

    Uses SAPM cell temperature model and a crude efficiency correction.
    """
    a, b, deltaT = -3.47, -0.0594, 36.0
    df['cell_temp'] = sapm_cell(
        poa_global=df['ghi'], temp_air=df['temp_air'], wind_speed=0.0,
        a=a, b=b, deltaT=deltaT,
    )
    p_coeff, eff= -0.005, 0.18  # example array
    df['pv_power_MW'] = (
        df['ghi'] * area * eff * (1 + p_coeff * (df['cell_temp'] - 25))
    ) * 1e-6
    return df

if __name__ == '__main__':
    lat, lon = 43.3026, 5.3691
    area = 1000 # in m2
    start = datetime(2023, 1, 23, 5)
    end   = datetime(2023, 12, 23, 5)

    df = fetch_data(lat, lon, start, end,
                    email='saifiitp16@gmail.com',
                    identifier='cams_radiation')
    df = compute_pv_power(df, area)