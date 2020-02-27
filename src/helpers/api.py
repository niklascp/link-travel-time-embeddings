import pandas as pd

API_BASE_URL = 'https://vehicletracker-api.azurewebsites.net/api/'

def calendar(from_date, to_date):
    url = f"{API_BASE_URL}services/calendar?fromDate={from_date.isoformat()}&toDate={to_date.isoformat()}"
    print(f"loading {url}")
    df = pd.read_json(url)
    df.set_index('date',inplace=True)
    return df

def link_travel_time(link_ref, from_time, to_time):
    url = f"{API_BASE_URL}services/link_travel_time?linkRef={link_ref}&fromTime={from_time.isoformat()}&toTime={to_time.isoformat()}"
    print(f"loading {url}")
    df = pd.read_json(url)
    df.index = pd.to_datetime(df['time'].cumsum(),unit='s')
    df.drop(columns = 'time',inplace=True)
    return df

def link_travel_time_n_preceding_normal_days(link_ref, time, n):
    url = f"{API_BASE_URL}services/link_travel_time_n_preceding_normal_days?linkRef={link_ref}&time={time.isoformat()}&n={n}"
    print(f"loading {url}")
    df = pd.read_json(url)
    df.index = pd.to_datetime(df['time'].cumsum(),unit='s')
    df.drop(columns = 'time',inplace=True)
    return df

def link_travel_time_special_days(link_ref, from_time, to_time):
    url = f"{API_BASE_URL}services/link_travel_time_special_days?linkRef={link_ref}&fromTime={from_time.isoformat()}&toTime={to_time.isoformat()}"
    print(f"loading {url}")
    df = pd.read_json(url)
    df.index = pd.to_datetime(df['time'].cumsum(),unit='s')
    df.drop(columns = 'time',inplace=True)
    return df
