import pandas as pd
import numpy as np


from datetime import timedelta


def create_list_lat_long(df_data):
    lats = df_data['latitude'].unique()
    lons = df_data['longitude'].unique()

    lat_long = []

    for lat in lats:
        for lon in lons:
            df = df_data[(df_data['latitude'] == lat) &
                                    (df_data['longitude'] == lon)]
            if df.shape[0] != 0:

                lat_long.append((lat,lon))

    return(lat_long)


def create_tp_daily_summary(df_data, year = 2025, threshold = 0.1):
    # creating `tp_mm`
    df = df_data.copy()
    df['tp_mm'] = 1000*df['tp']

    # Creating valid date column
    df['valid_time_dt'] = pd.to_datetime(df['valid_time'])
    df['valid_date'] = df['valid_time_dt'].dt.date
    mask_hour_0 = df['valid_time_dt'].dt.hour == 0
    df.loc[mask_hour_0, 'valid_date'] = df.loc[mask_hour_0, 'valid_date'] - timedelta(days=1)

    # thresholding and rounding tp_mm
    df.loc[df['tp_mm'] < threshold, 'tp_mm'] = 0.0
    df['tp_mm'] = np.round(df['tp_mm'], 1)

    # computing daily aggregate
    df['valid_date_dt'] = pd.to_datetime(df['valid_date'])
    df['year'] = df['valid_date_dt'].dt.year

    df_daily_tp_mm = (df[df['year'] == year]
                  .groupby('valid_date_dt')['tp_mm']
                  .max()
                  ).reset_index()
    return(df_daily_tp_mm)

def create_tp_monthly_aggregate(df_data):
    df_data['month_int'] = df_data['valid_date_dt'].dt.month
    df_data['month_str'] = df_data['valid_date_dt'].dt.strftime('%b')

    tp_monthly_aggregate = (df_data
                            .groupby('month_int')
                            .agg(
                                month_str = ('month_str', 'first'),
                                monthly_tp_mm = ('tp_mm', 'sum')
                                )
                                ).reset_index()
    
    return(tp_monthly_aggregate)

