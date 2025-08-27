import streamlit as st

import pandas as pd
import numpy as np


@st.cache_data  # 
def load_data(fname_yearly_mean, fname_yearly_points):
    df_yearly_mean = pd.read_parquet(fname_yearly_mean)
    df_yearly_points = pd.read_parquet(fname_yearly_points)

    df_yearly_mean = df_yearly_mean[df_yearly_mean['year'] > df_yearly_mean['year'].min()]
    return(df_yearly_mean, df_yearly_points)


@st.cache_data  # 
def load_parquet_data(fname_parquet):
    df_data = pd.read_parquet(fname_parquet)
    return(df_data)

# Cache the data processing function to avoid recomputing when maps rerun
@st.cache_data
def process_data_for_year(selected_year, london_yearly_mean, puglia_yearly_points, london_yearly_points):
    if selected_year == None:
        london_sum = london_yearly_mean['yearly_tp_mm'].sum()
        puglia_yearly_tp_per_point = (puglia_yearly_points
                                  .groupby('coordinates_str')
                                  .agg(
                                      tp_mm_sum = ('yearly_tp_mm', 'sum'),
                                      lat = ('latitude', 'first'),
                                      lon = ('longitude', 'first')
                                      )
                                      .rename(columns={'lat' : 'latitude', 'lon' : 'longitude'})
                                      ).reset_index()
        london_yearly_tp_per_point = (london_yearly_points
                                      .groupby('coordinates_str')
                                      .agg(
                                          tp_mm_sum = ('yearly_tp_mm', 'sum'),
                                          lat = ('latitude', 'first'),
                                          lon = ('longitude', 'first')
                                          )
                                          .rename(columns={'lat' : 'latitude', 'lon' : 'longitude'})
                                          ).reset_index()    
    else:
        london_sum = london_yearly_mean[london_yearly_mean['year'] == selected_year]['yearly_tp_mm'].values[0]

        puglia_yearly_tp_per_point = (puglia_yearly_points[puglia_yearly_points['year'] == selected_year]
                                .groupby('coordinates_str')
                                .agg(
                                    tp_mm_sum = ('yearly_tp_mm', 'sum'),
                                    lat = ('latitude', 'first'),
                                    lon = ('longitude', 'first')
                                    )
                                    .rename(columns={'lat' : 'latitude', 'lon' : 'longitude'})
                                    ).reset_index()
        
        london_yearly_tp_per_point = (london_yearly_points[london_yearly_points['year'] == selected_year]
                                .groupby('coordinates_str')
                                .agg(
                                    tp_mm_sum = ('yearly_tp_mm', 'sum'),
                                    lat = ('latitude', 'first'),
                                    lon = ('longitude', 'first')
                                    )
                                    .rename(columns={'lat' : 'latitude', 'lon' : 'longitude'})
                                    ).reset_index()
    
    return london_sum, puglia_yearly_tp_per_point, london_yearly_tp_per_point
