import streamlit as st

import pandas as pd
import numpy as np

from src.data_visualization import plot_highlighted_bars_seaborn

from streamlit_folium import st_folium
import folium

st.set_page_config(
    page_title="London and Apulia Rainfall Analysis",
    # page_icon="🚀",
    layout="wide",  # This sets wide mode
    initial_sidebar_state="expanded"
)


@st.cache_data  # 
def load_data(fname):
    df = pd.read_parquet(fname)
    df = df[df['year'] > df['year'].min()]
    return(df)


# london_yearly_mean = pd.read_parquet('./data/data_london_yearly_mean.parquet')
london_yearly_mean = load_data('./data/data_london_yearly_mean.parquet')
puglia_yearly_mean = load_data('./data/data_puglia_yearly_mean.parquet')

london_yearly_mean['location'] = 'london'
puglia_yearly_mean['location'] = 'apulia'

df_plot = pd.concat([london_yearly_mean, puglia_yearly_mean])

# create selector
min_year = df_plot['year'].min()
max_year = df_plot['year'].max()

years = list(range(min_year, max_year))  # 2018 to 2025
options = [None] + years

col1, col2 = st.columns([0.2,0.8])
with col1:
    selected_year = st.selectbox(
        "Select Year to Highlight:",
        options,
        index=0,  # Default to None
        format_func=lambda x: "   " if x is None else str(x)
    )

if selected_year == None:
    london_sum = np.int64(london_yearly_mean['yearly_tp_mm'].sum())
    puglia_sum = np.int64(puglia_yearly_mean['yearly_tp_mm'].sum())
    diff = np.round((london_yearly_mean['yearly_tp_mm'].sum() - puglia_yearly_mean['yearly_tp_mm'].sum()),0)
else:
    london_sum = np.round(london_yearly_mean[london_yearly_mean['year'] == selected_year]['yearly_tp_mm'].values[0],0)
    puglia_sum = np.round(puglia_yearly_mean[puglia_yearly_mean['year'] == selected_year]['yearly_tp_mm'].values[0],0)
    diff = np.round(london_sum - puglia_sum, 0)


col1, col2, col3, col4 = st.columns(4)
with col1:
    if selected_year == None:
        st.markdown("**London Total Precipitation**  \n(mm)")
        metric_subtitle = 'Year range: '
        st.metric(label = f"{metric_subtitle} {london_yearly_mean['year'].min()} to {london_yearly_mean['year'].max()}", 
                value = london_sum,
                delta=diff)
    else:
        st.markdown("**London Total Precipitation**  \n(mm)")
        metric_subtitle = 'Year: '
        st.metric(label = f"{metric_subtitle} {selected_year}", 
                value = london_sum,
                delta=diff)

with col2:
    if selected_year == None:
        st.markdown("**Apulia Total Precipitation**  \n(mm)")
        metric_subtitle = 'Year range: '
        st.metric(label = f"{metric_subtitle} {puglia_yearly_mean['year'].min()} to {puglia_yearly_mean['year'].max()}", value = puglia_sum)
    else:
        st.markdown("**Apulia Total Precipitation**  \n(mm)")
        metric_subtitle = 'Year: '
        st.metric(label = f"{metric_subtitle} {selected_year}", 
                  value = puglia_sum)
    
fig, ax = plot_highlighted_bars_seaborn(df_plot, selected_year=selected_year, figsize=(12,3))
# Create columns to control width
col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
with col2:
    #st.pyplot(fig)
    st.pyplot(fig,use_container_width=True)

# data = pd.read_pickle('./data/london_tp_daily_data.pkl')

# data_london_yearly_sum_all_points = pd.read_parquet('./data/data_london_yearly_sum_all_points.parquet')

# data = data[['date', 'year', 'month', 'day', 'tp_mm_daily_sum', 'month_int']]
# years = np.sort(data_london_yearly_sum_all_points['year'].unique())
# select_years = set(years)

# option = st.selectbox(
#     "Select Year",
#     select_years,
# )

# london_yearly_tp_points = data_london_yearly_sum_all_points[data_london_yearly_sum_all_points['year'] == option]
# tp_yearly_mean = london_yearly_tp_points['yearly_tp_mm'].mean()



# # Create a base map centered on your data
# center_lat = london_yearly_tp_points['latitude'].mean()
# center_lon = london_yearly_tp_points['longitude'].mean()
# m = folium.Map(location=[center_lat, center_lon], 
#                zoom_start=10,
#                min_zoom=10,
#                max_zoom=10)

# # Define color mapping based on tp_mm values
# def get_color(tp_mm_value, tp_yearly_mean):
#     if tp_mm_value < tp_yearly_mean:
#         return 'orange'
#     else:
#         return 'blue'

# # Add circles to the map
# for idx, row in london_yearly_tp_points.iterrows():
#     folium.CircleMarker(
#         location=[row['latitude'], row['longitude']],
#         radius=13,
#         popup=f"{np.round(row['latitude'],1), np.round(row['longitude'],1)} \ntotal precipation (mm): {row['yearly_tp_mm']}",
#         color=get_color(row['yearly_tp_mm'], tp_yearly_mean),
#         fill=True,
#         fillColor=get_color(row['yearly_tp_mm'], tp_yearly_mean),
#         fillOpacity=0.7
#     ).add_to(m)


# col1, col2 = st.columns(2)
# with col1:
#     st.metric(label = 'London Mean Total Precipitation (in mm)', value = np.round(tp_yearly_mean))
#     st_folium(m, width=700, height=500)