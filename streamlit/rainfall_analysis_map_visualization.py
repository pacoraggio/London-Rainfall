import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from src.data_visualization import plot_highlighted_bars_seaborn, create_map_with_comparison

st.set_page_config(
    page_title="London and Apulia Rainfall Analysis Map Visualization",
    # page_icon="🚀",
    layout="wide",  # This sets wide mode
    initial_sidebar_state="expanded"
)


############# CACHED FUNCTIONS
@st.cache_data  # 
def load_data(fname_yearly_mean, fname_yearly_points):
    df_yearly_mean = pd.read_parquet(fname_yearly_mean)
    df_yearly_mean = df_yearly_mean[df_yearly_mean['year'] > df_yearly_mean['year'].min()]

    df_yearly_points = pd.read_parquet(fname_yearly_points)

    return(df_yearly_mean, df_yearly_points)

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

def create_map_with_comparison_plotly(df, tp_threshold, zoom_start=8.5, 
                                    min_radius=3, max_radius=15):
    """
    This function creates a Plotly map with circles 
    denoting the points in the grid for the measured total precipitation.
    In blue, the visualized points are above the tp_threshold, red below.
    Marker size is scaled based on the data range.
    """
    
    # Create a copy to avoid modifying original dataframe
    df_plot = df.copy()
    
    # Get min and max values for scaling
    min_val = df_plot['tp_mm_sum'].min()
    max_val = df_plot['tp_mm_sum'].max()
    value_range = max_val - min_val
    
    # Define color mapping based on tp_mm values
    def get_color(tp_mm_value, compared_value):
        return 'blue' if tp_mm_value > compared_value else 'red'
    
    # Function to scale radius based on value
    def get_radius(value, min_val, max_val, min_radius, max_radius):
        if value_range == 0:  # Handle case where all values are the same
            return (min_radius + max_radius) / 2
        normalized = (value - min_val) / value_range
        return min_radius + normalized * (max_radius - min_radius)
    
    # Add color and size columns
    df_plot['color'] = df_plot['tp_mm_sum'].apply(lambda x: get_color(x, tp_threshold))
    df_plot['size'] = df_plot['tp_mm_sum'].apply(lambda x: get_radius(x, min_val, max_val, min_radius, max_radius))
    df_plot['hover_text'] = df_plot.apply(lambda row: f"tp_mm: {row['tp_mm_sum']:.2f}<br>min: {min_val:.2f}<br>max: {max_val:.2f}", axis=1)
    
    # Create the plot
    fig = go.Figure()
    
    # Add points for each color separately to get proper legend
    for color in ['blue', 'red']:
        df_color = df_plot[df_plot['color'] == color]
        if not df_color.empty:
            label = f"Above threshold ({np.round(tp_threshold,0)})" if color == 'blue' else f"Below threshold ({np.round(tp_threshold,0)})"
            fig.add_trace(go.Scattermapbox(
                lat=df_color['latitude'],
                lon=df_color['longitude'],
                mode='markers',
                marker=dict(
                    size=df_color['size'],
                    color=color,
                    opacity=0.7
                ),
                text=df_color['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name=label
            ))
    
    # Update layout
    center_lat = df_plot['latitude'].mean()
    center_lon = df_plot['longitude'].mean()
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_start
        ),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    return fig


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def create_dual_map_comparison_plotly(df1, df2, tp_threshold1, tp_threshold2, 
                                    zoom_start1=8.5, zoom_start2=8.5,
                                    min_radius1=3, max_radius1=15,
                                    min_radius2=3, max_radius2=15,
                                    title1="Map 1", title2="Map 2"):
    """
    Creates a single Plotly figure with two map subplots side by side.
    Each map shows circles denoting points in the grid for measured total precipitation.
    In blue, points above threshold; in red, points below threshold.
    Marker size is scaled based on the data range for each map.
    """
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title1, title2),
        specs=[[{"type": "scattermapbox"}, {"type": "scattermapbox"}]],
        horizontal_spacing=0.02
    )
    
    # Process first dataset
    df1_plot = df1.copy()
    min_val1 = df1_plot['tp_mm_sum'].min()
    max_val1 = df1_plot['tp_mm_sum'].max()
    value_range1 = max_val1 - min_val1
    
    # Process second dataset
    df2_plot = df2.copy()
    min_val2 = df2_plot['tp_mm_sum'].min()
    max_val2 = df2_plot['tp_mm_sum'].max()
    value_range2 = max_val2 - min_val2
    
    # Helper functions
    def get_color(tp_mm_value, compared_value):
        return 'blue' if tp_mm_value > compared_value else 'red'
    
    def get_radius(value, min_val, max_val, value_range, min_radius, max_radius):
        if value_range == 0:
            return (min_radius + max_radius) / 2
        normalized = (value - min_val) / value_range
        return min_radius + normalized * (max_radius - min_radius)
    
    # Process first map data
    df1_plot['color'] = df1_plot['tp_mm_sum'].apply(lambda x: get_color(x, tp_threshold1))
    df1_plot['size'] = df1_plot['tp_mm_sum'].apply(lambda x: get_radius(x, min_val1, max_val1, value_range1, min_radius1, max_radius1))
    df1_plot['hover_text'] = df1_plot.apply(lambda row: f"tp_mm: {row['tp_mm_sum']:.2f}<br>min: {min_val1:.2f}<br>max: {max_val1:.2f}", axis=1)
    
    # Process second map data
    df2_plot['color'] = df2_plot['tp_mm_sum'].apply(lambda x: get_color(x, tp_threshold2))
    df2_plot['size'] = df2_plot['tp_mm_sum'].apply(lambda x: get_radius(x, min_val2, max_val2, value_range2, min_radius2, max_radius2))
    df2_plot['hover_text'] = df2_plot.apply(lambda row: f"tp_mm: {row['tp_mm_sum']:.2f}<br>min: {min_val2:.2f}<br>max: {max_val2:.2f}", axis=1)
    
    # Add traces for first map
    for color in ['blue', 'red']:
        df_color = df1_plot[df1_plot['color'] == color]
        if not df_color.empty:
            label = f"Above threshold ({np.round(tp_threshold1,0)})" if color == 'blue' else f"Below threshold ({np.round(tp_threshold1,0)})"
            fig.add_trace(go.Scattermapbox(
                lat=df_color['latitude'],
                lon=df_color['longitude'],
                mode='markers',
                marker=dict(
                    size=df_color['size'],
                    color=color,
                    opacity=0.7
                ),
                text=df_color['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name=label,
                showlegend=True,  # Show legend for first map
                legendgroup=color  # Group by color for shared legend
            ), row=1, col=1)
    
    # Add traces for second map
    for color in ['blue', 'red']:
        df_color = df2_plot[df2_plot['color'] == color]
        if not df_color.empty:
            fig.add_trace(go.Scattermapbox(
                lat=df_color['latitude'],
                lon=df_color['longitude'],
                mode='markers',
                marker=dict(
                    size=df_color['size'],
                    color=color,
                    opacity=0.7
                ),
                text=df_color['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name=f"Above threshold ({np.round(tp_threshold2,0)})" if color == 'blue' else f"Below threshold ({np.round(tp_threshold2,0)})",
                showlegend=False,  # Hide legend for second map to avoid duplicates
                legendgroup=color  # Same legend group
            ), row=1, col=2)
    
    # Calculate centers for each map
    center_lat1 = df1_plot['latitude'].mean()
    center_lon1 = df1_plot['longitude'].mean()
    center_lat2 = df2_plot['latitude'].mean()
    center_lon2 = df2_plot['longitude'].mean()
    
    # Update layout with separate mapbox configurations
    fig.update_layout(
        mapbox1=dict(
            style="open-street-map",
            center=dict(lat=center_lat1, lon=center_lon1),
            zoom=zoom_start1
        ),
        mapbox2=dict(
            style="open-street-map",
            center=dict(lat=center_lat2, lon=center_lon2),
            zoom=zoom_start2
        ),
        height=600,
        margin={"r":0,"t":30,"l":0,"b":0},  # Slight top margin for subplot titles
        showlegend=True
    )
    
    return fig


# Alternative: Keep your original function and create a wrapper
# Alternative: Keep your original function and create a wrapper
# Alternative: Keep your original function and create a wrapper
def create_dual_map_from_existing(df1, df2, tp_threshold1, tp_threshold2, 
                                zoom_start1=8.5, zoom_start2=8.5,
                                min_radius1=3, max_radius1=15,
                                min_radius2=3, max_radius2=15,
                                title1="Map 1", title2="Map 2"):
    """
    Alternative approach: Use your existing function to get traces and combine them.
    """
    
    # Get individual figures using your existing function
    fig1 = create_map_with_comparison_plotly(df1, tp_threshold1, zoom_start1, min_radius1, max_radius1)
    fig2 = create_map_with_comparison_plotly(df2, tp_threshold2, zoom_start2, min_radius2, max_radius2)
    
    # Create subplot figure
    subplot_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title1, title2),
        specs=[[{"type": "scattermapbox"}, {"type": "scattermapbox"}]],
        horizontal_spacing=0.02
    )
    
    # Add traces from first figure to left subplot
    for trace in fig1.data:
        trace_copy = go.Scattermapbox(
            lat=trace.lat,
            lon=trace.lon,
            mode=trace.mode,
            marker=trace.marker,
            text=trace.text,
            hovertemplate=trace.hovertemplate,
            name=trace.name,
            showlegend=trace.showlegend,
            legendgroup=getattr(trace, 'legendgroup', None)
        )
        subplot_fig.add_trace(trace_copy, row=1, col=1)
    
    # Add traces from second figure to right subplot (hide legend)
    for trace in fig2.data:
        trace_copy = go.Scattermapbox(
            lat=trace.lat,
            lon=trace.lon,
            mode=trace.mode,
            marker=trace.marker,
            text=trace.text,
            hovertemplate=trace.hovertemplate,
            name=trace.name,
            showlegend=False,  # Hide legend for second map
            legendgroup=getattr(trace, 'legendgroup', None)
        )
        subplot_fig.add_trace(trace_copy, row=1, col=2)
    
    # Get mapbox settings from original figures and apply to subplots
    mapbox1_config = fig1.layout.mapbox
    mapbox2_config = fig2.layout.mapbox
    
    # Update layout with correct mapbox configurations
    subplot_fig.update_layout(
        mapbox1=dict(
            style=mapbox1_config.style,
            center=mapbox1_config.center,
            zoom=mapbox1_config.zoom
        ),
        mapbox2=dict(
            style=mapbox2_config.style,
            center=mapbox2_config.center,
            zoom=mapbox2_config.zoom
        ),
        height=600,
        margin={"r":0,"t":30,"l":0,"b":0},
        showlegend=True
    )
    
    return subplot_fig
# # Alternative version using Plotly Express (simpler but less customizable)
# def create_map_with_comparison_plotly_express(df, tp_threshold, zoom_start=8.5, 
#                                             min_radius=3, max_radius=15):
#     """
#     Simplified version using Plotly Express
#     """
#     # Create a copy to avoid modifying original dataframe
#     df_plot = df.copy()
    
#     # Get min and max values for scaling
#     min_val = df_plot['tp_mm_sum'].min()
#     max_val = df_plot['tp_mm_sum'].max()
#     value_range = max_val - min_val
    
#     # Add comparison column
#     df_plot['above_threshold'] = df_plot['tp_mm_sum'] > tp_threshold
#     df_plot['category'] = df_plot['above_threshold'].map({True: f'Above {tp_threshold}', False: f'Below {tp_threshold}'})
    
#     # Scale size to the desired range
#     if value_range > 0:
#         df_plot['scaled_size'] = min_radius + (df_plot['tp_mm_sum'] - min_val) / value_range * (max_radius - min_radius)
#     else:
#         df_plot['scaled_size'] = (min_radius + max_radius) / 2
    
#     # Create the map
#     fig = px.scatter_mapbox(
#         df_plot,
#         lat="latitude",
#         lon="longitude",
#         color="category",
#         size="scaled_size",
#         hover_data={
#             'tp_mm_sum': ':.2f',
#             'scaled_size': False,
#             'above_threshold': False,
#             'category': False
#         },
#         color_discrete_map={
#             f'Above {tp_threshold}': 'blue',
#             f'Below {tp_threshold}': 'red'
#         },
#         zoom=zoom_start,
#         height=600,
#         mapbox_style="open-street-map"
#     )
    
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
#     return fig

############


# london_yearly_mean = pd.read_parquet('./data/data_london_yearly_mean.parquet')
london_yearly_mean, london_yearly_points = load_data('./data/data_london_yearly_mean.parquet', './data/london_yearly_points.parquet')
puglia_yearly_mean, puglia_yearly_points = load_data('./data/data_puglia_yearly_mean.parquet', './data/puglia_yearly_points.parquet')


# create selector
min_year = london_yearly_mean['year'].min()
max_year = london_yearly_mean['year'].max()

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

london_sum, puglia_yearly_tp_per_point, london_yearly_tp_per_point = process_data_for_year(
    selected_year, london_yearly_mean, puglia_yearly_points, london_yearly_points
)

# st.write(london_yearly_tp_per_point.shape)
# st.write(london_yearly_tp_per_point.head())


# col1, col2 = st.columns(2)

# with col1:
#     fig = create_map_with_comparison_plotly(london_yearly_tp_per_point, london_sum, zoom_start=9, min_radius=20, max_radius=35)
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     fig = create_map_with_comparison_plotly(puglia_yearly_tp_per_point, london_sum, zoom_start=7, min_radius=15, max_radius=25)
#     st.plotly_chart(fig, use_container_width=True)

# # Option 1: Use the new dual map function
# fig = create_dual_map_comparison_plotly(
#     london_yearly_tp_per_point, puglia_yearly_tp_per_point,
#     london_sum, london_sum,  # You can use different thresholds if needed
#     zoom_start1=9, zoom_start2=7,
#     min_radius1=20, max_radius1=35,
#     min_radius2=15, max_radius2=25,
#     title1="London", title2="Puglia"
# )

# st.plotly_chart(fig, use_container_width=True)

# Option 2: Use the wrapper function with your existing function

show_map = st.checkbox("Show on Map")

if show_map:

    fig = create_dual_map_from_existing(
        london_yearly_tp_per_point, puglia_yearly_tp_per_point,
        london_sum, london_sum,
        zoom_start1=9, zoom_start2=7,
        min_radius1=20, max_radius1=35,
        min_radius2=15, max_radius2=25,
        title1="London", title2="Puglia"
    )

    st.plotly_chart(fig, use_container_width=True)


# # Simple point plotting
# df = pd.DataFrame({
#     'lat': [37.7749, 40.7128, 34.0522],
#     'lon': [-122.4194, -74.0060, -118.2437],
#     'size': [1000, 200, 150]  # optional for point sizes
# })

# st.map(df, size='size')
# # Using Plotly Express
# fig = px.scatter_mapbox(
#     df, 
#     lat="lat", 
#     lon="lon",
# #    hover_name="city",
#     zoom=3,
#     height=600,
#     mapbox_style="open-street-map"
# )
# st.plotly_chart(fig, use_container_width=True)