import streamlit as st

import pandas as pd
import numpy as np

from src.data_management import load_data, process_data_for_year
from src.data_visualization import create_dual_map_from_existing, plot_highlighted_bars_plotly, create_map_with_comparison, create_map_with_comparison_plotly


st.set_page_config(
    page_title="London and Apulia Rainfall Analysis",
    page_icon="🌧️",
    layout="wide",  # This sets wide mode
    initial_sidebar_state="expanded"
)

# st.set_page_config(
#         page_title="Precipitation Analysis",
#         page_icon="🌧️",
#         layout="wide"
#     )
    
st.title("London vs Apulia -- Annual Precipitation Analysis")

london_yearly_mean, london_yearly_points = load_data('./data/data_london_yearly_mean.parquet', './data/london_yearly_points.parquet')
puglia_yearly_mean, puglia_yearly_points = load_data('./data/data_puglia_yearly_mean.parquet', './data/puglia_yearly_points.parquet')

london_yearly_mean['location'] = 'london'
puglia_yearly_mean['location'] = 'apulia'

df_plot = pd.concat([london_yearly_mean, puglia_yearly_mean])

# create selector
min_year = df_plot['year'].min()
max_year = df_plot['year'].max()

years = list(range(min_year, max_year))  # 2018 to 2025
options = [None] + years

selected_year = st.sidebar.selectbox(
        "Select Year to Highlight:",
        options,
        index=0,  # Default to None
        format_func=lambda x: "   " if x is None else str(x)
    )


st.sidebar.markdown("---")

show_map = st.sidebar.checkbox("Show on Map")

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.subheader("📈 Statistics")
        
    # Calculate statistics
    london_data = df_plot[df_plot['location'] == 'london']['yearly_tp_mm']
    apulia_data = df_plot[df_plot['location'] == 'apulia']['yearly_tp_mm']
    
    st.metric("London Avg", f"{london_data.mean():.1f} mm")
    st.metric("Apulia Avg", f"{apulia_data.mean():.1f} mm")
    st.metric("Difference", f"{london_data.mean() - apulia_data.mean():.1f} mm")

with col2:
    # Create and display the chart
    figsize = (600,400)
    with st.spinner("Generating chart..."):
        fig = plot_highlighted_bars_plotly(df_plot, selected_year, figsize)
        
        # Display the chart
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'precipitation_chart',
                    'height': figsize[1],
                    'width': figsize[0],
                    'scale': 2
                }
            }
        )

with col3:
    if selected_year:
        st.markdown(f"**{selected_year} Data:**")
        year_data = df_plot[df_plot['year'] == selected_year]
        for _, row in year_data.iterrows():
            st.write(f"{row['location'].title()}: {row['yearly_tp_mm']:.1f} mm")


london_sum, puglia_yearly_tp_per_point, london_yearly_tp_per_point = process_data_for_year(
    selected_year, london_yearly_mean, puglia_yearly_points, london_yearly_points
)

if show_map:
    st.markdown("---")

    fig = create_dual_map_from_existing(
        london_yearly_tp_per_point, puglia_yearly_tp_per_point,
        london_sum, london_sum,
        zoom_start1=9, zoom_start2=7,
        min_radius1=20, max_radius1=35,
        min_radius2=15, max_radius2=25,
        title1="London", title2="Puglia"
    )

    st.plotly_chart(fig, use_container_width=True)

# Additional features
st.markdown("---")
with st.expander("ℹ️ About this visualization"):
    st.markdown("""
    This interactive chart shows annual precipitation data for London and Apulia:
    
    - **Interactive**: Hover over bars to see exact values
    - **Highlighting**: Select a specific year to highlight in the sidebar
    - **Download**: Use the camera icon in the chart toolbar to download as PNG
    - **Zoom**: Use the zoom tools to focus on specific time periods
    - **Legend**: Click location names to hide/show data series
    """)