import streamlit as st
import pandas as pd
import numpy as np

from src.data_management import load_data, process_data_for_year
from src.data_visualization import create_dual_map_from_existing, plot_highlighted_bars_plotly, create_map_with_comparison, create_map_with_comparison_plotly

st.set_page_config(
    page_title="London and Apulia Rainfall Analysis",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for fixed navigation
st.markdown("""
<style>
    .nav-container {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #e3e6ea;
        margin-bottom: 1rem;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        display: flex;
        gap: 0.5rem;
    }
    
    .nav-item {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        text-decoration: none;
        color: #0c5460;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
    }
    
    .nav-item:hover {
        background-color: #e3f2fd;
        border-color: #b3e5fc;
    }
    
    .nav-item.active {
        background-color: #ff6f00;
        color: white;
        border-color: #ff6f00;
    }
    
    /* Add padding to main content to account for fixed navbar */
    .main-content {
        padding-top: 60px;
    }
    
    /* Fix for Streamlit button styling */
    .stButton button {
        width: 100%;
        margin: 0 !important;
        padding: 0.5rem 1rem !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 3px !important;
        color: #0c5460 !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    
    .stButton button:hover {
        background-color: #e3f2fd !important;
        border-color: #b3e5fc !important;
    }
    
    /* Style for active button - you'll need to add logic for this */
    .stButton.active button {
        background-color: #ff6f00 !important;
        color: white !important;
        border-color: #ff6f00 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Create fixed navigation bar using HTML and CSS
nav_html = f"""
<div class="nav-container">
    <div class="nav-item {'active' if st.session_state.current_page == 'Home' else ''}">
        🏠 Introduction
    </div>
    <div class="nav-item {'active' if st.session_state.current_page == 'Questions' else ''}">
        📊 Annual Precipitation Analysis
    </div>
</div>
"""

st.markdown(nav_html, unsafe_allow_html=True)

# Alternative approach using columns with tight spacing
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Create navigation with minimal column gaps
col1, col2, spacer = st.columns([2, 3, 5])

with col1:
    if st.button("🏠 Introduction", key="home"):
        st.session_state.current_page = 'Home'

with col2:
    if st.button("📊 Annual Precipitation Analysis", key="questions"):
        st.session_state.current_page = 'Questions'

st.markdown("---")

# Display content based on current page
if st.session_state.current_page == 'Home':
    st.title("🌧️ Where it rains most? London or Apulia?")
    st.subheader("How a simple question turned into a streamlit app. A rainfall analysis comparing London and Apulia, Italy")
    st.markdown("Lately, I was chatting on Whatstapp with a friend of mine. As we live in different country, the weather is alwasy a subject " \
    " that we use as an update for our lives (*\"here is sunny\"*, *\"here it's raining and miserable\"*, *\"there are no middle seasons anymore\"*). " \
    "During the chat, out of the blue, my friend stated that in Apulia it rains more than in London but less often. " \
    "As he is usually right, and I am usually trying to contradict him, I started thinking how I verify this statement. " \
    "What normal people would do, would be asking ChatGPT, or Claude, or Google. But then I was curious about how we measure, " \
    "collect and analys data about the amount or precipitation falling")

    st.markdown("There are several sources reporting rainfall for different region, and very often they use different methodologies in measuring or reporting the data." \
                "As I wanted to compare two different regions quite far away from each other, I searched for a single source reporting different parts of the world")
    
elif st.session_state.current_page == 'Questions':
    st.title("London vs Apulia -- Annual Precipitation Analysis")

    london_yearly_mean, london_yearly_points = load_data('./data/data_london_yearly_mean.parquet', './data/london_yearly_points.parquet')
    puglia_yearly_mean, puglia_yearly_points = load_data('./data/data_puglia_yearly_mean.parquet', './data/puglia_yearly_points.parquet')

    london_yearly_mean['location'] = 'london'
    puglia_yearly_mean['location'] = 'apulia'

    df_plot = pd.concat([london_yearly_mean, puglia_yearly_mean])

    # create selector
    min_year = df_plot['year'].min()
    max_year = df_plot['year'].max()

    years = list(range(min_year, max_year))
    options = [None] + years

    selected_year = st.sidebar.selectbox(
        "Select Year to Highlight:",
        options,
        index=0,
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

st.markdown('</div>', unsafe_allow_html=True)