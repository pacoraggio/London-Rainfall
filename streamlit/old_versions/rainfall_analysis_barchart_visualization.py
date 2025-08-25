import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# from src.data_visualization import plot_highlighted_bars_plotly

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_highlighted_bars_plotly(df_plot, selected_year=None, figsize=(900, 500)):
    """
    Create a bar chart with selected year highlighted and others grayed out using Plotly.
    
    Parameters:
    - df_plot: DataFrame with columns 'year', 'yearly_tp_mm', 'location'
    - selected_year: Year to highlight (default: None - shows all years in full color)
    - figsize: Figure size tuple (width, height) (default: (900, 500))
    """
    
    # Define colors for locations
    colors = {'london': '#2E86AB', 'apulia': '#F24236'}
    
    # Get unique years and locations
    years = sorted(df_plot['year'].unique())
    # locations = sorted(df_plot['location'].unique())
    locations = ['london','apulia']
    
    # Create figure
    fig = go.Figure()
    
# Calculate bar width and positions
    bar_width = 0.3 # Reduced width for closer bars
    year_spacing = 3.0  # Increased spacing between years
    x_positions = np.arange(len(years)) * year_spacing

    # Plot bars for each location
    for i, location in enumerate(locations):
        location_data = df_plot[df_plot['location'] == location]
        
        # Get y values and prepare data for each year
        y_values = []
        x_pos = []
        opacities = []
        
        for j, year in enumerate(years):
            year_data = location_data[location_data['year'] == year]
            if not year_data.empty:
                y_values.append(year_data['yearly_tp_mm'].iloc[0])
                x_pos.append(j + (i - 0.5) * bar_width)
                
                # Set opacity based on whether it's the selected year
                if selected_year is None:
                    opacities.append(1.0)
                else:
                    opacities.append(1.0 if year == selected_year else 0.3)
        
        # Add bars for this location
        fig.add_trace(go.Bar(
            x=x_pos,
            y=y_values,
            name=location.capitalize(),
            marker=dict(
                color=colors[location],
                opacity=opacities
            ),
            width=bar_width,
            offsetgroup=location,  # This ensures proper grouping
            hovertemplate=f'{location.capitalize()}: %{{y:.1f}} m<extra></extra>',  # Custom hover format
            # text=[f'{val:.1f}' for val in y_values],  # Text labels on bars
            # textposition='outside',  # Position text above bars
            # textfont=dict(size=10, color='black')  # Text styling
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Annual Precipitation by Location (Highlighting {selected_year})' if selected_year else 'Annual Precipitation by Location',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis=dict(
            title='Year',
            tickmode='array',
            tickvals=list(range(len(years))),
            ticktext=years,
            tickangle=45
        ),
        yaxis=dict(
            title='Yearly Total Precipitation (mm)',
            gridcolor='rgba(128,128,128,0.3)'
        ),
        legend=dict(
            title='Location',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        ),
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        plot_bgcolor='white',
    )
    
    # Update x-axis to show grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
    
    return fig


st.set_page_config(
        page_title="Precipitation Analysis",
        page_icon="🌧️",
        layout="wide"
    )
    
st.title("London vs Apulia -- Annual Precipitation Analysis")
# st.markdown("Interactive visualizaton of yearly precipitation data for London and Apulia")

@st.cache_data  # 
def load_data(fname_yearly_mean, fname_yearly_points):
    df_yearly_mean = pd.read_parquet(fname_yearly_mean)
    df_yearly_mean = df_yearly_mean[df_yearly_mean['year'] > df_yearly_mean['year'].min()]

    df_yearly_points = pd.read_parquet(fname_yearly_points)

    return(df_yearly_mean, df_yearly_points)

# london_yearly_mean = pd.read_parquet('./data/data_london_yearly_mean.parquet')
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

st.markdown('---')

col1, col2, col3 = st.columns([0.4,0.2,0.4])

with col1:
    st.markdown("")
    st.markdown("")
    st.markdown("Interactive visualizaton of yearly precipitation data for London and Apulia")


with col2:
    selected_year = st.selectbox(
        "Select Year to Highlight:",
        options,
        index=0,  # Default to None
        format_func=lambda x: "   " if x is None else str(x)
    )


# Main content area

# st.sidebar.markdown("---")
# show_data = st.sidebar.checkbox("Show raw data", value=False)
# show_stats = st.sidebar.checkbox("Show statistics", value=True)
show_stats = 1
show_data = 0


# col1, col2 = st.columns([3, 1])
st.empty()

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
#        st.markdown("---")
        st.markdown(f"**{selected_year} Data:**")
        year_data = df_plot[df_plot['year'] == selected_year]
        for _, row in year_data.iterrows():
            st.write(f"{row['location'].title()}: {row['yearly_tp_mm']:.1f} mm")


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