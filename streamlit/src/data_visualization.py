import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import folium

def create_map_with_comparison(df, tp_threshold, zoom_start=8.5, min_zoom=8.5, 
                              min_radius=3, max_radius=15):
    ''' 
    This function create a folium map with circles 
    denoting the points in the grid for the measured total precipitation.
    In blue, the visualized points are above the tp_threshold, red below.
    Marker size is scaled based on the data range.
    '''
    
    # Create a base map centered on your data
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], 
                zoom_start=zoom_start,
                min_zoom=min_zoom)

    # Get min and max values for scaling
    min_val = df['tp_mm_sum'].min()
    max_val = df['tp_mm_sum'].max()
    value_range = max_val - min_val
    
    # Define color mapping based on tp_mm values
    def get_color(tp_mm_value, compared_value):
        if tp_mm_value > compared_value:
            return 'blue'
        else:
            return 'red'
    
    # Function to scale radius based on value
    def get_radius(value, min_val, max_val, min_radius, max_radius):
        if value_range == 0:  # Handle case where all values are the same
            return (min_radius + max_radius) / 2
        normalized = (value - min_val) / value_range
        return min_radius + normalized * (max_radius - min_radius)

    # Add circles to the map
    for idx, row in df.iterrows():
        radius = get_radius(row['tp_mm_sum'], min_val, max_val, min_radius, max_radius)
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            popup=f"tp_mm: {row['tp_mm_sum']} (min: {min_val:.2f}, max: {max_val:.2f})",
            color=get_color(row['tp_mm_sum'], tp_threshold),
            fill=True,
            fillColor=get_color(row['tp_mm_sum'], tp_threshold),
            fillOpacity=0.7
        ).add_to(m)

    return m



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
            'text': f'Annual Precipitation b Location (Highlighting {selected_year})' if selected_year else 'Annual Precipitation by Location',
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

def plot_highlighted_bars(df_plot, selected_year=None, figsize=(12, 6)):
    """
    Create a bar chart with selected year highlighted and others grayed out.
    
    Parameters:
    - df_plot: DataFrame with columns 'year', 'yearly_tp_mm', 'location'
    - selected_year: Year to highlight (default: None - shows all years in full color)
    - figsize: Figure size tuple (default: (12, 6))
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for locations
    colors = {'london': '#2E86AB', 'apulia': '#F24236'}
    
    # Get unique years and locations
    years = sorted(df_plot['year'].unique())
    locations = sorted(df_plot['location'].unique())
    
    # Calculate bar width and positions
    bar_width = 0.35
    x_positions = np.arange(len(years))
    
    # Plot bars for each location
    for i, location in enumerate(locations):
        location_data = df_plot[df_plot['location'] == location]
        
        # Get y values for each year (fill missing years with 0)
        y_values = []
        alphas = []
        
        for year in years:
            year_data = location_data[location_data['year'] == year]
            if not year_data.empty:
                y_values.append(year_data['yearly_tp_mm'].iloc[0])
            else:
                y_values.append(0)
            
            # Set alpha based on whether it's the selected year
            # If no year selected, show all in full color
            if selected_year is None:
                alphas.append(1.0)
            else:
                alphas.append(1.0 if year == selected_year else 0.3)
        
        # Calculate x positions for this location
        x_pos = x_positions + (i - 0.5) * bar_width
        
        # Plot bars with different alphas
        bars = ax.bar(x_pos, y_values, bar_width, 
                     label=location.capitalize(), 
                     color=colors[location], 
                     alpha=1.0)  # We'll set individual alphas below
        
        # Set individual alpha for each bar
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
    
    # Customize the plot
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Yearly Total Precipitation (mm)', fontsize=12)
    
    # Set title based on whether a year is selected
    if selected_year is None:
        ax.set_title('Annual Precipitation by Location', fontsize=14)
    else:
        ax.set_title(f'Annual Precipitation by Location (Highlighting {selected_year})', fontsize=14)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(years, rotation=45)
    
    # Add legend
    ax.legend(title='Location', loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

# Alternative approach using seaborn with custom styling
def plot_highlighted_bars_seaborn(df_plot, selected_year=None, figsize=(12, 6)):
    """
    Create a bar chart using seaborn with manual highlighting.
    """
    
    # Create a copy of the dataframe with highlight information
    df_styled = df_plot.copy()
    df_styled['highlight'] = df_styled['year'] == selected_year if selected_year is not None else True
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    colors = {'london': '#2E86AB', 'apulia': '#F24236'}
    
    # Plot highlighted and non-highlighted bars separately
    for location in df_styled['location'].unique():
        location_data = df_styled[df_styled['location'] == location]
        
        # Non-highlighted bars (grayed out)
        non_highlight = location_data[~location_data['highlight']]
        if not non_highlight.empty:
            sns.barplot(data=non_highlight, x='year', y='yearly_tp_mm', 
                       color=colors[location], alpha=0.3, ax=ax)
        
        # Highlighted bars (full color)
        highlight = location_data[location_data['highlight']]
        if not highlight.empty:
            sns.barplot(data=highlight, x='year', y='yearly_tp_mm', 
                       color=colors[location], alpha=1.0, ax=ax)
    
    # This approach has limitations with side-by-side bars in seaborn
    # So let's use the matplotlib approach which is more flexible
    
    plt.close(fig)  # Close this figure and use the matplotlib approach
    return plot_highlighted_bars(df_plot, selected_year, figsize)


# Interactive version with widget (if using Jupyter)
def create_interactive_plot(df_plot):
    """
    Create an interactive version using ipywidgets (for Jupyter notebooks)
    """
    try:
        from ipywidgets import interact, IntSlider
        
        years = sorted(df_plot['year'].unique())
        
        @interact(selected_year=IntSlider(
            value=years[0], 
            min=min(years), 
            max=max(years), 
            step=1,
            description='Year:'
        ))
        def interactive_plot(selected_year):
            fig, ax = plot_highlighted_bars(df_plot, selected_year)
            plt.show()
            
    except ImportError:
        print("ipywidgets not available. Use plot_highlighted_bars() function directly.")

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
        height=800,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    return fig

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

