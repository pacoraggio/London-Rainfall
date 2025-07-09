import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
