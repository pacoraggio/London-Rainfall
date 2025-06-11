import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

def plot_rainfall(df, 
                  start_year, 
                  latest_year, 
                  feature = 'total_rainfall', 
                  special_year=2022, 
                  title_label ='Total Rainfall',
                  year_shift = False):

    rainfall_se_start_year_latest_year = df[(df['year'] >= start_year) & (df['year'] <= latest_year)].copy()
    mean_previous_years = rainfall_se_start_year_latest_year[rainfall_se_start_year_latest_year['year'] < latest_year][feature].mean()

    years = sorted(rainfall_se_start_year_latest_year['year'].unique())
    colors = ['steelblue'] * len(years)

    # Set special colors
    for i, year in enumerate(years):
        if year == special_year:
            colors[i] = 'red'      # Color for 2022
        elif year == latest_year:
            colors[i] = 'orange'   # Color for last year

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        x='year',
        y=feature,
        hue='year',
        data=rainfall_se_start_year_latest_year,
        palette=colors,
        edgecolor='black',
        errorbar=None,
        legend=False
        );
    
    # Custom formatting
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.title(f'{title_label} in South East England ({start_year}-{latest_year})', fontsize=16)
    plt.axhline(y=mean_previous_years, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_previous_years:.2f}');

    # Move legend to the right
    plt.legend(bbox_to_anchor=(1., 1), loc='upper left')
    if(year_shift == True):
        new_labels = list(rainfall_se_start_year_latest_year['year'].astype(str)+'-'+(rainfall_se_start_year_latest_year['year']+1).astype(str).str[-2:])
        ax.set_xticks(range(len(new_labels)))
        ax.set_xticklabels(new_labels, rotation=45, ha='center')
    plt.show()


def plot_min_max_rainfall(df_rainfall, 
                          start_year=None, 
                          end_year=None,
                          figsize=(12, 6)):
    
    df_data = df_rainfall[(df_rainfall['year']>=start_year) &
                          (df_rainfall['year']<=end_year)]

    print('test')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.figure(figsize=(14, 7))

    # Fill between min/max
    ax.fill_between(df_data['year'], df_data['min_rainfall'], df_data['max_rainfall'], 
                    alpha=0.2, color='gray', label='Range (Min-Max)')

    # Plot extremes with thinner lines
    ax.plot(df_data['year'], df_data['min_rainfall'], color='blue', linewidth=1.5, 
            linestyle=':', alpha=0.8, label='Minimum')
    ax.plot(df_data['year'], df_data['max_rainfall'], color='red', linewidth=1.5, 
            linestyle=':', alpha=0.8, label='Maximum')

    # Emphasize central tendencies
    ax.plot(df_data['year'], df_data['avg_rainfall'], color='darkgreen', linewidth=1.5, 
            label='Mean', marker='o', markersize=5, markerfacecolor='white', 
            markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(df_data['year'], df_data['median_rainfall'], color='darkorange', linewidth=1.5, 
            label='Median', marker='D', markersize=5, markerfacecolor='white', 
            markeredgecolor='darkorange', markeredgewidth=2)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Precipitation (mm)', fontsize=12)

    # Create title with year range info
    year_range = f"{df_data['year'].min()}-{df_data['year'].max()}"
    ax.set_title(f'Annual Precipitation Analysis ({year_range})', fontsize=14, fontweight='bold')

    # Right side of the plot (most common)
    ax.legend(bbox_to_anchor=(1.0, 1.015), loc='upper left')
    print('test')
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    return fig, ax
    