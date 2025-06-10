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