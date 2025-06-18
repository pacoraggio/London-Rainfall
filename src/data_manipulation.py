import pandas as pd
import numpy as np

def transform_data_datetime(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['DATE'], format="%Y%m%d")
    df['year'] = df['datetime'].dt.year
    df['month int'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.strftime('%b')
    df['day'] = df['datetime'].dt.day

    return df

def sort_and_highlight_dataframe(df, sort_column, columns_to_display, 
                                 highlight_condition=None, filter_condition=None,
                                 ascending=False, add_rank=True, 
                                 highlight_color='#2d5a87', n_rows=None):
    """
    Sort a dataframe and display selected columns with highlighted rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to sort and display
    sort_column : str
        Column name to sort by
    columns_to_display : list
        List of column names to display in the final output
    highlight_condition : dict or list of dicts, optional
        Dictionary with column name as key and value(s) to highlight
        e.g., {'year': 2024} or {'year': [2024, 2023]}
        For multiple colors, use list of dicts:
        [{'condition': {'year': 2024}, 'color': '#ff0000'},
         {'condition': {'year': 2023}, 'color': '#00ff00'}]
    filter_condition : callable, optional
        Function that takes the dataframe and returns a boolean mask
        e.g., lambda df: (df['year'] >= 2010) & (df['year'] <= 2024)
    ascending : bool, default False
        Sort order (False for descending, True for ascending)
    add_rank : bool, default True
        Whether to add a 'Rank' column based on sort order
    highlight_color : str or dict, default '#2d5a87'
        Background color for highlighted rows. Can be:
        - Single color string (applied to all highlights)
        - Ignored if highlight_condition contains color specifications
    n_rows : int, optional
        Number of top rows to display (None for all rows)
    
    Returns:
    --------
    pandas.io.formats.style.Styler
        Styled dataframe with highlighted rows
    """
    import pandas as pd
    
    # Apply filter condition if provided
    if filter_condition is not None:
        filtered_df = df[filter_condition(df)].copy()
    else:
        filtered_df = df.copy()
    
    # Sort the dataframe
    sorted_df = filtered_df.sort_values(sort_column, ascending=ascending).reset_index(drop=True)
    
    # Find indices to highlight
    highlight_mapping = {}  # index -> color mapping
    
    if highlight_condition is not None:
        # Handle both old format (dict) and new format (list of dicts)
        if isinstance(highlight_condition, dict):
            # Old format - single condition, single color
            if 'condition' in highlight_condition and 'color' in highlight_condition:
                # New format but single item
                conditions = [highlight_condition]
            else:
                # Old format
                conditions = [{'condition': highlight_condition, 'color': highlight_color}]
        elif isinstance(highlight_condition, list):
            # New format - multiple conditions with colors
            conditions = highlight_condition
        else:
            conditions = []
        
        for cond_spec in conditions:
            condition = cond_spec['condition']
            color = cond_spec.get('color', highlight_color)
            
            for col, values in condition.items():
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    indices = sorted_df[sorted_df[col] == value].index.tolist()
                    for idx in indices:
                        highlight_mapping[idx] = color
    
    # Limit rows if specified
    if n_rows is not None:
        sorted_df = sorted_df.head(n_rows)
        # Filter highlight_mapping to only include visible rows
        highlight_mapping = {k: v for k, v in highlight_mapping.items() if k < n_rows}
    
    # Define highlighting function with multiple colors
    def highlight_rows(s):
        if s.name in highlight_mapping:
            return [f'background-color: {highlight_mapping[s.name]}' for _ in s]
        else:
            return ['' for _ in s]
    
    # Apply styling and return
    if(add_rank == True):
        sorted_df = sorted_df[columns_to_display]
        sorted_df['Rank'] = sorted_df.index + 1
        cols = sorted_df.columns.tolist()
        cols.remove('Rank')
        cols.insert(1, 'Rank')
        sorted_df = sorted_df[cols]

        sorted_df.rename(columns = {'year' : 'Year',
                                'total_rainfall' : 'Total Rainfall (mm)', 
                                'avg_rainfall' : 'Average Rainfall (mm)', 
                                'median_rainfall' : 'Median Rainfall (mm)',
                                'min_rainfall' : 'Min (mm)', 
                                'max_rainfall' : 'Max (mm)', 
                                'weather_year' : 'Reference Year'},
                                inplace = True)
        styled_df = sorted_df.style.hide(axis='index').apply(highlight_rows, axis=1)
        return(styled_df)

    sorted_df = sorted_df[columns_to_display]
    sorted_df.rename(columns = {'year' : 'Year',
                                'total_rainfall' : 'Total Rainfall (mm)', 
                                'avg_rainfall' : 'Average Rainfall (mm)', 
                                'median_rainfall' : 'Median Rainfall (mm)',
                                'min_rainfall' : 'Min (mm)', 
                                'max_rainfall' : 'Max (mm)', 
                                'weather_year' : 'Reference Year'},
                                inplace = True)

    styled_df = sorted_df.style.hide(axis='index').apply(highlight_rows, axis=1)   
    return styled_df


# Alternative simpler function for common use cases
def quick_sort_highlight(df, sort_by, show_cols, highlight_year=None, top_n=None):
    """
    Simplified version for quick sorting and highlighting by year.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    sort_by : str
        Column to sort by
    show_cols : list
        Columns to display
    highlight_year : int, optional
        Year to highlight
    top_n : int, optional
        Number of top rows to show
    """
    highlight_condition = {'year': highlight_year} if highlight_year else None
    
    return sort_and_highlight_dataframe(
        df=df,
        sort_column=sort_by,
        columns_to_display=show_cols,
        highlight_condition=highlight_condition,
        ascending=False,
        add_rank=True,
        n_rows=top_n
    )


# # example 1

# result = sort_and_highlight_dataframe(
#     df=rainfall_se,
#     sort_column='total_rainfall',
#     columns_to_display=['year', 'total_rainfall'],
#     highlight_condition={'year': [2000, 2023, 2022]},
#     highlight_color='#2d5a87',
#     n_rows=10
# )

# result

# Multiple colors
# result = sort_and_highlight_dataframe(
#     df=rainfall_se,
#     sort_column='total_rainfall',
#     columns_to_display=['year', 'total_rainfall'],
#     highlight_condition=[
#         {'condition': {'year': 1852}, 'color': '#ff4444'},  # Red
#         {'condition': {'year': 2014}, 'color': '#44ff44'},  # Green  
#         {'condition': {'year': 1872}, 'color': '#4444ff'}   # Blue
#     ],
#     n_rows=10
# )

# Mix conditions and colors

# result = sort_and_highlight_dataframe(
#     df=rainfall_se,
#     sort_column='total_rainfall',
#     columns_to_display=['year', 'total_rainfall'],
#     highlight_condition=[
#         {'condition': {'year': [2024, 2023]}, 'color': '#ff6b6b'},  # Recent years in red
#         {'condition': {'year': 1951}, 'color': '#4ecdc4'}           # 2010 in teal
#     ],
#     n_rows=15
# )