import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import cfgrib
import xarray as xr
import pandas as pd
import numpy as np

import glob
from pathlib import Path


def grib_to_csv(grib_file_path,
                city_lat,
                city_lon,
                output_variable='tp', # output variable 'tp' for total precipitation, 't2m' for temperature at 2m
                city_name = 'london', # city name for output file 'london', 'puglia'
                print_debug=True,
                ):
    """
    Read GRIB file, extract data for London coordinates, and save as CSV.
    
    Parameters:
    grib_file_path (str): Path to the GRIB file
    output_csv_path (str): Path for output CSV file (optional)
    
    Returns:
    pandas.DataFrame: DataFrame with the extracted data
    """
    if print_debug:
        print(f"Reading GRIB file: {grib_file_path}")
    
    try:
        # Open GRIB file with xarray
        print('breakpoint 0') 
        datasets = cfgrib.open_datasets(grib_file_path)
        for i, ds in enumerate(datasets):
            print(f"Dataset {i}:")
            print(f"Variables: {list(ds.data_vars.keys())}")
            print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
            print("---")
        
        ds = xr.open_dataset(grib_file_path, engine='cfgrib')
        print('breakpoint 1')        
        # Check longitude format (0-360 vs -180-180)
        city_min = ds.longitude.min().values
        city_max = ds.longitude.max().values
        if print_debug:
            print(f"Longitude range: {city_min} to {city_max}")
        
        target_city = city_lon
        if print_debug:
            print(f"Using -180-180 longitude format: {target_city}")
        
        # Find nearest grid point to London
        lat_idx = np.abs(ds.latitude - city_lat).argmin()
        lon_idx = np.abs(ds.longitude - target_city).argmin()
        
        actual_lat = ds.latitude[lat_idx].values
        actual_lon = ds.longitude[lon_idx].values
        
        if print_debug:
            print(f"Target coordinates: ({city_lat}, {target_city})")
            print(f"Nearest grid point: ({actual_lat}, {actual_lon})")
        
        # Extract data for London coordinates
        city_data = ds.isel(latitude=lat_idx, longitude=lon_idx)
        
        # Convert to pandas DataFrame
        df_list = []
        
        for var_name in city_data.data_vars:
            var_data = city_data[var_name]
            
            # Handle different time dimensions
            if 'time' in var_data.dims:
                # Create DataFrame with time index
                df_var = var_data.to_dataframe().reset_index()
                df_var = df_var.rename(columns={var_name: var_name})
                df_list.append(df_var)
            elif 'valid_time' in var_data.dims:
                # Handle valid_time dimension
                df_var = var_data.to_dataframe().reset_index()
                df_var = df_var.rename(columns={var_name: var_name})
                df_list.append(df_var)
            else:
                # Single value variables
                df_var = pd.DataFrame({
                    var_name: [var_data.values],
                    'latitude': [actual_lat],
                    'longitude': [actual_lon]
                })
                df_list.append(df_var)
        
        # Combine all variables into one DataFrame
        if len(df_list) > 1:
            # Merge on common columns (time, latitude, longitude)
            df = df_list[0]
            for df_var in df_list[1:]:
                common_cols = set(df.columns) & set(df_var.columns)
                if common_cols:
                    df = pd.merge(df, df_var, on=list(common_cols), how='outer')
                else:
                    # If no common columns, concatenate side by side
                    df = pd.concat([df, df_var], axis=1)
        else:
            df = df_list[0]
        
        df['target_latitude'] = city_lat
        df['target_longitude'] = target_city
        
        # Sort by time if time column exists
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            df = df.sort_values(time_cols[0])
            if print_debug:
                print(f"****\n - inside csv Time range: {df[time_col].min()} to {df[time_col].max()}")
            min_string = df[time_col].min().strftime('%Y%m%d')
            max_string = df[time_col].max().strftime('%Y%m%d')
            if print_debug:
                print(f" - min time as string: {min_string}")
                print(f" - max time as string: {max_string}")
            output_csv_name = list(ds.data_vars)[0] + '_' + min_string + '_' + max_string
            if print_debug:
                print(output_csv_name)
        
        output_csv_name = city_name + '_' + output_csv_name
        if print_debug:
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")
            print("grib to csv")
            print(f"ds.data_vars dtype: {type(ds.data_vars)}, df.data_vars name: {list(ds.data_vars)[0]}")
            output_csv_path = './output/' + output_variable + '//' + city_name + '//' + output_csv_name + '.csv'
            print(output_csv_path)
        # print("\nFirst few rows:")
        # print(df.head())
        
        # Save to CSV
        if output_csv_path is None:
            output_csv_path = Path(grib_file_path).stem + "_city_data.csv"
        
        output_csv_path = './output/' + output_variable + '//' + city_name + '//' + output_csv_name + '.csv'
        print(output_csv_path)
        if print_debug:
            print(f"Saving DataFrame to: {output_csv_path}")
        

        df.to_csv(output_csv_path, index=False)
        print(f"\nData saved to: {output_csv_path}")
        
        return df
        
    except Exception as e:
        print(f"Error processing GRIB file: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install xarray cfgrib pandas numpy")
        raise


def process_grib_file(grib_file_path,
                      city_lat=51.5074,  # Default to London latitude
                      city_lon=-0.1278,  # Default to London longitude
                      output_variable='tp', # Default for total precipitation 
                      city_name='london', # Default city name for london
                      print_debug = False):  
    
    grib_file = grib_file_path  # Replace with your GRIB file path
    output_csv_city = city_name  # Replace with desired output path

    # Check if file exists
    if not Path(grib_file).exists():
        print(f"GRIB file not found: {grib_file}")
        print("Please update the 'grib_file' variable with the correct path to your GRIB file.")
        return
    
    try:
        df = grib_to_csv(grib_file,
                         city_lat=city_lat,
                         city_lon=-city_lon, 
                         output_variable = output_variable, 
                         city_name=city_name, 
                         print_debug=print_debug)
        
        # Display summary statistics
        if print_debug:
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(f"Total records: {len(df)}")
        
        # Show time range if available
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            if print_debug:
                print(f"Time range: {df[time_col].min()} to {df[time_col].max()}")
        
        if print_debug:
            print(f"Variables extracted: {[col for col in df.columns if col not in ['latitude', 'longitude', 'actual_latitude', 'actual_longitude', 'target_latitude', 'target_longitude'] + time_cols]}")
        
        return df
        
    except Exception as e:
        print(f"Failed to process GRIB file: {e}")



def create_hourly_dataset(files_base_folder = './output',
                        weather_var = 'tp',
                        city = 'london'):
    folder_path = Path(files_base_folder + '/' + weather_var + '/' + city)

    csv_files = list(folder_path.glob("*.csv"))

    # To get just the filenames:
    csv_filenames = [file.name for file in csv_files]
    print(csv_filenames)

    # To get full paths as strings:
    csv_paths = [str(file) for file in csv_files]
    print(csv_paths)

    dfs = []

    for file in csv_paths:
        df = pd.read_csv(file)
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        df = (df[['valid_time', weather_var]]
            .sort_values('valid_time')
            .reset_index(drop=True)
            .dropna()
            ).copy()
        if (weather_var == 'tp'):
            df['tp_mm'] = df['tp'] * 1000
        print(df.shape)
        dfs.append(df)


    hourly_data = pd.concat(dfs, axis = 0)
    print(hourly_data.shape)
    # remove nans - redundant as they have already been removed in a previous step
    hourly_data.dropna(inplace=True)
    print(hourly_data.shape)
    # compute mean of duplicated records - same date and hour
    print(f"duplicated records: {hourly_data.duplicated().sum()}")
    hourly_data['date_hour'] = hourly_data['valid_time'].dt.floor('h')
    
    if weather_var == 'tp':
        df_24h = (hourly_data
                .groupby("date_hour")[['tp', 'tp_mm']]
                .mean()
                .reset_index()
                )
    else:
        df_24h = (hourly_data
                .groupby("date_hour")[weather_var]
                .mean()
                .reset_index()
                )
    
    df_24h.rename(columns = {'date_hour' : 'valid_time'}, inplace = True)
    print(f"duplicated records after mean: {df_24h.duplicated().sum()}")
    print(f"dataframe new shape: {df_24h.shape}")

    time_delta = (df_24h['valid_time'].max() - df_24h['valid_time'].min())
    n_expected_records = time_delta.components.days *24 +  time_delta.components.hours + 1
    print(f"number of expected records: {n_expected_records}")

    df_24h['hour_of_day'] = df_24h['valid_time'].dt.hour
    df_hourly = transform_data_datetime(df_24h, date_column='valid_time')
    
    
    filename = city + "_" + weather_var + '_' + "hourly_data"

    if weather_var == 'tp':
        features_list = ['valid_time', 'year', 'month', 'day', 'tp', 'tp_mm', 'hour_of_day', 'month int']
        df_hourly[features_list].to_pickle(str(folder_path) + '/' + filename + ".pkl")
    else:
        features_list = ['valid_time', 'year', 'month', 'day', weather_var, 'hour_of_day', 'month int']
        df_hourly[features_list].to_pickle(str(folder_path) + '/' + filename + ".pkl")
    return(df_hourly)

def create_daily_aggregate(file_folder = './output/tp/london/',
                           file_name = 'london_tp_hourly_data.pkl',
                           rainyday_threshold = 0):
    city = file_name.split('_')[0]
    weather_var = file_name.split('_')[1]
    print(f"city = {city}, var = {weather_var}")
    df = pd.read_pickle(file_folder + file_name)
    print(df.shape)
    print(f"columns = {df.columns}")
    if 'tp' in df.columns:
        df['date'] = df['valid_time'].dt.normalize()
        df_daily = (df
                   .groupby('date')
                   .agg(
                        year = ('year', 'first'),
                        month = ('month', 'first'),
                        day = ('day', 'first'),
                        
                        tp_daily_sum = ('tp', 'sum'),
                        tp_daily_mean = ('tp', 'mean'),
                        tp_daily_std = ('tp', 'std'),
                        tp_daily_median = ('tp', 'median'),
                        tp_daily_max = ('tp', 'max'),
                        tp_daily_min = ('tp', 'min'),

                        tp_mm_daily_sum = ('tp_mm', 'sum'),
                        tp_mm_daily_mean = ('tp_mm', 'mean'),
                        tp_mm_daily_std = ('tp_mm', 'std'),
                        tp_mm_daily_median = ('tp_mm', 'median'),
                        tp_mm_daily_min = ('tp_mm', 'min'),
                        tp_mm_daily_max = ('tp_mm', 'max')
                        ).reset_index()
                        )
        
        df_daily['month_int'] = df_daily['date'].dt.month

        df_daily['rainy_day'] = 0
        df_daily.loc[df_daily['tp_mm_daily_sum'] > rainyday_threshold, 'rainy_day'] = 1

        df_daily.sort_values('date', inplace = True)
        df_daily= df_daily.reset_index(drop = True)
        # saving result as .pkl file
        foutput = file_name.split('_')
        output_fname = file_folder + foutput[0] + '_' + foutput[1] + '_' + 'daily_data.pkl'
        print(output_fname)
        df_daily.to_pickle(output_fname)

        
        
        return(df_daily)

def transform_data_datetime(df, date_column='DATE', datetime_format=True):   
    df = df.copy()
    if datetime_format:
        df['year'] = df[date_column].dt.year
        df['month int'] = df[date_column].dt.month
        df['month'] = df[date_column].dt.strftime('%b')
        df['day'] = df[date_column].dt.day

        return df
    df['datetime'] = pd.to_datetime(df[date_column], format="%Y%m%d")
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