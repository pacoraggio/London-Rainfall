import cdsapi
import xarray as xr
import pandas as pd

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

def download_and_process_era5_precip(request_params, output_file="london_era5land_hourly_precip"):
    """
    Enhanced function to download and process ERA5-Land precipitation data with multiple fallback options
    """
    
    # Initialize the CDS API client
    c = cdsapi.Client()
    
    # Define the request parameters
    request_params = request_params
    
    grib_file = f"{output_file}.grib"
    netcdf_file = f"{output_file}.nc"
    
    print("Downloading ERA5-Land hourly precipitation data...")
    
    try:
        # Try downloading as GRIB first
        c.retrieve('reanalysis-era5-land', request_params, grib_file)
        print(f"Data downloaded to {grib_file}")
        print(f"File size: {os.path.getsize(grib_file) / (1024*1024):.2f} MB")
        
    
    except Exception as download_error:
        print(f"Download failed: {download_error}")
        return None



import zipfile
import os
import xarray as xr
from pathlib import Path

def extract_grib_from_zip(zip_file_path, extract_to=None):
    """
    Extract GRIB file from ZIP archive downloaded from CDS API
    
    Parameters:
    zip_file_path (str): Path to the ZIP file (mistakenly named .grib)
    extract_to (str): Directory to extract to (default: same directory as ZIP)
    
    Returns:
    str: Path to the extracted GRIB file
    """
    
    zip_path = Path(zip_file_path)
    
    if extract_to is None:
        extract_to = zip_path.parent
    else:
        extract_to = Path(extract_to)
        extract_to.mkdir(exist_ok=True)
    
    print(f"📦 Extracting ZIP file: {zip_file_path}")
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            print(f"📋 Files in archive: {file_list}")
            
            # Extract all files
            zip_ref.extractall(extract_to)
            print(f"✅ Extracted to: {extract_to}")
            
            # Find the GRIB file
            grib_files = []
            for filename in file_list:
                extracted_path = extract_to / filename
                if extracted_path.exists():
                    # Check if it's a GRIB file by reading first few bytes
                    with open(extracted_path, 'rb') as f:
                        first_bytes = f.read(4)
                        if first_bytes == b'GRIB':
                            grib_files.append(str(extracted_path))
                            print(f"🎯 Found GRIB file: {filename}")
            
            if grib_files:
                return grib_files[0]  # Return the first GRIB file found
            else:
                # If no GRIB magic bytes found, return the first file (might still be GRIB)
                first_file = str(extract_to / file_list[0])
                print(f"⚠️  No GRIB magic bytes found, returning first file: {first_file}")
                return first_file
                
    except zipfile.BadZipFile:
        print(f"❌ Error: {zip_file_path} is not a valid ZIP file")
        return None
    except Exception as e:
        print(f"❌ Error extracting ZIP file: {e}")
        return None

def load_extracted_grib(grib_file_path):
    """
    Load the extracted GRIB file using xarray
    
    Parameters:
    grib_file_path (str): Path to the extracted GRIB file
    
    Returns:
    xarray.Dataset: The loaded dataset
    """
    
    print(f"📊 Loading GRIB file: {grib_file_path}")
    
    # Try different methods to load the GRIB file
    methods = [
        ('cfgrib', {}),
        ('cfgrib with error handling', {'errors': 'ignore'}),
        ('cfgrib with filter', {'filter_by_keys': {'paramId': 228}}),
    ]
    
    for method_name, backend_kwargs in methods:
        try:
            print(f"🔄 Trying {method_name}...")
            ds = xr.open_dataset(grib_file_path, engine='cfgrib', backend_kwargs=backend_kwargs)
            print(f"✅ Success with {method_name}!")
            print(f"📏 Dataset dimensions: {dict(ds.dims)}")
            print(f"📊 Variables: {list(ds.data_vars)}")
            return ds
        except Exception as e:
            print(f"❌ {method_name} failed: {str(e)[:100]}...")
            continue
    
    print("❌ All methods failed to load the GRIB file")
    return None

def process_london_precipitation_data(zip_file_path):
    """
    Complete workflow to extract and process London precipitation data
    
    Parameters:
    zip_file_path (str): Path to the ZIP file downloaded from CDS
    
    Returns:
    xarray.Dataset: Processed precipitation dataset
    """
    
    print("🌧️  Processing London Precipitation Data")
    print("=" * 50)
    
    # Step 1: Extract the GRIB file from ZIP
    grib_file_path = extract_grib_from_zip(zip_file_path)
    
    if grib_file_path is None:
        print("❌ Failed to extract GRIB file")
        return None
    
    # Step 2: Load the GRIB file
    ds = load_extracted_grib(grib_file_path)
    
    if ds is None:
        print("❌ Failed to load GRIB file")
        return None
    
    # Step 3: Basic data exploration
    print("\n📈 Data Summary:")
    print("=" * 30)
    
    # Print coordinate information
    print("🗺️  Coordinates:")
    for coord in ds.coords:
        coord_data = ds.coords[coord]
        if coord_data.size > 1:
            print(f"   {coord}: {coord_data.min().values} to {coord_data.max().values} ({coord_data.size} points)")
        else:
            print(f"   {coord}: {coord_data.values}")
    
    # Print variable information
    print("\n📊 Data Variables:")
    for var in ds.data_vars:
        var_data = ds[var]
        print(f"   {var}: {var_data.dims} - {var_data.long_name if 'long_name' in var_data.attrs else 'No description'}")
        if hasattr(var_data, 'units'):
            print(f"      Units: {var_data.units}")
    
    return ds

def quick_visualization(ds, variable_name=None):
    """
    Create a quick visualization of the precipitation data
    
    Parameters:
    ds (xarray.Dataset): The precipitation dataset
    variable_name (str): Name of the variable to plot (auto-detect if None)
    """
    
    if variable_name is None:
        # Auto-detect precipitation variable
        possible_names = ['tp', 'total_precipitation', 'precip', 'precipitation']
        for name in possible_names:
            if name in ds.data_vars:
                variable_name = name
                break
        
        if variable_name is None:
            variable_name = list(ds.data_vars)[0]  # Use first variable
    
    print(f"📊 Creating visualization for variable: {variable_name}")
    
    try:
        import matplotlib.pyplot as plt
        
        # Get the variable
        var = ds[variable_name]
        
        # If time dimension exists, plot time series
        if 'time' in var.dims:
            # Average over spatial dimensions if they exist
            if 'latitude' in var.dims and 'longitude' in var.dims:
                var_avg = var.mean(dim=['latitude', 'longitude'])
            else:
                var_avg = var
            
            plt.figure(figsize=(12, 6))
            var_avg.plot()
            plt.title(f'Time Series: {variable_name}')
            plt.xlabel('Time')
            plt.ylabel(f'{variable_name} ({var.units if "units" in var.attrs else ""})')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # If spatial dimensions exist, plot map for first time step
        elif 'latitude' in var.dims and 'longitude' in var.dims:
            plt.figure(figsize=(10, 8))
            var.isel(time=0).plot()
            plt.title(f'Spatial Map: {variable_name} (First Time Step)')
            plt.show()
        
        print("✅ Visualization created successfully!")
        
    except ImportError:
        print("⚠️  Matplotlib not available for visualization")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

import pandas as pd
import numpy as np
from pathlib import Path

def explore_dataset_structure(ds):
    """Explore the structure of your dataset to understand what we're working with"""
    print("🔍 Dataset Structure Analysis")
    print("=" * 40)
    
    print(f"📏 Dimensions: {dict(ds.dims)}")
    print(f"📊 Variables: {list(ds.data_vars)}")
    print(f"🗺️  Coordinates: {list(ds.coords)}")
    
    # Show sample of each coordinate
    print("\n📍 Coordinate Details:")
    for coord_name, coord in ds.coords.items():
        if coord.size <= 10:
            print(f"   {coord_name}: {coord.values}")
        else:
            print(f"   {coord_name}: {coord.values[0]} to {coord.values[-1]} ({coord.size} points)")
    
    # Show variable details
    print("\n📊 Variable Details:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"   {var_name}:")
        print(f"      Shape: {var.shape}")
        print(f"      Dimensions: {var.dims}")
        if hasattr(var, 'units'):
            print(f"      Units: {var.units}")
        if hasattr(var, 'long_name'):
            print(f"      Description: {var.long_name}")
        print(f"      Data range: {float(var.min())} to {float(var.max())}")

def convert_to_long_format_dataframe(ds, variables=None):
    """
    Convert xarray dataset to long-format pandas DataFrame
    Each row represents one observation with all coordinates as columns
    
    Parameters:
    ds: xarray Dataset
    variables: list of variable names to include (None = all variables)
    
    Returns:
    pandas DataFrame in long format
    """
    
    print("📊 Converting to Long Format DataFrame...")
    
    if variables is None:
        variables = list(ds.data_vars)
    
    # Convert to DataFrame - this creates a long format automatically
    df = ds.to_dataframe()
    
    # Reset index to make all coordinates into columns
    df = df.reset_index()
    
    # Remove any NaN values if present
    df = df.dropna()
    
    print(f"✅ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"📊 Columns: {list(df.columns)}")
    
    return df

def convert_to_time_series_dataframe(ds, lat_lon_method='mean', variables=None):
    """
    Convert to time series DataFrame by aggregating spatial dimensions
    
    Parameters:
    ds: xarray Dataset
    lat_lon_method: How to handle lat/lon ('mean', 'median', 'sum', or specific lat/lon values)
    variables: list of variable names to include
    
    Returns:
    pandas DataFrame with time as index
    """
    
    print(f"📈 Converting to Time Series DataFrame (spatial aggregation: {lat_lon_method})...")
    
    if variables is None:
        variables = list(ds.data_vars)
    
    # Check if we have spatial dimensions
    spatial_dims = [dim for dim in ['latitude', 'longitude', 'lat', 'lon'] if dim in ds.dims]
    
    if spatial_dims and lat_lon_method in ['mean', 'median', 'sum']:
        # Aggregate spatial dimensions
        if lat_lon_method == 'mean':
            ds_agg = ds.mean(dim=spatial_dims)
        elif lat_lon_method == 'median':
            ds_agg = ds.median(dim=spatial_dims)
        elif lat_lon_method == 'sum':
            ds_agg = ds.sum(dim=spatial_dims)
        
        print(f"   Aggregated {len(spatial_dims)} spatial dimensions using {lat_lon_method}")
    else:
        ds_agg = ds
    
    # Convert to DataFrame
    df = ds_agg.to_dataframe()
    
    # If time is in the index, keep it there, otherwise reset index
    if 'time' in df.index.names:
        df = df.reset_index(level=[name for name in df.index.names if name != 'time'])
    else:
        df = df.reset_index()
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"✅ Created time series DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    return df

def convert_to_spatial_dataframe(ds, time_method='mean', variables=None):
    """
    Convert to spatial DataFrame by aggregating time dimension
    
    Parameters:
    ds: xarray Dataset
    time_method: How to handle time ('mean', 'sum', 'max', or specific time index)
    variables: list of variable names to include
    
    Returns:
    pandas DataFrame with lat/lon as columns
    """
    
    print(f"🗺️  Converting to Spatial DataFrame (time aggregation: {time_method})...")
    
    if variables is None:
        variables = list(ds.data_vars)
    
    # Check if we have time dimension
    if 'time' in ds.dims:
        if time_method == 'mean':
            ds_agg = ds.mean(dim='time')
        elif time_method == 'sum':
            ds_agg = ds.sum(dim='time')
        elif time_method == 'max':
            ds_agg = ds.max(dim='time')
        elif isinstance(time_method, int):
            ds_agg = ds.isel(time=time_method)
        else:
            ds_agg = ds.mean(dim='time')  # default
        
        print(f"   Aggregated time dimension using {time_method}")
    else:
        ds_agg = ds
    
    # Convert to DataFrame
    df = ds_agg.to_dataframe().reset_index()
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"✅ Created spatial DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    return df

def save_dataframe_to_csv(df, filename, add_metadata=True):
    """
    Save DataFrame to CSV with optional metadata
    
    Parameters:
    df: pandas DataFrame
    filename: output filename
    add_metadata: whether to add metadata header
    """
    
    filepath = Path(filename)
    
    print(f"💾 Saving DataFrame to: {filepath}")
    
    if add_metadata:
        # Create metadata header
        metadata_lines = [
            f"# Generated from ERA5 GRIB data",
            f"# Rows: {len(df)}",
            f"# Columns: {len(df.columns)}",
            f"# Column names: {', '.join(df.columns)}",
            f"# Generated on: {pd.Timestamp.now()}",
            "#"
        ]
        
        # Write metadata and data
        with open(filepath, 'w', newline='') as f:
            # Write metadata
            for line in metadata_lines:
                f.write(line + '\n')
            
            # Write CSV data
            df.to_csv(f, index=False)
    else:
        # Simple CSV save
        df.to_csv(filepath, index=False)
    
    file_size = filepath.stat().st_size / (1024*1024)  # MB
    print(f"✅ Saved successfully! File size: {file_size:.2f} MB")
    
    return str(filepath)

def create_multiple_csv_formats(ds, base_filename="london_precipitation"):
    """
    Create CSV files in multiple formats for different use cases
    
    Parameters:
    ds: xarray Dataset
    base_filename: base name for output files
    
    Returns:
    dict: mapping of format names to file paths
    """
    
    print("🚀 Creating multiple CSV formats...")
    print("=" * 40)
    
    output_files = {}
    
    # 1. Long format (all data points)
    try:
        df_long = convert_to_long_format_dataframe(ds)
        file_long = save_dataframe_to_csv(df_long, f"{base_filename}_long_format.csv")
        output_files['long_format'] = file_long
        print(f"📊 Long format sample:\n{df_long.head()}\n")
    except Exception as e:
        print(f"❌ Long format failed: {e}")
    
    # 2. Time series (spatial average)
    try:
        df_time = convert_to_time_series_dataframe(ds, lat_lon_method='mean')
        file_time = save_dataframe_to_csv(df_time, f"{base_filename}_time_series.csv")
        output_files['time_series'] = file_time
        print(f"📈 Time series sample:\n{df_time.head()}\n")
    except Exception as e:
        print(f"❌ Time series failed: {e}")
    
    # 3. Spatial average (time averaged)
    try:
        df_spatial = convert_to_spatial_dataframe(ds, time_method='mean')
        file_spatial = save_dataframe_to_csv(df_spatial, f"{base_filename}_spatial_average.csv")
        output_files['spatial_average'] = file_spatial
        print(f"🗺️  Spatial average sample:\n{df_spatial.head()}\n")
    except Exception as e:
        print(f"❌ Spatial average failed: {e}")
    
    return output_files

def quick_data_summary(df):
    """Generate a quick summary of the DataFrame"""
    print("📊 Data Summary")
    print("=" * 20)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())

# Main function to use with your dataset
def process_my_data(ds, filename = "london_era5_precipitation"):
    """
    Main function to process your specific dataset
    Call this with your loaded dataset: process_my_data(ds)
    """
    
    print("🌧️  Processing Your London Precipitation Data")
    print("=" * 50)
    
    # First, explore the structure
    explore_dataset_structure(ds)
    
    print("\n" + "="*50)
    
    # Create multiple formats
    output_files = create_multiple_csv_formats(ds, filename)
    
    print(f"\n🎉 Processing complete! Created {len(output_files)} files:")
    for format_name, filepath in output_files.items():
        print(f"   📄 {format_name}: {filepath}")
    
    # Return the long format DataFrame for immediate use
    if 'long_format' in output_files:
        df = pd.read_csv(output_files['long_format'], comment='#')
        print(f"\n💡 Returning long format DataFrame with {len(df)} rows")
        return df
    else:
        print("⚠️  Returning time series DataFrame as fallback")
        return convert_to_time_series_dataframe(ds)