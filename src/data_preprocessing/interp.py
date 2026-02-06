import rasterio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
import numpy as np
import xarray as xr
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling
import os
import glob
from matplotlib.colors import ListedColormap
from datetime import datetime
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d

def interpolate_netcdf_daily(nc_data, output_file, time_var_name, data_var_name, time_format=None):
    """
    Interpolates NetCDF data to a daily time resolution and saves to a new NetCDF file.

    Parameters:
    - input_file (str): Path to the input NetCDF file.
    - output_file (str): Path to the output NetCDF file.
    - time_var_name (str): Name of the time variable in the NetCDF file.
    - data_var_name (str): Name of the data variable to interpolate.
    - time_format (str): Format string for converting time to pandas datetime. If None, assume the time is already in datetime format.

    Returns:
    - None
    """
    
    
    time = nc_data.variables[time_var_name][:]
    data = nc_data.variables[data_var_name][:]
    
    
    if time_format:
       
        time_dates = pd.to_datetime(time, format=time_format)
    else:
        #
        try:
            time_dates = pd.to_datetime(time)
        except Exception as e:
            raise ValueError(f"Failed to convert time to datetime: {e}")
    
    nc_data.close()
    
    start_date = time_dates.min()
    end_date = time_dates.max()
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Convert original dates to numeric values for interpolation
    time_numeric = pd.to_numeric(time_dates)

   
    interp_func = interp1d(time_numeric, data, kind='linear', axis=0, fill_value='extrapolate')

    
    full_dates_numeric = pd.to_numeric(full_dates)

    # Interpolate data
    interpolated_data = interp_func(full_dates_numeric)

    
    data_array = xr.DataArray(interpolated_data, dims=['XTIME', 'south_north', 'west_east'], coords={'XTIME': full_dates})
    dataset = xr.Dataset({'LAI': data_array})

    return dataset


output_file = '../data/lai.nc'
time_var_name = 'XTIME'
data_var_name = 'LAI'

input_file = xr.open_dataset('../data/lai_regrided.nc')

data_Var = interpolate_netcdf_daily(input_file, output_file, time_var_name,data_var_name)

data_Var.to_netcdf(output_file)