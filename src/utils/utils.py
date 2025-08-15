import torch
import numpy as np
import logging
import yaml
import joblib
import pathlib as pl
import glob
import subprocess
from netCDF4 import Dataset
from datetime import datetime, timedelta
import netCDF4 as nc
import pandas as pd

def load_scalers(filename):
      
      try:
        scalers = joblib.load(filename)
        return scalers
          
      except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def unscale_pred(model,test_data,conf_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions =  evaluate_with_loader(model, test_data,device)
    
    with open(conf_path, 'r') as file:
            conf = yaml.safe_load(file)
        
    y_scaler = load_scalers(conf['yscaler'])
    
  
    predictions_reshaped = predictions.reshape(-1, 390 * 348)
    
    # Inverse transform the scaled predictions
    predictions_original = y_scaler[0].inverse_transform(predictions_reshaped)
    
    # Reshape back to the original dimensions
    predictions_original = predictions_original.reshape(*predictions.shape)
    return predictions_original

def write_to_netcdf(output_file, start_date, end_date, lat_data, lon_data, data):
    # Define the reference time
    reference_time = pd.Timestamp('2000-01-01')

    # Create a date range
    date_range = pd.date_range(start_date, end_date, freq='1D')

    # Convert dates to days since reference time
    time_values = (date_range - reference_time).days

    # Open the NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as dataset:
        # Define dimensions
        time_dim = dataset.createDimension('XTIME', len(time_values))
        lat_dim = dataset.createDimension('south_north', len(lat_data))
        lon_dim = dataset.createDimension('west_east', len(lon_data))

        # Create coordinate variables
        times = dataset.createVariable('XTIME', 'i4', ('XTIME',))  # 'i4' for integer type
        lats = dataset.createVariable('south_north', 'f4', ('south_north',))
        lons = dataset.createVariable('west_east', 'f4', ('west_east',))

        # Create the main variable
        main_var = dataset.createVariable('SNOW', 'f4', ('XTIME', 'south_north', 'west_east'))

        # Write data to variables
        times[:] = time_values
        lats[:] = lat_data
        lons[:] = lon_data
        main_var[:, :, :] = data

        # Add attributes
        times.units = f'days since {reference_time.strftime("%Y-%m-%d")}'
        times.calendar = 'gregorian'  # Optionally add calendar attribute if using other calendars
        lats.units = 'degrees_north'
        lons.units = 'degrees_east'
        main_var.units = 'mm'  # or appropriate units for your data

def data_split(start:str, end:str):
    assert int(start) <= int(end), f'Time step should be incremental'


    years = [int(start) + i for i in range(int(end) + 1 - int(start)) ]
    
    file_paths = []
    for yr in years:
        file_paths.extend(sorted(glob.glob(f'/bsuscratch/stanleyakor/uppercolorado/WY{yr}/wrf*')))

    x = xr.open_mfdataset(file_paths,  concat_dim = 'XTIME',combine='nested', parallel = True)

    return x
