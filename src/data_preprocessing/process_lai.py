import os
import glob
import rasterio
import numpy as np
import xarray as xr
import pandas as pd
from rasterio.warp import reproject, Resampling
from scipy.interpolate import interp1d

class LAIProcessor:
    """
    A class to process LAI GeoTIFF files:
    - Convert to NetCDF
    - Interpolate to daily resolution
    - Save to file
    """

    def __init__(self, tif_dir, output_file,
                 ideal_shape=(390, 348),
                 time_var_name="XTIME",
                 data_var_name="LAI",
                 time_format=None):
        """
        Initialize the processor with all required inputs.

        Args:
            tif_dir (str): Directory path containing GeoTIFFs.
            output_file (str): Path to save the NetCDF file.
            ideal_shape (tuple): Shape (height, width) for reprojection.
            time_var_name (str): Time variable name.
            data_var_name (str): Data variable name.
            time_format (str): Optional datetime format for parsing.
        """
        self.tif_dir = tif_dir
        self.output_file = output_file
        self.ideal_shape = ideal_shape
        self.time_var_name = time_var_name
        self.data_var_name = data_var_name
        self.time_format = time_format

        self.dataset = None
        self.ref_transform = None
        self.ref_crs = None

    def convert_tif_to_netcdf(self):
        """Convert LAI GeoTIFF files into an xarray Dataset."""
        geotiff_files = sorted(glob.glob(os.path.join(self.tif_dir, '*')))
        
        if not geotiff_files:
            raise ValueError(f"No GeoTIFF files found in {self.tif_dir}")

        data_list, time_list = [], []

        # Use first file as reference
        with rasterio.open(geotiff_files[0]) as src:
            self.ref_transform = src.transform
            self.ref_crs = src.crs

        for geotiff_file in geotiff_files:
            with rasterio.open(geotiff_file) as src:
                data = src.read(1)
                time = os.path.basename(geotiff_file).split('_')[1].split('.')[0]
               

                reprojected_data = np.empty(self.ideal_shape, dtype=data.dtype)
                reproject(
                    source=data,
                    destination=reprojected_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.ref_transform,
                    dst_crs=self.ref_crs,
                    resampling=Resampling.bilinear,
                )

                data_list.append(reprojected_data)
                time_list.append(time)

        da = xr.DataArray(
            data_list,
            dims=[self.time_var_name, 'south_north', 'west_east'],
            coords={self.time_var_name: time_list},
        )
        self.dataset = xr.Dataset({self.data_var_name: da})
        return self.dataset

    def interpolate_daily(self):
        _ = self.convert_tif_to_netcdf()
        """Interpolate dataset to daily resolution."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Run convert_tif_to_netcdf first.")

        time = self.dataset.variables[self.time_var_name].values
        data = self.dataset.variables[self.data_var_name].values

        # Convert time to datetime
        if self.time_format:
            time_dates = pd.to_datetime(time, format=self.time_format)
        else:
            try:
                time_dates = pd.to_datetime(time)
            except Exception as e:
                raise ValueError(f"Failed to convert time to datetime: {e}")

        start_date, end_date = time_dates.min(), time_dates.max()
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Convert dates to numeric for interpolation
        time_numeric = pd.to_numeric(time_dates)
        interp_func = interp1d(
            time_numeric, data, kind='linear', axis=0, fill_value='extrapolate'
        )
        full_dates_numeric = pd.to_numeric(full_dates)

        interpolated_data = interp_func(full_dates_numeric)

        da = xr.DataArray(
            interpolated_data,
            dims=[self.time_var_name, 'south_north', 'west_east'],
            coords={self.time_var_name: full_dates},
        )
        return xr.Dataset({self.data_var_name: da})
