import rasterio
import numpy as np
import xarray as xr
import glob
from datetime import datetime
from rasterio.warp import reproject, Resampling
import os

class ModisDataProcessor:
    """
    Processes  MODIS snow data from GeoTIFF files with cloud fraction filtering.
    Converts snow GeoTIFFs to NetCDF while handling cloudy days.
    """

    def __init__(self, 
                 snow_pattern, 
                 cloud_pattern, 
                 output_file, 
                 ideal_shape=(390, 348), 
                 cloud_threshold=40):
        """
        Initializes the processor with file patterns, output, and parameters.

        Args:
            snow_pattern (str): Glob pattern for snow GeoTIFF files.
            cloud_pattern (str): Glob pattern for cloud GeoTIFF files.
            output_file (str): Output NetCDF file path.
            ideal_shape (tuple): Target shape for reprojected data.
            cloud_threshold (int): Max cloud fraction (%) to consider a day 'clear'.
        """
        self.snow_pattern = snow_pattern
        self.cloud_pattern = cloud_pattern
        self.output_file = output_file
        self.ideal_shape = ideal_shape
        self.cloud_threshold = cloud_threshold
        self.ref_transform = None
        self.ref_crs = None
        print(f"Initialized SnowDataProcessor with cloud threshold {cloud_threshold}%")

    def _get_reference_geotiff_info(self, geotiff_files):
        """Set reference CRS and transform from the first GeoTIFF."""
        if not geotiff_files:
            raise ValueError("No GeoTIFF files found for reference.")
        with rasterio.open(geotiff_files[0]) as src:
            self.ref_transform = src.transform
            self.ref_crs = src.crs

    def check_cloudfrac(self):
        """Return list of dates with cloud fraction below the threshold."""
        geotiff_files = sorted(glob.glob(self.cloud_pattern))

        if not geotiff_files:
            print(f"Warning: No cloud GeoTIFFs found at {self.cloud_pattern}")
            return []

        clear_days = []
        for file in geotiff_files:
            with rasterio.open(file) as src:
                img = src.read(1)
                percent_clear = (np.sum(img == 1) / img.size) * 100
                if 0 < percent_clear <= self.cloud_threshold:
                    try:
                        date_str = os.path.basename(file).split('_')[1].split('.')[0]
                        datetime.strptime(date_str, '%Y-%m-%d')  # validate
                        clear_days.append(date_str)
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Cannot parse date from {file}: {e}")
        return clear_days

    def find_nearest_date(self, target_date_str, date_list_str):
        """Return the nearest date from a list to the target date."""
        if not date_list_str:
            raise ValueError("Date list cannot be empty.")
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in date_list_str]
        nearest = min(dates, key=lambda d: abs(d - target_date))
        return nearest.strftime('%Y-%m-%d')

    def convert_tif_netcdf(self):
        """Convert snow GeoTIFFs to NetCDF with cloud-aware selection."""
        snow_files = sorted(glob.glob(self.snow_pattern))
        if not snow_files:
            raise ValueError(f"No snow GeoTIFFs found at {self.snow_pattern}")

        self._get_reference_geotiff_info(snow_files)
        clear_days = self.check_cloudfrac()

        data_list = []
        time_list = []

        for snow_file in snow_files:
            time_str = os.path.basename(snow_file).split('.')[0]
            selected_date = time_str

            if clear_days and time_str not in clear_days:
                selected_date = self.find_nearest_date(time_str, clear_days)

            snow_file_to_open = os.path.join(os.path.dirname(snow_file), f"{selected_date}.tif")
            if not os.path.exists(snow_file_to_open):
                print(f"Error: Snow file {snow_file_to_open} not found. Skipping.")
                continue

            with rasterio.open(snow_file_to_open) as src:
                data = src.read(1)
                reprojected_data = np.empty(self.ideal_shape, dtype=data.dtype)
                reproject(
                    source=data,
                    destination=reprojected_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.ref_transform,
                    dst_crs=self.ref_crs,
                    resampling=Resampling.bilinear
                )
                data_list.append(reprojected_data)
                time_list.append(time_str)

        if not data_list:
            print("No data processed. NetCDF file will not be created.")
            return

        da = xr.DataArray(
            data_list,
            dims=['XTIME', 'south_north', 'west_east'],
            coords={'XTIME': time_list}
        )
       
        ds = xr.Dataset({'snow_cover': da})
        print(f"NetCDF file created: {self.output_file}")
        return ds
        # ds.to_netcdf(self.output_file)
        


# processor = SnowDataProcessor(
#     snow_pattern="snowcover/*",
#     cloud_pattern="cloud/*",
#     output_file="snow_data.nc",
#     ideal_shape=(390, 348),
#     cloud_threshold=35
# )

# ds = processor.convert_tif_netcdf()