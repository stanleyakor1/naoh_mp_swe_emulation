
import numpy as np
import xarray as xr
import rioxarray as rxr
import py3dep
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# ----------------------------
# Load MODIS NetCDF (target grid)
# ----------------------------
path_to_data = '/bsuscratch/stanleyakor/nscid/wy/wy2014/sca_full_20131001_20140930.nc'
ds = xr.open_dataset(path_to_data)
tpl = ds["NDSI_Snow_Cover"].isel(time=0).astype("float32")

lats = tpl.lat.values
lons = tpl.lon.values

dx = float(np.abs(np.nanmedian(np.diff(lons))))
dy = float(np.abs(np.nanmedian(np.diff(lats))))
transform = from_origin(float(lons.min()-dx/2), float(lats.max()+dy/2), dx, dy)

tpl = tpl.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
tpl = tpl.rio.write_crs("EPSG:4326", inplace=False)
tpl = tpl.rio.write_transform(transform, inplace=False)

# bbox in EPSG:4326
bbox4326 = (float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()))

# ----------------------------
# Download DEM
# ----------------------------
dem = py3dep.get_dem(bbox4326, resolution=30)
dem.name = "elevation"
dem.attrs["units"] = "m"

# Ensure DEM has CRS
if not dem.rio.crs:
    dem = dem.rio.write_crs("EPSG:4326")

print("DEM original CRS:", dem.rio.crs)


dem_4326 = dem.rio.reproject("EPSG:4326", resampling=Resampling.bilinear)


dem_4326 = dem_4326.rio.clip_box(*bbox4326)

# ----------------------------
# Regrid DEM -> MODIS grid
# ----------------------------
dem_on_grid = dem_4326.rio.reproject_match(tpl, resampling=Resampling.average)
dem_on_grid.name = "elevation"
dem_on_grid.attrs["units"] = "m"

# Save
dem_on_grid.to_dataset(name="elevation").to_netcdf("/bsuscratch/stanleyakor/nscid/dem/dem_on_modis_grid.nc")


print("Wrote: dem_on_modis_grid.nc")

