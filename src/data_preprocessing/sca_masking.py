
import xarray as xr
import numpy as np

path_to_data = 'MOD10A1_v061_bbox_20161001_20170930.nc' # enter path to netcdf file
data = xr.open_dataset(path_to_data)
mod_fsca = data['NDSI_Snow_Cover']

sca = xr.where(
    (mod_fsca >= 5) & (mod_fsca <= 100),
    1,
    xr.where(
        (mod_fsca >= 0) & (mod_fsca < 5),
        0,
        np.nan
    )
)

out_file_name = '/bsuscratch/stanleyakor/nscid/wy/wy2017/sca_20161001_20170930' # enter output dir and name
sca.to_netcdf(f'{out_file_name}.nc')
