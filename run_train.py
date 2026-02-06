import os
import logging
import xarray as xr
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
from src.data_preprocessing.normalization import XarrayNormalizer
from src.data_preprocessing.create_dataloaders import SweDataset
from src.data_preprocessing.split_data import split_by_time
from src.utils.utils import (write_to_netcdf, data_split, unscale_pred)
from src.training.trainer import (train_model,evaluate_model)
from src.models.swe_net import SWE_NET
from src.log.logg import setup_logging




setup_logging("INFO")

logger = logging.getLogger(__name__)


wy_start = "2002-10-01"
wy_end = "2017-09-30"

channel_order = ['PRCP_CUMSUM','TMIN', 'TMAX','ELEVATION']
# ------------------ Load and Combine Data ------------------
logger.info("Loading datasets...")
ds1 = xr.open_dataset('/bsuscratch/stanleyakor/nscid/data/wrf_forcings.nc')[channel_order]
ds2 = xr.open_dataset('/bsuscratch/stanleyakor/nscid/data/modis_snow_cover.nc')
ds3 = xr.open_dataset('/bsuscratch/stanleyakor/nscid/data/lai.nc').sel(XTIME=slice(wy_start, wy_end))
ds2['XTIME'] = pd.to_datetime(ds2['XTIME'].values)

logger.info("Merging datasets...")
data = xr.merge([ds1, ds2, ds3])
data = data.rename({"snow_mask": "BINARY_SNOW_CLASS"})


# ------------------ Data Splitting ------------------
logger.info("Splitting dataset by time...")
train_split = ("2002-10-01", "2013-09-30")
val_split = ("2013-10-01", "2015-09-30")
test_split =  ("2015-10-01", "2017-09-30")
split = split_by_time(data, train_split, val_split, test_split)



variables = ['PRCP_CUMSUM','TMIN', 'TMAX','ELEVATION', 'BINARY_SNOW_CLASS', 'LAI']
logger.info(f"Normalizing features: {variables}")
normalizer = XarrayNormalizer(split['train'])
train_features_norm = normalizer.fit_transform(
    variables=variables, 
    method="minmax", 
    save_scaler_path="/bsuscratch/stanleyakor/scalers/scalers.pkl"
)

normalizer_val = XarrayNormalizer(split['test'])
val_features_norm = normalizer_val.transform(
    variables=variables,
    load_scaler_path="/bsuscratch/stanleyakor/scalers/scalers.pkl"
)

# ------------------ Target Normalization ------------------
logger.info("Normalizing target...")
target_data = xr.open_dataset('/bsuscratch/stanleyakor/nscid/data/swe.nc')
split_target = split_by_time(target_data, train_split, val_split, test_split)

target_normalizer_train = XarrayNormalizer(split_target['train'])
train_target_norm = target_normalizer_train.fit_transform(
    variables=['SNOW'],
    method="minmax",
    save_scaler_path="/bsuscratch/stanleyakor/scalers/target_scalers.pkl"
)

target_normalizer_val = XarrayNormalizer(split_target['test'])
val_target_norm = target_normalizer_val.transform(
    variables=['SNOW'],
    load_scaler_path="/bsuscratch/stanleyakor/scalers/target_scalers.pkl"
)



# ------------------ Dataset and DataLoader ------------------
sequence_length = 5


channel_order = ['PRCP_CUMSUM','TMIN', 'TMAX','ELEVATION', 'BINARY_SNOW_CLASS', 'LAI']

logger.info("Setting up train/test DataLoaders...")
train_dataset = SweDataset(train_features_norm, train_target_norm, sequence_length, channel_order)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True) # batch_size = 8

test_dataset = SweDataset(val_features_norm, val_target_norm, sequence_length, channel_order)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------ Model Training ------------------
logger.info("Instantiating SWE_NET model...")
model = SWE_NET(input_dim=len(channel_order), hidden_dim=64, kernel_size=(3, 3),
                height=390, width=348, dropout_rate=0.3)

logger.info("Starting model training...")
model = train_model(model, train_loader, num_epochs=3, lr=1e-4,
                    checkpoint_path="saved_models/atmoshpheric_forcings_only_180.pth")

# ------------------ Prediction ------------------
logger.info("Making predictions on validation set...")
unscaled = unscale_pred(model, test_loader, "/bsuscratch/stanleyakor/scalers/target_scalers.pkl")

# ------------------ Write to NetCDF ------------------
logger.info("Saving predictions to NetCDF...")
static = xr.open_dataset('/bsuscratch/stanleyakor/uppercolorado/static_inputs/wrfout_d02_2000-04-08_00:00:00').isel(Time=0)
lat = static.XLAT.values[:, 0]
lon = static.XLONG.values[0, :]
fstart = f'2015-10-{sequence_length + 1}'
fend = '2017-09-30'
save_name = "data/swe"

write_to_netcdf(f'{save_name}_pred.nc', fstart, fend, lat, lon, unscaled)

# ------------------ Evaluation ------------------
logger.info("Loading true labels and computing spatial correlation...")
true_labels = data_split('2015', '2017').sel(XTIME=slice(fstart, fend))
pred = xr.open_dataset(f'{save_name}_pred.nc')
correlation = xr.corr(true_labels['SNOW'], pred['SNOW'], dim='XTIME')

logger.info("Writing correlation results to NetCDF...")
write_to_netcdf(f'{save_name}_correlation.nc', '2016-10-01', '2016-10-01', lat, lon, correlation)

# ------------------ Final Evaluation ------------------
logger.info("Computing mean time-series correlation...")
em_model = pred['SNOW'].mean(dim=('south_north', 'west_east'))
wrf_hydro = true_labels['SNOW'].mean(dim=('south_north', 'west_east'))
cor = np.corrcoef(wrf_hydro, em_model)[0, 1]
r2_value = r2_score(wrf_hydro, em_model)

logger.info(f"✅ R-squared = {cor ** 2:.4f}, Correlation = {cor:.4f}")
logger.info(f"✅ R-squared sklearn, Correlation = {r2_value:.4f}")
logger.info("🎉 HPC job completed successfully.")
