import os
import logging
import yaml
import xarray as xr
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from src.data_preprocessing.normalization import XarrayNormalizer
from src.data_preprocessing.create_dataloaders import SweDataset
from src.data_preprocessing.split_data import split_by_time
from src.utils.utils import write_to_netcdf, data_split, unscale_pred
from src.training.trainer import train_model
from src.models.swe_net import SWE_NET
from src.log.logg import setup_logging


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_parent_dir(filepath: str) -> None:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_fstart(base_month: str, sequence_length: int, day_offset_from_seq: int) -> str:
    """
    base_month: "YYYY-MM"
    fstart day = sequence_length + day_offset_from_seq
    returns "YYYY-MM-DD"
    """
    day = int(sequence_length) + int(day_offset_from_seq)
    return f"{base_month}-{day:02d}"


def main(config_path: str):
    cfg = load_config(config_path)

    # ------------------ Logging ------------------
    setup_logging(cfg["logging"]["level"])
    logger = logging.getLogger(__name__)

    wy_start = cfg["time"]["wy_start"]
    wy_end = cfg["time"]["wy_end"]

    # ------------------ Load and Combine Data ------------------
    logger.info("Loading datasets...")

    ds1_path = cfg["data"]["inputs"]["wrf_forcings_nc"]
    ds2_path = cfg["data"]["inputs"]["modis_snow_cover_nc"]
    ds3_path = cfg["data"]["inputs"]["lai_nc"]

    wrf_channels = cfg["data"]["inputs"]["wrf_channel_order"]

    ds1 = xr.open_dataset(ds1_path)[wrf_channels]
    ds2 = xr.open_dataset(ds2_path)
    ds3 = xr.open_dataset(ds3_path).sel(XTIME=slice(wy_start, wy_end))

    if "XTIME" in ds2:
        ds2["XTIME"] = pd.to_datetime(ds2["XTIME"].values)

    logger.info("Merging datasets...")
    data = xr.merge([ds1, ds2, ds3])

    rename_map = cfg["data"].get("rename_vars", {})
    if rename_map:
        data = data.rename(rename_map)

    # ------------------ Data Splitting ------------------
    logger.info("Splitting dataset by time...")
    train_split = tuple(cfg["splits"]["train"])
    val_split = tuple(cfg["splits"]["val"])
    test_split = tuple(cfg["splits"]["test"])

    split = split_by_time(data, train_split, val_split, test_split)

    # ------------------ Feature Normalization ------------------
    variables = cfg["normalization"]["features"]["variables"]
    feat_method = cfg["normalization"]["features"]["method"]
    feat_scaler_path = cfg["normalization"]["features"]["scaler_path"]

    _ensure_parent_dir(feat_scaler_path)

    logger.info(f"Normalizing features: {variables}")
    normalizer_train = XarrayNormalizer(split["train"])
    train_features_norm = normalizer_train.fit_transform(
        variables=variables,
        method=feat_method,
        save_scaler_path=feat_scaler_path,
    )

    feat_eval_key = cfg["normalization"]["features"]["eval_split_key"]
    normalizer_eval = XarrayNormalizer(split[feat_eval_key])
    val_features_norm = normalizer_eval.transform(
        variables=variables,
        load_scaler_path=feat_scaler_path,
    )

    # ------------------ Target Normalization ------------------
    logger.info("Normalizing target...")
    target_path = cfg["data"]["targets"]["swe_nc"]
    target_var = cfg["data"]["targets"]["variable"]

    target_data = xr.open_dataset(target_path)
    split_target = split_by_time(target_data, train_split, val_split, test_split)

    targ_method = cfg["normalization"]["target"]["method"]
    targ_scaler_path = cfg["normalization"]["target"]["scaler_path"]
    _ensure_parent_dir(targ_scaler_path)

    target_normalizer_train = XarrayNormalizer(split_target["train"])
    train_target_norm = target_normalizer_train.fit_transform(
        variables=[target_var],
        method=targ_method,
        save_scaler_path=targ_scaler_path,
    )

    targ_eval_key = cfg["normalization"]["target"]["eval_split_key"]
    target_normalizer_eval = XarrayNormalizer(split_target[targ_eval_key])
    val_target_norm = target_normalizer_eval.transform(
        variables=[target_var],
        load_scaler_path=targ_scaler_path,
    )

    # ------------------ Dataset and DataLoader ------------------
    sequence_length = int(cfg["dataloader"]["sequence_length"])
    channel_order = cfg["dataloader"]["channel_order"]

    logger.info("Setting up DataLoaders...")
    train_dataset = SweDataset(train_features_norm, train_target_norm, sequence_length, channel_order)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["dataloader"]["train"]["batch_size"]),
        shuffle=bool(cfg["dataloader"]["train"]["shuffle"]),
        drop_last=bool(cfg["dataloader"]["train"]["drop_last"]),
        num_workers=int(cfg["dataloader"]["train"].get("num_workers", 0)),
        pin_memory=bool(cfg["dataloader"]["train"].get("pin_memory", False)),
    )

    eval_dataset = SweDataset(val_features_norm, val_target_norm, sequence_length, channel_order)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["dataloader"]["eval"]["batch_size"]),
        shuffle=bool(cfg["dataloader"]["eval"]["shuffle"]),
        drop_last=bool(cfg["dataloader"]["eval"].get("drop_last", False)),
        num_workers=int(cfg["dataloader"]["eval"].get("num_workers", 0)),
        pin_memory=bool(cfg["dataloader"]["eval"].get("pin_memory", False)),
    )

    # ------------------ Model Training ------------------
    logger.info("Instantiating SWE_NET model...")
    mcfg = cfg["model"]
    model = SWE_NET(
        input_dim=len(channel_order),
        hidden_dim=int(mcfg["hidden_dim"]),
        kernel_size=tuple(mcfg["kernel_size"]),
        height=int(mcfg["height"]),
        width=int(mcfg["width"]),
        dropout_rate=float(mcfg["dropout_rate"]),
    )

    tcfg = cfg["training"]
    ckpt_path = tcfg["checkpoint_path"]
    _ensure_parent_dir(ckpt_path)

    logger.info("Starting model training...")
    model = train_model(
        model,
        train_loader,
        num_epochs=int(tcfg["num_epochs"]),
        lr=float(tcfg["lr"]),
        checkpoint_path=ckpt_path,
    )

    # ------------------ Prediction (unscale) ------------------
    logger.info("Making predictions on eval set...")
    unscaled = unscale_pred(model, eval_loader, targ_scaler_path)

    # ------------------ Write predictions to NetCDF ------------------
    out_cfg = cfg["outputs"]
    save_name = out_cfg["save_name_prefix"]  # e.g. "data/swe"
    pred_nc = f"{save_name}_pred.nc"

    static_path = cfg["data"]["static"]["wrf_static_file"]
    static_time_index = int(cfg["data"]["static"].get("time_index", 0))

    logger.info("Loading static lat/lon...")
    static = xr.open_dataset(static_path).isel(Time=static_time_index)
    lat = static[cfg["data"]["static"]["lat_var"]].values[:, 0]
    lon = static[cfg["data"]["static"]["lon_var"]].values[0, :]

    base_month = out_cfg["pred_time"]["base_month"]           # "YYYY-MM"
    offset = int(out_cfg["pred_time"]["day_offset_from_seq"]) # e.g. 1
    fend = out_cfg["pred_time"]["end"]                        # "YYYY-MM-DD"
    fstart = build_fstart(base_month, sequence_length, offset)

    _ensure_parent_dir(pred_nc)

    logger.info("Saving predictions to NetCDF...")
    write_to_netcdf(pred_nc, fstart, fend, lat, lon, unscaled)

    # ------------------ Evaluation: spatial correlation ------------------
    logger.info("Computing spatial correlation...")
    eval_true = data_split(out_cfg["eval_period"]["true_start_year"], out_cfg["eval_period"]["true_end_year"])
    eval_true = eval_true.sel(XTIME=slice(fstart, fend))

    pred = xr.open_dataset(pred_nc)
    correlation = xr.corr(eval_true[target_var], pred[target_var], dim="XTIME")

    corr_nc = f"{save_name}_correlation.nc"
    _ensure_parent_dir(corr_nc)

    corr_time = out_cfg["correlation_time"]
    logger.info("Writing correlation results to NetCDF...")
    write_to_netcdf(corr_nc, corr_time, corr_time, lat, lon, correlation)

    # ------------------ Final Evaluation: basin-mean correlation ------------------
    logger.info("Computing mean time-series correlation...")
    spatial_dims = tuple(out_cfg["spatial_dims"])
    em_model = pred[target_var].mean(dim=spatial_dims)
    wrf_hydro = eval_true[target_var].mean(dim=spatial_dims)
    r2_value = r2_score(wrf_hydro, em_model)

    logger.info(f"✅ R-squared sklearn, Correlation = {r2_value:.4f}")
    logger.info("🎉 HPC job completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)
