import glob
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import yaml
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


# DATA NORMALIZATION ROUTINE USING A MINMAXSCALER OR ZSCORE
class XarrayNormalizer:
    """
    A class to normalize selected variables in an xarray.Dataset using MinMaxScaler or StandardScaler.
    """

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize the normalizer with a dataset.

        Parameters:
        -----------
        dataset : xr.Dataset
            Dataset with dimensions (XTIME, south_north, west_east) per variable.
        """
        self.dataset = dataset
        self.scalers = {}

    def fit_transform(self, variables=None, method='minmax', save_scaler_path=None):
        """
        Fit scalers on selected variables and transform the data.

        Parameters:
        -----------
        variables : list of str, optional
            Variable names to normalize. If None, all variables are used.
        method : str
            'minmax' or 'zscore' normalization method.
        save_scaler_path : str or None
            Path to save the scalers (e.g., 'scalers.pkl').

        Returns:
        --------
        xr.Dataset
            A new dataset with the selected variables normalized.
        """
        if variables is None:
            variables = list(self.dataset.data_vars)

        scaler_cls = MinMaxScaler if method == 'minmax' else StandardScaler
        normalized_data = {}

        for var in variables:
            data = self.dataset[var].values
            reshaped = data.reshape(-1, 1)
            scaler = scaler_cls()
            scaled = scaler.fit_transform(reshaped)
            normalized_data[var] = (self.dataset[var].dims, scaled.reshape(data.shape))
            self.scalers[var] = scaler

        norm_ds = self.dataset.copy()
        for var, (dims, data) in normalized_data.items():
            norm_ds[var] = xr.DataArray(data, dims=dims, coords=self.dataset[var].coords)

        if save_scaler_path:
            joblib.dump(self.scalers, save_scaler_path)

        return norm_ds

    def transform(self, variables, load_scaler_path):
        """
        Apply pre-fitted scalers to selected variables.

        Parameters:
        -----------
        variables : list of str
            Variable names to normalize.
        load_scaler_path : str
            Path to the saved scalers file (e.g., 'scalers.pkl').

        Returns:
        --------
        xr.Dataset
            A new dataset with the selected variables normalized.
        """
        self.scalers = joblib.load(load_scaler_path)
        transformed_data = {}

        for var in variables:
            if var not in self.scalers:
                raise ValueError(f"Scaler for '{var}' not found in saved scalers.")
            data = self.dataset[var].values
            reshaped = data.reshape(-1, 1)
            scaled = self.scalers[var].transform(reshaped)
            transformed_data[var] = (self.dataset[var].dims, scaled.reshape(data.shape))

        norm_ds = self.dataset.copy()
        for var, (dims, data) in transformed_data.items():
            norm_ds[var] = xr.DataArray(data, dims=dims, coords=self.dataset[var].coords)

        return norm_ds
