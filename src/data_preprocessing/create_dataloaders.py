import numpy as np
import xarray as xr
import yaml
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# CREATE A SWE DATALOADER FROM THE NORMALIZED DATASET
class SweDataset(Dataset):
    """
    A PyTorch Dataset for SWE data stored in xarray.Dataset format.
    Handles feature/target separation, sequence slicing, and channel reordering.
    """

    def __init__(self, feature_ds, target_ds=None, sequence_length=7, variable_order=None):
        """
        Initialize the SWE Dataset.

        Parameters:
        -----------
        feature_ds : xr.Dataset
            xarray dataset containing all input features.
        target_ds : xr.Dataset or None
            xarray dataset containing targets (optional for test mode).
        sequence_length : int
            Number of timesteps to use in each sequence.
        variable_order : list of str or None
            Optional reordering of variables (channel order).
        """
        self.sequence_length = sequence_length

        # Select and reorder features
        if variable_order is None:
            variable_order = list(feature_ds.data_vars)

        self.x = self._stack_and_permute(feature_ds, variable_order)
        
        
        if target_ds is not None:
            self.y = torch.tensor(target_ds.to_array().values[0], dtype=torch.float32)
            self.y = self.y.unsqueeze(1)
        else:
            self.y = torch.zeros((self.x.shape[0], 1, self.x.shape[2], self.x.shape[3]))  # shape: (T, 1, H, W)

    def _stack_and_permute(self, dataset, variable_order):
        """
        Stack selected variables into one tensor and permute to (T, C, H, W).
        """
        arrays = [torch.tensor(dataset[var].values, dtype=torch.float32) for var in variable_order]
        stacked = torch.stack(arrays, dim=-1)  # (T, H, W, C)
        return stacked.permute(0, 3, 1, 2)     # (T, C, H, W)

    def __len__(self):
        return self.x.shape[0] - self.sequence_length

    def __getitem__(self, index):
        """
        Return a tuple of (input_sequence, target_frame).
        """
        return self.x[index:index + self.sequence_length], self.y[index + self.sequence_length]

