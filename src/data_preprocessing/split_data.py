import xarray as xr
import numpy as np

# SPLIT DATASET INTO TRAIN, VAL AND TEST SETS
def split_by_time(dataset, train_split, val_split, test_split):
    """
    Split dataset into train, val, and test using predefined time ranges.

    Returns:
    --------
    dict of xr.Dataset: {'train', 'val', 'test'}
    """
    train = dataset.sel(XTIME=slice(train_split[0], train_split[1]))
    val = dataset.sel(XTIME=slice(val_split[0], val_split[1]))
    test = dataset.sel(XTIME=slice(test_split[0], test_split[1]))
    return {'train': train, 'val': val, 'test': test}

