import xarray as xr
import numpy as np

# SPLIT DATASET INTO TRAIN, VAL AND TEST SETS
def split_by_time(dataset):
    """
    Split dataset into train, val, and test using predefined time ranges.

    Returns:
    --------
    dict of xr.Dataset: {'train', 'val', 'test'}
    """
    train = dataset.sel(XTIME=slice("2005-10-01", "2014-09-30"))
    val = dataset.sel(XTIME=slice("2014-10-01", "2015-09-30"))
    test = dataset.sel(XTIME=slice("2015-10-01", "2016-09-30"))
    return {'train': train, 'val': val, 'test': test}

